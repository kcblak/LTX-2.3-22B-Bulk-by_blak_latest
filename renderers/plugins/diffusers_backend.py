import gc
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from config import Config
from core import (
    AspectRatio,
    IRenderer,
    RenderMetrics,
    RenderParams,
    RenderResult,
    RenderedClip,
    RendererEncodingError,
    RendererInitializationError,
    RendererInputError,
    RendererOOMError,
    RendererOutputValidationError,
    RendererSessionInfo,
    Resolution,
    SUPPORTED_IMAGE_FORMATS,
)
from logging_system import get_logger
from renderers.base import register_renderer_plugin
from utils import compute_file_hash

logger = get_logger("render")


def _diffusers_available(_config: Config) -> bool:
    return True


def _diffusers_auto_priority(_config: Config) -> int:
    return 10


@register_renderer_plugin(
    name="diffusers",
    description="Diffusers-based LTX backend",
    availability_check=_diffusers_available,
    auto_priority_resolver=_diffusers_auto_priority,
)
class LTXVideoRenderer(IRenderer):
    """Reusable LTX rendering session optimized for long-running Kaggle jobs."""

    def __init__(self, config: Config):
        self.config = config
        self.pipeline = None
        self.device = self._get_device()
        self.dtype = self._select_dtype()
        self.session_info: Optional[RendererSessionInfo] = None
        self.render_count = 0

    def _get_device(self) -> torch.device:
        if self.config.use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        if self.config.use_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _select_dtype(self) -> torch.dtype:
        if self.device.type == "cuda":
            if self.config.precision.lower() == "fp16":
                return torch.float16
            if self.config.precision.lower() == "fp32":
                return torch.float32
            major, _minor = torch.cuda.get_device_capability()
            return torch.bfloat16 if major >= 8 else torch.float16
        if self.device.type == "mps":
            return torch.float16
        return torch.float32

    def _get_dimensions(
        self, resolution: Resolution, aspect_ratio: AspectRatio
    ) -> tuple[int, int]:
        base = resolution.pixels
        if aspect_ratio == AspectRatio.AR_16_9:
            width, height = base * 16 // 9, base
        elif aspect_ratio == AspectRatio.AR_9_16:
            width, height = base, base * 16 // 9
        else:
            width, height = base, base

        alignment = max(1, int(self.config.resolution_alignment))
        width = max(alignment, (width // alignment) * alignment)
        height = max(alignment, (height // alignment) * alignment)
        return width, height

    def _get_vram_stats(self) -> tuple[int, int]:
        if self.device.type != "cuda":
            return 0, 0
        total = torch.cuda.get_device_properties(0).total_memory
        free, _ = torch.cuda.mem_get_info()
        return total, free

    def _apply_runtime_optimizations(self) -> None:
        if self.device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(
                self.config.enable_memory_efficient_attention
            )
            torch.backends.cuda.enable_math_sdp(True)

        if (
            hasattr(self.pipeline, "enable_attention_slicing")
            and self.config.enable_attention_slicing
        ):
            self.pipeline.enable_attention_slicing()
        if hasattr(self.pipeline, "enable_vae_slicing") and self.config.enable_vae_slicing:
            self.pipeline.enable_vae_slicing()
        if hasattr(self.pipeline, "enable_vae_tiling") and self.config.enable_vae_tiling:
            self.pipeline.enable_vae_tiling()
        if (
            self.device.type == "cuda"
            and self.config.enable_model_cpu_offload
            and hasattr(self.pipeline, "enable_model_cpu_offload")
        ):
            self.pipeline.enable_model_cpu_offload()
        elif (
            self.device.type == "cuda"
            and self.config.enable_sequential_cpu_offload
            and hasattr(self.pipeline, "enable_sequential_cpu_offload")
        ):
            self.pipeline.enable_sequential_cpu_offload()
        elif hasattr(self.pipeline, "to"):
            self.pipeline = self.pipeline.to(self.device)

    def _warmup(self) -> bool:
        if not self.config.warmup_enabled or self.pipeline is None:
            return False

        try:
            warmup_size = max(
                self.config.resolution_alignment * 8,
                min(320, self.config.resolution_alignment * 10),
            )
            warmup_img = Image.new("RGB", (warmup_size, warmup_size), color=(0, 0, 0))
            generator = (
                torch.Generator(device=self.device).manual_seed(0)
                if self.device.type in {"cuda", "mps"}
                else torch.Generator().manual_seed(0)
            )
            _ = self.pipeline(
                prompt="warmup",
                image=warmup_img,
                num_inference_steps=self.config.warmup_steps,
                guidance_scale=1.0,
                width=warmup_size,
                height=warmup_size,
                num_frames=self.config.warmup_num_frames,
                generator=generator,
            )
            self._memory_housekeeping(force_empty_cache=False)
            return True
        except Exception as exc:
            logger.warning(
                f"Renderer warmup skipped after failure: {exc}",
                extra={"job_id": "N/A"},
            )
            return False

    def initialize(self) -> RendererSessionInfo:
        if self.session_info is not None and self.pipeline is not None:
            return self.session_info

        start = time.perf_counter()
        try:
            from assets import AssetManager
            from diffusers import LTXVideoPipeline

            logger.info(
                "Loading LTX rendering pipeline",
                extra={"job_id": "N/A"},
            )
            asset_report = AssetManager(self.config).ensure_assets(backend="diffusers")
            if not asset_report.ready:
                raise RendererInitializationError(
                    "Diffusers assets are not ready. See asset_report.json for details."
                )
            registry = ((self.config.extra or {}).get("model_registry") or {}).get("entries", {})
            model_entry = registry.get("diffusers_model")
            if model_entry and model_entry.get("status") == "found":
                model_path = model_entry["actual_path"]
            else:
                model_path = self.config.extra.get("diffusers_model_dir") or self.config.model_name
            self.pipeline = LTXVideoPipeline.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                local_files_only=True,
            )
            self._apply_runtime_optimizations()

            scheduler_name = None
            if getattr(self.pipeline, "scheduler", None) is not None:
                scheduler_name = self.pipeline.scheduler.__class__.__name__

            warmup_performed = self._warmup()
            total_vram, free_vram = self._get_vram_stats()
            self.session_info = RendererSessionInfo(
                renderer_name=self.__class__.__name__,
                device=str(self.device),
                precision=str(self.dtype).replace("torch.", ""),
                total_vram_bytes=total_vram,
                available_vram_bytes=free_vram,
                scheduler_name=scheduler_name,
                warmup_performed=warmup_performed,
                initialization_seconds=time.perf_counter() - start,
            )
            logger.info(
                (
                    "Renderer initialized "
                    f"device={self.session_info.device} "
                    f"precision={self.session_info.precision} "
                    f"scheduler={self.session_info.scheduler_name or 'unknown'} "
                    f"warmup={self.session_info.warmup_performed}"
                ),
                extra={"job_id": "N/A"},
            )
            return self.session_info
        except torch.cuda.OutOfMemoryError as exc:
            raise RendererInitializationError(
                f"Out of memory during renderer initialization: {exc}"
            ) from exc
        except Exception as exc:
            raise RendererInitializationError(
                f"Failed to initialize renderer: {exc}"
            ) from exc

    def validate_parameters(self, params: RenderParams) -> bool:
        if not params.prompt or not params.prompt.strip():
            raise RendererInputError("Prompt must not be empty")
        if not params.start_image.exists():
            raise RendererInputError(f"Start image not found: {params.start_image}")
        if params.end_image and not params.end_image.exists():
            raise RendererInputError(f"End image not found: {params.end_image}")
        if params.num_inference_steps <= 0:
            raise RendererInputError("num_inference_steps must be greater than zero")
        if params.guidance_scale <= 0:
            raise RendererInputError("guidance_scale must be greater than zero")
        if params.frame_rate <= 0:
            raise RendererInputError("frame_rate must be greater than zero")
        return True

    def _load_and_prepare_image(
        self, image_path: Path, width: int, height: int
    ) -> Image.Image:
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
            raise RendererInputError(
                f"Unsupported image format for {image_path.name}: {image_path.suffix}"
            )

        try:
            with Image.open(image_path) as image:
                image.load()
                image = image.convert("RGB")
                return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        except UnidentifiedImageError as exc:
            raise RendererInputError(f"Corrupted image file: {image_path}") from exc
        except OSError as exc:
            raise RendererInputError(f"Could not read image: {image_path}") from exc

    def _prepare_prompt(self, prompt: str) -> str:
        return " ".join(prompt.strip().split())

    def _make_generator(self, seed: int) -> Optional[torch.Generator]:
        if seed == -1:
            return None
        if self.device.type in {"cuda", "mps"}:
            return torch.Generator(device=self.device).manual_seed(seed)
        return torch.Generator().manual_seed(seed)

    def _memory_housekeeping(self, force_empty_cache: bool) -> None:
        gc.collect()
        if self.device.type == "cuda" and (
            force_empty_cache
            or self.render_count % max(1, self.config.empty_cache_interval) == 0
        ):
            torch.cuda.empty_cache()

    @staticmethod
    def _is_probable_oom_error(exc: Exception) -> bool:
        return "out of memory" in str(exc).lower()

    def _save_video(
        self,
        frames: list[np.ndarray],
        output_path: Path,
        frame_rate: int,
        codec: str,
        quality: int,
    ) -> None:
        try:
            import imageio.v2 as imageio

            output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                output_path,
                frames,
                fps=frame_rate,
                codec=codec,
                quality=quality,
            )
        except Exception as exc:
            raise RendererEncodingError(f"Failed to encode video: {exc}") from exc

    def _validate_output(
        self,
        output_path: Path,
        expected_width: int,
        expected_height: int,
        expected_frame_rate: int,
        expected_frame_count: int,
        codec: str,
    ) -> RenderedClip:
        import imageio.v2 as imageio

        if not output_path.exists():
            raise RendererOutputValidationError("Encoded output file does not exist")

        file_size = output_path.stat().st_size
        if file_size < self.config.min_output_size_bytes:
            raise RendererOutputValidationError(
                f"Encoded output file is too small: {file_size} bytes"
            )

        try:
            from imageio_ffmpeg import count_frames_and_secs

            counted_frames, duration_seconds = count_frames_and_secs(str(output_path))
        except Exception as exc:
            raise RendererOutputValidationError(
                f"Could not inspect video output: {exc}"
            ) from exc

        try:
            reader = imageio.get_reader(str(output_path))
            metadata = reader.get_meta_data()
            frame = reader.get_data(0)
            reader.close()
        except Exception as exc:
            raise RendererOutputValidationError(
                f"Encoded video is not readable: {exc}"
            ) from exc

        height, width = frame.shape[0], frame.shape[1]
        fps = int(round(metadata.get("fps", expected_frame_rate)))

        if width != expected_width or height != expected_height:
            raise RendererOutputValidationError(
                f"Output resolution mismatch: expected {expected_width}x{expected_height}, "
                f"got {width}x{height}"
            )
        if fps != expected_frame_rate:
            raise RendererOutputValidationError(
                f"Output frame rate mismatch: expected {expected_frame_rate}, got {fps}"
            )
        expected_duration = expected_frame_count / expected_frame_rate
        if (
            abs(duration_seconds - expected_duration)
            > self.config.expected_duration_tolerance_seconds
        ):
            raise RendererOutputValidationError(
                f"Output duration mismatch: expected about {expected_duration:.2f}s, "
                f"got {duration_seconds:.2f}s"
            )

        return RenderedClip(
            path=output_path,
            checksum_sha256=compute_file_hash(output_path),
            file_size_bytes=file_size,
            width=width,
            height=height,
            frame_rate=fps,
            frame_count=counted_frames,
            duration_seconds=duration_seconds,
            codec=codec,
        )

    def generate_clip(self, params: RenderParams, output_path: Path) -> RenderResult:
        total_start = time.perf_counter()
        metrics = RenderMetrics()

        try:
            self.initialize()
            self.validate_parameters(params)
            width, height = self._get_dimensions(params.resolution, params.aspect_ratio)

            image_start = time.perf_counter()
            start_img = self._load_and_prepare_image(params.start_image, width, height)
            end_img = (
                self._load_and_prepare_image(params.end_image, width, height)
                if params.end_image
                else None
            )
            metrics.image_loading_seconds = time.perf_counter() - image_start

            prompt_start = time.perf_counter()
            prompt = self._prepare_prompt(params.prompt)
            generator = self._make_generator(params.seed)
            metrics.prompt_preparation_seconds = time.perf_counter() - prompt_start

            logger.info(
                (
                    f"Rendering job={params.job_id} size={width}x{height} "
                    f"frames={params.duration.frames} steps={params.num_inference_steps} "
                    f"seed={params.seed}"
                ),
                extra={"job_id": params.job_id},
            )

            inference_start = time.perf_counter()
            pipeline_result = self.pipeline(
                prompt=prompt,
                image=start_img,
                end_image=end_img,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                width=width,
                height=height,
                num_frames=params.duration.frames,
                generator=generator,
            )
            metrics.inference_seconds = time.perf_counter() - inference_start

            frames = getattr(pipeline_result, "frames", None)
            if not frames or len(frames[0]) == 0:
                raise RendererOutputValidationError("Pipeline returned no frames")

            encoding_start = time.perf_counter()
            self._save_video(
                frames[0],
                output_path,
                params.frame_rate,
                params.output_codec,
                params.output_quality,
            )
            metrics.encoding_seconds = time.perf_counter() - encoding_start

            validation_start = time.perf_counter()
            clip = self._validate_output(
                output_path=output_path,
                expected_width=width,
                expected_height=height,
                expected_frame_rate=params.frame_rate,
                expected_frame_count=params.duration.frames,
                codec=params.output_codec,
            )
            metrics.validation_seconds = time.perf_counter() - validation_start
            metrics.total_seconds = time.perf_counter() - total_start

            self.render_count += 1
            self._memory_housekeeping(force_empty_cache=False)
            logger.info(
                f"Render completed in {metrics.total_seconds:.2f}s",
                extra={"job_id": params.job_id},
            )
            return RenderResult(
                success=True,
                job_id=params.job_id,
                clip=clip,
                metrics=metrics,
            )
        except torch.cuda.OutOfMemoryError as exc:
            self._memory_housekeeping(force_empty_cache=True)
            return RenderResult(
                success=False,
                job_id=params.job_id,
                metrics=metrics,
                error_type=RendererOOMError.__name__,
                error_message=str(exc),
            )
        except (
            RendererInputError,
            RendererEncodingError,
            RendererOutputValidationError,
            RendererInitializationError,
        ) as exc:
            self._memory_housekeeping(force_empty_cache=False)
            return RenderResult(
                success=False,
                job_id=params.job_id,
                metrics=metrics,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            )
        except Exception as exc:
            if self._is_probable_oom_error(exc):
                self._memory_housekeeping(force_empty_cache=True)
                return RenderResult(
                    success=False,
                    job_id=params.job_id,
                    metrics=metrics,
                    error_type=RendererOOMError.__name__,
                    error_message=str(exc),
                )
            self._memory_housekeeping(force_empty_cache=True)
            logger.error(
                f"Unexpected renderer failure: {exc}",
                extra={"job_id": params.job_id},
                exc_info=True,
            )
            return RenderResult(
                success=False,
                job_id=params.job_id,
                metrics=metrics,
                error_type="UnexpectedRenderError",
                error_message=str(exc),
            )
        finally:
            metrics.total_seconds = time.perf_counter() - total_start

    def cleanup(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.session_info = None
        self._memory_housekeeping(force_empty_cache=True)
        logger.info("Renderer resources released", extra={"job_id": "N/A"})
