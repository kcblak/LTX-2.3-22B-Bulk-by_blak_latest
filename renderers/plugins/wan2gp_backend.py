import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

from config import Config
from core import (
    RenderMetrics,
    RenderParams,
    RenderResult,
    RendererInitializationError,
    RendererOOMError,
    RendererSessionInfo,
)
from logging_system import get_logger
from renderers.base import register_renderer_plugin
from renderers.plugins.diffusers_backend import LTXVideoRenderer

logger = get_logger("render")


def _wan2gp_available(config: Config) -> bool:
    return config.wan2gp_dir.exists()


def _wan2gp_auto_priority(config: Config) -> int:
    return 100 if _wan2gp_available(config) else -1


@register_renderer_plugin(
    name="wan2gp",
    description="Notebook-aligned Wan2GP/MMGP/GGUF backend",
    availability_check=_wan2gp_available,
    auto_priority_resolver=_wan2gp_auto_priority,
)
class Wan2GPLTXRenderer(LTXVideoRenderer):
    """Notebook-grade Wan2GP/MMGP LTX backend behind the common renderer interface."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.model_handler: Any = None
        self.save_video_fn: Any = None
        self.offload_module: Any = None

    def _prepare_runtime_environment(self) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        wan2gp_dir = self.config.wan2gp_dir.resolve(strict=False)
        if str(wan2gp_dir) not in sys.path:
            sys.path.insert(0, str(wan2gp_dir))

    def _resolve_model_dir(self) -> Path:
        if self.config.wan2gp_model_dir is not None:
            return self.config.wan2gp_model_dir.resolve(strict=False)
        return (self.config.wan2gp_dir / "models").resolve(strict=False)

    def _resolve_text_encoder_filename(self, text_encoder_dir: Path) -> Optional[str]:
        if self.config.wan2gp_text_encoder_filename:
            return self.config.wan2gp_text_encoder_filename

        candidates = sorted(text_encoder_dir.glob("*.safetensors"))
        if not candidates:
            return None

        preferred_patterns = ("quanto", "q4", "bf16", "fp16")
        for pattern in preferred_patterns:
            for candidate in candidates:
                if pattern in candidate.name.lower():
                    return candidate.name
        return candidates[0].name

    def _resolve_scheduler_name(self) -> Optional[str]:
        pipe = getattr(self.model_handler, "pipe", None)
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None:
            return None
        return scheduler.__class__.__name__

    def _apply_mmgp_profile(self) -> None:
        pipe = getattr(self.model_handler, "pipe", None)
        if pipe is None or self.offload_module is None:
            return

        convert_dtype = (
            self.dtype
            if self.dtype in (torch.float16, torch.bfloat16)
            else torch.float16
        )
        self.offload_module.profile(
            pipe,
            profile_no=self.config.wan2gp_mmgp_profile,
            compile_args=False,
            quantizeTransformer=self.config.wan2gp_quantize_transformer,
            convertWeightsFloatTo=convert_dtype,
            loras=["transformer"],
            budgets=self.config.wan2gp_mmgp_budgets,
        )
        shared_state = getattr(self.offload_module, "shared_state", None)
        if isinstance(shared_state, dict):
            shared_state["_attention"] = "sdpa"

    def _load_msr_lora(self, model_dir: Path) -> None:
        if not self.config.wan2gp_msr_enabled:
            return

        lora_path = model_dir / self.config.wan2gp_lora_filename
        if not lora_path.exists():
            nested_lora_path = model_dir / "loras" / "ltx2" / self.config.wan2gp_lora_filename
            if nested_lora_path.exists():
                lora_path = nested_lora_path
        if not lora_path.exists():
            logger.warning(
                f"MSR LoRA not found, skipping: {lora_path}",
                extra={"job_id": "N/A"},
            )
            return

        try:
            from models.base import trans_lora
            from shared.utils.lora import preprocess_sd

            state_dict = preprocess_sd(
                str(lora_path), self.model_handler.model["transformer"]
            )
            trans_lora.load_loras(
                self.model_handler.model["transformer"],
                {"msr": state_dict},
                activate_all_loras=True,
                fuse=True,
            )
        except Exception as exc:
            logger.warning(
                f"MSR LoRA load skipped after failure: {exc}",
                extra={"job_id": "N/A"},
            )

    def initialize(self) -> RendererSessionInfo:
        if self.session_info is not None and self.model_handler is not None:
            return self.session_info

        self._prepare_runtime_environment()
        start = time.perf_counter()
        model_dir = self._resolve_model_dir()
        transformer_path = model_dir / self.config.wan2gp_transformer_filename
        text_encoder_dir = model_dir / self.config.wan2gp_text_encoder_dirname
        text_encoder_filename = self._resolve_text_encoder_filename(text_encoder_dir)

        try:
            import mmgp.offload as offload
            import shared.qtypes.gguf  # noqa: F401
            from models.ltx2.ltx2_handler import family_handler
            from shared.utils.audio_video import save_video

            self.offload_module = offload
            self.model_handler = family_handler
            self.save_video_fn = save_video

            model_def = {
                "type": self.config.wan2gp_base_model_type,
                "pipeline": self.config.wan2gp_pipeline_variant,
                "msr": self.config.wan2gp_msr_enabled,
                "frame_count": self.config.wan2gp_msr_frame_count,
                "text_encoder": self.config.wan2gp_text_encoder_dirname,
            }

            self.model_handler.load_model(
                model_def,
                checkpoint_path=str(transformer_path),
                VAE_path=None,
                precision="auto",
                dtype=torch.float16,
                VAE_dtype=torch.float16,
                text_encoder_filename=text_encoder_filename,
            )

            self.pipeline = getattr(self.model_handler, "pipe", self.model_handler)
            self._apply_mmgp_profile()
            self._load_msr_lora(model_dir)

            total_vram, free_vram = self._get_vram_stats()
            self.session_info = RendererSessionInfo(
                renderer_name=self.__class__.__name__,
                device=str(self.device),
                precision=str(self.dtype).replace("torch.", ""),
                total_vram_bytes=total_vram,
                available_vram_bytes=free_vram,
                scheduler_name=self._resolve_scheduler_name(),
                warmup_performed=False,
                initialization_seconds=time.perf_counter() - start,
            )
            logger.info(
                (
                    "Wan2GP renderer initialized "
                    f"checkpoint={transformer_path.name} "
                    f"text_encoder={text_encoder_filename or 'auto'} "
                    f"mmgp_profile={self.config.wan2gp_mmgp_profile}"
                ),
                extra={"job_id": "N/A"},
            )
            return self.session_info
        except torch.cuda.OutOfMemoryError as exc:
            raise RendererInitializationError(
                f"Out of memory during Wan2GP initialization: {exc}"
            ) from exc
        except Exception as exc:
            raise RendererInitializationError(
                f"Failed to initialize Wan2GP backend: {exc}"
            ) from exc

    def _build_reference_images(self, params: RenderParams) -> list[str]:
        refs = [str(params.start_image.resolve(strict=False))]
        if params.end_image is not None:
            refs.append(str(params.end_image.resolve(strict=False)))
        return refs

    def _build_video_prompt_type(self, params: RenderParams) -> str:
        return "KI" if params.end_image is not None else "I"

    def generate_clip(self, params: RenderParams, output_path: Path) -> RenderResult:
        total_start = time.perf_counter()
        metrics = RenderMetrics()

        try:
            self.initialize()
            self.validate_parameters(params)
            width, height = self._get_dimensions(params.resolution, params.aspect_ratio)

            image_start = time.perf_counter()
            self._load_and_prepare_image(params.start_image, width, height)
            if params.end_image is not None:
                self._load_and_prepare_image(params.end_image, width, height)
            reference_images = self._build_reference_images(params)
            metrics.image_loading_seconds = time.perf_counter() - image_start

            prompt_start = time.perf_counter()
            prompt = self._prepare_prompt(params.prompt)
            video_prompt_type = self._build_video_prompt_type(params)
            metrics.prompt_preparation_seconds = time.perf_counter() - prompt_start

            inference_start = time.perf_counter()
            with torch.inference_mode():
                video_tensor = self.model_handler.generate(
                    input_prompt=prompt,
                    input_ref_images=reference_images,
                    height=height,
                    width=width,
                    frame_num=params.duration.frames,
                    fps=params.frame_rate,
                    seed=params.seed,
                    callback=None,
                    VAE_tile_size=self.config.wan2gp_vae_tile_size,
                    input_video_strength=1.0,
                    denoising_strength=1.0,
                    guide_scale=float(params.guidance_scale),
                    sampling_steps=int(params.num_inference_steps),
                    guide_phases=int(self.config.wan2gp_guide_phases),
                    n_prompt=self.config.wan2gp_negative_prompt,
                    video_prompt_type=video_prompt_type,
                    audio_prompt_type="" if not self.config.wan2gp_audio_enabled else "A",
                )
            metrics.inference_seconds = time.perf_counter() - inference_start

            encoding_start = time.perf_counter()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_video_fn(
                video_tensor, {"save_file": str(output_path)}, fps=params.frame_rate
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
            self._memory_housekeeping(force_empty_cache=False)
            return RenderResult(
                success=False,
                job_id=params.job_id,
                metrics=metrics,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            )
        finally:
            metrics.total_seconds = time.perf_counter() - total_start

    def cleanup(self) -> None:
        self.model_handler = None
        self.save_video_fn = None
        self.offload_module = None
        super().cleanup()
