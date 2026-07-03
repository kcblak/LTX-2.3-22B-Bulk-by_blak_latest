from pathlib import Path
from typing import Optional, Tuple
import torch
from PIL import Image
import numpy as np

from ..core.models import Job, Config
from ..core.enums import Resolution, AspectRatio, Duration
from ..utils.logger import setup_logger


class ModelService:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__, config.logs_dir)
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

    def load_model(self):
        self.logger.info("Loading model...")
        try:
            from diffusers import LTXVideoPipeline
            import torch

            self.pipeline = LTXVideoPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
            )
            self.pipeline = self.pipeline.to(self.device)
            self.logger.info("Model loaded successfully!")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _get_resolution_dimensions(
        self, resolution: Resolution, aspect_ratio: AspectRatio
    ) -> Tuple[int, int]:
        base_resolutions = {
            Resolution.RES_480P: 480,
            Resolution.RES_720P: 720,
            Resolution.RES_1080P: 1080,
        }
        base = base_resolutions[resolution]
        if aspect_ratio == AspectRatio.AR_16_9:
            return (base * 16 // 9, base)
        elif aspect_ratio == AspectRatio.AR_9_16:
            return (base, base * 16 // 9)
        else:
            return (base, base)

    def _get_num_frames(self, duration: Duration) -> int:
        duration_map = {
            Duration.DUR_5S: 121,
            Duration.DUR_10S: 241,
            Duration.DUR_15S: 361,
        }
        return duration_map[duration]

    def _load_image(self, image_path: Path) -> Optional[Image.Image]:
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def generate_video(self, job: Job, output_path: Path) -> bool:
        if not self.pipeline:
            self.load_model()

        self.logger.info(f"Generating video for job {job.job_id}")

        width, height = self._get_resolution_dimensions(
            job.resolution, job.aspect_ratio
        )
        num_frames = self._get_num_frames(job.duration)
        start_image = self._load_image(job.start_image)
        end_image = self._load_image(job.end_image) if job.end_image else None

        if not start_image:
            self.logger.error(f"Failed to load start image for job {job.job_id}")
            return False

        try:
            generator = torch.Generator(device=self.device)
            if job.seed != -1:
                generator.manual_seed(job.seed)

            video = self.pipeline(
                prompt=job.prompt,
                image=start_image,
                end_image=end_image,
                num_inference_steps=job.steps,
                guidance_scale=job.guide_scale,
                width=width,
                height=height,
                num_frames=num_frames,
                generator=generator,
            ).frames[0]

            self._save_video(video, output_path)
            self.logger.info(f"Video generated successfully for job {job.job_id}")
            return True
        except Exception as e:
            self.logger.error(f"Video generation failed for job {job.job_id}: {e}")
            return False

    def _save_video(self, video: np.ndarray, output_path: Path):
        import imageio

        imageio.mimwrite(output_path, video, fps=24, quality=8)
        self.logger.info(f"Video saved to {output_path}")
