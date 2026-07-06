import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from config import Config
from core import RenderParams
from renderers.ltx_renderer import LTXVideoRenderer


class _FakePipeline:
    def __init__(self):
        class FakeScheduler:
            pass

        self.scheduler = FakeScheduler()
        self.to_calls = []
        self.enable_attention_slicing_called = False
        self.enable_vae_slicing_called = False
        self.enable_vae_tiling_called = False

    def to(self, device):
        self.to_calls.append(str(device))
        return self

    def enable_attention_slicing(self):
        self.enable_attention_slicing_called = True

    def enable_vae_slicing(self):
        self.enable_vae_slicing_called = True

    def enable_vae_tiling(self):
        self.enable_vae_tiling_called = True

    def __call__(self, **_kwargs):
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(9)]
        return types.SimpleNamespace(frames=[frames])


class LTXVideoRendererTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config = Config(
            output_dir=self.root / "outputs",
            temp_dir=self.root / "temp",
            log_dir=self.root / "logs",
            warmup_enabled=False,
        )
        self.renderer = LTXVideoRenderer(self.config)
        self.start_image = self.root / "start.png"
        Image.new("RGB", (96, 96), color=(255, 0, 0)).save(self.start_image)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _params(self) -> RenderParams:
        return RenderParams(
            job_id="job-1",
            prompt="A cinematic slow dolly shot.",
            start_image=self.start_image,
            end_image=None,
            duration=types.SimpleNamespace(frames=9),
            resolution=types.SimpleNamespace(pixels=64),
            aspect_ratio=types.SimpleNamespace(label="1:1", width_ratio=1, height_ratio=1),
            seed=123,
            guidance_scale=3.0,
            num_inference_steps=4,
            frame_rate=24,
            output_codec="libx264",
            output_container="mp4",
            output_quality=8,
        )

    def test_initialize_creates_reusable_session_info(self):
        fake_pipeline = _FakePipeline()
        fake_diffusers = types.SimpleNamespace(
            LTXVideoPipeline=types.SimpleNamespace(
                from_pretrained=MagicMock(return_value=fake_pipeline)
            )
        )

        with patch.dict(sys.modules, {"diffusers": fake_diffusers}):
            session = self.renderer.initialize()

        self.assertEqual(session.renderer_name, "LTXVideoRenderer")
        self.assertIsNotNone(self.renderer.pipeline)
        self.assertTrue(fake_pipeline.to_calls or self.renderer.device.type != "cuda")

    def test_validate_parameters_rejects_missing_input(self):
        params = self._params()
        params.start_image = self.root / "missing.png"

        with self.assertRaises(Exception):
            self.renderer.validate_parameters(params)

    def test_image_preprocessing_converts_and_resizes(self):
        source = self.root / "palette.png"
        Image.new("P", (150, 75)).save(source)

        prepared = self.renderer._load_and_prepare_image(source, 64, 64)

        self.assertEqual(prepared.mode, "RGB")
        self.assertEqual(prepared.size, (64, 64))

    def test_generate_clip_returns_structured_success_result(self):
        self.renderer.pipeline = _FakePipeline()
        self.renderer.session_info = types.SimpleNamespace(device="cpu")
        params = self._params()
        output = self.root / "temp.mp4"

        with patch.object(self.renderer, "_save_video") as save_video, patch.object(
            self.renderer,
            "_validate_output",
            return_value=types.SimpleNamespace(
                path=output,
                checksum_sha256="abc",
                file_size_bytes=2048,
                width=64,
                height=64,
                frame_rate=24,
                frame_count=9,
                duration_seconds=0.375,
                codec="libx264",
            ),
        ):
            result = self.renderer.generate_clip(params, output)

        self.assertTrue(result.success)
        self.assertEqual(result.job_id, "job-1")
        self.assertEqual(result.clip.checksum_sha256, "abc")
        self.assertGreaterEqual(result.metrics.total_seconds, 0.0)
        save_video.assert_called_once()

    def test_generate_clip_maps_oom_to_renderer_result(self):
        class _OOMPipeline:
            def __call__(self, **_kwargs):
                raise RuntimeError("CUDA out of memory")

        self.renderer.pipeline = _OOMPipeline()
        self.renderer.session_info = types.SimpleNamespace(device="cpu")
        params = self._params()

        result = self.renderer.generate_clip(params, self.root / "temp.mp4")

        self.assertFalse(result.success)
        self.assertIn("memory", result.error_message.lower())

    def test_output_validation_checks_resolution_and_duration(self):
        output = self.root / "clip.mp4"
        output.write_bytes(b"x" * 2048)

        fake_reader = MagicMock()
        fake_reader.get_meta_data.return_value = {"fps": 24}
        fake_reader.get_data.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        fake_imageio = types.ModuleType("imageio.v2")
        fake_imageio.get_reader = MagicMock(return_value=fake_reader)
        fake_imageio_root = types.ModuleType("imageio")
        fake_imageio_root.v2 = fake_imageio
        fake_imageio_ffmpeg = types.ModuleType("imageio_ffmpeg")
        fake_imageio_ffmpeg.count_frames_and_secs = MagicMock(return_value=(9, 9 / 24))

        with patch.dict(
            sys.modules,
            {"imageio": fake_imageio_root, "imageio.v2": fake_imageio, "imageio_ffmpeg": fake_imageio_ffmpeg},
        ), patch(
            "renderers.plugins.diffusers_backend.compute_file_hash",
            return_value="hash",
        ):
            clip = self.renderer._validate_output(output, 64, 64, 24, 9, "libx264")

        self.assertEqual(clip.width, 64)
        self.assertEqual(clip.frame_count, 9)
        self.assertEqual(clip.checksum_sha256, "hash")

    def test_cleanup_releases_pipeline(self):
        self.renderer.pipeline = _FakePipeline()
        self.renderer.session_info = types.SimpleNamespace(device="cpu")

        self.renderer.cleanup()

        self.assertIsNone(self.renderer.pipeline)
        self.assertIsNone(self.renderer.session_info)


if __name__ == "__main__":
    unittest.main()
