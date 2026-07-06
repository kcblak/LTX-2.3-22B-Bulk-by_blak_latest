import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from config import Config
from core import RenderParams
from renderers.wan2gp_ltx_renderer import Wan2GPLTXRenderer


class _FakeFamilyHandler:
    def __init__(self):
        class FakeScheduler:
            pass

        self.pipe = types.SimpleNamespace(scheduler=FakeScheduler())
        self.model = {"transformer": object()}
        self.load_model = MagicMock()
        self.generate = MagicMock(return_value="fake-video")


class Wan2GPLTXRendererTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.wan_dir = self.root / "Wan2GP"
        self.model_dir = self.wan_dir / "models"
        self.text_encoder_dir = self.model_dir / "gemma-3-12b-it-qat-q4_0-unquantized"
        self.text_encoder_dir.mkdir(parents=True)
        (self.model_dir / "ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf").write_bytes(b"gguf")
        (self.model_dir / "LTX-2.3-Licon-MSR-V1.safetensors").write_bytes(b"lora")
        (self.text_encoder_dir / "gemma-q4.safetensors").write_bytes(b"text")
        self.start_image = self.root / "start.png"
        Image.new("RGB", (96, 96), color=(0, 255, 0)).save(self.start_image)
        self.config = Config(
            renderer_backend="wan2gp",
            wan2gp_dir=self.wan_dir,
            wan2gp_model_dir=self.model_dir,
            output_dir=self.root / "outputs",
            temp_dir=self.root / "temp",
            log_dir=self.root / "logs",
        )
        self.renderer = Wan2GPLTXRenderer(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _params(self) -> RenderParams:
        return RenderParams(
            job_id="job-1",
            prompt="A highly detailed shot.",
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

    def _module_map(self, family_handler):
        offload = types.ModuleType("mmgp.offload")
        offload.profile = MagicMock()
        offload.shared_state = {}

        mmgp = types.ModuleType("mmgp")
        mmgp.offload = offload

        gguf = types.ModuleType("shared.qtypes.gguf")
        qtypes = types.ModuleType("shared.qtypes")
        qtypes.gguf = gguf

        audio_video = types.ModuleType("shared.utils.audio_video")

        def save_video(_video, args_out, fps):
            Path(args_out["save_file"]).write_bytes(b"x" * 4096)

        audio_video.save_video = save_video

        lora = types.ModuleType("shared.utils.lora")
        lora.preprocess_sd = MagicMock(return_value={"mock": 1})

        shared_utils = types.ModuleType("shared.utils")
        shared_utils.audio_video = audio_video
        shared_utils.lora = lora

        shared = types.ModuleType("shared")
        shared.qtypes = qtypes
        shared.utils = shared_utils

        ltx2_handler = types.ModuleType("models.ltx2.ltx2_handler")
        ltx2_handler.family_handler = family_handler

        ltx2 = types.ModuleType("models.ltx2")
        ltx2.ltx2_handler = ltx2_handler

        trans_lora = types.ModuleType("models.base.trans_lora")
        trans_lora.load_loras = MagicMock()

        models_base = types.ModuleType("models.base")
        models_base.trans_lora = trans_lora

        models = types.ModuleType("models")
        models.ltx2 = ltx2
        models.base = models_base

        return {
            "mmgp": mmgp,
            "mmgp.offload": offload,
            "shared": shared,
            "shared.qtypes": qtypes,
            "shared.qtypes.gguf": gguf,
            "shared.utils": shared_utils,
            "shared.utils.audio_video": audio_video,
            "shared.utils.lora": lora,
            "models": models,
            "models.ltx2": ltx2,
            "models.ltx2.ltx2_handler": ltx2_handler,
            "models.base": models_base,
            "models.base.trans_lora": trans_lora,
        }

    def test_initialize_loads_wan2gp_runtime(self):
        family_handler = _FakeFamilyHandler()
        module_map = self._module_map(family_handler)

        with patch.dict(sys.modules, module_map):
            session = self.renderer.initialize()

        self.assertEqual(session.renderer_name, "Wan2GPLTXRenderer")
        family_handler.load_model.assert_called_once()
        self.assertEqual(self.renderer.offload_module.shared_state.get("_attention"), "sdpa")

    def test_generate_clip_uses_wan2gp_generation_path(self):
        family_handler = _FakeFamilyHandler()
        module_map = self._module_map(family_handler)
        params = self._params()
        output = self.root / "temp.mp4"

        with patch.dict(sys.modules, module_map):
            self.renderer.initialize()
            with patch.object(
                self.renderer,
                "_validate_output",
                return_value=types.SimpleNamespace(
                    path=output,
                    checksum_sha256="abc",
                    file_size_bytes=4096,
                    width=64,
                    height=64,
                    frame_rate=24,
                    frame_count=9,
                    duration_seconds=9 / 24,
                    codec="libx264",
                ),
            ):
                result = self.renderer.generate_clip(params, output)

        self.assertTrue(result.success)
        family_handler.generate.assert_called_once()
        self.assertEqual(result.clip.checksum_sha256, "abc")


if __name__ == "__main__":
    unittest.main()
