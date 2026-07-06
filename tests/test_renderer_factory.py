import tempfile
import unittest
from pathlib import Path

from config import Config
from renderers.factory import create_renderer, get_available_renderer_backends
from renderers.ltx_renderer import LTXVideoRenderer
from renderers.wan2gp_ltx_renderer import Wan2GPLTXRenderer


class RendererFactoryTests(unittest.TestCase):
    def test_auto_uses_diffusers_when_wan2gp_dir_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(renderer_backend="auto", wan2gp_dir=Path(temp_dir) / "missing")
            renderer = create_renderer(config)

        self.assertIsInstance(renderer, LTXVideoRenderer)

    def test_auto_uses_wan2gp_when_runtime_dir_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wan_dir = Path(temp_dir) / "Wan2GP"
            wan_dir.mkdir()
            config = Config(renderer_backend="auto", wan2gp_dir=wan_dir)
            renderer = create_renderer(config)

        self.assertIsInstance(renderer, Wan2GPLTXRenderer)

    def test_explicit_backend_selection_is_honored(self):
        config = Config(renderer_backend="diffusers")
        self.assertIsInstance(create_renderer(config), LTXVideoRenderer)

    def test_plugin_registry_exposes_known_backends(self):
        backends = get_available_renderer_backends()

        self.assertIn("diffusers", backends)
        self.assertIn("wan2gp", backends)


if __name__ == "__main__":
    unittest.main()
