import tempfile
import unittest
from pathlib import Path

from config.loader import load_config


class ConfigLoaderTests(unittest.TestCase):
    def test_precedence_orders_overrides_correctly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_path = root / "default.yaml"
            project_path = root / "project.yaml"

            default_path.write_text(
                "\n".join(
                    [
                        'profile: "debug"',
                        "paths:",
                        '  output_dir: "default_outputs"',
                        "model:",
                        "  num_inference_steps: 8",
                        "drive:",
                        "  enable_drive_upload: false",
                        "features:",
                        "  gradio_ui: false",
                        "execution:",
                        '  execution_profile: "kaggle_bulk"',
                    ]
                ),
                encoding="utf-8",
            )
            project_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  num_inference_steps: 12",
                        "logging:",
                        '  log_level: "DEBUG"',
                        "features:",
                        "  gradio_ui: true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(
                default_config_path=default_path,
                project_config_path=project_path,
                env_overrides={"num_inference_steps": 16},
                cli_overrides={"num_inference_steps": 20, "output_dir": root / "cli_outputs"},
                runtime_overrides={"num_inference_steps": 24, "benchmark_mode": True},
            )

            self.assertEqual(config.profile, "debug")
            self.assertEqual(config.num_inference_steps, 24)
            self.assertEqual(config.log_level, "DEBUG")
            self.assertEqual(config.output_dir, (root / "cli_outputs").resolve(strict=False))
            self.assertFalse(config.enable_drive_upload)
            self.assertTrue(config.benchmark_mode)
            self.assertEqual(config.project_config_path, project_path)
            self.assertEqual(config.execution_profile, "kaggle_bulk")
            self.assertTrue(config.features["gradio_ui"])


if __name__ == "__main__":
    unittest.main()
