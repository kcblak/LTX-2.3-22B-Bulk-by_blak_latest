import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from orchestration.runtime_assets import ensure_requirement_file, inspect_requirement_file


class RuntimeAssetTests(unittest.TestCase):
    def test_inspect_requirement_file_reports_missing_and_installed_packages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_path = Path(temp_dir) / "requirements.txt"
            requirements_path.write_text("pip>=24.0\nmissing-package>=1.0\n", encoding="utf-8")

            with patch(
                "orchestration.runtime_assets._installed_distribution_versions",
                return_value={"pip": "24.0"},
            ):
                report = inspect_requirement_file(requirements_path)

            statuses = {status.package_name: status for status in report.inspected}
            self.assertTrue(statuses["pip"].installed)
            self.assertFalse(statuses["missing-package"].installed)

    def test_ensure_requirement_file_installs_only_missing_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_path = Path(temp_dir) / "requirements.txt"
            requirements_path.write_text("pip>=24.0\nmissing-package>=1.0\n", encoding="utf-8")

            with patch(
                "orchestration.runtime_assets._installed_distribution_versions",
                return_value={"pip": "24.0"},
            ), patch("orchestration.runtime_assets.subprocess.check_call") as mock_check_call:
                report = ensure_requirement_file(requirements_path)

            mock_check_call.assert_called_once()
            self.assertEqual(report.installed_requirements, ["missing-package>=1.0"])
            self.assertEqual(report.skipped_requirements, ["pip>=24.0"])

    def test_config_exposes_wan2gp_runtime_asset_defaults(self):
        config = Config()

        self.assertTrue(config.wan2gp_repo_url.endswith("Wan2GP.git"))
        self.assertTrue(config.wan2gp_required_companion_files)
        self.assertTrue(config.wan2gp_required_text_encoder_files)

    def test_ensure_git_checkout_strips_wrapping_backticks_and_spaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "repo"
            with patch("orchestration.runtime_assets.subprocess.check_call") as mock_check_call:
                from orchestration.runtime_assets import ensure_git_checkout

                report = ensure_git_checkout(
                    destination=destination,
                    repo_url=" `https://github.com/example/project.git` ",
                    ref=" `main` ",
                )

            mock_check_call.assert_called_once_with(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "main",
                    "https://github.com/example/project.git",
                    str(destination.resolve(strict=False)),
                ]
            )
            self.assertEqual(report.repo_url, "https://github.com/example/project.git")
            self.assertEqual(report.ref, "main")


if __name__ == "__main__":
    unittest.main()
