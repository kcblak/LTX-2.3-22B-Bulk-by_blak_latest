import tempfile
import unittest
from subprocess import CompletedProcess
from pathlib import Path
from unittest.mock import patch

from config import Config
from orchestration.runtime_assets import (
    DependencyProfileReport,
    DependencyStatus,
    FeatureRuntimeStatus,
    PipInstallResult,
    detect_execution_profile,
    detect_renderer_dependency_profile,
    ensure_dependency_profile,
    ensure_runtime_dependency_profile,
    inspect_feature_dependency_profile,
    inspect_dependency_profile,
    inspect_requirement_file,
    resolve_enabled_features,
    verify_runtime_dependencies,
)


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

    def test_inspect_dependency_profile_filters_platform_specific_packages(self):
        report = inspect_dependency_profile(
            "development",
            platform_name="linux",
            kaggle_mode=True,
        )

        skipped = {item.package_name: item.reason for item in report.skipped_packages}
        self.assertIn("pywin32", skipped)
        self.assertIn("Rejected by Kaggle", skipped["pywin32"])

    def test_ensure_dependency_profile_installs_missing_bootstrap_packages(self):
        with patch(
            "orchestration.runtime_assets._installed_distribution_versions",
            return_value={},
        ), patch(
            "orchestration.runtime_assets.subprocess.run"
        ) as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "ok"
            mock_run.return_value.stderr = ""
            report = ensure_dependency_profile("bootstrap", kaggle_mode=True, platform_name="linux")

        mock_run.assert_called_once()
        self.assertEqual(report.installed_packages[0].package_name, "PyYAML")

    def test_ensure_dependency_profile_records_required_failure_diagnostics(self):
        with patch(
            "orchestration.runtime_assets._installed_distribution_versions",
            return_value={},
        ), patch(
            "orchestration.runtime_assets.subprocess.run"
        ) as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = "downloading"
            mock_run.return_value.stderr = "no matching distribution found"
            report = ensure_dependency_profile("bootstrap", kaggle_mode=True, platform_name="linux")

        self.assertEqual(len(report.failed_required), 1)
        failure = report.failed_required[0]
        self.assertEqual(failure.package_name, "PyYAML")
        self.assertIn("no matching distribution found", failure.stderr)
        self.assertTrue(failure.suggested_resolution)
        self.assertEqual(failure.python_version.count("."), 2)

    def test_config_exposes_wan2gp_runtime_asset_defaults(self):
        config = Config()

        self.assertTrue(config.wan2gp_repo_url.endswith("Wan2GP.git"))
        self.assertTrue(config.wan2gp_required_companion_files)
        self.assertTrue(config.wan2gp_required_text_encoder_files)

    def test_detect_execution_profile_uses_config_override(self):
        config = Config(execution_profile="production")
        self.assertEqual(detect_execution_profile(config), "production")

    def test_detect_execution_profile_maps_legacy_alias(self):
        config = Config(execution_profile="production_server")
        self.assertEqual(detect_execution_profile(config), "production")

    def test_detect_renderer_dependency_profile_prefers_wan2gp_when_runtime_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(wan2gp_dir=Path(temp_dir), renderer_backend="auto")
            self.assertEqual(detect_renderer_dependency_profile(config), "wan2gp")

    def test_resolve_enabled_features_applies_renderer_and_runtime_flags(self):
        config = Config(
            execution_profile="kaggle_bulk",
            renderer_backend="wan2gp",
            enable_drive_upload=False,
            enable_stitching=False,
            resume_enabled=True,
            features={"gradio_ui": True},
        )
        enabled, disabled = resolve_enabled_features(config)

        self.assertIn("core_renderer", enabled)
        self.assertIn("wan2gp_runtime", enabled)
        self.assertIn("gguf_runtime", enabled)
        self.assertIn("gradio_ui", enabled)
        self.assertIn("google_drive_integration", disabled)
        self.assertIn("video_stitching", disabled)

    def test_inspect_feature_dependency_profile_reports_enabled_and_disabled_features(self):
        config = Config(
            execution_profile="kaggle_bulk",
            renderer_backend="wan2gp",
            enable_drive_upload=False,
            enable_stitching=False,
            features={"gradio_ui": True},
        )
        with patch("orchestration.runtime_assets._current_platform_name", return_value="linux"), patch(
            "orchestration.runtime_assets._is_kaggle_runtime",
            return_value=True,
        ):
            report = inspect_feature_dependency_profile(config)

        self.assertEqual(report.profile_name, "kaggle_bulk")
        self.assertIn("wan2gp_runtime", report.enabled_features)
        self.assertIn("google_drive_integration", report.disabled_features)
        skipped = {item.package_name: item.reason for item in report.skipped_packages}
        self.assertIn("gradio", skipped)

    def test_ensure_runtime_dependency_profile_keeps_optional_failure_nonfatal(self):
        base_report = DependencyProfileReport(
            profile_name="kaggle_bulk",
            platform_name="linux",
            kaggle_mode=True,
            enabled_features=["gradio_ui"],
            feature_statuses=[
                FeatureRuntimeStatus(
                    feature_key="gradio_ui",
                    enabled=True,
                    required=False,
                    state="PENDING",
                    package_keys=["gradio"],
                )
            ],
            statuses=[
                DependencyStatus(
                    requirement="gradio>=4.0.0",
                    package_name="gradio",
                    import_name="gradio",
                    classification="optional",
                    feature_keys=["gradio_ui"],
                    installed=False,
                )
            ],
        )
        verification_report = DependencyProfileReport(
            profile_name="kaggle_bulk",
            platform_name="linux",
            kaggle_mode=True,
            enabled_features=["gradio_ui"],
            feature_statuses=[
                FeatureRuntimeStatus(
                    feature_key="gradio_ui",
                    enabled=True,
                    required=False,
                    state="DISABLED",
                    reason="pip install failed",
                    package_keys=["gradio"],
                )
            ],
            statuses=[],
            verification_checks=[],
        )
        with patch(
            "orchestration.runtime_assets.inspect_feature_dependency_profile",
            return_value=base_report,
        ), patch(
            "orchestration.runtime_assets.verify_runtime_dependencies",
            return_value=verification_report,
        ), patch(
            "orchestration.runtime_assets.subprocess.run"
        ) as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = "stdout"
            mock_run.return_value.stderr = "stderr"
            report = ensure_runtime_dependency_profile(Config())

        self.assertFalse(report.failed_required)
        self.assertEqual(len(report.failed_optional), 1)
        self.assertEqual(report.failed_optional[0].package_name, "gradio")

    def test_verify_runtime_dependencies_reports_required_feature_failure(self):
        config = Config(renderer_backend="diffusers", use_cuda=False, enable_drive_upload=False)
        base_report = DependencyProfileReport(
            profile_name="kaggle_bulk",
            platform_name="linux",
            kaggle_mode=True,
            enabled_features=["core_renderer"],
            feature_statuses=[
                FeatureRuntimeStatus(
                    feature_key="core_renderer",
                    enabled=True,
                    required=True,
                    state="PENDING",
                    package_keys=["torch", "imageio"],
                )
            ],
            statuses=[],
        )
        with patch(
            "orchestration.runtime_assets.inspect_feature_dependency_profile",
            return_value=base_report,
        ), patch(
            "orchestration.runtime_assets.shutil.which",
            return_value=None,
        ), patch(
            "orchestration.runtime_assets.importlib.import_module",
            side_effect=ImportError("missing"),
        ):
            report = verify_runtime_dependencies(config)

        failed = {
            status.feature_key: status.reason
            for status in report.feature_statuses
            if status.state == "FAILED"
        }
        self.assertIn("core_renderer", failed)

    def test_ensure_git_checkout_strips_wrapping_backticks_and_spaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "repo"
            with patch(
                "orchestration.runtime_assets.subprocess.run",
                return_value=CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="",
                    stderr="",
                ),
            ) as mock_run:
                from orchestration.runtime_assets import ensure_git_checkout

                report = ensure_git_checkout(
                    destination=destination,
                    repo_url=" `https://github.com/example/project.git` ",
                    ref=" `main` ",
                )

            first_command = mock_run.call_args_list[0].args[0]
            self.assertEqual(
                first_command,
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "main",
                    "https://github.com/example/project.git",
                    str(destination.resolve(strict=False)),
                ],
            )
            self.assertEqual(report.repo_url, "https://github.com/example/project.git")
            self.assertEqual(report.ref, "main")
            self.assertTrue(report.cloned)

    def test_ensure_git_checkout_uses_dataset_source_without_git(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "repo"
            destination.mkdir()
            (destination / "main.py").write_text("print('ok')", encoding="utf-8")

            from orchestration.runtime_assets import ensure_git_checkout

            report = ensure_git_checkout(
                destination=destination,
                repo_url="https://github.com/example/project.git",
                ref="main",
                source="dataset",
            )

        self.assertTrue(report.used_local_copy)
        self.assertEqual(report.repository_state, "dataset_source")
        self.assertFalse(report.command_results)

    def test_ensure_git_checkout_uses_local_copy_when_remote_is_offline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "repo"
            (destination / ".git").mkdir(parents=True)

            command_results = [
                CompletedProcess(args=[], returncode=0, stdout=".git\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="main\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="## main\n", stderr=""),
                CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="https://github.com/example/project.git\n",
                    stderr="",
                ),
                CompletedProcess(args=[], returncode=128, stdout="", stderr="offline"),
            ]

            with patch(
                "orchestration.runtime_assets.subprocess.run",
                side_effect=command_results,
            ):
                from orchestration.runtime_assets import ensure_git_checkout

                report = ensure_git_checkout(
                    destination=destination,
                    repo_url="https://github.com/example/project.git",
                    ref="main",
                )

        self.assertTrue(report.used_local_copy)
        self.assertEqual(report.repository_state, "offline_local_checkout")
        self.assertFalse(report.remote_reachable)

    def test_ensure_git_checkout_repairs_wrong_remote_and_falls_back_branch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "repo"
            (destination / ".git").mkdir(parents=True)

            command_results = [
                CompletedProcess(args=[], returncode=0, stdout=".git\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="## HEAD (detached at abc123)\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="https://github.com/example/wrong.git\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="ref: refs/heads/main\tHEAD\nabcd\tHEAD\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="  origin/main\n  origin/dev\n", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            ]

            with patch(
                "orchestration.runtime_assets.subprocess.run",
                side_effect=command_results,
            ):
                from orchestration.runtime_assets import ensure_git_checkout

                report = ensure_git_checkout(
                    destination=destination,
                    repo_url="https://github.com/example/project.git",
                    ref="feature-does-not-exist",
                )

        self.assertEqual(report.target_ref, "main")
        self.assertTrue(report.updated)
        self.assertIn(
            "Replaced incorrect origin remote URL.",
            report.recovery_actions,
        )


if __name__ == "__main__":
    unittest.main()
