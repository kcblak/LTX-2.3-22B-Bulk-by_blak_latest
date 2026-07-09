import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from orchestration.kaggle import (
    KaggleNotebookLauncher,
    detect_source_root,
    discover_drive_credentials,
    discover_kaggle_project,
    validate_repository_layout,
)
from orchestration.runner import PreparationResult
from orchestration.runtime_assets import (
    DependencyProfileReport,
    DependencyVerificationCheck,
    FeatureRuntimeStatus,
)


class KaggleLauncherTests(unittest.TestCase):
    def test_validate_repository_layout_reports_missing_critical_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            result = validate_repository_layout(repo_root)

            self.assertFalse(result.ready)
            self.assertIn("main.py", result.critical_missing)

    def test_discover_kaggle_project_prefers_candidate_with_matching_images_and_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "input"
            project_root = input_root / "project-a"
            project_root.mkdir(parents=True)
            (project_root / "images").mkdir()
            (project_root / "images" / "start.png").write_bytes(b"png")
            (project_root / "project.yaml").write_text("profile: debug\n", encoding="utf-8")
            (project_root / "jobs.csv").write_text(
                "prompt,start_image,end_image,duration,resolution,aspect_ratio,seed,guidance_scale,num_inference_steps\n"
                "test,images/start.png,,5 Seconds,480p,1:1 Square,1,3.0,8\n",
                encoding="utf-8",
            )

            discovered = discover_kaggle_project(input_root)

            self.assertEqual(discovered.jobs_csv_path, project_root / "jobs.csv")
            self.assertEqual(discovered.reference_images_dir, project_root)
            self.assertEqual(discovered.project_config_path, project_root / "project.yaml")
            self.assertGreaterEqual(discovered.image_match_count, 1)

    def test_discover_kaggle_project_reports_missing_image_references(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "input"
            project_root = input_root / "project-a"
            project_root.mkdir(parents=True)
            (project_root / "jobs.csv").write_text(
                "prompt,start_image,end_image,duration,resolution,aspect_ratio,seed,guidance_scale,num_inference_steps\n"
                "test,missing/start.png,,5 Seconds,480p,1:1 Square,1,3.0,8\n",
                encoding="utf-8",
            )

            discovered = discover_kaggle_project(input_root)

            self.assertEqual(discovered.referenced_image_count, 1)
            self.assertEqual(discovered.missing_image_count, 1)
            self.assertIn("missing/start.png", discovered.missing_image_refs)

    def test_discover_drive_credentials_uses_environment_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            previous = os.environ.get("LTX_DRIVE_CREDENTIALS_JSON")
            os.environ["LTX_DRIVE_CREDENTIALS_JSON"] = '{"type":"service_account"}'
            try:
                discovered = discover_drive_credentials(repo_root)
            finally:
                if previous is None:
                    os.environ.pop("LTX_DRIVE_CREDENTIALS_JSON", None)
                else:
                    os.environ["LTX_DRIVE_CREDENTIALS_JSON"] = previous

            self.assertTrue(discovered.enabled)
            self.assertEqual(discovered.source, "env:LTX_DRIVE_CREDENTIALS_JSON")

    def test_detect_source_root_prefers_src_layout_when_markers_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            source_root = repo_root / "src"
            (source_root / "config").mkdir(parents=True)
            (source_root / "orchestration").mkdir()
            (source_root / "main.py").write_text("", encoding="utf-8")

            detected = detect_source_root(repo_root)

            self.assertEqual(detected, source_root)

    def test_validate_repository_layout_uses_src_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            source_root = repo_root / "src"
            for relative_path in [
                "main.py",
                "bootstrap.py",
                "config/default.yaml",
                "config/loader.py",
                "engine/pipeline.py",
                "renderers/base.py",
                "renderers/factory.py",
                "reports/report_generator.py",
                "validation/validators.py",
                "drive/gdrive.py",
                "drive/sync_engine.py",
                "stitching/service.py",
                "stitching/ffmpeg_wrapper.py",
                "orchestration/__init__.py",
            ]:
                target = source_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")

            result = validate_repository_layout(repo_root)

            self.assertTrue(result.ready)
            self.assertEqual(result.optional_missing, [])

    def test_execute_bootstrap_flow_records_failed_and_skipped_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            launcher = KaggleNotebookLauncher(root, input_root=root, working_root=root)
            with patch.object(
                KaggleNotebookLauncher,
                "bootstrap_context",
                side_effect=FileNotFoundError("jobs.csv missing"),
            ), patch.object(KaggleNotebookLauncher, "display_preparation", return_value=None):
                state = launcher.execute_bootstrap_flow()

            stages = {
                stage.stage_name: stage
                for stage in state.bootstrap_report.stages
            }
            self.assertIsNone(state.context)
            self.assertEqual(stages["bootstrap_context"].status, "FAILED")
            self.assertEqual(stages["prepare_runtime"].status, "SKIPPED")
            self.assertEqual(stages["run_preflight"].status, "SKIPPED")
            self.assertEqual(stages["display_preparation"].status, "PASS")
            self.assertIn("Repository", state.bootstrap_report.sections)
            self.assertFalse(state.bootstrap_report.ready_to_launch)

    def test_execute_bootstrap_flow_is_ready_in_clean_working_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            working_root = temp_root / "working"
            input_root = temp_root / "input"
            project_root = input_root / "project-a"
            image_dir = project_root / "images"
            image_dir.mkdir(parents=True)
            working_root.mkdir(parents=True)
            (image_dir / "start.png").write_bytes(b"png")
            (project_root / "jobs.csv").write_text(
                "prompt,start_image,end_image,duration,resolution,aspect_ratio,seed,guidance_scale,num_inference_steps\n"
                "test,images/start.png,,5 Seconds,480p,1:1 Square,1,3.0,8\n",
                encoding="utf-8",
            )

            dependency_report = DependencyProfileReport(
                profile_name="kaggle_bulk",
                platform_name="linux",
                kaggle_mode=True,
                renderer_backend="diffusers",
                enabled_features=[
                    "core_renderer",
                    "diffusers_backend",
                    "csv_batch_rendering",
                    "resume_engine",
                    "reporting",
                ],
                disabled_features={"google_drive_integration": "Disabled by execution profile or configuration"},
                feature_statuses=[
                    FeatureRuntimeStatus(feature_key="core_renderer", enabled=True, required=True, state="PASS", reason="Feature runtime verified"),
                    FeatureRuntimeStatus(feature_key="diffusers_backend", enabled=True, required=True, state="PASS", reason="Feature runtime verified"),
                    FeatureRuntimeStatus(feature_key="csv_batch_rendering", enabled=True, required=True, state="PASS", reason="Feature runtime verified"),
                    FeatureRuntimeStatus(feature_key="resume_engine", enabled=True, required=False, state="PASS", reason="Feature runtime verified"),
                    FeatureRuntimeStatus(feature_key="reporting", enabled=True, required=True, state="PASS", reason="Feature runtime verified"),
                ],
                verification_checks=[
                    DependencyVerificationCheck(name="python", success=True, details="3.11.0", feature_key="core_renderer"),
                    DependencyVerificationCheck(name="ffmpeg", success=True, details="ffmpeg", feature_key="core_renderer"),
                    DependencyVerificationCheck(name="cuda", success=True, details="12.1", feature_key="core_renderer"),
                    DependencyVerificationCheck(name="gpu", success=True, details="NVIDIA T4", feature_key="core_renderer"),
                    DependencyVerificationCheck(name="renderer_initialization", success=True, details="renderers.factory import succeeded", feature_key="core_renderer"),
                    DependencyVerificationCheck(name="diffusers", success=True, details="diffusers import succeeded", feature_key="diffusers_backend"),
                ],
                python_version="3.11.0",
                torch_version="2.2.0",
                cuda_version="12.1",
            )
            preparation = PreparationResult(
                diagnostics={"status": "PASS"},
                preflight={"blocking_failures": [], "warnings": []},
                ready=True,
                started_at=datetime.now().isoformat(),
                completed_at=datetime.now().isoformat(),
            )
            environment_info = {
                "platform": "Linux-kaggle",
                "python_version": "3.11.0",
                "is_kaggle": True,
                "gpu": {"available": True, "name": "NVIDIA T4", "free_vram_bytes": 15 * 1024**3, "total_vram_bytes": 16 * 1024**3},
                "ram": {"total_bytes": 30 * 1024**3},
                "disk": {"free_bytes": 50 * 1024**3},
                "dataset_mounts": [str(project_root)],
            }

            self.assertFalse(any(working_root.iterdir()))
            with patch("orchestration.kaggle.ensure_runtime_dependency_profile", return_value=dependency_report), patch(
                "orchestration.kaggle.collect_environment_info",
                return_value=environment_info,
            ), patch(
                "orchestration.kaggle.ApplicationRunner.prepare",
                return_value=preparation,
            ), patch("orchestration.kaggle._render_markdown", return_value=None):
                launcher = KaggleNotebookLauncher(
                    repo_root,
                    input_root=input_root,
                    working_root=working_root,
                )
                state = launcher.execute_bootstrap_flow()

            self.assertTrue(state.bootstrap_report.ready_to_launch)
            self.assertEqual(
                state.bootstrap_report.sections["Dataset"].status,
                "PASS",
            )
            self.assertEqual(
                state.bootstrap_report.sections["Runtime"].status,
                "PASS",
            )
            self.assertEqual(state.context.discovery.jobs_csv_path, project_root / "jobs.csv")
            self.assertFalse(any(working_root.iterdir()))


if __name__ == "__main__":
    unittest.main()
