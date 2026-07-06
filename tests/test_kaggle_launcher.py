import os
import tempfile
import unittest
from pathlib import Path

from orchestration.kaggle import (
    discover_drive_credentials,
    discover_kaggle_project,
    validate_repository_layout,
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


if __name__ == "__main__":
    unittest.main()
