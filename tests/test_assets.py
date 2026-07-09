import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from assets.disk_manager import DiskManager
from config import Config


class FakeDriveClient:
    def __init__(self, file_map: dict[str, Path]) -> None:
        self.file_map = file_map
        self.downloaded: list[str] = []

    def ensure_project_structure(self, project_name: str):
        return SimpleNamespace(folders={"models": "folder-1"})

    def find_file_by_name(self, name: str, folder_id: str):
        if name not in self.file_map:
            return None
        return SimpleNamespace(file_id=name)

    def download_file(self, file_id: str, destination_path: Path) -> Path:
        destination_path.write_bytes(self.file_map[file_id].read_bytes())
        self.downloaded.append(file_id)
        return destination_path

    def upload_file(self, local_path: Path, remote_name: str, folder_id: str):
        return None


class AssetsTests(unittest.TestCase):
    def test_disk_manager_rejects_when_space_is_low(self):
        manager = DiskManager(safety_margin_gb=2.0)
        with patch("assets.disk_manager.shutil.disk_usage") as usage:
            usage.return_value = SimpleNamespace(total=0, used=0, free=1024)
            plan = manager.plan_download(destination_root=Path("."), download_bytes=10 * 1024)
        self.assertFalse(plan.ok)

    def test_asset_manager_reuses_cached_files_without_downloading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = Config()
            config.model_cache_dir = root / "models"
            config.hf_cache_dir = root / "hf_cache"
            config.asset_temp_dir = root / "tmp"
            config.asset_report_dir = root / "reports"
            config.asset_download_manifest_path = root / "download_manifest.json"
            config.features["wan2gp_runtime"] = True
            config.wan2gp_required_companion_files = ["a.bin"]
            config.wan2gp_required_text_encoder_files = ["b.json"]
            config.wan2gp_text_encoder_dirname = "encoder"
            config.wan2gp_msr_enabled = False

            cache_root = config.model_cache_dir / "wan2gp"
            (cache_root / "encoder").mkdir(parents=True, exist_ok=True)
            (cache_root / config.wan2gp_transformer_filename).write_bytes(b"x")
            (cache_root / "a.bin").write_bytes(b"y")
            (cache_root / "encoder" / "b.json").write_bytes(b"z")

            from assets import AssetManager

            try:
                import huggingface_hub
            except Exception:
                self.skipTest("huggingface_hub is not installed")

            with patch("huggingface_hub.hf_hub_download") as download:
                report = AssetManager(config).ensure_assets(backend="wan2gp")
                self.assertTrue(report.ready)
                download.assert_not_called()

    def test_asset_manager_can_pull_from_drive_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = Config()
            config.model_cache_dir = root / "models"
            config.hf_cache_dir = root / "hf_cache"
            config.asset_temp_dir = root / "tmp"
            config.asset_report_dir = root / "reports"
            config.asset_download_manifest_path = root / "download_manifest.json"
            config.enable_drive_model_cache = True
            config.drive_model_cache_folder_name = "models"
            config.features["wan2gp_runtime"] = True
            config.wan2gp_required_companion_files = []
            config.wan2gp_required_text_encoder_files = []
            config.wan2gp_msr_enabled = False
            config.wan2gp_transformer_filename = "transformer.gguf"

            source_file = root / "drive_transformer.gguf"
            source_file.write_bytes(b"data")
            drive = FakeDriveClient({"transformer.gguf": source_file})

            from assets import AssetManager

            try:
                import huggingface_hub
            except Exception:
                self.skipTest("huggingface_hub is not installed")

            with patch("huggingface_hub.hf_hub_download") as download:
                report = AssetManager(config, drive_client=drive).ensure_assets(backend="wan2gp")
                self.assertTrue(report.ready)
                self.assertIn("transformer.gguf", drive.downloaded)
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
