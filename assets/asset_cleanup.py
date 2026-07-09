from __future__ import annotations

import shutil
from pathlib import Path


class AssetCleanup:
    def __init__(self, temp_root: Path) -> None:
        self.temp_root = temp_root

    def ensure_temp_root(self) -> None:
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def delete_path(self, path: Path) -> None:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except Exception:
            return

    def cleanup_temp_root(self) -> None:
        self.delete_path(self.temp_root)
