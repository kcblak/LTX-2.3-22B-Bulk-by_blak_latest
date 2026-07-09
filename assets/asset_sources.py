from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SOURCE_DATASET = "dataset"
SOURCE_USER_DIR = "user_dir"
SOURCE_LOCAL_CACHE = "local_cache"
SOURCE_HF_CACHE = "hf_cache"
SOURCE_DRIVE_CACHE = "drive_cache"
SOURCE_HF_DOWNLOAD = "hf_download"
SOURCE_DRIVE_DOWNLOAD = "drive_download"


@dataclass(frozen=True)
class AssetCandidate:
    source: str
    path: Path
    description: str = ""
    remote_id: Optional[str] = None

