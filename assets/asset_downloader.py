from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from assets.asset_manifest import AssetSpec


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    source: str
    path: Optional[Path] = None
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, str | float | bool | None]:
        return {
            "ok": self.ok,
            "source": self.source,
            "path": str(self.path) if self.path else None,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


class AssetDownloader:
    def __init__(
        self,
        *,
        hf_cache_dir: Optional[Path],
        temp_root: Path,
        drive_client: Optional[object] = None,
        drive_models_folder_id: Optional[str] = None,
    ) -> None:
        self.hf_cache_dir = hf_cache_dir
        self.temp_root = temp_root
        self.drive_client = drive_client
        self.drive_models_folder_id = drive_models_folder_id

    def download_from_huggingface(self, spec: AssetSpec) -> DownloadResult:
        started_at = datetime.now().isoformat()
        started = time.monotonic()
        try:
            if not spec.hf_repo_id or not spec.hf_filename:
                return DownloadResult(
                    ok=False,
                    source="hf_download",
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    duration_seconds=round(time.monotonic() - started, 2),
                    error="hf_repo_id or hf_filename not configured",
                )
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    repo_id=spec.hf_repo_id,
                    filename=spec.hf_filename,
                    cache_dir=str(self.hf_cache_dir) if self.hf_cache_dir else None,
                    resume_download=True,
                )
            )
            return DownloadResult(
                ok=True,
                source="hf_download",
                path=path,
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                duration_seconds=round(time.monotonic() - started, 2),
            )
        except Exception as exc:
            return DownloadResult(
                ok=False,
                source="hf_download",
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                duration_seconds=round(time.monotonic() - started, 2),
                error=str(exc),
            )

    def download_from_drive(self, spec: AssetSpec) -> DownloadResult:
        started_at = datetime.now().isoformat()
        started = time.monotonic()
        try:
            if self.drive_client is None or not self.drive_models_folder_id:
                return DownloadResult(
                    ok=False,
                    source="drive_download",
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    duration_seconds=round(time.monotonic() - started, 2),
                    error="drive client not configured",
                )
            metadata = self.drive_client.find_file_by_name(spec.filename, self.drive_models_folder_id)
            if metadata is None:
                return DownloadResult(
                    ok=False,
                    source="drive_download",
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    duration_seconds=round(time.monotonic() - started, 2),
                    error="file not found in drive cache",
                )
            self.temp_root.mkdir(parents=True, exist_ok=True)
            destination = self.temp_root / spec.filename
            path = self.drive_client.download_file(metadata.file_id, destination)
            return DownloadResult(
                ok=True,
                source="drive_download",
                path=Path(path),
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                duration_seconds=round(time.monotonic() - started, 2),
            )
        except Exception as exc:
            return DownloadResult(
                ok=False,
                source="drive_download",
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                duration_seconds=round(time.monotonic() - started, 2),
                error=str(exc),
            )

