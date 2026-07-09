from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class AssetSpec:
    key: str
    filename: str
    backend: str
    required: bool
    priority: int = 100
    expected_size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    source_url: Optional[str] = None
    hf_repo_id: Optional[str] = None
    hf_filename: Optional[str] = None
    dependencies: tuple[str, ...] = ()

    def desired_path(self, root: Path) -> Path:
        return root / self.filename


@dataclass
class AssetManifest:
    backend: str
    assets: list[AssetSpec]

    def required_assets(self) -> list[AssetSpec]:
        return [asset for asset in self.assets if asset.required]

    def optional_assets(self) -> list[AssetSpec]:
        return [asset for asset in self.assets if not asset.required]

    def keys(self) -> set[str]:
        return {asset.key for asset in self.assets}


@dataclass
class AssetStateEntry:
    asset_key: str
    status: str
    source: str
    path: str
    size_bytes: int
    sha256: str
    started_at: str
    completed_at: str
    download_seconds: float = 0.0
    verify_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_key": self.asset_key,
            "status": self.status,
            "source": self.source,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "download_seconds": self.download_seconds,
            "verify_seconds": self.verify_seconds,
            "notes": list(self.notes),
        }


@dataclass
class DownloadManifest:
    entries: dict[str, AssetStateEntry] = field(default_factory=dict)

    def mark(self, entry: AssetStateEntry) -> None:
        self.entries[entry.asset_key] = entry

    def get(self, asset_key: str) -> Optional[AssetStateEntry]:
        return self.entries.get(asset_key)

    def to_dict(self) -> dict[str, Any]:
        return {key: entry.to_dict() for key, entry in self.entries.items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DownloadManifest":
        manifest = cls()
        for key, raw in payload.items():
            if not isinstance(raw, dict):
                continue
            manifest.entries[key] = AssetStateEntry(
                asset_key=raw.get("asset_key", key),
                status=raw.get("status", "unknown"),
                source=raw.get("source", "unknown"),
                path=raw.get("path", ""),
                size_bytes=int(raw.get("size_bytes", 0) or 0),
                sha256=str(raw.get("sha256", "") or ""),
                started_at=str(raw.get("started_at", "") or ""),
                completed_at=str(raw.get("completed_at", "") or ""),
                download_seconds=float(raw.get("download_seconds", 0.0) or 0.0),
                verify_seconds=float(raw.get("verify_seconds", 0.0) or 0.0),
                notes=list(raw.get("notes", []) or []),
            )
        return manifest

