from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from assets.asset_manifest import AssetSpec
from assets.asset_sources import AssetCandidate


@dataclass(frozen=True)
class CacheDecision:
    action: str
    source: str
    source_path: str
    destination_path: str
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "action": self.action,
            "source": self.source,
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "note": self.note,
        }


class AssetCache:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root

    def destination_for(self, spec: AssetSpec) -> Path:
        return spec.desired_path(self.cache_root)

    def ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def link_or_copy(self, candidate: AssetCandidate, destination: Path) -> CacheDecision:
        self.ensure_parent(destination)
        if destination.exists():
            return CacheDecision(
                action="exists",
                source=candidate.source,
                source_path=str(candidate.path),
                destination_path=str(destination),
            )
        try:
            os.link(candidate.path, destination)
            return CacheDecision(
                action="hardlink",
                source=candidate.source,
                source_path=str(candidate.path),
                destination_path=str(destination),
            )
        except Exception:
            shutil.copy2(candidate.path, destination)
            return CacheDecision(
                action="copy",
                source=candidate.source,
                source_path=str(candidate.path),
                destination_path=str(destination),
            )

    def remove(self, path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            return

