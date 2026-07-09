from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from assets.asset_manifest import AssetSpec


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str
    size_bytes: int = 0
    sha256: str = ""

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


class AssetValidator:
    def __init__(self, *, chunk_size: int = 8 * 1024 * 1024) -> None:
        self.chunk_size = int(max(1024, chunk_size))

    def compute_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(self.chunk_size)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def validate(
        self,
        spec: AssetSpec,
        path: Path,
        *,
        expected_size_bytes: Optional[int] = None,
        expected_sha256: Optional[str] = None,
    ) -> ValidationResult:
        if not path.exists():
            return ValidationResult(ok=False, reason="missing")
        try:
            size_bytes = path.stat().st_size
        except Exception as exc:
            return ValidationResult(ok=False, reason=f"stat_failed:{exc}")
        expected_size = expected_size_bytes if expected_size_bytes is not None else spec.expected_size_bytes
        if expected_size is not None and size_bytes != expected_size:
            return ValidationResult(
                ok=False,
                reason="size_mismatch",
                size_bytes=size_bytes,
            )
        sha256 = ""
        expected_hash = expected_sha256 or spec.sha256
        if expected_hash:
            try:
                sha256 = self.compute_sha256(path)
            except Exception as exc:
                return ValidationResult(ok=False, reason=f"hash_failed:{exc}", size_bytes=size_bytes)
            if sha256.lower() != expected_hash.lower():
                return ValidationResult(ok=False, reason="checksum_mismatch", size_bytes=size_bytes, sha256=sha256)
        return ValidationResult(ok=True, reason="ok", size_bytes=size_bytes, sha256=sha256)
