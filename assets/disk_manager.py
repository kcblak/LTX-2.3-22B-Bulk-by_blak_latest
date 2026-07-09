from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DiskPlan:
    required_bytes: int
    available_bytes: int
    safety_margin_bytes: int
    download_bytes: int
    temp_overhead_bytes: int

    @property
    def ok(self) -> bool:
        return self.available_bytes >= self.required_bytes

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "required_bytes": self.required_bytes,
            "available_bytes": self.available_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "download_bytes": self.download_bytes,
            "temp_overhead_bytes": self.temp_overhead_bytes,
            "ok": self.ok,
        }


class DiskManager:
    def __init__(self, *, safety_margin_gb: float = 2.0) -> None:
        self.safety_margin_bytes = int(max(0.0, safety_margin_gb) * 1024**3)

    def available_bytes(self, path: Path) -> int:
        usage = shutil.disk_usage(path)
        return int(usage.free)

    def plan_download(
        self,
        *,
        destination_root: Path,
        download_bytes: int,
        temp_overhead_bytes: Optional[int] = None,
    ) -> DiskPlan:
        download_bytes = int(max(0, download_bytes))
        temp_overhead_bytes = (
            int(max(0, temp_overhead_bytes))
            if temp_overhead_bytes is not None
            else int(download_bytes * 0.2)
        )
        required = download_bytes + temp_overhead_bytes + self.safety_margin_bytes
        available = self.available_bytes(destination_root)
        return DiskPlan(
            required_bytes=required,
            available_bytes=available,
            safety_margin_bytes=self.safety_margin_bytes,
            download_bytes=download_bytes,
            temp_overhead_bytes=temp_overhead_bytes,
        )

