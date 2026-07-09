from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from assets.asset_manifest import AssetStateEntry, AssetManifest


@dataclass
class AssetReport:
    backend: str
    ready: bool
    started_at: str
    completed_at: str
    disk_plan: dict[str, Any] = field(default_factory=dict)
    entries: list[AssetStateEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "ready": self.ready,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "disk_plan": dict(self.disk_plan),
            "entries": [entry.to_dict() for entry in self.entries],
            "notes": list(self.notes),
        }


class AssetReportWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def write_json(self, report: AssetReport, filename: str = "asset_report.json") -> Path:
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_text(self, report: AssetReport, filename: str = "asset_report.txt") -> Path:
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("ASSET REPORT")
        lines.append("=" * 80)
        lines.append(f"backend: {report.backend}")
        lines.append(f"ready: {report.ready}")
        lines.append(f"started_at: {report.started_at}")
        lines.append(f"completed_at: {report.completed_at}")
        if report.disk_plan:
            lines.append("")
            lines.append("disk_plan:")
            for key, value in report.disk_plan.items():
                lines.append(f"  {key}: {value}")
        if report.notes:
            lines.append("")
            lines.append("notes:")
            for note in report.notes:
                lines.append(f" - {note}")
        lines.append("")
        lines.append("assets:")
        for entry in report.entries:
            lines.append(
                f" - {entry.asset_key}: {entry.status} source={entry.source} size={entry.size_bytes}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_markdown(self, report: AssetReport, filename: str = "asset_report.md") -> Path:
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("# Asset Report")
        lines.append("")
        lines.append(f"- backend: {report.backend}")
        lines.append(f"- ready: {report.ready}")
        lines.append(f"- started_at: {report.started_at}")
        lines.append(f"- completed_at: {report.completed_at}")
        if report.disk_plan:
            lines.append("")
            lines.append("## Disk Plan")
            for key, value in report.disk_plan.items():
                lines.append(f"- {key}: {value}")
        if report.notes:
            lines.append("")
            lines.append("## Notes")
            for note in report.notes:
                lines.append(f"- {note}")
        lines.append("")
        lines.append("## Assets")
        lines.append("")
        lines.append("| Asset | Status | Source | Size (bytes) | SHA256 | |")
        lines.append("|---|---|---|---:|---|---|")
        for entry in report.entries:
            lines.append(
                f"| {entry.asset_key} | {entry.status} | {entry.source} | {entry.size_bytes} | {entry.sha256} | |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

