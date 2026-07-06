import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Config
from logging_system import get_logger
from validation import (
    CSVValidator,
    ConfigValidator,
    DependencyValidator,
    DiskValidator,
    DriveValidator,
    EnvironmentValidator,
    GPUValidator,
    ImageValidator,
    ManifestValidator,
    ModelAssetValidator,
    OutputValidator,
    ValidationReport,
    ValidationSeverity,
)

logger = get_logger("diagnostics")


@dataclass
class PreflightResult:
    started_at: str
    completed_at: str
    reports: list[ValidationReport]
    estimated_runtime_seconds: float
    estimated_storage_bytes: int
    estimated_upload_bytes: int
    expected_clip_count: int

    @property
    def blocking_failures(self) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for report in self.reports:
            for finding in report.findings:
                if finding.blocking and finding.severity == ValidationSeverity.FAIL:
                    findings.append(
                        {
                            "validator": report.validator_name,
                            "message": finding.message,
                            "recommendation": finding.recommendation,
                        }
                    )
        return findings

    @property
    def warnings(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for report in self.reports:
            for finding in report.findings:
                if finding.severity == ValidationSeverity.WARNING:
                    items.append(
                        {
                            "validator": report.validator_name,
                            "message": finding.message,
                            "recommendation": finding.recommendation,
                        }
                    )
        return items

    @property
    def ready(self) -> bool:
        return not self.blocking_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "ready": self.ready,
            "expected_clip_count": self.expected_clip_count,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
            "estimated_storage_bytes": self.estimated_storage_bytes,
            "estimated_upload_bytes": self.estimated_upload_bytes,
            "blocking_failures": self.blocking_failures,
            "warnings": self.warnings,
            "reports": [report.to_dict() for report in self.reports],
        }


class PreflightAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.validators = [
            ConfigValidator(),
            EnvironmentValidator(),
            DependencyValidator(),
            ManifestValidator(),
            OutputValidator(),
            DiskValidator(),
            GPUValidator(),
            CSVValidator(),
            ImageValidator(),
            ModelAssetValidator(),
        ]
        if self.config.enable_drive_upload:
            self.validators.append(DriveValidator())

    def _load_csv_rows(self) -> list[dict[str, str]]:
        if not self.config.jobs_csv_path.exists():
            return []
        with self.config.jobs_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _estimate_outputs(self, rows: list[dict[str, str]]) -> tuple[float, int, int]:
        clip_count = len(rows)
        total_runtime_seconds = 0.0
        total_storage_bytes = 0

        resolution_map = {"360p": 640 * 360, "480p": 854 * 480, "720p": 1280 * 720}
        duration_map = {"5s": 121, "10s": 241, "15s": 361}

        for row in rows:
            resolution_label = (row.get("resolution") or "480p").strip().lower()
            duration_label = (row.get("duration") or "5s").strip().lower()
            steps = int(row.get("steps") or self.config.num_inference_steps)
            pixels = resolution_map.get(resolution_label, 854 * 480)
            frames = duration_map.get(duration_label, 121)

            total_runtime_seconds += max(60.0, steps * frames * 0.75)
            total_storage_bytes += int(pixels * frames * 0.07)

        return total_runtime_seconds, total_storage_bytes, total_storage_bytes

    def analyze(self) -> PreflightResult:
        started = datetime.now().isoformat()
        reports = [validator.run(self.config) for validator in self.validators]
        rows = self._load_csv_rows()
        estimated_runtime_seconds, estimated_storage_bytes, estimated_upload_bytes = (
            self._estimate_outputs(rows)
        )
        result = PreflightResult(
            started_at=started,
            completed_at=datetime.now().isoformat(),
            reports=reports,
            estimated_runtime_seconds=estimated_runtime_seconds,
            estimated_storage_bytes=estimated_storage_bytes,
            estimated_upload_bytes=estimated_upload_bytes,
            expected_clip_count=len(rows),
        )
        self.config.extra["preflight_report"] = result.to_dict()
        return result

    def save(self, result: PreflightResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = int(max(0, seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        value = float(num_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if value < 1024.0 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} TB"

    def build_console_summary(self, result: PreflightResult) -> str:
        environment = self.config.extra.get("environment_info", {})
        gpu_info = environment.get("gpu", {})
        disk_info = environment.get("disk", {})
        free_disk = self._format_bytes(int(disk_info.get("free_bytes", 0)))
        gpu_name = gpu_info.get("name", "Unavailable") if gpu_info.get("available") else "Unavailable"
        vram = self._format_bytes(int(gpu_info.get("total_vram_bytes", 0)))

        missing_images = 0
        corrupted_images = 0
        duplicate_prompts = 0
        for report in result.reports:
            for finding in report.findings:
                details = finding.details
                missing_images += int(details.get("missing_images", 0))
                corrupted_images += int(details.get("corrupted_images", 0))
                duplicate_prompts += int(details.get("duplicate_prompts", 0))

        lines = [
            "=" * 50,
            "",
            "PROJECT PREFLIGHT",
            "",
            "=" * 50,
            f"Jobs                  {result.expected_clip_count}",
            f"Missing Images        {missing_images}",
            f"Corrupted Images      {corrupted_images}",
            f"Duplicate Prompts     {duplicate_prompts}",
            f"GPU                   {gpu_name}",
            f"VRAM                  {vram}",
            f"Disk Free             {free_disk}",
            f"Estimated Runtime     {self._format_duration(result.estimated_runtime_seconds)}",
            f"Estimated Storage     {self._format_bytes(result.estimated_storage_bytes)}",
            f"Estimated Upload      {self._format_bytes(result.estimated_upload_bytes)}",
            f"Drive                 {'Connected' if not any(r.validator_name == 'DriveValidator' and r.has_blocking_failures for r in result.reports) else 'FAIL'}",
            f"Configuration         {'PASS' if not any(r.validator_name == 'ConfigValidator' and r.has_blocking_failures for r in result.reports) else 'FAIL'}",
            f"Manifest              {'PASS' if not any(r.validator_name == 'ManifestValidator' and r.has_blocking_failures for r in result.reports) else 'FAIL'}",
            "=" * 50,
            "READY" if result.ready else "BLOCKED",
            "=" * 50,
        ]
        return "\n".join(lines)
