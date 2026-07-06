import json
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bootstrap import collect_environment_info
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
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
)

logger = get_logger("diagnostics")


@dataclass
class DiagnosticsResult:
    started_at: str
    completed_at: str
    status: str
    reports: list[ValidationReport]
    environment: dict[str, Any]
    config_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "environment": self.environment,
            "config_summary": self.config_summary,
            "reports": [report.to_dict() for report in self.reports],
        }


class DiagnosticsRunner:
    def __init__(self, config: Config):
        self.config = config
        self.validators = [
            ConfigValidator(),
            EnvironmentValidator(),
            DependencyValidator(),
            GPUValidator(),
            DiskValidator(),
            OutputValidator(),
            ManifestValidator(),
            CSVValidator(),
            ImageValidator(),
            ModelAssetValidator(),
        ]
        if self.config.enable_drive_upload:
            self.validators.append(DriveValidator())

    def run(self) -> DiagnosticsResult:
        started_at = datetime.now().isoformat()
        environment = collect_environment_info(self.config)
        reports = [validator.run(self.config) for validator in self.validators]
        if self.config.diagnostics_network_check:
            reports.append(self._network_report())
        status = self._derive_status(reports)
        result = DiagnosticsResult(
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            status=status,
            reports=reports,
            environment=environment,
            config_summary=self._build_config_summary(),
        )
        self.config.extra["diagnostics"] = result.to_dict()
        return result

    def save(self, result: DiagnosticsResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
        logger.info("Diagnostics report saved", extra={"job_id": "N/A"})

    def _derive_status(self, reports: list[ValidationReport]) -> str:
        if any(report.has_blocking_failures for report in reports):
            return "FAIL"
        if any(report.has_warnings for report in reports):
            return "WARNING"
        return "PASS"

    def _build_config_summary(self) -> dict[str, Any]:
        return {
            "config_version": self.config.config_version,
            "profile": self.config.profile,
            "renderer_backend": self.config.renderer_backend,
            "model_name": self.config.model_name,
            "output_dir": str(self.config.output_dir),
            "jobs_csv_path": str(self.config.jobs_csv_path),
            "reference_images_dir": str(self.config.reference_images_dir),
            "drive_enabled": self.config.enable_drive_upload,
            "benchmark_mode": self.config.benchmark_mode,
            "heartbeat_enabled": self.config.heartbeat_enabled,
        }

    def _network_report(self) -> ValidationReport:
        finding_severity = ValidationSeverity.PASS
        message = "Internet connectivity check succeeded"
        recommendation = ""
        try:
            with socket.create_connection(("8.8.8.8", 53), timeout=5):
                pass
        except OSError as exc:
            finding_severity = ValidationSeverity.WARNING
            message = f"Internet connectivity check failed: {exc}"
            recommendation = "Verify outbound connectivity if Drive uploads or model downloads are required."
        return ValidationReport(
            validator_name="NetworkValidator",
            findings=[
                ValidationFinding(
                    validator_name="NetworkValidator",
                    severity=finding_severity,
                    message=message,
                    recommendation=recommendation,
                    blocking=False,
                    execution_time_seconds=0.0,
                    details={},
                )
            ],
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            execution_time_seconds=0.0,
        )
