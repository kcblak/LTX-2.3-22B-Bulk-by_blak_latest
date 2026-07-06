from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationSeverity(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class ValidationFinding:
    validator_name: str
    severity: ValidationSeverity
    message: str
    recommendation: str = ""
    blocking: bool = False
    execution_time_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    validator_name: str
    findings: list[ValidationFinding] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    execution_time_seconds: float = 0.0

    @property
    def has_blocking_failures(self) -> bool:
        return any(
            finding.blocking and finding.severity == ValidationSeverity.FAIL
            for finding in self.findings
        )

    @property
    def has_failures(self) -> bool:
        return any(finding.severity == ValidationSeverity.FAIL for finding in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(
            finding.severity == ValidationSeverity.WARNING for finding in self.findings
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_name": self.validator_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time_seconds": self.execution_time_seconds,
            "findings": [
                {
                    "validator_name": finding.validator_name,
                    "severity": finding.severity.value,
                    "message": finding.message,
                    "recommendation": finding.recommendation,
                    "blocking": finding.blocking,
                    "execution_time_seconds": finding.execution_time_seconds,
                    "details": finding.details,
                }
                for finding in self.findings
            ],
        }
