from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from bootstrap import bootstrap
from config import Config
from diagnostics import DiagnosticsRunner
from engine.pipeline import Pipeline
from logging_system import get_logger, setup_logging
from observability import EventBus, RuntimeMonitor
from observability.failures import classify_exception
from preflight import PreflightAnalyzer


@dataclass
class PreparationResult:
    diagnostics: dict[str, Any]
    preflight: dict[str, Any]
    ready: bool
    started_at: str
    completed_at: str


@dataclass
class ApplicationRunResult:
    success: bool
    exit_code: int
    started_at: str
    completed_at: str
    config: Config
    diagnostics: dict[str, Any]
    preflight: dict[str, Any]
    summary: dict[str, Any]
    failure: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "diagnostics": self.diagnostics,
            "preflight": self.preflight,
            "summary": self.summary,
            "failure": self.failure,
            "config": self.config.to_dict(),
        }


class ApplicationRunner:
    def __init__(
        self,
        config: Config,
        *,
        event_bus: Optional[EventBus] = None,
        runtime_monitor: Optional[RuntimeMonitor] = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus or EventBus()
        self.runtime_monitor = runtime_monitor or RuntimeMonitor(config)
        self.logger = get_logger("main")
        self._logging_initialized = False
        self._prepared: Optional[PreparationResult] = None

    def _initialize_logging(self) -> None:
        if self._logging_initialized:
            return
        setup_logging(self.config)
        self.logger = get_logger("main")
        self.logger.info("=" * 80, extra={"job_id": "N/A"})
        self.logger.info("LTX VIDEO BULK RENDERER", extra={"job_id": "N/A"})
        self.logger.info("=" * 80, extra={"job_id": "N/A"})
        self._logging_initialized = True

    def prepare(self) -> PreparationResult:
        if self._prepared is not None:
            return self._prepared

        self._initialize_logging()
        started_at = datetime.now().isoformat()
        diagnostics_runner = DiagnosticsRunner(self.config)
        preflight = PreflightAnalyzer(self.config)

        diagnostics_result = diagnostics_runner.run()
        diagnostics_runner.save(diagnostics_result, self.config.diagnostics_path)

        bootstrap(self.config)

        preflight_result = preflight.analyze()
        preflight.save(preflight_result, self.config.validation_report_path)
        self.config.extra["validation_report"] = preflight_result.to_dict()
        self.logger.info(
            "\n" + preflight.build_console_summary(preflight_result),
            extra={"job_id": "N/A"},
        )

        self._prepared = PreparationResult(
            diagnostics=diagnostics_result.to_dict(),
            preflight=preflight_result.to_dict(),
            ready=not preflight_result.blocking_failures
            and not (
                self.config.preflight_abort_on_warning and preflight_result.warnings
            ),
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
        )
        return self._prepared

    def _result_from_failure(
        self,
        *,
        started_at: str,
        diagnostics: Optional[dict[str, Any]],
        preflight: Optional[dict[str, Any]],
        exc: Exception,
    ) -> ApplicationRunResult:
        failure = classify_exception(exc)
        self.logger.critical(f"FAILED: {exc}", extra={"job_id": "N/A"}, exc_info=True)
        self.logger.critical(
            (
                f"Failure category={failure.category} "
                f"recommendation={failure.recommendation}"
            ),
            extra={"job_id": "N/A"},
        )
        return ApplicationRunResult(
            success=False,
            exit_code=1,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            config=self.config,
            diagnostics=diagnostics or {},
            preflight=preflight or {},
            summary={},
            failure={
                "category": failure.category,
                "summary": failure.summary,
                "recommendation": failure.recommendation,
                "exception_type": exc.__class__.__name__,
            },
        )

    def _result_from_preparation(self, preparation: PreparationResult) -> ApplicationRunResult:
        blocking_failures = preparation.preflight.get("blocking_failures", [])
        warnings = preparation.preflight.get("warnings", [])
        if blocking_failures:
            return ApplicationRunResult(
                success=False,
                exit_code=1,
                started_at=preparation.started_at,
                completed_at=datetime.now().isoformat(),
                config=self.config,
                diagnostics=preparation.diagnostics,
                preflight=preparation.preflight,
                summary={},
                failure={
                    "category": "preflight",
                    "summary": (
                        "Preflight failed with blocking issues. "
                        "See validation_report.json for details."
                    ),
                    "recommendation": "Resolve blocking validation failures and rerun.",
                    "blocking_failures": blocking_failures,
                },
            )
        if self.config.preflight_abort_on_warning and warnings:
            return ApplicationRunResult(
                success=False,
                exit_code=1,
                started_at=preparation.started_at,
                completed_at=datetime.now().isoformat(),
                config=self.config,
                diagnostics=preparation.diagnostics,
                preflight=preparation.preflight,
                summary={},
                failure={
                    "category": "preflight-warning",
                    "summary": (
                        "Preflight warnings are configured to abort the run. "
                        "See validation_report.json for details."
                    ),
                    "recommendation": "Resolve warnings or disable preflight_abort_on_warning.",
                    "warnings": warnings,
                },
            )
        return ApplicationRunResult(
            success=True,
            exit_code=0,
            started_at=preparation.started_at,
            completed_at=datetime.now().isoformat(),
            config=self.config,
            diagnostics=preparation.diagnostics,
            preflight=preparation.preflight,
            summary={},
            failure={},
        )

    def run_pipeline_only(self) -> ApplicationRunResult:
        started_at = datetime.now().isoformat()
        try:
            preparation = self.prepare()
            preparation_gate = self._result_from_preparation(preparation)
            if not preparation_gate.success:
                return preparation_gate

            pipeline = Pipeline(
                self.config,
                event_bus=self.event_bus,
                runtime_monitor=self.runtime_monitor,
            )
            summary = pipeline.run()
            self.logger.info("=" * 80, extra={"job_id": "N/A"})
            self.logger.info("SUCCESS: All jobs processed!", extra={"job_id": "N/A"})
            self.logger.info("=" * 80, extra={"job_id": "N/A"})
            return ApplicationRunResult(
                success=True,
                exit_code=0,
                started_at=preparation.started_at,
                completed_at=datetime.now().isoformat(),
                config=self.config,
                diagnostics=preparation.diagnostics,
                preflight=preparation.preflight,
                summary=summary,
                failure={},
            )
        except Exception as exc:
            prepared = self._prepared
            return self._result_from_failure(
                started_at=prepared.started_at if prepared else started_at,
                diagnostics=prepared.diagnostics if prepared else None,
                preflight=prepared.preflight if prepared else None,
                exc=exc,
            )

    def run(self, *, preflight_only: bool = False) -> ApplicationRunResult:
        started_at = datetime.now().isoformat()
        try:
            preparation = self.prepare()
            gated_result = self._result_from_preparation(preparation)
            if not gated_result.success or preflight_only:
                if preflight_only and gated_result.success:
                    self.logger.info(
                        "Preflight-only mode complete",
                        extra={"job_id": "N/A"},
                    )
                return gated_result
            return self.run_pipeline_only()
        except Exception as exc:
            prepared = self._prepared
            return self._result_from_failure(
                started_at=prepared.started_at if prepared else started_at,
                diagnostics=prepared.diagnostics if prepared else None,
                preflight=prepared.preflight if prepared else None,
                exc=exc,
            )
