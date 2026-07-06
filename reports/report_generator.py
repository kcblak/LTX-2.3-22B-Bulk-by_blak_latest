import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from config import Config
from core import IReporter, JobStatus
from jobs.job_queue import JobQueue
from logging_system import get_logger

logger = get_logger("reports")

FINAL_SUCCESS_STATUSES = {JobStatus.COMPLETED, JobStatus.UPLOADED}
FINAL_FAILURE_STATUSES = {
    JobStatus.FAILED_VALIDATION,
    JobStatus.FAILED_RENDER,
    JobStatus.FAILED_UPLOAD,
    JobStatus.FAILED_VERIFY,
}


class ReportGenerator(IReporter):
    """Generates operational reports and benchmark artifacts."""

    def __init__(
        self,
        config: Config,
        job_queue: JobQueue,
        runtime_monitor: Optional[Any] = None,
    ):
        self.config = config
        self.job_queue = job_queue
        self.runtime_monitor = runtime_monitor

    def _aggregate_render_metrics(self) -> dict[str, float]:
        aggregate_metrics = {
            "image_loading_seconds": 0.0,
            "prompt_preparation_seconds": 0.0,
            "inference_seconds": 0.0,
            "encoding_seconds": 0.0,
            "validation_seconds": 0.0,
            "total_seconds": 0.0,
        }
        jobs_with_metrics = 0
        for job in self.job_queue.jobs:
            if not job.render_metrics:
                continue
            jobs_with_metrics += 1
            for key in aggregate_metrics:
                aggregate_metrics[key] += float(job.render_metrics.get(key, 0.0))
        aggregate_metrics["jobs_with_metrics"] = float(jobs_with_metrics)
        return aggregate_metrics

    def _average_metrics(self, aggregate_metrics: dict[str, float]) -> dict[str, float]:
        jobs_with_metrics = int(aggregate_metrics.get("jobs_with_metrics", 0))
        return {
            f"average_{key}": (
                value / jobs_with_metrics if jobs_with_metrics else 0.0
            )
            for key, value in aggregate_metrics.items()
            if key != "jobs_with_metrics"
        }

    def _total_duration(self) -> timedelta:
        total_duration = timedelta(0)
        for job in self.job_queue.jobs:
            if job.started_at and job.completed_at:
                total_duration += job.completed_at - job.started_at
        return total_duration

    def _job_counts(self) -> dict[str, int]:
        total = len(self.job_queue.jobs)
        completed = len([job for job in self.job_queue.jobs if job.status in FINAL_SUCCESS_STATUSES])
        failed = len([job for job in self.job_queue.jobs if job.status in FINAL_FAILURE_STATUSES])
        uploaded = len(self.job_queue.get_jobs_by_status(JobStatus.UPLOADED))
        return {
            "total_jobs": total,
            "completed_jobs": completed,
            "uploaded_jobs": uploaded,
            "failed_jobs": failed,
            "failed_validation": len(self.job_queue.get_jobs_by_status(JobStatus.FAILED_VALIDATION)),
            "failed_render": len(self.job_queue.get_jobs_by_status(JobStatus.FAILED_RENDER)),
            "failed_upload": len(self.job_queue.get_jobs_by_status(JobStatus.FAILED_UPLOAD)),
            "failed_verify": len(self.job_queue.get_jobs_by_status(JobStatus.FAILED_VERIFY)),
            "pending_jobs": len(self.job_queue.get_jobs_by_status(JobStatus.PENDING)),
            "success_rate": (completed / total if total else 0.0),
            "failure_rate": (failed / total if total else 0.0),
        }

    def _build_benchmark_section(self, average_metrics: dict[str, float]) -> dict[str, Any]:
        benchmark = {
            "enabled": self.config.benchmark_mode,
            "max_jobs": self.config.benchmark_max_jobs,
            "compare_previous": self.config.benchmark_compare_previous,
            "current_run": {
                "run_id": self.config.run_id,
                "profile": self.config.profile,
                "renderer_backend": self.config.renderer_backend,
                "average_total_job_seconds": average_metrics.get("average_total_seconds", 0.0),
            },
        }
        history_path = self.config.benchmark_history_path
        previous = None
        if self.config.benchmark_compare_previous and history_path.exists():
            try:
                previous = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])[-1]
            except Exception:
                previous = None
        if previous is not None:
            delta = benchmark["current_run"]["average_total_job_seconds"] - float(
                previous.get("average_total_job_seconds", 0.0)
            )
            benchmark["comparison"] = {
                "previous_run_id": previous.get("run_id"),
                "previous_average_total_job_seconds": previous.get("average_total_job_seconds", 0.0),
                "delta_average_total_job_seconds": delta,
            }
        return benchmark

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a consolidated operational report."""
        aggregate_metrics = self._aggregate_render_metrics()
        average_metrics = self._average_metrics(aggregate_metrics)
        counts = self._job_counts()
        total_duration = self._total_duration()
        runtime_export = self.runtime_monitor.export() if self.runtime_monitor is not None else {}

        summary = {
            "version": self.config.app_version,
            "generated_at": datetime.now().isoformat(),
            "project": {
                "project_id": self.config.project_id,
                "run_id": self.config.run_id,
                "app_version": self.config.app_version,
                "profile": self.config.profile,
                "renderer_backend": self.config.renderer_backend,
                "model_name": self.config.model_name,
                "benchmark_mode": self.config.benchmark_mode,
            },
            "summary": {
                **counts,
                "total_processing_time": str(total_duration),
                **average_metrics,
            },
            "environment": self.config.extra.get("environment_info", {}),
            "configuration": self.config.to_dict(),
            "drive": self.config.extra.get("drive_metrics", {}),
            "preflight": self.config.extra.get("preflight_report", {}),
            "diagnostics": self.config.extra.get("diagnostics", {}),
            "validation": self.config.extra.get("validation_report", {}),
            "performance": {
                "aggregate": aggregate_metrics,
                "runtime_monitor": runtime_export,
            },
            "benchmark": self._build_benchmark_section(average_metrics),
            "stitching": self.config.extra.get("stitching", {}),
            "jobs": [job.to_dict() for job in self.job_queue.jobs],
        }
        return summary

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_json_report(self, path: Path, summary: Optional[Dict[str, Any]] = None) -> None:
        report = summary or self.generate_summary()
        self._write_json(path, report)
        logger.info("JSON report saved", extra={"job_id": "N/A"})

    def save_text_report(self, path: Path, summary: Optional[Dict[str, Any]] = None) -> None:
        summary = summary or self.generate_summary()
        lines = [
            "=" * 80,
            "LTX VIDEO BULK RENDERER - EXECUTION REPORT",
            "=" * 80,
            "",
            f"Generated at: {summary['generated_at']}",
            f"Project ID: {summary['project']['project_id']}",
            f"Run ID: {summary['project']['run_id']}",
            f"Profile: {summary['project']['profile']}",
            f"Renderer Backend: {summary['project']['renderer_backend']}",
            "",
            "SUMMARY",
            "-" * 80,
        ]
        for key, value in summary["summary"].items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        diagnostics_status = summary.get("diagnostics", {}).get("status")
        if diagnostics_status:
            lines.append(f"Diagnostics Status: {diagnostics_status}")
        preflight_ready = summary.get("preflight", {}).get("ready")
        if preflight_ready is not None:
            lines.append(f"Preflight Ready: {preflight_ready}")
        lines.append("")
        lines.append("=" * 80)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Text report saved", extra={"job_id": "N/A"})

    def save_csv_report(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "job_id",
                    "status",
                    "prompt",
                    "started_at",
                    "completed_at",
                    "error_message",
                    "render_total_seconds",
                    "upload_total_seconds",
                    "output_path",
                ],
            )
            writer.writeheader()
            for job in self.job_queue.jobs:
                writer.writerow(
                    {
                        "job_id": job.job_id,
                        "status": job.status.name,
                        "prompt": job.prompt,
                        "started_at": job.started_at.isoformat() if job.started_at else "",
                        "completed_at": job.completed_at.isoformat() if job.completed_at else "",
                        "error_message": job.error_message or "",
                        "render_total_seconds": job.render_metrics.get("total_seconds", 0.0),
                        "upload_total_seconds": job.upload_metrics.get("total_seconds", 0.0),
                        "output_path": str(job.output_path) if job.output_path else "",
                    }
                )
        logger.info("CSV report saved", extra={"job_id": "N/A"})

    def save_performance_report(
        self,
        path: Path,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        data = (summary or self.generate_summary())["performance"]
        self._write_json(path, data)
        logger.info("Performance report saved", extra={"job_id": "N/A"})

    def save_benchmark_json(
        self,
        path: Path,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        data = (summary or self.generate_summary())["benchmark"]
        self._write_json(path, data)
        logger.info("Benchmark JSON saved", extra={"job_id": "N/A"})

    def save_benchmark_csv(
        self,
        path: Path,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        benchmark = (summary or self.generate_summary())["benchmark"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_id",
                    "profile",
                    "renderer_backend",
                    "average_total_job_seconds",
                    "previous_run_id",
                    "previous_average_total_job_seconds",
                    "delta_average_total_job_seconds",
                ],
            )
            writer.writeheader()
            comparison = benchmark.get("comparison", {})
            writer.writerow(
                {
                    "run_id": self.config.run_id,
                    "profile": self.config.profile,
                    "renderer_backend": self.config.renderer_backend,
                    "average_total_job_seconds": benchmark["current_run"]["average_total_job_seconds"],
                    "previous_run_id": comparison.get("previous_run_id", ""),
                    "previous_average_total_job_seconds": comparison.get("previous_average_total_job_seconds", ""),
                    "delta_average_total_job_seconds": comparison.get("delta_average_total_job_seconds", ""),
                }
            )
        logger.info("Benchmark CSV saved", extra={"job_id": "N/A"})

    def save_performance_summary(
        self,
        path: Path,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        summary = summary or self.generate_summary()
        perf = summary["performance"]
        benchmark = summary["benchmark"]
        lines = [
            "LTX BULK RENDERER PERFORMANCE SUMMARY",
            "",
            f"Run ID: {self.config.run_id}",
            f"Profile: {self.config.profile}",
            f"Renderer Backend: {self.config.renderer_backend}",
            f"Average Job Time: {benchmark['current_run']['average_total_job_seconds']:.2f}s",
            f"Completed Jobs: {summary['summary']['completed_jobs']}",
            f"Failed Jobs: {summary['summary']['failed_jobs']}",
            f"Average Upload Time: {perf['runtime_monitor'].get('latest', {}).get('average_upload_time_seconds', 0.0):.2f}s",
            f"Average Render Time: {perf['runtime_monitor'].get('latest', {}).get('average_render_time_seconds', 0.0):.2f}s",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Performance summary saved", extra={"job_id": "N/A"})

    def save_validation_report(self, path: Path) -> None:
        data = self.config.extra.get("validation_report", {})
        self._write_json(path, data)
        logger.info("Validation report saved", extra={"job_id": "N/A"})

    def save_diagnostics_report(self, path: Path) -> None:
        data = self.config.extra.get("diagnostics", {})
        self._write_json(path, data)
        logger.info("Diagnostics report saved", extra={"job_id": "N/A"})

    def save_all(self) -> Dict[str, Any]:
        summary = self.generate_summary()
        self.save_json_report(self.config.report_path, summary)
        self.save_text_report(self.config.summary_path, summary)
        self.save_csv_report(self.config.project_report_csv_path)
        self.save_performance_report(self.config.performance_report_path, summary)
        self.save_benchmark_json(self.config.benchmark_json_path, summary)
        self.save_benchmark_csv(self.config.benchmark_csv_path, summary)
        self.save_performance_summary(self.config.performance_summary_path, summary)
        self.save_validation_report(self.config.validation_report_path)
        self.save_diagnostics_report(self.config.diagnostics_path)
        self._update_benchmark_history(summary["benchmark"])
        return summary

    def _update_benchmark_history(self, benchmark_section: dict[str, Any]) -> None:
        if not self.config.benchmark_mode:
            return
        path = self.config.benchmark_history_path
        history: dict[str, Any] = {"history": []}
        if path.exists():
            try:
                history = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                history = {"history": []}
        history.setdefault("history", []).append(
            {
                "run_id": self.config.run_id,
                "profile": self.config.profile,
                "renderer_backend": self.config.renderer_backend,
                "average_total_job_seconds": benchmark_section["current_run"]["average_total_job_seconds"],
                "generated_at": datetime.now().isoformat(),
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Benchmark history updated", extra={"job_id": "N/A"})
