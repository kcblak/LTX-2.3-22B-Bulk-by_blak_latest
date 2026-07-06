import json
import tempfile
import unittest
from pathlib import Path

from config import Config
from core import AspectRatio, Duration, JobStatus, Resolution
from jobs.job import Job
from observability.runtime import RuntimeMonitor
from reports.report_generator import ReportGenerator


class _FakeJobQueue:
    def __init__(self, jobs):
        self.jobs = jobs

    def get_jobs_by_status(self, status):
        return [job for job in self.jobs if job.status == status]


class ObservabilityReportingTests(unittest.TestCase):
    def test_runtime_monitor_writes_heartbeat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = Config(
                output_dir=root / "outputs",
                log_dir=root / "logs",
                temp_dir=root / "temp",
                heartbeat_path=root / "outputs" / "heartbeat.json",
                heartbeat_enabled=True,
            )
            monitor = RuntimeMonitor(config)
            monitor.set_status("RUNNING")
            monitor.set_current_job("JOB-001")
            monitor.set_upload_queue_length(2)
            monitor.record_render_time(12.5)

            snapshot = monitor.write_heartbeat(force=True)

            self.assertEqual(snapshot["status"], "RUNNING")
            self.assertEqual(snapshot["current_job"], "JOB-001")
            self.assertTrue(config.heartbeat_path.exists())
            written = json.loads(config.heartbeat_path.read_text(encoding="utf-8"))
            self.assertEqual(written["upload_queue"], 2)

    def test_report_generator_saves_artifacts_and_benchmark_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = Config(
                output_dir=root / "outputs",
                log_dir=root / "logs",
                temp_dir=root / "temp",
                report_path=root / "outputs" / "project_report.json",
                summary_path=root / "outputs" / "summary.txt",
                project_report_csv_path=root / "outputs" / "project_report.csv",
                performance_report_path=root / "outputs" / "performance.json",
                diagnostics_path=root / "outputs" / "diagnostics.json",
                validation_report_path=root / "outputs" / "validation_report.json",
                benchmark_history_path=root / "outputs" / "benchmark_history.json",
                benchmark_mode=True,
                benchmark_max_jobs=1,
                enable_drive_upload=False,
            )
            job = Job(
                job_id="JOB-001",
                prompt="A scene",
                start_image=root / "image.png",
                end_image=None,
                duration=Duration.D5,
                resolution=Resolution.R480,
                aspect_ratio=AspectRatio.AR_1_1,
                seed=1,
                guidance_scale=3.0,
                num_inference_steps=8,
                status=JobStatus.COMPLETED,
                render_metrics={"total_seconds": 10.0, "inference_seconds": 7.0},
            )
            monitor = RuntimeMonitor(config)
            monitor.set_status("COMPLETED")
            config.extra["diagnostics"] = {"status": "PASS"}
            config.extra["validation_report"] = {"ready": True}
            queue = _FakeJobQueue([job])

            reporter = ReportGenerator(config, queue, monitor)
            reporter.save_all()

            self.assertTrue(config.report_path.exists())
            self.assertTrue(config.summary_path.exists())
            self.assertTrue(config.project_report_csv_path.exists())
            self.assertTrue(config.performance_report_path.exists())
            self.assertTrue(config.diagnostics_path.exists())
            self.assertTrue(config.validation_report_path.exists())
            history = json.loads(config.benchmark_history_path.read_text(encoding="utf-8"))
            self.assertEqual(len(history["history"]), 1)


if __name__ == "__main__":
    unittest.main()
