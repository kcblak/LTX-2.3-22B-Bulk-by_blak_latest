import json
import tempfile
import unittest
from pathlib import Path

from benchmarking import BenchmarkComparator
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


class StressAndBenchmarkingTests(unittest.TestCase):
    def test_report_generator_handles_large_job_sets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = Config(
                output_dir=root / "outputs",
                log_dir=root / "logs",
                temp_dir=root / "temp",
                enable_drive_upload=False,
            )
            jobs = [
                Job(
                    job_id=f"job-{index}",
                    sequence_index=index + 1,
                    prompt=f"prompt-{index}",
                    start_image=root / f"start-{index}.png",
                    end_image=None,
                    duration=Duration.D5,
                    resolution=Resolution.R480,
                    aspect_ratio=AspectRatio.AR_1_1,
                    seed=index,
                    guidance_scale=3.0,
                    num_inference_steps=8,
                    status=JobStatus.COMPLETED if index % 2 == 0 else JobStatus.FAILED_RENDER,
                    render_metrics={"total_seconds": 1.0},
                )
                for index in range(1000)
            ]
            reporter = ReportGenerator(config, _FakeJobQueue(jobs), RuntimeMonitor(config))
            summary = reporter.generate_summary()

            self.assertEqual(summary["summary"]["total_jobs"], 1000)
            self.assertEqual(summary["summary"]["completed_jobs"], 500)

    def test_benchmark_comparator_reports_delta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_path = root / "benchmark.json"
            baseline_path = root / "baseline.json"
            current_path.write_text(
                json.dumps({"current_run": {"average_total_job_seconds": 12.0}}),
                encoding="utf-8",
            )
            baseline_path.write_text(
                json.dumps(
                    {
                        "environment": "kaggle-l4",
                        "renderer_backend": "wan2gp",
                        "average_total_job_seconds": 15.0,
                    }
                ),
                encoding="utf-8",
            )

            result = BenchmarkComparator().compare(current_path, baseline_path)

            self.assertTrue(result["improved"])
            self.assertEqual(result["delta_average_total_job_seconds"], -3.0)


if __name__ == "__main__":
    unittest.main()
