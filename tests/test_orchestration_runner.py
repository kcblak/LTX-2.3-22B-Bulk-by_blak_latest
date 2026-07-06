import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from orchestration.runner import ApplicationRunner


class _FakeDiagnosticsResult:
    def __init__(self, status: str = "PASS"):
        self._payload = {"status": status}

    def to_dict(self):
        return dict(self._payload)


class _FakePreflightResult:
    def __init__(self, *, ready: bool = True, warnings=None):
        self.blocking_failures = []
        self.warnings = list(warnings or [])
        self._payload = {
            "ready": ready,
            "expected_clip_count": 2,
            "blocking_failures": [],
            "warnings": self.warnings,
        }

    def to_dict(self):
        return dict(self._payload)


class _FakeDiagnosticsRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        return _FakeDiagnosticsResult()

    def save(self, result, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


class _FakePreflightAnalyzer:
    def __init__(self, config):
        self.config = config
        self.result = _FakePreflightResult()

    def analyze(self):
        return self.result

    def save(self, result, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    def build_console_summary(self, result):
        return "PREPARED"


class _FakePipeline:
    def __init__(self, config, event_bus=None, runtime_monitor=None):
        self.config = config

    def run(self):
        return {"summary": {"completed_jobs": 2, "failed_jobs": 0}}


class ApplicationRunnerTests(unittest.TestCase):
    def _build_config(self, root: Path) -> Config:
        return Config(
            output_dir=root / "outputs",
            log_dir=root / "logs",
            temp_dir=root / "temp",
            diagnostics_path=root / "outputs" / "diagnostics.json",
            validation_report_path=root / "outputs" / "validation_report.json",
            report_path=root / "outputs" / "report.json",
            summary_path=root / "outputs" / "summary.txt",
            project_report_csv_path=root / "outputs" / "project_report.csv",
            performance_report_path=root / "outputs" / "performance.json",
            benchmark_json_path=root / "outputs" / "benchmark.json",
            benchmark_csv_path=root / "outputs" / "benchmark.csv",
            performance_summary_path=root / "outputs" / "performance_summary.txt",
            enable_drive_upload=False,
        )

    @patch("orchestration.runner.Pipeline", _FakePipeline)
    @patch("orchestration.runner.PreflightAnalyzer", _FakePreflightAnalyzer)
    @patch("orchestration.runner.DiagnosticsRunner", _FakeDiagnosticsRunner)
    @patch("orchestration.runner.bootstrap", lambda config: None)
    def test_run_preflight_only_returns_success_without_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(Path(temp_dir))
            runner = ApplicationRunner(config)
            result = runner.run(preflight_only=True)

            self.assertTrue(result.success)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.preflight["expected_clip_count"], 2)
            self.assertEqual(result.summary, {})

    @patch("orchestration.runner.Pipeline", _FakePipeline)
    @patch("orchestration.runner.PreflightAnalyzer", _FakePreflightAnalyzer)
    @patch("orchestration.runner.DiagnosticsRunner", _FakeDiagnosticsRunner)
    @patch("orchestration.runner.bootstrap", lambda config: None)
    def test_run_pipeline_only_returns_pipeline_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(Path(temp_dir))
            runner = ApplicationRunner(config)
            runner.prepare()

            result = runner.run_pipeline_only()

            self.assertTrue(result.success)
            self.assertEqual(result.summary["summary"]["completed_jobs"], 2)


if __name__ == "__main__":
    unittest.main()
