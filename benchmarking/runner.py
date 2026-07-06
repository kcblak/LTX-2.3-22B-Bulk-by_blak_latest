import json
from pathlib import Path
from typing import Any


class BenchmarkComparator:
    """Compares a generated benchmark artifact against a reference baseline."""

    def compare(self, current_path: Path, baseline_path: Path) -> dict[str, Any]:
        current = json.loads(current_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        current_time = float(current.get("current_run", {}).get("average_total_job_seconds", 0.0))
        baseline_time = float(baseline.get("average_total_job_seconds", 0.0))
        delta = current_time - baseline_time
        return {
            "current_average_total_job_seconds": current_time,
            "baseline_average_total_job_seconds": baseline_time,
            "delta_average_total_job_seconds": delta,
            "improved": delta < 0,
            "baseline_environment": baseline.get("environment"),
            "baseline_renderer_backend": baseline.get("renderer_backend"),
        }
