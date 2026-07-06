import json
import threading
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from jobs.job import Job
from config import Config
from core import APP_VERSION
from logging_system import get_logger

logger = get_logger("jobs.manifest")


class Manifest:
    """Manages the job manifest file for persistence and resumption."""

    def __init__(self, manifest_path: Path, reference_images_dir: Path):
        self.manifest_path = manifest_path
        self.reference_images_dir = reference_images_dir
        self._lock = threading.RLock()

    def exists(self) -> bool:
        """Check if manifest file exists."""
        return self.manifest_path.exists()

    def load(self) -> List[Job]:
        """Load jobs from manifest file."""
        with self._lock:
            if not self.exists():
                return []
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            jobs_data = data.get("jobs", [])
            jobs = []
            for job_dict in jobs_data:
                job = Job.from_dict(job_dict, self.reference_images_dir)
                jobs.append(job)
        logger.info(f"Loaded {len(jobs)} jobs from manifest", extra={"job_id": "N/A"})
        return jobs

    def save(self, jobs: List[Job]) -> None:
        """Save jobs to manifest file."""
        with self._lock:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "app_version": APP_VERSION,
                "updated_at": datetime.now().isoformat(),
                "jobs": [job.to_dict() for job in jobs],
            }
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved {len(jobs)} jobs to manifest", extra={"job_id": "N/A"})

    def update_job(self, job: Job, jobs: List[Job]) -> None:
        """Update a job in the manifest."""
        for idx, j in enumerate(jobs):
            if j.job_id == job.job_id:
                jobs[idx] = job
                break
        self.save(jobs)
