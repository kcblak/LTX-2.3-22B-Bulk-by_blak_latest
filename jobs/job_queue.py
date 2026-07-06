import threading
from typing import List, Optional

from jobs.job import Job
from core import JobStatus
from config import Config
from jobs.csv_parser import parse_jobs_from_csv
from jobs.manifest import Manifest
from logging_system import get_logger

logger = get_logger("jobs.queue")


class JobQueue:
    """Manages the job queue and state."""

    def __init__(self, config: Config):
        self.config = config
        self.manifest = Manifest(config.manifest_path, config.reference_images_dir)
        self.jobs: List[Job] = []
        self._lock = threading.RLock()
        self._initialize_jobs()

    def _initialize_jobs(self) -> None:
        """Initialize jobs from manifest or CSV."""
        if self.config.resume_enabled and self.manifest.exists():
            self.jobs = self.manifest.load()
        else:
            self.jobs = parse_jobs_from_csv(self.config.jobs_csv_path, self.config.reference_images_dir)
        if self.config.benchmark_mode and self.config.benchmark_max_jobs > 0:
            self.jobs = self.jobs[: self.config.benchmark_max_jobs]
            logger.info(
                f"Benchmark mode enabled, limiting job queue to {len(self.jobs)} jobs",
                extra={"job_id": "N/A"},
            )
            self.manifest.save(self.jobs)

    def get_next_job(self) -> Optional[Job]:
        """Get the next job to process."""
        with self._lock:
            for job in self.jobs:
                if job.status in [
                    JobStatus.PENDING,
                    JobStatus.FAILED_VALIDATION,
                    JobStatus.FAILED_RENDER,
                    JobStatus.FAILED_VERIFY,
                    JobStatus.RETRYING,
                ]:
                    return job
        return None

    def update_job(self, job: Job) -> None:
        """Update a job in the queue and save manifest."""
        with self._lock:
            self.manifest.update_job(job, self.jobs)

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            for job in self.jobs:
                if job.job_id == job_id:
                    return job
        return None

    def get_jobs_by_statuses(self, statuses: List[JobStatus]) -> List[Job]:
        with self._lock:
            return [j for j in self.jobs if j.status in statuses]

    def get_jobs_needing_upload(self) -> List[Job]:
        return self.get_jobs_by_statuses(
            [
                JobStatus.UPLOAD_PENDING,
                JobStatus.UPLOADING,
                JobStatus.VERIFYING_REMOTE,
                JobStatus.FAILED_UPLOAD,
            ]
        )

    def get_jobs_by_status(self, status: JobStatus) -> List[Job]:
        """Get all jobs with a specific status."""
        with self._lock:
            return [j for j in self.jobs if j.status == status]

    def all_completed(self) -> bool:
        """Check if all jobs are in a final state."""
        final_statuses = [
            JobStatus.COMPLETED,
            JobStatus.UPLOADED,
            JobStatus.FAILED_VALIDATION,
            JobStatus.FAILED_RENDER,
            JobStatus.FAILED_UPLOAD,
            JobStatus.FAILED_VERIFY,
        ]
        with self._lock:
            return all(job.status in final_statuses for job in self.jobs)
