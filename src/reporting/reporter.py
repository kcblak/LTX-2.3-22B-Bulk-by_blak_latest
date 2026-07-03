import json
from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path

from ..core.models import Config
from ..core.job_manager import JobManager
from ..core.enums import JobStatus
from ..utils.logger import setup_logger


class Reporter:
    def __init__(self, config: Config, job_manager: JobManager):
        self.config = config
        self.job_manager = job_manager
        self.logger = setup_logger(__name__, config.logs_dir)

    def generate_report(self) -> Dict[str, Any]:
        total_jobs = len(self.job_manager.jobs)
        completed_jobs = len(self.job_manager.get_jobs_by_status(JobStatus.COMPLETED))
        verified_jobs = len(self.job_manager.get_jobs_by_status(JobStatus.VERIFIED))
        uploaded_jobs = len(self.job_manager.get_jobs_by_status(JobStatus.UPLOADED))
        failed_jobs = len(self.job_manager.get_jobs_by_status(JobStatus.FAILED))
        pending_jobs = len(self.job_manager.get_jobs_by_status(JobStatus.PENDING))
        running_jobs = len(self.job_manager.get_jobs_by_status(JobStatus.RUNNING))

        total_duration = timedelta(0)
        for job in self.job_manager.jobs:
            if job.started_at and job.completed_at:
                total_duration += job.completed_at - job.started_at

        report = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "verified_jobs": verified_jobs,
                "uploaded_jobs": uploaded_jobs,
                "failed_jobs": failed_jobs,
                "pending_jobs": pending_jobs,
                "running_jobs": running_jobs,
                "total_processing_time": str(total_duration),
            },
            "jobs": [
                {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "output_path": str(job.output_path) if job.output_path else None,
                    "error": job.error_message,
                }
                for job in self.job_manager.jobs
            ],
        }

        return report

    def save_report(self):
        report = self.generate_report()
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Report saved to {self.config.report_path}")
