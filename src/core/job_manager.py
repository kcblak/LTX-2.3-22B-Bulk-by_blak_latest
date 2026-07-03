import csv
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from .models import Job, Config
from .enums import JobStatus, Resolution, AspectRatio, Duration
from ..utils.logger import setup_logger


class JobManager:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__, config.logs_dir)
        self.jobs: List[Job] = []
        self._load_jobs()

    def _generate_job_id(self, job_data: Dict[str, Any]) -> str:
        job_str = json.dumps(job_data, sort_keys=True)
        return hashlib.md5(job_str.encode()).hexdigest()

    def _load_jobs(self):
        if self.config.manifest_path.exists():
            self.logger.info(f"Loading existing jobs from manifest: {self.config.manifest_path}")
            self._load_from_manifest()
        else:
            self.logger.info(f"Loading new jobs from CSV: {self.config.jobs_csv_path}")
            self._load_from_csv()

    def _load_from_csv(self):
        with open(self.config.jobs_csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, row in enumerate(reader):
                job_data = {k: v for k, v in row.items()}
                job_id = self._generate_job_id(job_data)
                job = Job(
                    job_id=job_id,
                    prompt=job_data["prompt"],
                    start_image=self.config.reference_images_dir / job_data["start_image"],
                    end_image=self.config.reference_images_dir / job_data["end_image"]
                    if job_data["end_image"]
                    else None,
                    duration=Duration(job_data["duration"]),
                    resolution=Resolution(job_data["resolution"]),
                    aspect_ratio=AspectRatio(job_data["aspect_ratio"]),
                    seed=int(job_data["seed"]),
                    guide_scale=float(job_data["guide_scale"]),
                    steps=int(job_data["steps"]),
                )
                self.jobs.append(job)
        self.save_manifest()

    def _load_from_manifest(self):
        with open(self.config.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            for job_dict in manifest_data["jobs"]:
                job = Job(
                    job_id=job_dict["job_id"],
                    prompt=job_dict["prompt"],
                    start_image=Path(job_dict["start_image"]),
                    end_image=Path(job_dict["end_image"]) if job_dict["end_image"] else None,
                    duration=Duration(job_dict["duration"]),
                    resolution=Resolution(job_dict["resolution"]),
                    aspect_ratio=AspectRatio(job_dict["aspect_ratio"]),
                    seed=job_dict["seed"],
                    guide_scale=job_dict["guide_scale"],
                    steps=job_dict["steps"],
                    status=JobStatus(job_dict["status"]),
                    created_at=datetime.fromisoformat(job_dict["created_at"]),
                    started_at=datetime.fromisoformat(job_dict["started_at"])
                    if job_dict["started_at"]
                    else None,
                    completed_at=datetime.fromisoformat(job_dict["completed_at"])
                    if job_dict["completed_at"]
                    else None,
                    verified_at=datetime.fromisoformat(job_dict["verified_at"])
                    if job_dict["verified_at"]
                    else None,
                    uploaded_at=datetime.fromisoformat(job_dict["uploaded_at"])
                    if job_dict["uploaded_at"]
                    else None,
                    output_path=Path(job_dict["output_path"]) if job_dict["output_path"] else None,
                    error_message=job_dict.get("error_message"),
                )
                self.jobs.append(job)

    def save_manifest(self):
        manifest_data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "prompt": job.prompt,
                    "start_image": str(job.start_image),
                    "end_image": str(job.end_image) if job.end_image else None,
                    "duration": job.duration.value,
                    "resolution": job.resolution.value,
                    "aspect_ratio": job.aspect_ratio.value,
                    "seed": job.seed,
                    "guide_scale": job.guide_scale,
                    "steps": job.steps,
                    "status": job.status.value,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "verified_at": job.verified_at.isoformat() if job.verified_at else None,
                    "uploaded_at": job.uploaded_at.isoformat() if job.uploaded_at else None,
                    "output_path": str(job.output_path) if job.output_path else None,
                    "error_message": job.error_message,
                }
                for job in self.jobs
            ],
        }
        self.config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    def get_next_job(self) -> Job | None:
        for job in self.jobs:
            if job.status in [JobStatus.PENDING, JobStatus.FAILED]:
                return job
        return None

    def update_job_status(self, job_id: str, status: JobStatus, **kwargs):
        job = next((j for j in self.jobs if j.job_id == job_id), None)
        if job:
            job.status = status
            if status == JobStatus.RUNNING:
                job.started_at = datetime.now()
            elif status == JobStatus.COMPLETED:
                job.completed_at = datetime.now()
            elif status == JobStatus.VERIFIED:
                job.verified_at = datetime.now()
            elif status == JobStatus.UPLOADED:
                job.uploaded_at = datetime.now()
            for key, value in kwargs.items():
                setattr(job, key, value)
            self.save_manifest()

    def get_jobs_by_status(self, status: JobStatus) -> List[Job]:
        return [j for j in self.jobs if j.status == status]

    def all_jobs_completed(self) -> bool:
        return all(j.status in [JobStatus.VERIFIED, JobStatus.UPLOADED] for j in self.jobs)
