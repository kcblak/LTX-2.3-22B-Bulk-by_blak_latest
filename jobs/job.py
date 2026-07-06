from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core import JobStatus, Duration, Resolution, AspectRatio


@dataclass
class Job:
    job_id: str
    sequence_index: int
    prompt: str
    start_image: Path
    end_image: Optional[Path]
    duration: Duration
    resolution: Resolution
    aspect_ratio: AspectRatio
    seed: int
    guidance_scale: float
    num_inference_steps: int
    status: JobStatus = JobStatus.PENDING
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    uploaded_at: Optional[datetime] = None
    upload_attempts: int = 0
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    cache_key: str = ""
    cache_hit: bool = False
    render_metrics: dict[str, float] = field(default_factory=dict)
    output_metadata: dict[str, Any] = field(default_factory=dict)
    upload_metrics: dict[str, float] = field(default_factory=dict)
    remote_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "sequence_index": self.sequence_index,
            "prompt": self.prompt,
            "start_image": str(self.start_image),
            "end_image": str(self.end_image) if self.end_image else None,
            "duration": self.duration.label,
            "resolution": self.resolution.label,
            "aspect_ratio": self.aspect_ratio.label,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "status": self.status.name,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "upload_attempts": self.upload_attempts,
            "output_path": str(self.output_path) if self.output_path else None,
            "error_message": self.error_message,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "render_metrics": self.render_metrics,
            "output_metadata": self.output_metadata,
            "upload_metrics": self.upload_metrics,
            "remote_metadata": self.remote_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict, reference_images_dir: Path) -> "Job":
        # Resolve paths relative to reference_images_dir
        start_image = Path(data["start_image"])
        if not start_image.is_absolute():
            start_image = reference_images_dir / start_image
        start_image = start_image.resolve(strict=False)

        end_image = None
        if data.get("end_image"):
            end_image = Path(data["end_image"])
            if not end_image.is_absolute():
                end_image = reference_images_dir / end_image
            end_image = end_image.resolve(strict=False)

        return cls(
            job_id=data["job_id"],
            sequence_index=int(data.get("sequence_index", 0)),
            prompt=data["prompt"],
            start_image=start_image,
            end_image=end_image,
            duration=Duration.from_string(data["duration"]),
            resolution=Resolution.from_string(data["resolution"]),
            aspect_ratio=AspectRatio.from_string(data["aspect_ratio"]),
            seed=data["seed"],
            guidance_scale=data["guidance_scale"],
            num_inference_steps=data["num_inference_steps"],
            status=JobStatus[data["status"]],
            retry_count=data.get("retry_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            verified_at=datetime.fromisoformat(data["verified_at"]) if data.get("verified_at") else None,
            uploaded_at=datetime.fromisoformat(data["uploaded_at"]) if data.get("uploaded_at") else None,
            upload_attempts=data.get("upload_attempts", 0),
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            error_message=data.get("error_message"),
            cache_key=data.get("cache_key", ""),
            cache_hit=bool(data.get("cache_hit", False)),
            render_metrics=data.get("render_metrics", {}),
            output_metadata=data.get("output_metadata", {}),
            upload_metrics=data.get("upload_metrics", {}),
            remote_metadata=data.get("remote_metadata", {}),
        )
