from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from .enums import JobStatus, Resolution, AspectRatio, Duration


@dataclass
class Job:
    job_id: str
    prompt: str
    start_image: Path
    end_image: Optional[Path]
    duration: Duration
    resolution: Resolution
    aspect_ratio: AspectRatio
    seed: int
    guide_scale: float
    steps: int
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    uploaded_at: Optional[datetime] = None
    output_path: Optional[Path] = None
    error_message: Optional[str] = None


@dataclass
class Config:
    jobs_csv_path: Path
    reference_images_dir: Path
    outputs_dir: Path
    logs_dir: Path
    manifest_path: Path
    report_path: Path
    model_name: str = "Lightricks/LTX-Video-2.3-22B-Ref-Distilled-1.1"
    enable_gdrive_upload: bool = False
    gdrive_credentials_path: Optional[Path] = None
    gdrive_folder_id: Optional[str] = None
    enable_stitching: bool = False
    max_retries: int = 3
    parallel_uploads: bool = True
    cleanup_temp_files: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
