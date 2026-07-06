from pathlib import Path
from typing import Optional
from core import IStorage, StorageError, SUPPORTED_IMAGE_FORMATS, SUPPORTED_VIDEO_FORMATS
from utils import safe_mkdir, safe_remove
from logging_system import get_logger


class LocalStorage(IStorage):
    """Local filesystem storage implementation."""

    def __init__(self, output_dir: Path, temp_dir: Path):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.clips_dir = output_dir / "clips"
        self.stitched_dir = output_dir / "stitched"
        self.previews_dir = output_dir / "previews"
        self.logger = get_logger("storage")
        self._initialize_dirs()

    def _initialize_dirs(self) -> None:
        """Initialize storage directories."""
        try:
            safe_mkdir(self.output_dir)
            safe_mkdir(self.temp_dir)
            safe_mkdir(self.clips_dir)
            safe_mkdir(self.stitched_dir)
            safe_mkdir(self.previews_dir)
            self.logger.debug("Storage directories initialized", extra={"job_id": "N/A"})
        except Exception as e:
            raise StorageError(f"Failed to initialize storage: {e}")

    def get_temp_path(self, job_id: str, suffix: str = "") -> Path:
        """Get a temporary file path for a job."""
        filename = f"{job_id}{suffix}" if suffix else job_id
        return self.temp_dir / filename

    def get_output_path(
        self,
        job_id: str,
        extension: str = "mp4",
        sequence_index: Optional[int] = None,
    ) -> Path:
        """Get an output file path for a job."""
        if sequence_index is None:
            return self.clips_dir / f"{job_id}.{extension}"
        return self.clips_dir / f"clip_{sequence_index:06d}_{job_id}.{extension}"

    def verify_file(self, path: Path, min_size_bytes: int = 1024) -> bool:
        """Verify a file exists and is valid."""
        if not path.exists():
            self.logger.error(f"File not found: {path}", extra={"job_id": "N/A"})
            return False
        if not path.is_file():
            self.logger.error(f"Path is not a file: {path}", extra={"job_id": "N/A"})
            return False
        if path.stat().st_size < min_size_bytes:
            self.logger.error(f"File too small: {path} ({path.stat().st_size} bytes)", extra={"job_id": "N/A"})
            return False
        return True

    def cleanup_temp(self, job_id: Optional[str] = None) -> None:
        """Clean up temporary files."""
        try:
            if job_id:
                temp_files = list(self.temp_dir.glob(f"{job_id}*"))
                for f in temp_files:
                    safe_remove(f)
                    self.logger.debug(f"Cleaned up temp file: {f}", extra={"job_id": job_id})
            else:
                for f in self.temp_dir.iterdir():
                    safe_remove(f)
                self.logger.debug("Cleaned up all temp files", extra={"job_id": "N/A"})
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp files: {e}", extra={"job_id": job_id or "N/A"})
