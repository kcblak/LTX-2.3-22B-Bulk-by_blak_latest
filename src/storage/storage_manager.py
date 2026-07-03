from pathlib import Path
from typing import Optional
import shutil

from ..core.models import Config, Job
from ..utils.logger import setup_logger


class StorageManager:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__, config.logs_dir)
        self._initialize_directories()

    def _initialize_directories(self):
        self.config.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.config.outputs_dir / "clips"
        self.temp_dir = self.config.outputs_dir / "temp"
        self.clips_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    def get_job_output_path(self, job: Job) -> Path:
        return self.clips_dir / f"{job.job_id}.mp4"

    def get_temp_path(self, job: Job, suffix: str = "") -> Path:
        return self.temp_dir / f"{job.job_id}{suffix}"

    def save_job_output(self, job: Job, temp_output_path: Path) -> Path:
        output_path = self.get_job_output_path(job)
        shutil.move(str(temp_output_path), str(output_path))
        self.logger.info(f"Saved job output to: {output_path}")
        return output_path

    def verify_output(self, output_path: Path) -> bool:
        if not output_path.exists():
            self.logger.error(f"Output file does not exist: {output_path}")
            return False
        if output_path.stat().st_size < 1024:
            self.logger.error(f"Output file is too small: {output_path}")
            return False
        return True

    def cleanup_temp_files(self, job: Job):
        temp_files = list(self.temp_dir.glob(f"{job.job_id}*"))
        for f in temp_files:
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
                self.logger.debug(f"Cleaned up temp file: {f}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up {f}: {e}")

    def cleanup_all_temp(self):
        for f in self.temp_dir.iterdir():
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            except Exception as e:
                self.logger.warning(f"Failed to clean up {f}: {e}")
