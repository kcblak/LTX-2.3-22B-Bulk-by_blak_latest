from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.models import Config, Job
from ..core.job_manager import JobManager
from ..core.enums import JobStatus
from ..services.model_service import ModelService
from ..services.gdrive_service import GDriveService
from ..storage.storage_manager import StorageManager
from ..reporting.reporter import Reporter
from ..utils.logger import setup_logger


class PipelineOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__, config.logs_dir)
        self.job_manager = JobManager(config)
        self.storage_manager = StorageManager(config)
        self.model_service = ModelService(config)
        self.gdrive_service = GDriveService(config)
        self.reporter = Reporter(config, self.job_manager)
        self.upload_executor: Optional[ThreadPoolExecutor] = None

    def _upload_worker(self, job: Job):
        if self.gdrive_service.upload_job_output(job):
            self.job_manager.update_job_status(job.job_id, JobStatus.UPLOADED)
        else:
            self.logger.warning(f"Upload failed for job {job.job_id}")

    def process_job(self, job: Job):
        self.logger.info(f"Processing job {job.job_id}")
        self.job_manager.update_job_status(job.job_id, JobStatus.RUNNING)

        temp_output_path = self.storage_manager.get_temp_path(job, ".mp4")

        success = self.model_service.generate_video(job, temp_output_path)
        if not success:
            self.job_manager.update_job_status(
                job.job_id,
                JobStatus.FAILED,
                error_message="Video generation failed",
            )
            return

        output_path = self.storage_manager.save_job_output(job, temp_output_path)
        self.job_manager.update_job_status(
            job.job_id,
            JobStatus.COMPLETED,
            output_path=output_path,
        )

        if self.storage_manager.verify_output(output_path):
            self.job_manager.update_job_status(job.job_id, JobStatus.VERIFIED)

            if self.config.enable_gdrive_upload:
                if self.config.parallel_uploads and self.upload_executor:
                    self.upload_executor.submit(self._upload_worker, job)
                else:
                    self._upload_worker(job)
        else:
            self.job_manager.update_job_status(
                job.job_id,
                JobStatus.FAILED,
                error_message="Output verification failed",
            )

        if self.config.cleanup_temp_files:
            self.storage_manager.cleanup_temp_files(job)

    def run(self):
        self.logger.info("Starting pipeline execution")
        self.reporter.save_report()

        if self.config.enable_gdrive_upload and self.config.parallel_uploads:
            self.upload_executor = ThreadPoolExecutor(max_workers=2)

        try:
            self.model_service.load_model()

            while True:
                next_job = self.job_manager.get_next_job()
                if not next_job:
                    break

                self.process_job(next_job)
                self.reporter.save_report()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            if self.upload_executor:
                self.logger.info("Waiting for uploads to complete...")
                self.upload_executor.shutdown(wait=True)
            self.reporter.save_report()

        self.logger.info("Pipeline execution completed!")
        return self.reporter.generate_report()
