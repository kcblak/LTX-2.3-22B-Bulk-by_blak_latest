from datetime import datetime
from pathlib import Path
import shutil
from typing import Optional

from cache_system import RenderCacheStore
from config import Config
from core import JobStatus, RenderParams
from drive.gdrive import GoogleDriveClient
from drive.sync_engine import DriveSyncEngine
from jobs.job_queue import JobQueue
from logging_system import get_logger
from observability.failures import classify_exception, classify_render_failure
from renderers.factory import create_renderer
from reports.report_generator import ReportGenerator
from storage.local_storage import LocalStorage
from stitching import VideoStitcher
from utils import compute_file_hash

logger = get_logger("engine")


class Pipeline:
    """The main pipeline orchestrator."""

    def __init__(self, config: Config, event_bus=None, runtime_monitor=None):
        self.config = config
        self.event_bus = event_bus
        self.runtime_monitor = runtime_monitor
        self.job_queue = JobQueue(config)
        self.storage = LocalStorage(config.output_dir, config.temp_dir)
        self.renderer = create_renderer(config)
        self.cache_store = RenderCacheStore(config)
        self.drive_client = GoogleDriveClient(config) if config.enable_drive_upload else None
        self.sync_engine = (
            DriveSyncEngine(
                config,
                self.drive_client,
                self.job_queue,
                event_bus=self.event_bus,
                runtime_monitor=self.runtime_monitor,
            )
            if self.drive_client is not None
            else None
        )
        self.reporter = ReportGenerator(config, self.job_queue, self.runtime_monitor)

    def run(self) -> None:
        logger.info("Starting pipeline...", extra={"job_id": "N/A"})
        pipeline_failed = False
        if self.runtime_monitor is not None:
            self.runtime_monitor.attach(job_queue=self.job_queue)
            self.runtime_monitor.set_status("STARTING")
            self.runtime_monitor.start()
        if self.event_bus is not None:
            self.event_bus.publish(
                "ProjectStarted",
                job_id="N/A",
                project_id=self.config.project_id,
                run_id=self.config.run_id,
            )

        try:
            session = self.renderer.initialize()
            self.config.extra["renderer_session"] = {
                "renderer_name": session.renderer_name,
                "device": session.device,
                "precision": session.precision,
                "total_vram_bytes": session.total_vram_bytes,
                "available_vram_bytes": session.available_vram_bytes,
                "scheduler_name": session.scheduler_name,
                "warmup_performed": session.warmup_performed,
                "initialization_seconds": session.initialization_seconds,
            }
            logger.info(
                (
                    f"Renderer session ready: device={session.device} "
                    f"precision={session.precision} warmup={session.warmup_performed}"
                ),
                extra={"job_id": "N/A"},
            )
            if self.runtime_monitor is not None:
                self.runtime_monitor.set_status("READY")
            if self.event_bus is not None:
                self.event_bus.publish(
                    "RendererInitialized",
                    job_id="N/A",
                    renderer_name=session.renderer_name,
                    device=session.device,
                    precision=session.precision,
                    initialization_seconds=session.initialization_seconds,
                )

            if self.sync_engine is not None:
                self.sync_engine.connect()
                if self.runtime_monitor is not None:
                    self.runtime_monitor.attach(
                        job_queue=self.job_queue,
                        heartbeat_remote_sync=self.sync_engine.sync_artifact,
                    )
                self.sync_engine.start()
                self.sync_engine.reconcile_jobs()
                if self.runtime_monitor is not None:
                    self.runtime_monitor.set_upload_queue_length(
                        self.sync_engine.upload_queue.qsize()
                    )

            while True:
                job = self.job_queue.get_next_job()
                if not job:
                    break

                self._process_job(job)

        except Exception as e:
            pipeline_failed = True
            failure = classify_exception(e)
            if self.runtime_monitor is not None:
                self.runtime_monitor.record_error(
                    failure.category,
                    failure.summary,
                    failure.recommendation,
                )
                self.runtime_monitor.set_status("FAILED")
            if self.event_bus is not None:
                self.event_bus.publish(
                    "ProjectFailed",
                    job_id="N/A",
                    category=failure.category,
                    error_message=failure.summary,
                    recommendation=failure.recommendation,
                )
            logger.error(f"Pipeline failed: {e}", extra={"job_id": "N/A"}, exc_info=True)
            raise
        finally:
            if self.sync_engine is not None:
                logger.info("Waiting for upload queue to drain...", extra={"job_id": "N/A"})
                self.sync_engine.shutdown(graceful=True)
                self.config.extra["drive_metrics"] = self.sync_engine.get_metrics()
            self.renderer.cleanup()
            if self.config.cleanup_temp:
                self.storage.cleanup_temp()
            if self.runtime_monitor is not None:
                if self.runtime_monitor.get_snapshot().get("status") != "FAILED":
                    self.runtime_monitor.set_status("COMPLETED")
                self.runtime_monitor.set_current_job(None)
                self.runtime_monitor.stop()
                self.config.extra["runtime_monitor"] = self.runtime_monitor.export()
            self.reporter.save_all()
            if self.config.enable_stitching:
                stitcher = VideoStitcher(
                    self.config,
                    self.job_queue,
                    drive_client=self.drive_client,
                    drive_project_paths=self.sync_engine.project_paths if self.sync_engine else None,
                )
                stitching_result = stitcher.run()
                self.config.extra["stitching"] = stitching_result.to_dict()
                if stitching_result.success and self.sync_engine is not None and self.config.stitch_upload_outputs:
                    self.sync_engine.sync_artifact(self.config.stitched_output_path, "stitched")
                    if self.config.thumbnail_path.exists():
                        self.sync_engine.sync_artifact(self.config.thumbnail_path, "thumbnails")
                    if self.config.preview_480p_path.exists():
                        self.sync_engine.sync_artifact(self.config.preview_480p_path, "previews")
                    if self.config.preview_720p_path.exists():
                        self.sync_engine.sync_artifact(self.config.preview_720p_path, "previews")
            if self.sync_engine is not None:
                self.sync_engine.sync_artifact(self.config.manifest_path, "manifests")
                self.sync_engine.sync_artifact(self.config.report_path, "reports")
                self.sync_engine.sync_artifact(self.config.summary_path, "reports")
                self.sync_engine.sync_artifact(self.config.validation_report_path, "reports")
                self.sync_engine.sync_artifact(self.config.performance_report_path, "reports")
                self.sync_engine.sync_artifact(self.config.performance_summary_path, "reports")
                self.sync_engine.sync_artifact(self.config.diagnostics_path, "reports")
                self.sync_engine.sync_artifact(self.config.project_report_csv_path, "reports")
                self.sync_engine.sync_artifact(self.config.benchmark_json_path, "reports")
                self.sync_engine.sync_artifact(self.config.benchmark_csv_path, "reports")
                for log_file in self.config.log_dir.glob("*.log"):
                    self.sync_engine.sync_artifact(log_file, "logs")
                for log_file in self.config.log_dir.glob("*.gz"):
                    self.sync_engine.sync_artifact(log_file, "logs")
                if self.config.jobs_csv_path.exists():
                    self.sync_engine.sync_artifact(self.config.jobs_csv_path, "input")
                default_config = (
                    Path(__file__).resolve().parents[1] / "config" / "default.yaml"
                )
                if default_config.exists():
                    self.sync_engine.sync_artifact(default_config, "config")
                if self.config.project_config_path and self.config.project_config_path.exists():
                    self.sync_engine.sync_artifact(self.config.project_config_path, "config")

        summary = self.reporter.generate_summary()
        logger.info("Pipeline complete!", extra={"job_id": "N/A"})
        logger.info("Final Summary:", extra={"job_id": "N/A"})
        for key, val in summary["summary"].items():
            logger.info(f"  {key}: {val}", extra={"job_id": "N/A"})
        if self.event_bus is not None and not pipeline_failed:
            self.event_bus.publish(
                "ProjectCompleted",
                job_id="N/A",
                project_id=self.config.project_id,
                run_id=self.config.run_id,
                summary=summary["summary"],
            )

    def _process_job(self, job) -> None:
        job_id = job.job_id
        logger.info(f"Processing job {job_id}", extra={"job_id": job_id})
        if self.runtime_monitor is not None:
            self.runtime_monitor.set_current_job(job_id)
            self.runtime_monitor.set_status("PROCESSING")
        if self.event_bus is not None:
            self.event_bus.publish("JobQueued", job_id=job_id)
            self.event_bus.publish("JobStarted", job_id=job_id)

        job.status = JobStatus.VALIDATING
        self.job_queue.update_job(job)

        try:
            params = RenderParams(
                job_id=job_id,
                prompt=job.prompt,
                start_image=job.start_image,
                end_image=job.end_image,
                duration=job.duration,
                resolution=job.resolution,
                aspect_ratio=job.aspect_ratio,
                seed=job.seed,
                guidance_scale=job.guidance_scale,
                num_inference_steps=job.num_inference_steps,
                frame_rate=self.config.frame_rate,
                output_codec=self.config.output_codec,
                output_container=self.config.output_container,
                output_quality=self.config.output_quality,
            )
            self.renderer.validate_parameters(params)
        except Exception as e:
            failure = classify_exception(e)
            job.status = JobStatus.FAILED_VALIDATION
            job.error_message = (
                f"{failure.category}: {failure.summary}. "
                f"Recommendation: {failure.recommendation}"
            )
            self.job_queue.update_job(job)
            logger.error(f"Validation failed: {e}", extra={"job_id": job_id})
            if self.runtime_monitor is not None:
                self.runtime_monitor.record_error(
                    failure.category,
                    failure.summary,
                    failure.recommendation,
                )
            if self.event_bus is not None:
                self.event_bus.publish(
                    "JobFailed",
                    job_id=job_id,
                    category=failure.category,
                    error_message=failure.summary,
                    recommendation=failure.recommendation,
                )
            if self.runtime_monitor is not None:
                self.runtime_monitor.set_current_job(None)
            return

        job.status = JobStatus.READY
        self.job_queue.update_job(job)

        temp_path = self.storage.get_temp_path(
            job_id, f".{self.config.output_container}"
        )
        job.started_at = datetime.now()
        job.status = JobStatus.LOADING
        self.job_queue.update_job(job)
        job.status = JobStatus.GENERATING
        self.job_queue.update_job(job)

        output_path = self.storage.get_output_path(
            job_id,
            extension=self.config.output_container,
            sequence_index=job.sequence_index,
        )
        job.cache_key = self.cache_store.build_cache_key(job)
        cached_path = self.cache_store.lookup(job.cache_key) if self.config.cache_enabled else None
        if cached_path is not None:
            if cached_path.resolve(strict=False) != output_path.resolve(strict=False):
                shutil.copy2(cached_path, output_path)
            job.output_path = output_path
            job.cache_hit = True
            job.output_metadata = {
                "checksum_sha256": compute_file_hash(output_path, algorithm="sha256"),
                "file_size_bytes": output_path.stat().st_size,
            }
            job.completed_at = datetime.now()
            job.status = (
                JobStatus.UPLOAD_PENDING
                if self.sync_engine is not None
                else JobStatus.COMPLETED
            )
            self.job_queue.update_job(job)
            if self.sync_engine is not None:
                self.sync_engine.enqueue_job(job)
            if self.runtime_monitor is not None:
                self.runtime_monitor.record_render_time(0.0)
                self.runtime_monitor.set_current_job(None)
            logger.info(
                f"Reused cached clip for job {job_id}",
                extra={"job_id": job_id},
            )
            return

        result = self.renderer.generate_clip(params, temp_path)
        job.render_metrics = {
            "image_loading_seconds": result.metrics.image_loading_seconds,
            "prompt_preparation_seconds": result.metrics.prompt_preparation_seconds,
            "inference_seconds": result.metrics.inference_seconds,
            "encoding_seconds": result.metrics.encoding_seconds,
            "validation_seconds": result.metrics.validation_seconds,
            "total_seconds": result.metrics.total_seconds,
        }

        if not result.success or result.clip is None:
            failure = classify_render_failure(result.error_type, result.error_message)
            if result.error_type == "RendererInputError":
                job.status = JobStatus.FAILED_VALIDATION
            elif result.error_type == "RendererOutputValidationError":
                job.status = JobStatus.FAILED_VERIFY
            else:
                job.status = JobStatus.FAILED_RENDER
            job.error_message = (
                f"{failure.category}: {failure.summary}. "
                f"Recommendation: {failure.recommendation}"
            )
            self.job_queue.update_job(job)
            if self.runtime_monitor is not None:
                self.runtime_monitor.record_error(
                    failure.category,
                    failure.summary,
                    failure.recommendation,
                )
            if self.event_bus is not None:
                self.event_bus.publish(
                    "JobFailed",
                    job_id=job_id,
                    category=failure.category,
                    error_message=failure.summary,
                    recommendation=failure.recommendation,
                )
            if self.runtime_monitor is not None:
                self.runtime_monitor.set_current_job(None)
            return

        job.status = JobStatus.ENCODING
        self.job_queue.update_job(job)
        shutil.move(str(temp_path), str(output_path))
        job.output_path = output_path
        job.cache_hit = False
        job.output_metadata = {
            "checksum_sha256": result.clip.checksum_sha256,
            "file_size_bytes": result.clip.file_size_bytes,
            "width": result.clip.width,
            "height": result.clip.height,
            "frame_rate": result.clip.frame_rate,
            "frame_count": result.clip.frame_count,
            "duration_seconds": result.clip.duration_seconds,
            "codec": result.clip.codec,
        }
        job.completed_at = datetime.now()
        job.status = (
            JobStatus.UPLOAD_PENDING
            if self.sync_engine is not None
            else JobStatus.VERIFIED if self.config.auto_verify else JobStatus.COMPLETED
        )
        self.job_queue.update_job(job)
        if self.config.auto_verify:
            job.verified_at = datetime.now()
        if self.sync_engine is not None:
            self.sync_engine.enqueue_job(job)
        else:
            job.status = JobStatus.COMPLETED
            self.job_queue.update_job(job)
        if self.config.cache_enabled:
            self.cache_store.register(job.cache_key, output_path, job.output_metadata)
        if self.runtime_monitor is not None:
            self.runtime_monitor.record_render_time(result.metrics.total_seconds)
            self.runtime_monitor.set_current_job(None)
        if self.event_bus is not None:
            self.event_bus.publish(
                "JobCompleted",
                job_id=job_id,
                output_path=str(job.output_path),
                render_seconds=result.metrics.total_seconds,
            )

        logger.info(f"Job {job_id} complete", extra={"job_id": job_id})
