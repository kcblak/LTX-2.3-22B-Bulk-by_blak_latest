from datetime import datetime
import threading
import time
from queue import Empty, Queue
from typing import Optional

from config import Config
from core import (
    DriveProjectPaths,
    JobStatus,
    RemoteFileMetadata,
    UploadMetrics,
    UploadResult,
    UploadTask,
)
from jobs.job import Job
from jobs.job_queue import JobQueue
from logging_system import get_logger
from utils import compute_file_hash, safe_remove

logger = get_logger("upload")


class DriveSyncEngine:
    """Background synchronization engine that mirrors local progress into Drive."""

    def __init__(
        self,
        config: Config,
        drive_client,
        job_queue: JobQueue,
        event_bus=None,
        runtime_monitor=None,
    ):
        self.config = config
        self.drive_client = drive_client
        self.job_queue = job_queue
        self.event_bus = event_bus
        self.runtime_monitor = runtime_monitor
        self.project_paths: Optional[DriveProjectPaths] = None
        self.upload_queue: Queue[Optional[UploadTask]] = Queue()
        self.stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._pending_jobs: set[str] = set()
        self._pending_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "uploaded_files": 0,
            "failed_uploads": 0,
            "skipped_duplicates": 0,
            "total_retries": 0,
            "total_upload_seconds": 0.0,
            "total_verification_seconds": 0.0,
            "queue_max_depth": 0,
        }

    def connect(self) -> DriveProjectPaths:
        self.drive_client.connect()
        project_name = self.config.drive_project_name
        if project_name == "default_project":
            project_name = self.config.output_dir.resolve(strict=False).parent.name
        self.project_paths = self.drive_client.ensure_project_structure(project_name)
        return self.project_paths

    def start(self) -> None:
        if not self.config.enable_drive_upload:
            return
        if self.project_paths is None:
            self.connect()
        worker_count = max(1, self.config.drive_max_parallel_uploads)
        for index in range(worker_count):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"drive-upload-{index}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def reconcile_jobs(self) -> None:
        if not self.config.enable_drive_upload:
            return
        if self.project_paths is None:
            self.connect()

        for job in self.job_queue.jobs:
            if job.status == JobStatus.COMPLETED and job.remote_metadata:
                continue

            if job.status in {
                JobStatus.VERIFIED,
                JobStatus.UPLOAD_PENDING,
                JobStatus.UPLOADING,
                JobStatus.VERIFYING_REMOTE,
                JobStatus.FAILED_UPLOAD,
                JobStatus.UPLOADED,
            }:
                if self._remote_clip_matches(job):
                    self._mark_uploaded(job, duplicate_skipped=True)
                    continue
                if job.output_path and job.output_path.exists():
                    self.enqueue_job(job)
                elif job.status in {
                    JobStatus.UPLOAD_PENDING,
                    JobStatus.UPLOADING,
                    JobStatus.VERIFYING_REMOTE,
                    JobStatus.FAILED_UPLOAD,
                }:
                    job.status = JobStatus.FAILED_UPLOAD
                    job.error_message = "Local output missing during upload resume"
                    self.job_queue.update_job(job)

    def enqueue_job(self, job: Job) -> None:
        if self.project_paths is None:
            self.connect()
        if not job.output_path or not job.output_path.exists():
            job.status = JobStatus.FAILED_UPLOAD
            job.error_message = "No local output available for upload"
            self.job_queue.update_job(job)
            return

        with self._pending_lock:
            if job.job_id in self._pending_jobs:
                return
            self._pending_jobs.add(job.job_id)

        local_md5 = compute_file_hash(job.output_path, algorithm="md5")
        local_size = job.output_path.stat().st_size
        task = UploadTask(
            job_id=job.job_id,
            local_path=job.output_path,
            remote_name=job.output_path.name,
            remote_folder_key="clips",
            local_size_bytes=local_size,
            local_md5=local_md5,
            cleanup_policy=self.config.drive_cleanup_policy,
        )

        job.status = JobStatus.UPLOAD_PENDING
        job.remote_metadata.setdefault("remote_folder_key", "clips")
        self.job_queue.update_job(job)

        queue_depth = self.upload_queue.qsize() + 1
        with self._metrics_lock:
            self._metrics["queue_max_depth"] = max(
                self._metrics["queue_max_depth"], queue_depth
            )
        self.upload_queue.put(task)
        if self.runtime_monitor is not None:
            self.runtime_monitor.set_upload_queue_length(self.upload_queue.qsize())

    def _worker_loop(self) -> None:
        while True:
            try:
                task = self.upload_queue.get(timeout=0.5)
            except Empty:
                if self.stop_event.is_set():
                    return
                continue

            if task is None:
                self.upload_queue.task_done()
                return

            try:
                self._process_upload_task(task)
            finally:
                with self._pending_lock:
                    self._pending_jobs.discard(task.job_id)
                self.upload_queue.task_done()
                if self.runtime_monitor is not None:
                    self.runtime_monitor.set_upload_queue_length(self.upload_queue.qsize())

    def _process_upload_task(self, task: UploadTask) -> None:
        job = self.job_queue.get_job(task.job_id)
        if job is None:
            return

        folder_id = self.project_paths.folders[task.remote_folder_key]
        existing = self.drive_client.find_file_by_name(task.remote_name, folder_id)
        if existing and self._remote_matches_task(existing, task):
            self._mark_uploaded(job, remote_metadata=existing, duplicate_skipped=True)
            return

        metrics = UploadMetrics(queue_depth_at_submit=self.upload_queue.qsize())
        attempts = 0
        backoff = max(0.1, self.config.drive_retry_base_seconds)
        started = time.perf_counter()
        last_error = None

        while attempts < max(1, self.config.drive_upload_retries):
            attempts += 1
            job.upload_attempts = attempts
            job.status = JobStatus.UPLOADING
            self.job_queue.update_job(job)
            if self.event_bus is not None:
                self.event_bus.publish(
                    "UploadStarted",
                    job_id=job.job_id,
                    attempt=attempts,
                    local_path=str(task.local_path),
                )

            try:
                upload_started = time.perf_counter()
                uploaded = self.drive_client.upload_file(
                    local_path=task.local_path,
                    remote_name=task.remote_name,
                    folder_id=folder_id,
                )
                metrics.upload_seconds += time.perf_counter() - upload_started
                if uploaded is None:
                    raise RuntimeError("Upload returned no remote metadata")

                job.status = JobStatus.VERIFYING_REMOTE
                self.job_queue.update_job(job)

                verify_started = time.perf_counter()
                verified = self.drive_client.verify_upload(
                    uploaded.file_id,
                    local_size_bytes=task.local_size_bytes,
                    local_md5=task.local_md5,
                )
                metrics.verification_seconds += time.perf_counter() - verify_started
                if verified is None:
                    raise RuntimeError("Remote verification failed")

                metrics.retry_count = attempts - 1
                metrics.total_seconds = time.perf_counter() - started
                if metrics.upload_seconds > 0:
                    metrics.average_upload_mbps = (
                        (task.local_size_bytes * 8) / 1_000_000
                    ) / metrics.upload_seconds
                with self._metrics_lock:
                    self._metrics["total_retries"] += metrics.retry_count
                self._mark_uploaded(job, remote_metadata=verified, metrics=metrics)
                return
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    f"Upload attempt {attempts} failed for job {task.job_id}: {exc}",
                    extra={"job_id": task.job_id},
                )
                if attempts >= self.config.drive_upload_retries:
                    break
                with self._metrics_lock:
                    self._metrics["total_retries"] += 1
                time.sleep(min(backoff, self.config.drive_retry_max_seconds))
                backoff *= 2

        metrics.retry_count = max(0, attempts - 1)
        metrics.total_seconds = time.perf_counter() - started
        self._mark_failed(job, last_error or "Upload failed", metrics)

    def _remote_matches_task(
        self, metadata: RemoteFileMetadata, task: UploadTask
    ) -> bool:
        if metadata.size_bytes != task.local_size_bytes:
            return False
        if metadata.md5_checksum and metadata.md5_checksum != task.local_md5:
            return False
        return True

    def _remote_clip_matches(self, job: Job) -> bool:
        if self.project_paths is None:
            return False
        output_path = job.output_path
        if output_path is None:
            return False
        remote = self.drive_client.find_file_by_name(
            output_path.name,
            self.project_paths.folders["clips"],
        )
        if remote is None:
            return False

        local_size = int(job.output_metadata.get("file_size_bytes", 0) or 0)
        if local_size and remote.size_bytes != local_size:
            return False
        local_md5 = job.remote_metadata.get("local_md5")
        if local_md5 and remote.md5_checksum and remote.md5_checksum != local_md5:
            return False
        job.remote_metadata.update(
            {
                "file_id": remote.file_id,
                "name": remote.name,
                "size_bytes": remote.size_bytes,
                "md5_checksum": remote.md5_checksum,
                "folder_id": remote.folder_id,
                "web_view_link": remote.web_view_link,
            }
        )
        return True

    def _mark_uploaded(
        self,
        job: Job,
        remote_metadata: Optional[RemoteFileMetadata] = None,
        metrics: Optional[UploadMetrics] = None,
        duplicate_skipped: bool = False,
    ) -> None:
        if metrics is None:
            metrics = UploadMetrics(duplicate_skipped=duplicate_skipped)
        metrics.duplicate_skipped = duplicate_skipped
        job.uploaded_at = datetime.now()
        job.status = JobStatus.UPLOADED
        if remote_metadata:
            job.remote_metadata.update(
                {
                    "file_id": remote_metadata.file_id,
                    "name": remote_metadata.name,
                    "size_bytes": remote_metadata.size_bytes,
                    "md5_checksum": remote_metadata.md5_checksum,
                    "folder_id": remote_metadata.folder_id,
                    "web_view_link": remote_metadata.web_view_link,
                }
            )
        job.upload_metrics = {
            "upload_seconds": metrics.upload_seconds,
            "verification_seconds": metrics.verification_seconds,
            "total_seconds": metrics.total_seconds,
            "average_upload_mbps": metrics.average_upload_mbps,
            "retry_count": metrics.retry_count,
            "queue_depth_at_submit": metrics.queue_depth_at_submit,
            "duplicate_skipped": float(metrics.duplicate_skipped),
        }
        job.status = JobStatus.COMPLETED
        self.job_queue.update_job(job)
        if self.runtime_monitor is not None:
            self.runtime_monitor.record_upload_time(metrics.total_seconds)
            self.runtime_monitor.set_upload_queue_length(self.upload_queue.qsize())
        if self.event_bus is not None:
            self.event_bus.publish(
                "UploadCompleted",
                job_id=job.job_id,
                remote_file_id=job.remote_metadata.get("file_id"),
                upload_seconds=metrics.total_seconds,
                duplicate_skipped=duplicate_skipped,
            )
        self._apply_cleanup_policy(job)
        with self._metrics_lock:
            self._metrics["uploaded_files"] += 1
            self._metrics["total_upload_seconds"] += metrics.upload_seconds
            self._metrics["total_verification_seconds"] += metrics.verification_seconds
            if duplicate_skipped:
                self._metrics["skipped_duplicates"] += 1

    def _mark_failed(self, job: Job, error_message: str, metrics: UploadMetrics) -> None:
        job.status = JobStatus.FAILED_UPLOAD
        job.error_message = error_message
        job.upload_metrics = {
            "upload_seconds": metrics.upload_seconds,
            "verification_seconds": metrics.verification_seconds,
            "total_seconds": metrics.total_seconds,
            "average_upload_mbps": metrics.average_upload_mbps,
            "retry_count": metrics.retry_count,
            "queue_depth_at_submit": metrics.queue_depth_at_submit,
            "duplicate_skipped": float(metrics.duplicate_skipped),
        }
        self.job_queue.update_job(job)
        if self.runtime_monitor is not None:
            self.runtime_monitor.set_upload_queue_length(self.upload_queue.qsize())
        if self.event_bus is not None:
            self.event_bus.publish(
                "UploadFailed",
                job_id=job.job_id,
                error_message=error_message,
                retry_count=metrics.retry_count,
            )
        with self._metrics_lock:
            self._metrics["failed_uploads"] += 1

    def _apply_cleanup_policy(self, job: Job) -> None:
        if not job.output_path or not job.output_path.exists():
            return
        if self.config.enable_stitching:
            return
        if self.config.drive_cleanup_policy == "keep_everything":
            return
        if self.config.drive_cleanup_policy == "delete_uploaded_clips":
            safe_remove(job.output_path)
            return
        if self.config.drive_cleanup_policy == "delete_temp_only":
            return
        if self.config.drive_cleanup_policy == "delete_everything_except_logs":
            safe_remove(job.output_path)

    def sync_artifact(
        self, local_path, folder_key: str, remote_name: Optional[str] = None
    ) -> Optional[RemoteFileMetadata]:
        if not self.config.enable_drive_upload or self.project_paths is None:
            return None
        if not local_path.exists():
            return None
        folder_id = self.project_paths.folders[folder_key]
        remote_name = remote_name or local_path.name
        local_md5 = compute_file_hash(local_path, algorithm="md5")
        existing = self.drive_client.find_file_by_name(remote_name, folder_id)
        if existing and existing.size_bytes == local_path.stat().st_size:
            if not existing.md5_checksum or existing.md5_checksum == local_md5:
                return existing
        uploaded = self.drive_client.upload_file(local_path, remote_name, folder_id)
        if uploaded is None:
            return None
        return self.drive_client.verify_upload(
            uploaded.file_id,
            local_size_bytes=local_path.stat().st_size,
            local_md5=local_md5,
        )

    def wait_for_completion(self) -> None:
        self.upload_queue.join()

    def shutdown(self, graceful: bool = True) -> None:
        if graceful:
            self.wait_for_completion()
        self.stop_event.set()
        for _ in self._workers:
            self.upload_queue.put(None)
        for worker in self._workers:
            worker.join(timeout=5)
        if self.runtime_monitor is not None:
            self.runtime_monitor.set_upload_queue_length(0)

    def get_metrics(self) -> dict:
        with self._metrics_lock:
            return dict(self._metrics)
