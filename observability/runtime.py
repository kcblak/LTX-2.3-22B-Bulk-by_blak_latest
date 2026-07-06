import json
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

from config import Config
from logging_system import get_logger

logger = get_logger("performance")

FINAL_JOB_STATUSES = {
    "COMPLETED",
    "FAILED_RENDER",
    "FAILED_UPLOAD",
    "FAILED_VERIFY",
    "FAILED_VALIDATION",
}


class RuntimeMonitor:
    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_job_id: Optional[str] = None
        self._status: str = "INITIALIZING"
        self._upload_queue_length = 0
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._render_times: deque[float] = deque(maxlen=256)
        self._upload_times: deque[float] = deque(maxlen=256)
        self._errors: deque[dict[str, Any]] = deque(maxlen=100)
        self._snapshots: list[dict[str, Any]] = []
        self._heartbeat_remote_sync = None
        self._job_queue = None
        self._last_snapshot: Optional[dict[str, Any]] = None
        self._last_heartbeat_time = 0.0
        self._started_at = datetime.now().isoformat()
        self._started_monotonic = time.monotonic()

    def attach(self, *, job_queue=None, heartbeat_remote_sync=None) -> None:
        self._job_queue = job_queue
        self._heartbeat_remote_sync = heartbeat_remote_sync

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="runtime-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.write_heartbeat(force=True)

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def set_current_job(self, job_id: Optional[str]) -> None:
        with self._lock:
            self._current_job_id = job_id

    def set_upload_queue_length(self, length: int) -> None:
        with self._lock:
            self._upload_queue_length = max(0, length)

    def record_render_time(self, seconds: float) -> None:
        with self._lock:
            self._render_times.append(seconds)
            self._completed_jobs += 1

    def record_upload_time(self, seconds: float) -> None:
        with self._lock:
            self._upload_times.append(seconds)

    def record_error(self, category: str, message: str, recommendation: str = "") -> None:
        with self._lock:
            self._failed_jobs += 1
            self._errors.append(
                {
                    "category": category,
                    "message": message,
                    "recommendation": recommendation,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def collect_snapshot(self) -> dict[str, Any]:
        snapshot = self.get_snapshot()
        with self._lock:
            self._last_snapshot = snapshot
            self._snapshots.append(snapshot)
        return snapshot

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            pending_jobs = 0
            total_jobs = 0
            if self._job_queue is not None:
                total_jobs = len(self._job_queue.jobs)
                pending_jobs = len(
                    [
                        job
                        for job in self._job_queue.jobs
                        if job.status.name not in FINAL_JOB_STATUSES
                    ]
                )
            elapsed_seconds = max(0.0, time.monotonic() - self._started_monotonic)
            throughput_jobs_per_hour = (
                (self._completed_jobs / elapsed_seconds) * 3600.0
                if elapsed_seconds > 0 and self._completed_jobs > 0
                else 0.0
            )
            eta_seconds = (
                (pending_jobs / self._completed_jobs) * elapsed_seconds
                if pending_jobs > 0 and self._completed_jobs > 0
                else None
            )
            snapshot = {
                "project_id": self.config.project_id,
                "run_id": self.config.run_id,
                "started_at": self._started_at,
                "status": self._status,
                "current_job": self._current_job_id,
                "total_jobs": total_jobs,
                "completed_jobs": self._completed_jobs,
                "failed_jobs": self._failed_jobs,
                "remaining_jobs": pending_jobs,
                "upload_queue": self._upload_queue_length,
                "elapsed_seconds": round(elapsed_seconds, 2),
                "throughput_jobs_per_hour": round(throughput_jobs_per_hour, 2),
                "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
                "average_render_time_seconds": (
                    sum(self._render_times) / len(self._render_times)
                    if self._render_times
                    else 0.0
                ),
                "average_upload_time_seconds": (
                    sum(self._upload_times) / len(self._upload_times)
                    if self._upload_times
                    else 0.0
                ),
                "error_count": len(self._errors),
                "last_update": datetime.now().isoformat(),
            }
        snapshot.update(self._sample_system_metrics())
        return snapshot

    def _sample_system_metrics(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "cpu_percent": None,
            "ram_total_mb": None,
            "ram_used_mb": None,
            "gpu_utilization_percent": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "gpu": None,
        }
        try:
            import psutil

            data["cpu_percent"] = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            data["ram_total_mb"] = round(memory.total / (1024**2), 2)
            data["ram_used_mb"] = round(memory.used / (1024**2), 2)
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                props = torch.cuda.get_device_properties(0)
                data["vram_used_mb"] = round(max(allocated, reserved) / (1024**2), 2)
                data["vram_total_mb"] = round(props.total_memory / (1024**2), 2)
                data["gpu"] = props.name
        except Exception:
            pass
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
            first_line = result.stdout.strip().splitlines()[0]
            utilization, memory_used, memory_total, name = [
                part.strip() for part in first_line.split(",", 3)
            ]
            data["gpu_utilization_percent"] = float(utilization)
            data["vram_used_mb"] = float(memory_used)
            data["vram_total_mb"] = float(memory_total)
            data["gpu"] = name
        except Exception:
            pass
        return data

    def _run_loop(self) -> None:
        poll_interval = max(1, self.config.health_poll_interval_seconds)
        heartbeat_interval = max(1, self.config.heartbeat_interval_seconds)
        while not self._stop_event.wait(poll_interval):
            snapshot = self.collect_snapshot()
            if time.monotonic() - self._last_heartbeat_time >= heartbeat_interval:
                self.write_heartbeat(snapshot=snapshot)

    def write_heartbeat(
        self,
        *,
        force: bool = False,
        snapshot: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        current_snapshot = snapshot or self.collect_snapshot()
        if not self.config.heartbeat_enabled and not force:
            return current_snapshot
        heartbeat_path = self.config.heartbeat_path
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        with heartbeat_path.open("w", encoding="utf-8") as handle:
            json.dump(current_snapshot, handle, indent=2, ensure_ascii=False)
        self._last_heartbeat_time = time.monotonic()
        logger.info(
            "Heartbeat generated",
            extra={"job_id": self._current_job_id or "N/A"},
        )
        if (
            self.config.sync_heartbeat_to_drive
            and self._heartbeat_remote_sync is not None
            and self.config.enable_drive_upload
        ):
            try:
                self._heartbeat_remote_sync(heartbeat_path, "logs", heartbeat_path.name)
            except Exception:
                logger.debug(
                    "Remote heartbeat sync skipped after failure",
                    extra={"job_id": self._current_job_id or "N/A"},
                )
        return current_snapshot

    def export(self) -> dict[str, Any]:
        with self._lock:
            latest = self._last_snapshot or self.get_snapshot()
            return {
                "snapshots": list(self._snapshots),
                "errors": list(self._errors),
                "latest": latest,
            }
