import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import Config
from jobs.job import Job
from logging_system import get_logger
from utils import compute_file_hash, stable_hash

logger = get_logger("cache")


@dataclass
class CacheEntry:
    cache_key: str
    output_path: str
    checksum_sha256: str
    file_size_bytes: int
    model_version: str
    renderer_backend: str
    config_signature: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "output_path": self.output_path,
            "checksum_sha256": self.checksum_sha256,
            "file_size_bytes": self.file_size_bytes,
            "model_version": self.model_version,
            "renderer_backend": self.renderer_backend,
            "config_signature": self.config_signature,
            "created_at": self.created_at,
        }


class RenderCacheStore:
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = config.cache_dir
        self.index_path = config.cache_index_path
        self._lock = threading.RLock()
        self._entries: dict[str, CacheEntry] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def build_cache_key(self, job: Job) -> str:
        start_checksum = compute_file_hash(job.start_image, algorithm="sha256")
        end_checksum = (
            compute_file_hash(job.end_image, algorithm="sha256")
            if job.end_image is not None and job.end_image.exists()
            else None
        )
        payload = {
            "prompt_hash": stable_hash(job.prompt),
            "start_image_checksum": start_checksum,
            "end_image_checksum": end_checksum,
            "renderer_backend": self.config.renderer_backend,
            "inference_settings": {
                "guidance_scale": job.guidance_scale,
                "num_inference_steps": job.num_inference_steps,
                "frame_rate": self.config.frame_rate,
                "output_codec": self.config.output_codec,
                "output_container": self.config.output_container,
                "output_quality": self.config.output_quality,
                "precision": self.config.precision,
            },
            "model_version": self.config.model_name,
            "resolution": job.resolution.label,
            "aspect_ratio": job.aspect_ratio.label,
            "duration": job.duration.label,
        }
        return stable_hash(payload)

    def _config_signature(self) -> str:
        return stable_hash(
            {
                "renderer_backend": self.config.renderer_backend,
                "model_name": self.config.model_name,
                "frame_rate": self.config.frame_rate,
                "output_codec": self.config.output_codec,
                "output_container": self.config.output_container,
                "output_quality": self.config.output_quality,
                "precision": self.config.precision,
            }
        )

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            entries = raw.get("entries", {})
            self._entries = {
                key: CacheEntry(**value)
                for key, value in entries.items()
            }
        except Exception as exc:
            logger.warning(
                f"Render cache index could not be loaded: {exc}",
                extra={"job_id": "N/A"},
            )
            self._entries = {}

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "entries": {
                key: entry.to_dict()
                for key, entry in self._entries.items()
            },
        }
        self.index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def lookup(self, cache_key: str) -> Optional[Path]:
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None:
                return None
            if entry.model_version != self.config.model_name:
                self.invalidate(cache_key)
                return None
            if entry.renderer_backend != self.config.renderer_backend:
                self.invalidate(cache_key)
                return None
            if entry.config_signature != self._config_signature():
                self.invalidate(cache_key)
                return None
            output_path = Path(entry.output_path)
            if not output_path.exists():
                self.invalidate(cache_key)
                return None
            if output_path.stat().st_size != entry.file_size_bytes:
                self.invalidate(cache_key)
                return None
            if compute_file_hash(output_path, algorithm="sha256") != entry.checksum_sha256:
                self.invalidate(cache_key)
                return None
            return output_path

    def register(self, cache_key: str, output_path: Path, metadata: dict[str, Any]) -> None:
        with self._lock:
            checksum = metadata.get("checksum_sha256") or compute_file_hash(
                output_path,
                algorithm="sha256",
            )
            file_size_bytes = int(
                metadata.get("file_size_bytes") or output_path.stat().st_size
            )
            self._entries[cache_key] = CacheEntry(
                cache_key=cache_key,
                output_path=str(output_path.resolve(strict=False)),
                checksum_sha256=checksum,
                file_size_bytes=file_size_bytes,
                model_version=self.config.model_name,
                renderer_backend=self.config.renderer_backend,
                config_signature=self._config_signature(),
                created_at=datetime.now().isoformat(),
            )
            self._save()

    def invalidate(self, cache_key: str) -> None:
        with self._lock:
            if cache_key in self._entries:
                self._entries.pop(cache_key, None)
                self._save()
