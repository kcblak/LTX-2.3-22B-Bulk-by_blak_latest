import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, TypeVar, Tuple
from functools import wraps
from core import StorageError

T = TypeVar("T")
_FILE_HASH_CACHE: dict[tuple[str, str, int, float], str] = {}


def generate_job_id(job_data: Dict[str, Any]) -> str:
    """Generate a unique job ID from job data."""
    job_str = json.dumps(job_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(job_str.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute the hash of a file."""
    stat = file_path.stat()
    cache_key = (
        str(file_path.resolve(strict=False)),
        algorithm,
        stat.st_size,
        stat.st_mtime,
    )
    if cache_key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[cache_key]
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    digest = hash_obj.hexdigest()
    _FILE_HASH_CACHE[cache_key] = digest
    return digest


def stable_hash(value: Any, algorithm: str = "sha256") -> str:
    """Compute a stable hash for a JSON-serializable value."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(payload.encode("utf-8"))
    return hash_obj.hexdigest()


def safe_remove(path: Path) -> None:
    """Safely remove a file or directory."""
    try:
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        raise StorageError(f"Failed to remove {path}: {e}")


def safe_mkdir(path: Path) -> None:
    """Safely create a directory."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise StorageError(f"Failed to create directory {path}: {e}")


class Timer:
    """Context manager for timing code execution."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            self.elapsed_time = time.perf_counter() - self.start_time
