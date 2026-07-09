import os
import platform
import sys
import shutil
from pathlib import Path
from config import Config
from core import BootstrapError
from logging_system import get_logger
from orchestration.runtime_assets import verify_runtime_dependencies

logger = get_logger("bootstrap")


def check_python_version() -> None:
    """Check Python version."""
    if sys.version_info < (3, 11):
        raise BootstrapError("Python 3.11 or higher required")


def check_dependencies(config: Config) -> None:
    """Check required dependencies are installed and the runtime stack is valid."""
    report = verify_runtime_dependencies(config)
    if report.missing_required:
        raise BootstrapError(
            "Missing required dependencies for "
            f"profile={report.profile_name} renderer={report.renderer_backend or 'unknown'}: "
            + ", ".join(report.missing_required)
        )
    failed_features = [
        status for status in report.feature_statuses if status.enabled and status.required and status.state == "FAILED"
    ]
    if failed_features:
        messages = [f"{status.feature_key}: {status.reason}" for status in failed_features]
        raise BootstrapError(
            "Runtime dependency verification failed for "
            f"profile={report.profile_name} renderer={report.renderer_backend or 'unknown'}: "
            + "; ".join(messages)
        )


def check_disk_space() -> None:
    """Check available disk space."""
    temp_dir = Path.cwd()
    try:
        usage = shutil.disk_usage(temp_dir)
        free_gb = usage.free / (1024**3)
        if free_gb < 10:
            raise BootstrapError(f"Insufficient disk space (only {free_gb:.2f}GB free, need at least 10GB)")
    except BootstrapError:
        raise
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}", extra={"job_id": "N/A"})


def collect_environment_info(config: Config) -> dict:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "is_kaggle": Path("/kaggle").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ,
        "dataset_mounts": [],
        "gpu": {},
        "ram": {},
        "disk": {},
    }

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        info["dataset_mounts"] = [str(path) for path in kaggle_input.iterdir()]

    try:
        usage = shutil.disk_usage(Path.cwd())
        info["disk"] = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    except Exception:
        info["disk"] = {}

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            info["gpu"] = {
                "available": True,
                "name": props.name,
                "total_vram_bytes": total_bytes,
                "free_vram_bytes": free_bytes,
                "device_count": torch.cuda.device_count(),
            }
        else:
            info["gpu"] = {"available": False}
    except ImportError:
        info["gpu"] = {"available": False}

    try:
        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            info["ram"] = {"total_bytes": page_size * phys_pages}
    except Exception:
        info["ram"] = {}
    config.extra["environment_info"] = info
    return info


def check_gpu(config: Config) -> None:
    """Check GPU availability."""
    environment = config.extra.get("environment_info") or collect_environment_info(config)
    gpu_available = bool(environment.get("gpu", {}).get("available"))
    if config.use_cuda and not gpu_available:
        raise BootstrapError("CUDA GPU is required but not available")
    if not gpu_available:
        logger.warning("CUDA not available, using CPU (this will be slow)", extra={"job_id": "N/A"})


def check_model_assets(config: Config) -> None:
    from assets import AssetManager

    report = AssetManager(config).ensure_assets()
    if not report.ready:
        raise BootstrapError(
            "Model assets are not ready. See asset_report.json for details."
        )


def bootstrap(config: Config) -> None:
    """Run all bootstrap checks."""
    logger.info("Starting bootstrap checks...", extra={"job_id": "N/A"})
    check_python_version()
    check_dependencies(config)
    check_disk_space()
    collect_environment_info(config)
    check_gpu(config)
    check_model_assets(config)
    logger.info("Bootstrap checks passed!", extra={"job_id": "N/A"})
