from dataclasses import dataclass
from typing import Optional

from core import (
    BootstrapError,
    ConfigurationError,
    DriveError,
    RendererEncodingError,
    RendererInitializationError,
    RendererInputError,
    RendererOOMError,
    RendererOutputValidationError,
    StorageError,
    ValidationError,
)


@dataclass(frozen=True)
class FailureInfo:
    category: str
    summary: str
    recommendation: str


def classify_exception(exc: Exception) -> FailureInfo:
    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()

    if isinstance(exc, ConfigurationError):
        return FailureInfo(
            category="Configuration",
            summary=message,
            recommendation="Review configuration overrides, profile selection, and file paths before starting the run.",
        )
    if isinstance(exc, ValidationError):
        return FailureInfo(
            category="Validation",
            summary=message,
            recommendation="Correct the reported project input issues and rerun preflight before rendering.",
        )
    if isinstance(exc, BootstrapError):
        return FailureInfo(
            category="Environment",
            summary=message,
            recommendation="Verify Python, dependencies, disk, and GPU availability on the worker node.",
        )
    if isinstance(exc, RendererOOMError) or "out of memory" in lowered or "cuda oom" in lowered:
        return FailureInfo(
            category="GPU",
            summary=message,
            recommendation="Reduce resolution, lower inference steps, or switch to the low_vram profile.",
        )
    if isinstance(exc, RendererInputError):
        return FailureInfo(
            category="Renderer",
            summary=message,
            recommendation="Inspect the job input parameters and source images for incompatible values.",
        )
    if isinstance(exc, RendererOutputValidationError):
        return FailureInfo(
            category="Encoding",
            summary=message,
            recommendation="Inspect ffmpeg availability, output size thresholds, and encoded clip validation settings.",
        )
    if isinstance(exc, RendererEncodingError):
        return FailureInfo(
            category="Encoding",
            summary=message,
            recommendation="Verify ffmpeg, codec configuration, and temporary disk space before retrying.",
        )
    if isinstance(exc, RendererInitializationError):
        return FailureInfo(
            category="Renderer",
            summary=message,
            recommendation="Verify model assets, backend dependencies, and GPU memory budget for initialization.",
        )
    if isinstance(exc, DriveError):
        return FailureInfo(
            category="Google Drive",
            summary=message,
            recommendation="Check Drive credentials, folder permissions, and network connectivity.",
        )
    if isinstance(exc, StorageError):
        return FailureInfo(
            category="Filesystem",
            summary=message,
            recommendation="Verify local output paths, free disk space, and write permissions.",
        )
    if "cuda" in lowered:
        return FailureInfo(
            category="CUDA",
            summary=message,
            recommendation="Confirm the Kaggle accelerator is attached and the installed CUDA stack matches the runtime.",
        )
    if "ffmpeg" in lowered or "codec" in lowered:
        return FailureInfo(
            category="Encoding",
            summary=message,
            recommendation="Check ffmpeg installation, codec selection, and output container compatibility.",
        )
    if "drive" in lowered or "google" in lowered or "upload" in lowered:
        return FailureInfo(
            category="Google Drive",
            summary=message,
            recommendation="Inspect authentication, API access, and upload retry settings.",
        )
    if "network" in lowered or "timeout" in lowered or "connection" in lowered:
        return FailureInfo(
            category="Network",
            summary=message,
            recommendation="Retry after confirming internet connectivity and remote service availability.",
        )
    return FailureInfo(
        category="Unexpected",
        summary=message,
        recommendation="Inspect diagnostics, structured logs, and the error report for the full stack trace.",
    )


def classify_render_failure(
    error_type: Optional[str],
    error_message: Optional[str],
) -> FailureInfo:
    error_name = error_type or "RenderError"
    message = error_message or error_name

    if error_name == "RendererInputError":
        return FailureInfo(
            category="Renderer",
            summary=message,
            recommendation="Review the prompt, image paths, and per-job generation parameters.",
        )
    if error_name == "RendererOOMError":
        return FailureInfo(
            category="GPU",
            summary=message,
            recommendation="Reduce resolution, switch to the low_vram profile, or lower inference steps.",
        )
    if error_name == "RendererOutputValidationError":
        return FailureInfo(
            category="Encoding",
            summary=message,
            recommendation="Verify ffmpeg output settings and output validation thresholds.",
        )
    if error_name == "RendererEncodingError":
        return FailureInfo(
            category="Encoding",
            summary=message,
            recommendation="Check ffmpeg, codec support, and local disk write access.",
        )
    if error_name == "RendererInitializationError":
        return FailureInfo(
            category="Renderer",
            summary=message,
            recommendation="Verify backend dependencies and model assets before rerunning the job.",
        )
    return FailureInfo(
        category="Renderer",
        summary=message,
        recommendation="Check the renderer logs and diagnostics report for the full failure context.",
    )
