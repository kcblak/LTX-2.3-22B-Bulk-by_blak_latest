from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from config import Config


_REQUIREMENT_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")
_SUPPORTED_PLATFORMS = ("linux", "windows", "darwin")


@dataclass(frozen=True)
class DependencySpec:
    key: str
    requirement: str
    import_name: Optional[str]
    classification: str = "required"
    supported_platforms: tuple[str, ...] = _SUPPORTED_PLATFORMS
    kaggle_supported: bool = True
    description: str = ""
    min_python: Optional[tuple[int, int]] = None
    max_python: Optional[tuple[int, int]] = None
    development_wheel: bool = False

    @property
    def package_name(self) -> str:
        match = _REQUIREMENT_NAME_PATTERN.match(self.requirement)
        if match:
            return match.group(1)
        return self.requirement

    @property
    def optional(self) -> bool:
        return self.classification == "optional"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    description: str
    dependencies: tuple[str, ...] = ()
    package_keys: tuple[str, ...] = ()
    validation_checks: tuple[str, ...] = ()
    failure_policy: str = "disable"


@dataclass
class RequirementStatus:
    requirement: str
    package_name: str
    installed: bool
    version: str = ""


@dataclass
class RequirementFileReport:
    path: Path
    inspected: list[RequirementStatus] = field(default_factory=list)
    installed_requirements: list[str] = field(default_factory=list)
    skipped_requirements: list[str] = field(default_factory=list)


@dataclass
class GitCommandResult:
    command: list[str]
    cwd: Optional[str]
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    note: str = ""


@dataclass
class GitCheckoutReport:
    destination: Path
    repo_url: str
    ref: str
    cloned: bool
    updated: bool
    source: str = "github"
    update_policy: str = "auto"
    used_local_copy: bool = False
    local_copy_reason: str = ""
    repository_state: str = "unknown"
    current_branch: str = ""
    default_branch: str = ""
    remote_name: str = "origin"
    remote_url: str = ""
    remote_reachable: bool = False
    working_tree: str = "unknown"
    target_ref: str = ""
    validation_messages: list[str] = field(default_factory=list)
    recovery_actions: list[str] = field(default_factory=list)
    command_results: list[GitCommandResult] = field(default_factory=list)
    failure_reason: str = ""


@dataclass
class PipInstallResult:
    requirement: str
    package_name: str
    classification: str
    optional: bool
    feature_keys: list[str]
    profile_name: str
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    reason: str = ""
    requested_version: str = ""
    installed_version_before: str = ""
    installed_version_after: str = ""
    platform_name: str = ""
    python_version: str = ""
    cuda_version: str = ""
    duration_seconds: float = 0.0
    suggested_resolution: str = ""
    command: list[str] = field(default_factory=list)


@dataclass
class DependencyStatus:
    requirement: str
    package_name: str
    import_name: Optional[str]
    classification: str
    feature_keys: list[str]
    installed: bool
    version: str = ""
    selected: bool = True
    reason: str = ""


@dataclass
class DependencyVerificationCheck:
    name: str
    success: bool
    details: str = ""
    feature_key: str = "runtime"


@dataclass
class FeatureRuntimeStatus:
    feature_key: str
    enabled: bool
    required: bool
    state: str
    reason: str = ""
    package_keys: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class DependencyProfileReport:
    profile_name: str
    platform_name: str
    kaggle_mode: bool
    renderer_backend: str = ""
    enabled_features: list[str] = field(default_factory=list)
    disabled_features: dict[str, str] = field(default_factory=dict)
    feature_statuses: list[FeatureRuntimeStatus] = field(default_factory=list)
    selected_requirements: list[str] = field(default_factory=list)
    statuses: list[DependencyStatus] = field(default_factory=list)
    installed_packages: list[PipInstallResult] = field(default_factory=list)
    skipped_packages: list[PipInstallResult] = field(default_factory=list)
    failed_required: list[PipInstallResult] = field(default_factory=list)
    failed_optional: list[PipInstallResult] = field(default_factory=list)
    verification_checks: list[DependencyVerificationCheck] = field(default_factory=list)
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""

    @property
    def missing_required(self) -> list[str]:
        return [
            status.requirement
            for status in self.statuses
            if status.selected
            and not status.installed
            and status.classification not in {"optional", "experimental"}
        ]

    @property
    def missing_optional(self) -> list[str]:
        return [
            status.requirement
            for status in self.statuses
            if status.selected and not status.installed and status.classification == "optional"
        ]

    @property
    def verification_passed(self) -> bool:
        if self.feature_statuses:
            return all(
                status.state != "FAILED"
                for status in self.feature_statuses
                if status.enabled and status.required
            )
        return all(check.success for check in self.verification_checks)

    @property
    def success(self) -> bool:
        return not self.failed_required and self.verification_passed


@dataclass
class ModelAssetReport:
    model_dir: Path
    downloaded_files: list[str] = field(default_factory=list)
    existing_files: list[str] = field(default_factory=list)
    runtime_report: Optional[GitCheckoutReport] = None
    dependency_report: Optional[DependencyProfileReport] = None


_DEPENDENCY_CATALOG: dict[str, DependencySpec] = {
    "pyyaml": DependencySpec(
        key="pyyaml",
        requirement="PyYAML>=6.0.1",
        import_name="yaml",
        description="Repository config loading",
    ),
    "pillow": DependencySpec(
        key="pillow",
        requirement="Pillow>=10.0.0",
        import_name="PIL",
        description="Image validation and processing",
    ),
    "numpy": DependencySpec(
        key="numpy",
        requirement="numpy>=1.26.0",
        import_name="numpy",
        description="Core array operations",
    ),
    "torch": DependencySpec(
        key="torch",
        requirement="torch>=2.2.0",
        import_name="torch",
        description="CUDA runtime and rendering backend",
    ),
    "torchvision": DependencySpec(
        key="torchvision",
        requirement="torchvision>=0.17.0",
        import_name="torchvision",
        description="Torch vision utilities",
    ),
    "imageio": DependencySpec(
        key="imageio",
        requirement="imageio>=2.34.0",
        import_name="imageio",
        description="Video IO",
    ),
    "imageio-ffmpeg": DependencySpec(
        key="imageio-ffmpeg",
        requirement="imageio-ffmpeg>=0.4.9",
        import_name="imageio_ffmpeg",
        description="FFmpeg bridge",
    ),
    "psutil": DependencySpec(
        key="psutil",
        requirement="psutil>=5.9.8",
        import_name="psutil",
        description="Runtime monitoring",
    ),
    "huggingface-hub": DependencySpec(
        key="huggingface-hub",
        requirement="huggingface-hub>=0.24.0",
        import_name="huggingface_hub",
        description="Model asset downloads",
    ),
    "safetensors": DependencySpec(
        key="safetensors",
        requirement="safetensors>=0.4.3",
        import_name="safetensors",
        description="Model weights",
    ),
    "diffusers": DependencySpec(
        key="diffusers",
        requirement="diffusers>=0.31.0",
        import_name="diffusers",
        description="Diffusers renderer backend",
    ),
    "transformers": DependencySpec(
        key="transformers",
        requirement="transformers>=4.44.0",
        import_name="transformers",
        description="Text encoder runtime",
    ),
    "accelerate": DependencySpec(
        key="accelerate",
        requirement="accelerate>=0.33.0",
        import_name="accelerate",
        description="Backend acceleration helpers",
    ),
    "mmgp": DependencySpec(
        key="mmgp",
        requirement="mmgp>=3.2.0",
        import_name="mmgp",
        description="Wan2GP memory management runtime",
    ),
    "gguf": DependencySpec(
        key="gguf",
        requirement="gguf>=0.10.0",
        import_name="gguf",
        description="GGUF model format support",
    ),
    "google-api-python-client": DependencySpec(
        key="google-api-python-client",
        requirement="google-api-python-client>=2.130.0",
        import_name="googleapiclient",
        classification="optional",
        description="Google Drive API access",
    ),
    "google-auth": DependencySpec(
        key="google-auth",
        requirement="google-auth>=2.29.0",
        import_name="google.auth",
        classification="optional",
        description="Google authentication",
    ),
    "google-auth-oauthlib": DependencySpec(
        key="google-auth-oauthlib",
        requirement="google-auth-oauthlib>=1.2.0",
        import_name="google_auth_oauthlib",
        classification="optional",
        description="OAuth helpers for Google Drive",
    ),
    "coverage": DependencySpec(
        key="coverage",
        requirement="coverage>=7.6.0",
        import_name="coverage",
        classification="optional",
        description="Testing utilities",
    ),
    "pytest": DependencySpec(
        key="pytest",
        requirement="pytest>=8.3.0",
        import_name="pytest",
        classification="optional",
        description="Test runner",
    ),
    "ruff": DependencySpec(
        key="ruff",
        requirement="ruff>=0.6.0",
        import_name="ruff",
        classification="optional",
        description="Development linting",
    ),
    "mypy": DependencySpec(
        key="mypy",
        requirement="mypy>=1.11.0",
        import_name="mypy",
        classification="optional",
        description="Static typing checks",
    ),
    "pywin32": DependencySpec(
        key="pywin32",
        requirement="pywin32>=306",
        import_name="win32api",
        classification="platform-specific",
        supported_platforms=("windows",),
        kaggle_supported=False,
        description="Windows-only shell integration",
    ),
    "torchaudio": DependencySpec(
        key="torchaudio",
        requirement="torchaudio>=2.2.0",
        import_name="torchaudio",
        classification="optional",
        description="Audio processing runtime",
    ),
    "librosa": DependencySpec(
        key="librosa",
        requirement="librosa>=0.10.0",
        import_name="librosa",
        classification="optional",
        description="Audio feature extraction",
    ),
    "soundfile": DependencySpec(
        key="soundfile",
        requirement="soundfile>=0.12.1",
        import_name="soundfile",
        classification="optional",
        description="Audio file IO",
    ),
    "gradio": DependencySpec(
        key="gradio",
        requirement="gradio>=4.0.0",
        import_name="gradio",
        classification="optional",
        kaggle_supported=False,
        description="Interactive UI",
    ),
    "openai-whisper": DependencySpec(
        key="openai-whisper",
        requirement="openai-whisper>=20240930",
        import_name="whisper",
        classification="optional",
        kaggle_supported=False,
        description="Whisper speech recognition runtime",
    ),
    "speechbrain": DependencySpec(
        key="speechbrain",
        requirement="speechbrain>=1.0.0",
        import_name="speechbrain",
        classification="optional",
        kaggle_supported=False,
        description="Speech processing toolkit",
    ),
    "pyannote-audio": DependencySpec(
        key="pyannote-audio",
        requirement="pyannote.audio>=3.3.0",
        import_name="pyannote.audio",
        classification="optional",
        kaggle_supported=False,
        description="Advanced speech diarization toolkit",
    ),
    "opencv-python": DependencySpec(
        key="opencv-python",
        requirement="opencv-python>=4.10.0",
        import_name="cv2",
        classification="optional",
        description="Image editing and CV utilities",
    ),
    "rembg": DependencySpec(
        key="rembg",
        requirement="rembg>=2.0.59",
        import_name="rembg",
        classification="optional",
        kaggle_supported=False,
        description="Background removal",
    ),
    "gfpgan": DependencySpec(
        key="gfpgan",
        requirement="gfpgan>=1.3.8",
        import_name="gfpgan",
        classification="experimental",
        kaggle_supported=False,
        description="Face restoration",
    ),
    "realesrgan": DependencySpec(
        key="realesrgan",
        requirement="realesrgan>=0.3.0",
        import_name="realesrgan",
        classification="experimental",
        kaggle_supported=False,
        description="Image upscaling and restoration",
    ),
}

_DEPENDENCY_PROFILES: dict[str, tuple[str, ...]] = {
    "bootstrap": ("pyyaml",),
    "diffusers": (
        "pyyaml",
        "pillow",
        "numpy",
        "torch",
        "torchvision",
        "imageio",
        "imageio-ffmpeg",
        "psutil",
        "huggingface-hub",
        "safetensors",
        "diffusers",
        "transformers",
        "accelerate",
    ),
    "wan2gp": (
        "pyyaml",
        "pillow",
        "numpy",
        "torch",
        "torchvision",
        "imageio",
        "imageio-ffmpeg",
        "psutil",
        "huggingface-hub",
        "safetensors",
        "transformers",
        "accelerate",
        "mmgp",
        "gguf",
    ),
    "development": ("coverage", "ruff", "pywin32", "gradio", "gfpgan"),
    "testing": ("coverage",),
}

_OPTIONAL_FEATURE_KEYS: dict[str, tuple[str, ...]] = {
    "drive": (
        "google-api-python-client",
        "google-auth",
        "google-auth-oauthlib",
    )
}

_EXECUTION_PROFILES: dict[str, dict[str, bool]] = {
    "bootstrap": {
        "core_renderer": False,
    },
    "kaggle_bulk": {
        "core_renderer": True,
        "csv_batch_rendering": True,
        "resume_engine": True,
        "video_stitching": True,
        "reporting": True,
        "google_drive_integration": True,
        "wan2gp_runtime": True,
        "gguf_runtime": True,
        "msr_models": True,
        "diffusers_backend": False,
    },
    "kaggle_interactive": {
        "core_renderer": True,
        "csv_batch_rendering": True,
        "resume_engine": True,
        "video_stitching": True,
        "reporting": True,
        "google_drive_integration": False,
        "wan2gp_runtime": True,
        "gguf_runtime": True,
        "msr_models": True,
        "diffusers_backend": False,
    },
    "local_development": {
        "core_renderer": True,
        "csv_batch_rendering": True,
        "resume_engine": True,
        "video_stitching": True,
        "reporting": True,
        "google_drive_integration": True,
        "development_tools": True,
        "testing_tools": True,
        "image_editing": True,
        "gradio_ui": True,
    },
    "production": {
        "core_renderer": True,
        "csv_batch_rendering": True,
        "resume_engine": True,
        "video_stitching": True,
        "reporting": True,
        "google_drive_integration": True,
        "wan2gp_runtime": True,
        "gguf_runtime": True,
        "msr_models": True,
        "diffusers_backend": False,
    },
    "testing": {
        "core_renderer": True,
        "csv_batch_rendering": True,
        "resume_engine": True,
        "video_stitching": False,
        "reporting": True,
        "google_drive_integration": False,
        "development_tools": False,
        "testing_tools": True,
        "gradio_ui": False,
    },
}

_EXECUTION_PROFILE_ALIASES: dict[str, str] = {
    "production_server": "production",
    "workstation": "local_development",
}

_FEATURE_CATALOG: dict[str, FeatureSpec] = {
    "core_renderer": FeatureSpec(
        key="core_renderer",
        description="Core renderer runtime, image IO, and configuration loading",
        package_keys=("pyyaml", "pillow", "numpy", "torch", "torchvision", "imageio", "imageio-ffmpeg", "psutil"),
        validation_checks=("torch", "imageio", "ffmpeg"),
        failure_policy="abort",
    ),
    "wan2gp_runtime": FeatureSpec(
        key="wan2gp_runtime",
        description="Wan2GP backend and model runtime",
        dependencies=("core_renderer",),
        package_keys=("transformers", "accelerate", "huggingface-hub", "safetensors", "mmgp"),
        validation_checks=("transformers", "accelerate", "mmgp", "wan2gp"),
        failure_policy="abort",
    ),
    "diffusers_backend": FeatureSpec(
        key="diffusers_backend",
        description="Diffusers backend support",
        dependencies=("core_renderer",),
        package_keys=("diffusers", "transformers", "accelerate", "huggingface-hub", "safetensors"),
        validation_checks=("diffusers", "transformers", "accelerate"),
        failure_policy="abort",
    ),
    "gguf_runtime": FeatureSpec(
        key="gguf_runtime",
        description="GGUF model format runtime support",
        dependencies=("core_renderer",),
        package_keys=("gguf",),
        validation_checks=("gguf",),
        failure_policy="abort",
    ),
    "msr_models": FeatureSpec(
        key="msr_models",
        description="MSR LoRA and model variant support",
        dependencies=("wan2gp_runtime", "gguf_runtime"),
        validation_checks=("msr_models",),
        failure_policy="abort",
    ),
    "google_drive_integration": FeatureSpec(
        key="google_drive_integration",
        description="Google Drive synchronization and upload support",
        dependencies=("reporting",),
        package_keys=("google-api-python-client", "google-auth", "google-auth-oauthlib"),
        validation_checks=("google_drive",),
        failure_policy="disable",
    ),
    "csv_batch_rendering": FeatureSpec(
        key="csv_batch_rendering",
        description="CSV-driven batch rendering workload",
        dependencies=("core_renderer",),
        validation_checks=("csv_batch_rendering",),
        failure_policy="abort",
    ),
    "resume_engine": FeatureSpec(
        key="resume_engine",
        description="Manifest and cache-based resume engine",
        dependencies=("csv_batch_rendering",),
        validation_checks=("resume_engine",),
        failure_policy="disable",
    ),
    "video_stitching": FeatureSpec(
        key="video_stitching",
        description="Final video stitching and preview generation",
        dependencies=("core_renderer",),
        validation_checks=("video_stitching", "ffmpeg"),
        failure_policy="disable",
    ),
    "reporting": FeatureSpec(
        key="reporting",
        description="Structured reports, summaries, and diagnostics",
        validation_checks=("reporting",),
        failure_policy="abort",
    ),
    "gradio_ui": FeatureSpec(
        key="gradio_ui",
        description="Interactive Gradio UI",
        dependencies=("core_renderer",),
        package_keys=("gradio",),
        validation_checks=("gradio",),
        failure_policy="disable",
    ),
    "whisper": FeatureSpec(
        key="whisper",
        description="Whisper inference support",
        dependencies=("audio_processing",),
        package_keys=("openai-whisper",),
        validation_checks=("whisper",),
        failure_policy="disable",
    ),
    "audio_processing": FeatureSpec(
        key="audio_processing",
        description="Audio ingestion and processing stack",
        dependencies=("core_renderer",),
        package_keys=("torchaudio", "librosa", "soundfile"),
        validation_checks=("audio_processing",),
        failure_policy="disable",
    ),
    "speech_recognition": FeatureSpec(
        key="speech_recognition",
        description="Speech recognition, diarization, and transcription",
        dependencies=("whisper",),
        package_keys=("speechbrain", "pyannote-audio"),
        validation_checks=("speech_recognition",),
        failure_policy="disable",
    ),
    "face_restoration": FeatureSpec(
        key="face_restoration",
        description="Face restoration and enhancement",
        dependencies=("image_editing",),
        package_keys=("gfpgan", "realesrgan"),
        validation_checks=("face_restoration",),
        failure_policy="disable",
    ),
    "background_removal": FeatureSpec(
        key="background_removal",
        description="Background removal workflows",
        dependencies=("image_editing",),
        package_keys=("rembg",),
        validation_checks=("background_removal",),
        failure_policy="disable",
    ),
    "image_editing": FeatureSpec(
        key="image_editing",
        description="Image editing and computer vision workflows",
        dependencies=("core_renderer",),
        package_keys=("opencv-python",),
        validation_checks=("image_editing",),
        failure_policy="disable",
    ),
    "development_tools": FeatureSpec(
        key="development_tools",
        description="Development linting and typing tools",
        package_keys=("ruff", "mypy"),
        validation_checks=("development_tools",),
        failure_policy="disable",
    ),
    "testing_tools": FeatureSpec(
        key="testing_tools",
        description="Testing and coverage tools",
        package_keys=("pytest", "coverage"),
        validation_checks=("testing_tools",),
        failure_policy="disable",
    ),
}


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _current_platform_name() -> str:
    normalized = platform.system().strip().lower()
    if normalized.startswith("win"):
        return "windows"
    if normalized.startswith("darwin"):
        return "darwin"
    return "linux"


def _is_kaggle_runtime() -> bool:
    return Path("/kaggle").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _strip_wrapping_delimiters(value: str) -> str:
    cleaned = value.strip()
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'", "`"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _sanitize_git_text(value: str, *, field_name: str) -> str:
    cleaned = _strip_wrapping_delimiters(value)
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _installed_distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        versions[_normalize_package_name(name)] = distribution.version
    return versions


def _is_import_available(import_name: Optional[str]) -> bool:
    if not import_name:
        return True
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _run_checked_command(
    command: list[str],
    *,
    cwd: Optional[Path] = None,
    field_name: str = "command",
) -> subprocess.CompletedProcess[str]:
    command_result = _run_command_result(command, cwd=cwd)
    if not command_result.success:
        command_text = " ".join(command)
        stdout = command_result.stdout.strip()
        stderr = command_result.stderr.strip()
        raise RuntimeError(
            f"{field_name} failed (exit_code={command_result.exit_code}): {command_text}\n"
            f"stdout:\n{stdout or '<empty>'}\n"
            f"stderr:\n{stderr or '<empty>'}"
        )
    return subprocess.CompletedProcess(
        command,
        command_result.exit_code,
        stdout=command_result.stdout,
        stderr=command_result.stderr,
    )


def _run_command_result(
    command: list[str],
    *,
    cwd: Optional[Path] = None,
    timeout_seconds: int = 20,
    note: str = "",
) -> GitCommandResult:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return GitCommandResult(
            command=list(command),
            cwd=str(cwd) if cwd is not None else None,
            success=result.returncode == 0,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=round(time.monotonic() - started, 2),
            note=note,
        )
    except subprocess.TimeoutExpired as exc:
        return GitCommandResult(
            command=list(command),
            cwd=str(cwd) if cwd is not None else None,
            success=False,
            exit_code=-1,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            duration_seconds=round(time.monotonic() - started, 2),
            timed_out=True,
            note=note or "Command timed out",
        )


def _normalize_repository_source(value: str) -> str:
    normalized = (value or "github").strip().lower()
    if normalized not in {"github", "dataset"}:
        raise ValueError("repository source must be 'github' or 'dataset'")
    return normalized


def _normalize_update_policy(value: str) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized not in {"auto", "never", "force"}:
        raise ValueError("repository update policy must be 'auto', 'never', or 'force'")
    return normalized


def _append_git_result(report: GitCheckoutReport, result: GitCommandResult) -> GitCommandResult:
    report.command_results.append(result)
    return result


def _path_looks_like_source_tree(destination: Path) -> bool:
    if not destination.exists() or not destination.is_dir():
        return False
    try:
        return any(destination.iterdir())
    except Exception:
        return False


def _git_command(
    report: GitCheckoutReport,
    command: list[str],
    *,
    cwd: Optional[Path] = None,
    timeout_seconds: int = 20,
    note: str = "",
) -> GitCommandResult:
    return _append_git_result(
        report,
        _run_command_result(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            note=note,
        ),
    )


def _describe_working_tree(status_output: str) -> str:
    lines = [line for line in status_output.splitlines() if line.strip()]
    if any(line.startswith("##") and "detached" in line.lower() for line in lines):
        return "detached"
    if len(lines) <= 1:
        return "clean"
    return "dirty"


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _remote_branch_names(output: str) -> list[str]:
    branches: list[str] = []
    for line in output.splitlines():
        cleaned = line.strip()
        if not cleaned or "->" in cleaned:
            continue
        if cleaned.startswith("origin/"):
            branches.append(cleaned.split("/", 1)[1])
    return branches


def _parse_default_branch(output: str) -> str:
    for line in output.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("ref: refs/heads/") and cleaned.endswith("HEAD"):
            branch = cleaned[len("ref: refs/heads/") :].split("\t", 1)[0].strip()
            if branch:
                return branch
    return ""


def _build_profile_specs(
    profile_name: str,
    *,
    optional_features: Optional[set[str]] = None,
) -> list[DependencySpec]:
    if profile_name not in _DEPENDENCY_PROFILES:
        available = ", ".join(sorted(_DEPENDENCY_PROFILES))
        raise ValueError(f"Unknown dependency profile '{profile_name}'. Available: {available}")
    keys = list(_DEPENDENCY_PROFILES[profile_name])
    for feature_name in sorted(optional_features or set()):
        keys.extend(_OPTIONAL_FEATURE_KEYS.get(feature_name, ()))
    return [_DEPENDENCY_CATALOG[key] for key in keys]


def _resolve_execution_profile_name(profile_name: str) -> str:
    normalized = profile_name.strip().lower()
    return _EXECUTION_PROFILE_ALIASES.get(normalized, normalized)


def detect_execution_profile(config: Config) -> str:
    requested = _resolve_execution_profile_name(config.execution_profile or "")
    if requested and requested != "auto":
        return requested
    return "kaggle_bulk" if _is_kaggle_runtime() else "production"


def _resolve_feature_overrides(config: Config) -> dict[str, bool]:
    execution_profile = detect_execution_profile(config)
    resolved = dict(_EXECUTION_PROFILES.get(execution_profile, _EXECUTION_PROFILES["kaggle_bulk"]))
    resolved.update(dict(config.features or {}))

    renderer_profile = detect_renderer_dependency_profile(config)
    resolved["core_renderer"] = True
    resolved["reporting"] = resolved.get("reporting", True)
    resolved["csv_batch_rendering"] = resolved.get("csv_batch_rendering", True)
    resolved["wan2gp_runtime"] = renderer_profile == "wan2gp"
    resolved["diffusers_backend"] = renderer_profile == "diffusers"
    resolved["gguf_runtime"] = resolved.get("gguf_runtime", True) and resolved["wan2gp_runtime"]
    resolved["msr_models"] = bool(config.wan2gp_msr_enabled) and resolved["wan2gp_runtime"]
    resolved["google_drive_integration"] = bool(config.enable_drive_upload) and resolved.get(
        "google_drive_integration", True
    )
    resolved["resume_engine"] = bool(config.resume_enabled) and resolved.get("resume_engine", True)
    resolved["video_stitching"] = bool(config.enable_stitching) and resolved.get(
        "video_stitching", True
    )
    return resolved


def resolve_enabled_features(config: Config) -> tuple[list[str], dict[str, str]]:
    feature_flags = _resolve_feature_overrides(config)
    enabled: list[str] = []
    disabled: dict[str, str] = {}
    visiting: set[str] = set()

    def _visit(feature_key: str) -> None:
        if feature_key in visiting or feature_key in enabled:
            return
        visiting.add(feature_key)
        spec = _FEATURE_CATALOG[feature_key]
        for dependency in spec.dependencies:
            _visit(dependency)
        visiting.remove(feature_key)
        if feature_key not in enabled:
            enabled.append(feature_key)

    for feature_key, is_enabled in feature_flags.items():
        if feature_key not in _FEATURE_CATALOG:
            disabled[feature_key] = "Unknown feature key"
            continue
        if is_enabled:
            _visit(feature_key)
        else:
            disabled[feature_key] = "Disabled by execution profile or configuration"
    return enabled, disabled


def _package_feature_map(enabled_features: list[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for feature_key in enabled_features:
        spec = _FEATURE_CATALOG[feature_key]
        for package_key in spec.package_keys:
            mapping.setdefault(package_key, [])
            if feature_key not in mapping[package_key]:
                mapping[package_key].append(feature_key)
    return mapping


def _python_version_tuple() -> tuple[int, int]:
    return sys.version_info[:2]


def _python_version_string() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _parse_requested_version(requirement: str) -> str:
    match = re.search(r"(==|>=|<=|~=|>|<)\s*([^;,\s]+)", requirement)
    return match.group(2) if match else ""


def _detect_torch_cuda_version() -> tuple[str, str]:
    try:
        import torch

        return getattr(torch, "__version__", ""), torch.version.cuda or ""
    except Exception:
        return "", ""


def _find_dependency_spec(package_name: str) -> Optional[DependencySpec]:
    normalized = _normalize_package_name(package_name)
    for spec in _DEPENDENCY_CATALOG.values():
        if _normalize_package_name(spec.package_name) == normalized:
            return spec
    return None


def _suggest_dependency_resolution(
    spec: Optional[DependencySpec],
    *,
    optional: bool,
    platform_name: str,
    kaggle_mode: bool,
    reason: str,
) -> str:
    suggestions: list[str] = []
    if "Unsupported on platform" in reason:
        suggestions.append(
            f"Use a profile compatible with {platform_name} or disable the feature that requested {spec.package_name if spec else 'this package'}."
        )
    if "Rejected by Kaggle compatibility filter" in reason:
        suggestions.append("Disable the feature for Kaggle or rerun in a local/production environment.")
    if "Experimental dependency" in reason:
        suggestions.append("Enable dependency_allow_experimental only if the workload explicitly requires it.")
    if "Development wheels are disabled" in reason:
        suggestions.append("Enable dependency_allow_development_wheels only for controlled development environments.")
    if "pip install failed" in reason:
        suggestions.append("Inspect pip stdout/stderr, verify the version is available for the active Python/CUDA runtime, and retry.")
    if kaggle_mode:
        suggestions.append("Confirm the package has Linux wheels compatible with Kaggle Python 3.11.")
    if optional:
        suggestions.append("This is optional; the launcher can continue with the feature disabled.")
    else:
        suggestions.append("This dependency is required for the selected renderer/profile and must be resolved before launch.")
    return " ".join(dict.fromkeys(suggestions))


def _build_pip_install_result(
    *,
    requirement: str,
    package_name: str,
    classification: str,
    feature_keys: list[str],
    profile_name: str,
    result: subprocess.CompletedProcess[str],
    duration_seconds: float,
    platform_name: str,
    kaggle_mode: bool,
    installed_version_before: str = "",
    reason: str = "",
) -> PipInstallResult:
    spec = _find_dependency_spec(package_name)
    installed_version_after = installed_version_before
    if result.returncode == 0:
        installed_version_after = _installed_distribution_versions().get(
            _normalize_package_name(package_name),
            installed_version_before,
        )
    _, cuda_version = _detect_torch_cuda_version()
    optional = classification != "required"
    resolved_reason = reason or ("" if result.returncode == 0 else "pip install failed")
    return PipInstallResult(
        requirement=requirement,
        package_name=package_name,
        classification=classification,
        optional=optional,
        feature_keys=list(feature_keys),
        profile_name=profile_name,
        success=result.returncode == 0,
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        reason=resolved_reason,
        requested_version=_parse_requested_version(requirement),
        installed_version_before=installed_version_before,
        installed_version_after=installed_version_after,
        platform_name=platform_name,
        python_version=_python_version_string(),
        cuda_version=cuda_version,
        duration_seconds=round(duration_seconds, 2),
        suggested_resolution=_suggest_dependency_resolution(
            spec,
            optional=optional,
            platform_name=platform_name,
            kaggle_mode=kaggle_mode,
            reason=resolved_reason,
        ),
        command=[sys.executable, "-m", "pip", "install", requirement],
    )


def _classify_spec_for_runtime(
    spec: DependencySpec,
    *,
    platform_name: str,
    kaggle_mode: bool,
    allow_experimental: bool,
    allow_development_wheels: bool = False,
) -> tuple[bool, str]:
    if platform_name not in spec.supported_platforms:
        return False, f"Unsupported on platform '{platform_name}'"
    if kaggle_mode and not spec.kaggle_supported:
        return False, "Rejected by Kaggle compatibility filter"
    if spec.classification == "experimental" and not allow_experimental:
        return False, "Experimental dependency not enabled"
    if spec.development_wheel and not allow_development_wheels:
        return False, "Development wheels are disabled"
    python_version = _python_version_tuple()
    if spec.min_python and python_version < spec.min_python:
        return False, f"Requires Python {spec.min_python[0]}.{spec.min_python[1]}+"
    if spec.max_python and python_version > spec.max_python:
        return False, f"Requires Python <= {spec.max_python[0]}.{spec.max_python[1]}"
    return True, ""


def detect_renderer_dependency_profile(config: Config) -> str:
    backend = (config.renderer_backend or "auto").strip().lower()
    if backend == "auto":
        return "wan2gp" if config.wan2gp_dir.exists() else "diffusers"
    if backend in {"diffusers", "wan2gp"}:
        return backend
    return "diffusers"


def inspect_requirement_file(requirements_path: Path) -> RequirementFileReport:
    installed_versions = _installed_distribution_versions()
    report = RequirementFileReport(path=requirements_path)
    if not requirements_path.exists():
        return report

    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement = raw_line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        if requirement.startswith(("-", "git+", "http://", "https://")):
            continue
        requirement = requirement.split(" #", 1)[0].strip()
        match = _REQUIREMENT_NAME_PATTERN.match(requirement)
        if not match:
            continue
        package_name = _normalize_package_name(match.group(1))
        version = installed_versions.get(package_name, "")
        report.inspected.append(
            RequirementStatus(
                requirement=requirement,
                package_name=package_name,
                installed=bool(version),
                version=version,
            )
        )
    return report


def inspect_dependency_profile(
    profile_name: str,
    *,
    optional_features: Optional[set[str]] = None,
    allow_experimental: bool = False,
    platform_name: Optional[str] = None,
    kaggle_mode: Optional[bool] = None,
) -> DependencyProfileReport:
    resolved_platform = platform_name or _current_platform_name()
    resolved_kaggle = _is_kaggle_runtime() if kaggle_mode is None else kaggle_mode
    installed_versions = _installed_distribution_versions()
    report = DependencyProfileReport(
        profile_name=profile_name,
        platform_name=resolved_platform,
        kaggle_mode=resolved_kaggle,
        python_version=_python_version_string(),
    )

    for spec in _build_profile_specs(profile_name, optional_features=optional_features):
        compatible, reason = _classify_spec_for_runtime(
            spec,
            platform_name=resolved_platform,
            kaggle_mode=resolved_kaggle,
            allow_experimental=allow_experimental,
        )
        if not compatible:
            report.statuses.append(
                DependencyStatus(
                    requirement=spec.requirement,
                    package_name=spec.package_name,
                    import_name=spec.import_name,
                    classification=spec.classification,
                    feature_keys=[],
                    installed=False,
                    selected=False,
                    reason=reason,
                )
            )
            report.skipped_packages.append(
                PipInstallResult(
                    requirement=spec.requirement,
                    package_name=spec.package_name,
                    classification=spec.classification,
                    optional=spec.optional,
                    feature_keys=[],
                    profile_name=profile_name,
                    success=False,
                    exit_code=0,
                    reason=reason,
                    requested_version=_parse_requested_version(spec.requirement),
                    platform_name=resolved_platform,
                    python_version=_python_version_string(),
                    cuda_version=_detect_torch_cuda_version()[1],
                    suggested_resolution=_suggest_dependency_resolution(
                        spec,
                        optional=spec.optional,
                        platform_name=resolved_platform,
                        kaggle_mode=resolved_kaggle,
                        reason=reason,
                    ),
                )
            )
            continue

        normalized_name = _normalize_package_name(spec.package_name)
        version = installed_versions.get(normalized_name, "")
        installed = bool(version) or _is_import_available(spec.import_name)
        report.selected_requirements.append(spec.requirement)
        report.statuses.append(
            DependencyStatus(
                requirement=spec.requirement,
                package_name=spec.package_name,
                import_name=spec.import_name,
                classification=spec.classification,
                feature_keys=[],
                installed=installed,
                version=version,
            )
        )
    return report


def ensure_dependency_profile(
    profile_name: str,
    *,
    optional_features: Optional[set[str]] = None,
    allow_experimental: bool = False,
    platform_name: Optional[str] = None,
    kaggle_mode: Optional[bool] = None,
) -> DependencyProfileReport:
    report = inspect_dependency_profile(
        profile_name,
        optional_features=optional_features,
        allow_experimental=allow_experimental,
        platform_name=platform_name,
        kaggle_mode=kaggle_mode,
    )

    for skipped in list(report.skipped_packages):
        if skipped.classification not in {"optional", "experimental"}:
            report.failed_required.append(skipped)

    if report.failed_required:
        return report

    for status in report.statuses:
        if not status.selected or status.installed:
            continue
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", status.requirement],
            capture_output=True,
            text=True,
        )
        pip_result = _build_pip_install_result(
            requirement=status.requirement,
            package_name=status.package_name,
            classification=status.classification,
            feature_keys=list(status.feature_keys),
            profile_name=profile_name,
            result=result,
            duration_seconds=time.monotonic() - started,
            platform_name=report.platform_name,
            kaggle_mode=report.kaggle_mode,
            installed_version_before=status.version,
        )
        if pip_result.success:
            report.installed_packages.append(pip_result)
            continue
        if pip_result.optional:
            report.failed_optional.append(pip_result)
            continue
        report.failed_required.append(pip_result)
        break

    refreshed = inspect_dependency_profile(
        profile_name,
        optional_features=optional_features,
        allow_experimental=allow_experimental,
        platform_name=platform_name,
        kaggle_mode=kaggle_mode,
    )
    report.statuses = refreshed.statuses
    report.selected_requirements = refreshed.selected_requirements
    report.skipped_packages = refreshed.skipped_packages
    return report


def _feature_required(feature_key: str) -> bool:
    return _FEATURE_CATALOG[feature_key].failure_policy == "abort"


def _module_check(
    feature_key: str,
    module_name: str,
    *,
    display_name: Optional[str] = None,
) -> DependencyVerificationCheck:
    try:
        importlib.import_module(module_name)
        return DependencyVerificationCheck(
            name=display_name or module_name,
            success=True,
            details=f"{module_name} import succeeded",
            feature_key=feature_key,
        )
    except Exception as exc:
        return DependencyVerificationCheck(
            name=display_name or module_name,
            success=False,
            details=str(exc),
            feature_key=feature_key,
        )


def inspect_feature_dependency_profile(config: Config) -> DependencyProfileReport:
    execution_profile = detect_execution_profile(config)
    platform_name = _current_platform_name()
    kaggle_mode = _is_kaggle_runtime()
    renderer_backend = detect_renderer_dependency_profile(config)
    enabled_features, disabled_features = resolve_enabled_features(config)
    installed_versions = _installed_distribution_versions()
    package_features = _package_feature_map(enabled_features)

    report = DependencyProfileReport(
        profile_name=execution_profile,
        platform_name=platform_name,
        kaggle_mode=kaggle_mode,
        renderer_backend=renderer_backend,
        enabled_features=enabled_features,
        disabled_features=disabled_features,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    for package_key, feature_keys in package_features.items():
        spec = _DEPENDENCY_CATALOG[package_key]
        compatible, reason = _classify_spec_for_runtime(
            spec,
            platform_name=platform_name,
            kaggle_mode=kaggle_mode,
            allow_experimental=config.dependency_allow_experimental,
            allow_development_wheels=config.dependency_allow_development_wheels,
        )
        required = any(_feature_required(feature_key) for feature_key in feature_keys)
        effective_classification = "required" if required else spec.classification
        if not compatible:
            report.statuses.append(
                DependencyStatus(
                    requirement=spec.requirement,
                    package_name=spec.package_name,
                    import_name=spec.import_name,
                    classification=effective_classification,
                    feature_keys=list(feature_keys),
                    installed=False,
                    selected=False,
                    reason=reason,
                )
            )
            report.skipped_packages.append(
                PipInstallResult(
                    requirement=spec.requirement,
                    package_name=spec.package_name,
                    classification=effective_classification,
                    optional=not required,
                    feature_keys=list(feature_keys),
                    profile_name=execution_profile,
                    success=False,
                    exit_code=0,
                    reason=reason,
                    requested_version=_parse_requested_version(spec.requirement),
                    platform_name=platform_name,
                    python_version=_python_version_string(),
                    cuda_version=_detect_torch_cuda_version()[1],
                    suggested_resolution=_suggest_dependency_resolution(
                        spec,
                        optional=not required,
                        platform_name=platform_name,
                        kaggle_mode=kaggle_mode,
                        reason=reason,
                    ),
                )
            )
            continue

        normalized_name = _normalize_package_name(spec.package_name)
        version = installed_versions.get(normalized_name, "")
        installed = bool(version) or _is_import_available(spec.import_name)
        report.selected_requirements.append(spec.requirement)
        report.statuses.append(
            DependencyStatus(
                requirement=spec.requirement,
                package_name=spec.package_name,
                import_name=spec.import_name,
                classification=effective_classification,
                feature_keys=list(feature_keys),
                installed=installed,
                version=version,
            )
        )

    for feature_key in enabled_features:
        spec = _FEATURE_CATALOG[feature_key]
        report.feature_statuses.append(
            FeatureRuntimeStatus(
                feature_key=feature_key,
                enabled=True,
                required=spec.failure_policy == "abort",
                state="PENDING",
                package_keys=list(spec.package_keys),
                dependencies=list(spec.dependencies),
            )
        )
    for feature_key, reason in disabled_features.items():
        if feature_key not in _FEATURE_CATALOG:
            continue
        spec = _FEATURE_CATALOG[feature_key]
        report.feature_statuses.append(
            FeatureRuntimeStatus(
                feature_key=feature_key,
                enabled=False,
                required=spec.failure_policy == "abort",
                state="DISABLED",
                reason=reason,
                package_keys=list(spec.package_keys),
                dependencies=list(spec.dependencies),
            )
        )
    return report


def _apply_feature_runtime_statuses(report: DependencyProfileReport) -> None:
    status_map: dict[str, FeatureRuntimeStatus] = {
        status.feature_key: status for status in report.feature_statuses
    }
    checks_by_feature: dict[str, list[DependencyVerificationCheck]] = {}
    for check in report.verification_checks:
        checks_by_feature.setdefault(check.feature_key, []).append(check)
    package_failures: dict[str, list[str]] = {}
    for item in report.failed_required + report.failed_optional + report.skipped_packages:
        for feature_key in item.feature_keys:
            package_failures.setdefault(feature_key, []).append(
                item.reason or f"{item.package_name} installation failed"
            )

    for feature_key in report.enabled_features:
        feature_status = status_map.get(feature_key)
        if feature_status is None:
            continue
        feature_checks = checks_by_feature.get(feature_key, [])
        failures = list(package_failures.get(feature_key, []))
        failures.extend(check.details for check in feature_checks if not check.success)
        if failures:
            feature_status.state = "FAILED" if feature_status.required else "DISABLED"
            feature_status.reason = "; ".join(dict.fromkeys(failures))
        else:
            feature_status.state = "PASS"
            feature_status.reason = "Feature runtime verified"


def verify_runtime_dependencies(config: Config) -> DependencyProfileReport:
    report = inspect_feature_dependency_profile(config)

    torch_version = ""
    cuda_version = ""
    try:
        import torch

        torch_version = getattr(torch, "__version__", "")
        cuda_version = torch.version.cuda or ""
    except Exception:
        pass
    report.torch_version = torch_version
    report.cuda_version = cuda_version

    checks: list[DependencyVerificationCheck] = [
        DependencyVerificationCheck(
            name="python",
            success=sys.version_info >= (3, 11),
            details=_python_version_string(),
            feature_key="core_renderer",
        ),
        DependencyVerificationCheck(
            name="ffmpeg",
            success=shutil.which("ffmpeg") is not None,
            details=shutil.which("ffmpeg") or "ffmpeg not on PATH",
            feature_key="core_renderer",
        )
    ]

    feature_modules: dict[str, tuple[tuple[str, str], ...]] = {
        "core_renderer": (("imageio", "imageio"), ("torch", "torch")),
        "wan2gp_runtime": (
            ("transformers", "transformers"),
            ("accelerate", "accelerate"),
            ("mmgp", "mmgp"),
        ),
        "diffusers_backend": (
            ("diffusers", "diffusers"),
            ("transformers", "transformers"),
            ("accelerate", "accelerate"),
        ),
        "gguf_runtime": (("gguf", "gguf"),),
        "google_drive_integration": (
            ("googleapiclient", "googleapiclient"),
            ("google.auth", "google.auth"),
        ),
        "gradio_ui": (("gradio", "gradio"),),
        "audio_processing": (
            ("torchaudio", "torchaudio"),
            ("librosa", "librosa"),
            ("soundfile", "soundfile"),
        ),
        "whisper": (("whisper", "whisper"),),
        "speech_recognition": (
            ("speechbrain", "speechbrain"),
            ("pyannote.audio", "pyannote.audio"),
        ),
        "face_restoration": (("gfpgan", "gfpgan"), ("realesrgan", "realesrgan")),
        "background_removal": (("rembg", "rembg"),),
        "image_editing": (("cv2", "cv2"),),
        "development_tools": (("ruff", "ruff"), ("mypy", "mypy")),
        "testing_tools": (("pytest", "pytest"), ("coverage", "coverage")),
    }

    for feature_key in report.enabled_features:
        if feature_key == "core_renderer":
            try:
                import torch

                if config.use_cuda:
                    checks.append(
                        DependencyVerificationCheck(
                            name="cuda",
                            success=bool(torch.cuda.is_available()),
                            details=torch.version.cuda or "CUDA unavailable",
                            feature_key="core_renderer",
                        )
                    )
                    checks.append(
                        DependencyVerificationCheck(
                            name="gpu",
                            success=bool(torch.cuda.is_available()),
                            details=torch.cuda.get_device_name(0)
                            if torch.cuda.is_available()
                            else "GPU unavailable",
                            feature_key="core_renderer",
                        )
                    )
                else:
                    checks.append(
                        DependencyVerificationCheck(
                            name="cuda",
                            success=True,
                            details="CUDA not required by configuration",
                            feature_key="core_renderer",
                        )
                    )
                    checks.append(
                        DependencyVerificationCheck(
                            name="gpu",
                            success=True,
                            details="GPU not required by configuration",
                            feature_key="core_renderer",
                        )
                    )
            except Exception as exc:
                checks.append(
                    DependencyVerificationCheck(
                        name="cuda",
                        success=not config.use_cuda,
                        details=str(exc),
                        feature_key="core_renderer",
                    )
                )
                checks.append(
                    DependencyVerificationCheck(
                        name="gpu",
                        success=not config.use_cuda,
                        details=str(exc),
                        feature_key="core_renderer",
                    )
                )
            checks.append(
                _module_check(
                    "core_renderer",
                    "renderers.factory",
                    display_name="renderer_initialization",
                )
            )
        if feature_key == "wan2gp_runtime":
            checks.append(
                DependencyVerificationCheck(
                    name="wan2gp",
                    success=config.wan2gp_dir.exists()
                    and any(
                        (config.wan2gp_dir / relative_path).exists()
                        for relative_path in ("wan", "wan_generate_video.py", "ltx_video", "requirements.txt")
                    ),
                    details=str(config.wan2gp_dir.resolve(strict=False)),
                    feature_key=feature_key,
                )
            )
        if feature_key == "msr_models":
            checks.append(
                DependencyVerificationCheck(
                    name="msr_models",
                    success=bool(config.wan2gp_msr_enabled),
                    details="MSR feature enabled by configuration"
                    if config.wan2gp_msr_enabled
                    else "MSR feature disabled by configuration",
                    feature_key=feature_key,
                )
            )
        if feature_key == "reporting":
            checks.append(
                DependencyVerificationCheck(
                    name="reporting",
                    success=True,
                    details="Repository reporting is built-in",
                    feature_key=feature_key,
                )
            )
        if feature_key == "csv_batch_rendering":
            checks.append(
                DependencyVerificationCheck(
                    name="csv_batch_rendering",
                    success=True,
                    details="CSV batch rendering is repository-native",
                    feature_key=feature_key,
                )
            )
        if feature_key == "resume_engine":
            checks.append(
                DependencyVerificationCheck(
                    name="resume_engine",
                    success=True,
                    details="Resume engine is repository-native",
                    feature_key=feature_key,
                )
            )
        if feature_key == "video_stitching":
            checks.append(
                DependencyVerificationCheck(
                    name="video_stitching",
                    success=shutil.which("ffmpeg") is not None,
                    details=shutil.which("ffmpeg") or "ffmpeg not on PATH",
                    feature_key=feature_key,
                )
            )
        for module_name, display_name in feature_modules.get(feature_key, ()):
            checks.append(_module_check(feature_key, module_name, display_name=display_name))

    report.verification_checks = checks
    _apply_feature_runtime_statuses(report)
    return report


def ensure_runtime_dependency_profile(config: Config) -> DependencyProfileReport:
    report = inspect_feature_dependency_profile(config)
    for skipped in list(report.skipped_packages):
        if not skipped.optional:
            report.failed_required.append(skipped)

    if not report.failed_required:
        for status in report.statuses:
            if not status.selected or status.installed:
                continue
            started = time.monotonic()
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", status.requirement],
                capture_output=True,
                text=True,
            )
            pip_result = _build_pip_install_result(
                requirement=status.requirement,
                package_name=status.package_name,
                classification=status.classification,
                feature_keys=list(status.feature_keys),
                profile_name=report.profile_name,
                result=result,
                duration_seconds=time.monotonic() - started,
                platform_name=report.platform_name,
                kaggle_mode=report.kaggle_mode,
                installed_version_before=status.version,
            )
            if pip_result.success:
                report.installed_packages.append(pip_result)
                continue
            if pip_result.optional:
                report.failed_optional.append(pip_result)
                continue
            report.failed_required.append(pip_result)
            break

    verification = verify_runtime_dependencies(config)
    report.statuses = verification.statuses
    report.selected_requirements = verification.selected_requirements
    report.verification_checks = verification.verification_checks
    report.feature_statuses = verification.feature_statuses
    report.enabled_features = verification.enabled_features
    report.disabled_features = verification.disabled_features
    report.renderer_backend = verification.renderer_backend
    report.python_version = verification.python_version
    report.torch_version = verification.torch_version
    report.cuda_version = verification.cuda_version
    report.skipped_packages = verification.skipped_packages
    return report


def ensure_git_checkout(
    destination: Path,
    repo_url: str,
    ref: str = "main",
    *,
    source: str = "github",
    update_policy: str = "auto",
) -> GitCheckoutReport:
    destination = destination.resolve(strict=False)
    repo_url = _sanitize_git_text(repo_url, field_name="repo_url")
    ref = _sanitize_git_text(ref, field_name="ref")
    source = _normalize_repository_source(source)
    update_policy = _normalize_update_policy(update_policy)
    report = GitCheckoutReport(
        destination=destination,
        repo_url=repo_url,
        ref=ref,
        target_ref=ref,
        cloned=False,
        updated=False,
        source=source,
        update_policy=update_policy,
    )

    def use_local_copy(reason: str, state: str) -> GitCheckoutReport:
        report.used_local_copy = True
        report.local_copy_reason = reason
        report.repository_state = state
        report.validation_messages.append(reason)
        return report

    def reclone(reason: str) -> bool:
        report.recovery_actions.append(reason)
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        clone_result = _git_command(
            report,
            ["git", "clone", "--depth", "1", "--branch", report.target_ref, repo_url, str(destination)],
            note="clone repository",
            timeout_seconds=60,
        )
        if not clone_result.success:
            report.failure_reason = (
                f"git clone failed (exit_code={clone_result.exit_code})\n"
                f"stdout:\n{clone_result.stdout or '<empty>'}\n"
                f"stderr:\n{clone_result.stderr or '<empty>'}"
            )
            return False
        report.cloned = True
        report.repository_state = "cloned"
        report.remote_name = "origin"
        report.remote_url = repo_url
        report.remote_reachable = True
        report.current_branch = report.target_ref
        report.default_branch = report.target_ref
        report.working_tree = "clean"
        return True

    if source == "dataset":
        if _path_looks_like_source_tree(destination):
            return use_local_copy(
                "Using dataset-provided repository source tree without Git operations.",
                "dataset_source",
            )
        report.failure_reason = (
            f"Repository source is configured as dataset, but no valid source tree exists at {destination}."
        )
        return report

    if update_policy == "force" and destination.exists():
        report.recovery_actions.append("Force policy requested a fresh clone.")
        if not reclone("Removing existing checkout before fresh clone."):
            return report
        return report

    if not destination.exists():
        if not reclone("Repository directory was missing."):
            return report
        return report

    if not (destination / ".git").exists():
        if _path_looks_like_source_tree(destination):
            return use_local_copy(
                "Destination contains a source tree without .git metadata; treating it as a dataset/local copy.",
                "local_source_tree",
            )
        if not reclone("Destination existed but was not a Git checkout."):
            return report
        return report

    git_dir_check = _git_command(
        report,
        ["git", "-C", str(destination), "rev-parse", "--git-dir"],
        note="validate git metadata",
    )
    if not git_dir_check.success:
        if not reclone("Git metadata was invalid or corrupted; recreating checkout."):
            return report
        return report

    report.repository_state = "git_checkout"
    branch_result = _git_command(
        report,
        ["git", "-C", str(destination), "branch", "--show-current"],
        note="detect current branch",
    )
    report.current_branch = _first_nonempty_line(branch_result.stdout) if branch_result.success else ""
    if not report.current_branch:
        report.validation_messages.append("Checkout is currently detached.")

    status_result = _git_command(
        report,
        ["git", "-C", str(destination), "status", "--short", "--branch"],
        note="inspect working tree",
    )
    report.working_tree = (
        _describe_working_tree(status_result.stdout)
        if status_result.success
        else "unknown"
    )

    remote_result = _git_command(
        report,
        ["git", "-C", str(destination), "remote", "get-url", "origin"],
        note="read origin remote",
    )
    if remote_result.success:
        report.remote_url = _first_nonempty_line(remote_result.stdout)
    else:
        report.validation_messages.append("Origin remote missing.")
        add_remote = _git_command(
            report,
            ["git", "-C", str(destination), "remote", "add", "origin", repo_url],
            note="repair missing origin",
        )
        if add_remote.success:
            report.recovery_actions.append("Recreated missing origin remote.")
            report.remote_url = repo_url
        else:
            set_remote = _git_command(
                report,
                ["git", "-C", str(destination), "remote", "set-url", "origin", repo_url],
                note="repair origin url",
            )
            if set_remote.success:
                report.recovery_actions.append("Repaired origin remote URL.")
                report.remote_url = repo_url

    if report.remote_url and report.remote_url != repo_url:
        set_url_result = _git_command(
            report,
            ["git", "-C", str(destination), "remote", "set-url", "origin", repo_url],
            note="replace incorrect origin url",
        )
        if set_url_result.success:
            report.recovery_actions.append("Replaced incorrect origin remote URL.")
            report.remote_url = repo_url

    ls_remote_result = _git_command(
        report,
        ["git", "-C", str(destination), "ls-remote", "--symref", "origin", "HEAD"],
        note="check remote reachability",
        timeout_seconds=15,
    )
    report.remote_reachable = ls_remote_result.success
    if ls_remote_result.success:
        report.default_branch = _parse_default_branch(ls_remote_result.stdout) or ref
    else:
        report.validation_messages.append(
            "Remote repository is not reachable; offline recovery mode enabled."
        )

    if update_policy == "never":
        if _path_looks_like_source_tree(destination):
            return use_local_copy(
                "Repository update policy is set to never; using the local checkout as-is.",
                "local_checkout_policy_never",
            )
        report.failure_reason = f"Repository update policy is 'never' but no usable source tree exists at {destination}."
        return report

    if not report.remote_reachable:
        if _path_looks_like_source_tree(destination):
            return use_local_copy(
                "Remote is unreachable, but a local repository copy is available.",
                "offline_local_checkout",
            )
        report.failure_reason = (
            "Remote repository is unreachable and no valid local source tree is available."
        )
        return report

    remote_branches_result = _git_command(
        report,
        ["git", "-C", str(destination), "branch", "-r"],
        note="list remote branches",
    )
    remote_branches = (
        _remote_branch_names(remote_branches_result.stdout)
        if remote_branches_result.success
        else []
    )
    if remote_branches and ref not in remote_branches:
        fallback_branch = report.default_branch or (remote_branches[0] if remote_branches else ref)
        report.validation_messages.append(
            f"Configured branch '{ref}' was not available remotely; falling back to '{fallback_branch}'."
        )
        report.recovery_actions.append(
            f"Selected fallback branch '{fallback_branch}' because '{ref}' was unavailable."
        )
        report.target_ref = fallback_branch
    else:
        report.target_ref = ref

    fetch_result = _git_command(
        report,
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", report.target_ref],
        note="fetch target branch",
        timeout_seconds=60,
    )
    if not fetch_result.success:
        if _path_looks_like_source_tree(destination):
            return use_local_copy(
                "Fetch failed, but a valid local source tree is available.",
                "local_source_after_fetch_failure",
            )
        if not reclone("Fetch failed; attempting fresh clone recovery."):
            return report
        return report

    checkout_result = _git_command(
        report,
        ["git", "-C", str(destination), "checkout", "-B", report.target_ref, f"origin/{report.target_ref}"],
        note="checkout target branch",
        timeout_seconds=30,
    )
    if not checkout_result.success:
        if not reclone("Checkout failed; attempting fresh clone recovery."):
            return report
        return report

    report.updated = True
    report.current_branch = report.target_ref
    report.default_branch = report.default_branch or report.target_ref
    report.remote_url = repo_url
    report.repository_state = "healthy"
    report.working_tree = "clean"
    return report


def ensure_wan2gp_runtime(config: Config) -> GitCheckoutReport:
    return ensure_git_checkout(
        destination=config.wan2gp_dir,
        repo_url=config.wan2gp_repo_url,
        ref=config.wan2gp_repo_ref,
    )


def ensure_wan2gp_model_assets(config: Config, *, drive_client: Optional[object] = None) -> dict[str, Any]:
    from assets import AssetManager

    manager = AssetManager(config, drive_client=drive_client)
    report = manager.ensure_assets(backend="wan2gp")
    return report.to_dict()
