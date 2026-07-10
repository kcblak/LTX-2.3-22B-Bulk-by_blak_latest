from __future__ import annotations

import csv
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from config import Config, load_config
from core import APP_NAME, APP_VERSION
from orchestration.runner import ApplicationRunResult, ApplicationRunner, PreparationResult
from orchestration.runtime_assets import (
    detect_execution_profile,
    detect_renderer_dependency_profile,
    ensure_dependency_profile,
    ensure_model_registry,
    ensure_runtime_dependency_profile,
    ensure_wan2gp_model_assets,
    ensure_wan2gp_runtime,
    inspect_dependency_profile,
    verify_runtime_dependencies,
)
import bootstrap as _bootstrap_module
collect_environment_info = _bootstrap_module.collect_environment_info
CRITICAL_REPOSITORY_FILES = [
    Path("main.py"),
    Path("bootstrap.py"),
    Path("config/default.yaml"),
    Path("config/loader.py"),
    Path("engine/pipeline.py"),
    Path("renderers/base.py"),
    Path("renderers/factory.py"),
    Path("reports/report_generator.py"),
    Path("validation/validators.py"),
]
SOURCE_ROOT_MARKERS = [
    Path("main.py"),
    Path("config"),
    Path("orchestration"),
]
OPTIONAL_REPOSITORY_FILES = [
    Path("drive/gdrive.py"),
    Path("drive/sync_engine.py"),
    Path("stitching/service.py"),
    Path("stitching/ffmpeg_wrapper.py"),
]
DRIVE_SECRET_NAMES = [
    "LTX_DRIVE_CREDENTIALS_JSON",
    "GOOGLE_DRIVE_CREDENTIALS_JSON",
    "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    "gdrive_credentials_json",
    "google_drive_credentials_json",
]
DRIVE_ENV_JSON_KEYS = [
    "LTX_DRIVE_CREDENTIALS_JSON",
    "GOOGLE_DRIVE_CREDENTIALS_JSON",
    "GOOGLE_APPLICATION_CREDENTIALS_JSON",
]
DRIVE_ENV_PATH_KEYS = [
    "LTX_DRIVE_CREDENTIALS_PATH",
    "GOOGLE_DRIVE_CREDENTIALS_PATH",
    "GOOGLE_APPLICATION_CREDENTIALS",
]
FINAL_SUCCESS_STATUSES = {"COMPLETED", "UPLOADED"}
FINAL_FAILURE_STATUSES = {
    "FAILED_VALIDATION",
    "FAILED_RENDER",
    "FAILED_UPLOAD",
    "FAILED_VERIFY",
}


@dataclass
class PackageInspection:
    module_name: str
    package_name: str
    installed: bool
    version: str = ""


@dataclass
class DependencyInspection:
    packages: list[PackageInspection]
    missing_packages: list[str]
    ffmpeg_path: Optional[str]
    ffmpeg_version: str
    cuda_version: str
    cuda_available: bool
    gpu_name: str


@dataclass
class RepositoryValidation:
    repo_root: Path
    critical_missing: list[str]
    optional_missing: list[str]

    @property
    def ready(self) -> bool:
        return not self.critical_missing


@dataclass
class ProjectDiscovery:
    input_root: Path
    dataset_roots: list[Path]
    jobs_csv_path: Path
    reference_images_dir: Path
    project_config_path: Optional[Path]
    manifest_seed_path: Optional[Path]
    cache_seed_path: Optional[Path]
    preset_paths: list[Path]
    project_name: str
    image_match_count: int
    referenced_image_count: int = 0
    missing_image_refs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def missing_image_count(self) -> int:
        return len(self.missing_image_refs)


@dataclass
class DriveCredentialDiscovery:
    enabled: bool
    source: str
    raw_json: Optional[str]
    path: Optional[Path]
    notes: list[str] = field(default_factory=list)


@dataclass
class ResumeSummary:
    manifest_path: Optional[Path]
    cache_index_path: Optional[Path]
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    remaining_jobs: int
    skipped_jobs: int
    cache_entries: int
    notes: list[str] = field(default_factory=list)


@dataclass
class NotebookLaunchContext:
    repo_validation: RepositoryValidation
    dependency_inspection: DependencyInspection
    discovery: ProjectDiscovery
    drive_credentials: DriveCredentialDiscovery
    config: Config
    environment_info: dict[str, Any]
    resume_summary: ResumeSummary
    source_root: Path
    runtime_preparation: dict[str, Any] = field(default_factory=dict)
    preparation: Optional[PreparationResult] = None


@dataclass
class BootstrapSection:
    name: str
    status: str
    message: str
    details: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": list(self.details),
            "suggestions": list(self.suggestions),
        }


@dataclass
class StageExecutionReport:
    stage_name: str
    status: str
    started_at: str
    completed_at: str
    duration_seconds: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    error_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "details": dict(self.details),
            "suggestions": list(self.suggestions),
            "error_type": self.error_type,
        }


@dataclass
class BootstrapReport:
    stages: list[StageExecutionReport] = field(default_factory=list)
    sections: dict[str, BootstrapSection] = field(default_factory=dict)
    ready_to_launch: bool = False
    summary_lines: list[str] = field(default_factory=list)

    def stage_by_name(self) -> dict[str, StageExecutionReport]:
        return {stage.stage_name: stage for stage in self.stages}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_to_launch": self.ready_to_launch,
            "summary_lines": list(self.summary_lines),
            "stages": [stage.to_dict() for stage in self.stages],
            "sections": {
                section_name: section.to_dict()
                for section_name, section in self.sections.items()
            },
        }


@dataclass
class LauncherExecutionState:
    context: Optional[NotebookLaunchContext] = None
    runtime_preparation: dict[str, Any] = field(default_factory=dict)
    preparation: Optional[PreparationResult] = None
    result: Optional[ApplicationRunResult] = None
    bootstrap_report: BootstrapReport = field(default_factory=BootstrapReport)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": None
            if self.context is None
            else {
                "repo_validation": {
                    "ready": self.context.repo_validation.ready,
                    "critical_missing": list(self.context.repo_validation.critical_missing),
                    "optional_missing": list(self.context.repo_validation.optional_missing),
                },
                "discovery": {
                    "jobs_csv_path": str(self.context.discovery.jobs_csv_path),
                    "reference_images_dir": str(self.context.discovery.reference_images_dir),
                    "project_config_path": (
                        str(self.context.discovery.project_config_path)
                        if self.context.discovery.project_config_path
                        else None
                    ),
                    "preset_paths": [str(path) for path in self.context.discovery.preset_paths],
                    "referenced_image_count": self.context.discovery.referenced_image_count,
                    "image_match_count": self.context.discovery.image_match_count,
                    "missing_image_refs": list(self.context.discovery.missing_image_refs),
                    "notes": list(self.context.discovery.notes),
                },
                "drive_credentials": {
                    "enabled": self.context.drive_credentials.enabled,
                    "source": self.context.drive_credentials.source,
                    "notes": list(self.context.drive_credentials.notes),
                },
                "resume_summary": {
                    "manifest_path": (
                        str(self.context.resume_summary.manifest_path)
                        if self.context.resume_summary.manifest_path
                        else None
                    ),
                    "cache_index_path": (
                        str(self.context.resume_summary.cache_index_path)
                        if self.context.resume_summary.cache_index_path
                        else None
                    ),
                    "completed_jobs": self.context.resume_summary.completed_jobs,
                    "remaining_jobs": self.context.resume_summary.remaining_jobs,
                    "skipped_jobs": self.context.resume_summary.skipped_jobs,
                    "cache_entries": self.context.resume_summary.cache_entries,
                    "notes": list(self.context.resume_summary.notes),
                },
                "source_root": str(self.context.source_root),
            },
            "runtime_preparation": dict(self.runtime_preparation),
            "preparation": None
            if self.preparation is None
            else {
                "ready": self.preparation.ready,
                "diagnostics": dict(self.preparation.diagnostics),
                "preflight": dict(self.preparation.preflight),
                "started_at": self.preparation.started_at,
                "completed_at": self.preparation.completed_at,
            },
            "result": None if self.result is None else self.result.to_dict(),
            "bootstrap_report": self.bootstrap_report.to_dict(),
        }


def _safe_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return ""


def _format_bytes(num_bytes: Optional[float]) -> str:
    if num_bytes is None:
        return "N/A"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    total_seconds = int(max(0, seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "ltx-project"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON file {path}: {exc}") from exc


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse CSV file {path}: {exc}") from exc


def _serialize_pip_result(result: Any) -> dict[str, Any]:
    return {
        "package": result.package_name,
        "requirement": result.requirement,
        "requested_version": getattr(result, "requested_version", ""),
        "installed_version_before": getattr(result, "installed_version_before", ""),
        "installed_version_after": getattr(result, "installed_version_after", ""),
        "classification": result.classification,
        "optional": result.optional,
        "feature_keys": list(getattr(result, "feature_keys", [])),
        "profile_name": result.profile_name,
        "platform": getattr(result, "platform_name", ""),
        "python_version": getattr(result, "python_version", ""),
        "cuda_version": getattr(result, "cuda_version", ""),
        "duration_seconds": getattr(result, "duration_seconds", 0.0),
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "reason": result.reason,
        "suggested_resolution": getattr(result, "suggested_resolution", ""),
        "command": list(getattr(result, "command", [])),
    }


def inspect_runtime_dependencies(config: Optional[Config] = None) -> DependencyInspection:
    report = (
        verify_runtime_dependencies(config)
        if config is not None
        else inspect_dependency_profile("bootstrap")
    )
    packages: list[PackageInspection] = []
    missing_packages = list(report.missing_required)
    for status in report.statuses:
        if not status.selected:
            continue
        version = _safe_version(status.package_name) if status.installed else ""
        packages.append(
            PackageInspection(
                module_name=status.import_name or status.package_name,
                package_name=status.package_name,
                installed=status.installed,
                version=version,
            )
        )

    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_version = ""
    if ffmpeg_path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            )
            ffmpeg_version = result.stdout.splitlines()[0].strip()
        except Exception:
            ffmpeg_version = "Unavailable"

    cuda_version = "Unavailable"
    cuda_available = False
    gpu_name = "Unavailable"
    try:
        import torch

        cuda_version = torch.version.cuda or "Unavailable"
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_properties(0).name
    except Exception:
        pass

    return DependencyInspection(
        packages=packages,
        missing_packages=missing_packages,
        ffmpeg_path=ffmpeg_path,
        ffmpeg_version=ffmpeg_version,
        cuda_version=cuda_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
    )


def ensure_notebook_dependencies() -> DependencyInspection:
    report = ensure_dependency_profile("bootstrap")
    if report.failed_required:
        failure = report.failed_required[0]
        raise RuntimeError(
            f"Bootstrap dependency install failed for {failure.package_name} "
            f"(requested={getattr(failure, 'requested_version', 'n/a') or 'n/a'}, "
            f"installed_before={getattr(failure, 'installed_version_before', 'n/a') or 'n/a'}, "
            f"python={getattr(failure, 'python_version', 'n/a') or 'n/a'}, "
            f"cuda={getattr(failure, 'cuda_version', 'n/a') or 'n/a'}, "
            f"exit_code={failure.exit_code}).\n"
            f"stdout:\n{failure.stdout or '<empty>'}\n"
            f"stderr:\n{failure.stderr or '<empty>'}\n"
            f"suggested_resolution:\n{getattr(failure, 'suggested_resolution', 'Review pip diagnostics and retry.')}"
        )
    return inspect_runtime_dependencies()


def detect_source_root(repo_root: Path) -> Path:
    candidates = [repo_root, repo_root / "src"]
    for candidate in candidates:
        if all((candidate / marker).exists() for marker in SOURCE_ROOT_MARKERS):
            return candidate
    return repo_root


def validate_repository_layout(repo_root: Path) -> RepositoryValidation:
    source_root = detect_source_root(repo_root)
    critical_missing = [
        str(relative_path)
        for relative_path in CRITICAL_REPOSITORY_FILES
        if not (source_root / relative_path).exists()
    ]
    optional_missing = [
        str(relative_path)
        for relative_path in OPTIONAL_REPOSITORY_FILES
        if not (source_root / relative_path).exists()
    ]
    return RepositoryValidation(
        repo_root=repo_root,
        critical_missing=critical_missing,
        optional_missing=optional_missing,
    )


def _candidate_files(root: Path, *, file_name: str) -> list[Path]:
    if not root.exists():
        return []
    matches: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() == file_name.lower():
            matches.append(path)
    return sorted(matches)


def _find_nearest_config(jobs_csv_path: Path, input_root: Path) -> Optional[Path]:
    current = jobs_csv_path.parent
    while True:
        for filename in ("project.yaml", "project.yml"):
            candidate = current / filename
            if candidate.exists():
                return candidate
        if current == input_root or current.parent == current:
            break
        current = current.parent
    return None


def _find_nearest_presets(jobs_csv_path: Path, input_root: Path) -> list[Path]:
    presets: list[Path] = []
    current = jobs_csv_path.parent
    while True:
        for pattern in ("preset*.yaml", "preset*.yml", "preset*.json"):
            presets.extend(sorted(current.glob(pattern)))
        if current == input_root or current.parent == current:
            break
        current = current.parent
    deduped: dict[str, Path] = {str(path): path for path in presets}
    return list(deduped.values())


def _score_reference_root(root: Path, image_refs: list[str]) -> int:
    score = 0
    for ref in image_refs:
        if not ref:
            continue
        if (root / ref).exists():
            score += 1
    return score


def _discover_reference_root(
    jobs_csv_path: Path,
    input_root: Path,
) -> tuple[Path, int, list[str], int]:
    rows = _read_csv_rows(jobs_csv_path)
    image_refs: list[str] = []
    for row in rows:
        for key in ("start_image", "end_image"):
            value = (row.get(key) or "").strip()
            if value:
                image_refs.append(value)
    image_refs = list(dict.fromkeys(image_refs))

    if not image_refs:
        return jobs_csv_path.parent, 0, [], 0

    candidate_roots: list[Path] = []
    current = jobs_csv_path.parent
    while True:
        candidate_roots.append(current)
        if current == input_root or current.parent == current:
            break
        current = current.parent

    best_root = jobs_csv_path.parent
    best_score = -1
    for candidate_root in candidate_roots:
        score = _score_reference_root(candidate_root, image_refs)
        if score > best_score:
            best_root = candidate_root
            best_score = score
    missing_refs = sorted(
        ref for ref in image_refs if ref and not (best_root / ref).exists()
    )
    return best_root, max(0, best_score), missing_refs, len(image_refs)


def discover_kaggle_project(input_root: Path = Path("/kaggle/input")) -> ProjectDiscovery:
    if not input_root.exists():
        raise FileNotFoundError(f"Kaggle input root not found: {input_root}")

    dataset_roots = sorted([path for path in input_root.iterdir() if path.is_dir()])
    jobs_candidates = _candidate_files(input_root, file_name="jobs.csv")
    if not jobs_candidates:
        raise FileNotFoundError("No jobs.csv file was found under /kaggle/input")

    best_candidate: Optional[ProjectDiscovery] = None
    candidate_errors: list[str] = []
    for jobs_csv_path in jobs_candidates:
        try:
            reference_root, image_match_count, missing_image_refs, referenced_image_count = _discover_reference_root(
                jobs_csv_path,
                input_root,
            )
            project_config_path = _find_nearest_config(jobs_csv_path, input_root)
            preset_paths = _find_nearest_presets(jobs_csv_path, input_root)
            current = jobs_csv_path.parent
            manifest_seed_path: Optional[Path] = None
            cache_seed_path: Optional[Path] = None
            while True:
                manifest_candidate = current / "manifest.json"
                cache_candidate = current / "render_cache.json"
                if manifest_seed_path is None and manifest_candidate.exists():
                    manifest_seed_path = manifest_candidate
                if cache_seed_path is None and cache_candidate.exists():
                    cache_seed_path = cache_candidate
                outputs_manifest = current / "outputs" / "manifest.json"
                outputs_cache = current / "outputs" / "cache" / "render_cache.json"
                if manifest_seed_path is None and outputs_manifest.exists():
                    manifest_seed_path = outputs_manifest
                if cache_seed_path is None and outputs_cache.exists():
                    cache_seed_path = outputs_cache
                if current == input_root or current.parent == current:
                    break
                current = current.parent

            dataset_name = jobs_csv_path.relative_to(input_root).parts[0]
            candidate = ProjectDiscovery(
                input_root=input_root,
                dataset_roots=dataset_roots,
                jobs_csv_path=jobs_csv_path,
                reference_images_dir=reference_root,
                project_config_path=project_config_path,
                manifest_seed_path=manifest_seed_path,
                cache_seed_path=cache_seed_path,
                preset_paths=preset_paths,
                project_name=_slugify(dataset_name),
                image_match_count=image_match_count,
                referenced_image_count=referenced_image_count,
                missing_image_refs=missing_image_refs,
                notes=[],
            )
            if project_config_path is not None:
                candidate.notes.append(f"Using project config: {project_config_path}")
            if manifest_seed_path is not None:
                candidate.notes.append(f"Found resume manifest seed: {manifest_seed_path}")
            if cache_seed_path is not None:
                candidate.notes.append(f"Found cache seed: {cache_seed_path}")
            if preset_paths:
                candidate.notes.append(
                    "Found optional presets: " + ", ".join(str(path) for path in preset_paths)
                )
            if missing_image_refs:
                candidate.notes.append(
                    f"Missing referenced images: {len(missing_image_refs)}"
                )
        except Exception as exc:
            candidate_errors.append(f"{jobs_csv_path}: {exc}")
            continue

        if best_candidate is None:
            best_candidate = candidate
            continue

        current_score = (
            -candidate.missing_image_count,
            candidate.image_match_count,
            1 if candidate.project_config_path else 0,
            1 if candidate.manifest_seed_path else 0,
            -len(candidate.jobs_csv_path.parts),
        )
        best_score = (
            -best_candidate.missing_image_count,
            best_candidate.image_match_count,
            1 if best_candidate.project_config_path else 0,
            1 if best_candidate.manifest_seed_path else 0,
            -len(best_candidate.jobs_csv_path.parts),
        )
        if current_score > best_score:
            best_candidate = candidate

    if best_candidate is None:
        raise FileNotFoundError(
            "Unable to determine a valid project candidate from /kaggle/input. "
            + ("; ".join(candidate_errors) if candidate_errors else "No valid jobs.csv candidates.")
        )
    if candidate_errors:
        best_candidate.notes.extend(
            ["Rejected candidate: " + error for error in candidate_errors]
        )
    return best_candidate


def discover_drive_credentials(
    repo_root: Path,
    *,
    input_root: Path = Path("/kaggle/input"),
    working_root: Path = Path("/kaggle/working"),
) -> DriveCredentialDiscovery:
    notes: list[str] = []
    for env_key in DRIVE_ENV_JSON_KEYS:
        raw_json = os.environ.get(env_key)
        if raw_json:
            notes.append(f"Using credentials from environment variable {env_key}")
            return DriveCredentialDiscovery(
                enabled=True,
                source=f"env:{env_key}",
                raw_json=raw_json,
                path=None,
                notes=notes,
            )

    for env_key in DRIVE_ENV_PATH_KEYS:
        raw_path = os.environ.get(env_key)
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if candidate.exists():
            notes.append(f"Using credentials file from environment variable {env_key}")
            return DriveCredentialDiscovery(
                enabled=True,
                source=f"env-path:{env_key}",
                raw_json=candidate.read_text(encoding="utf-8"),
                path=candidate,
                notes=notes,
            )

    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for secret_name in DRIVE_SECRET_NAMES:
            try:
                raw_json = client.get_secret(secret_name)
            except Exception:
                continue
            if raw_json:
                notes.append(f"Using Kaggle secret {secret_name}")
                return DriveCredentialDiscovery(
                    enabled=True,
                    source=f"kaggle-secret:{secret_name}",
                    raw_json=raw_json,
                    path=None,
                    notes=notes,
                )
    except Exception:
        notes.append("Kaggle secrets client unavailable or no matching secret found")

    search_roots = [working_root, repo_root, input_root]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            filename = path.name.lower()
            if not any(token in filename for token in ("credential", "service", "drive", "gdrive")):
                continue
            try:
                raw_json = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if '"type": "service_account"' in raw_json or '"refresh_token"' in raw_json:
                notes.append(f"Using credentials file discovered at {path}")
                return DriveCredentialDiscovery(
                    enabled=True,
                    source=f"file:{path}",
                    raw_json=raw_json,
                    path=path,
                    notes=notes,
                )

    notes.append("No Drive credentials detected; launcher will continue in local-only mode")
    return DriveCredentialDiscovery(
        enabled=False,
        source="local-only",
        raw_json=None,
        path=None,
        notes=notes,
    )


def _restore_seed_file(source: Optional[Path], destination: Path) -> Optional[Path]:
    if source is None or not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    shutil.copy2(source, destination)
    return destination


def _summarize_resume(manifest_path: Optional[Path], cache_index_path: Optional[Path]) -> ResumeSummary:
    notes: list[str] = []
    manifest_data = {"jobs": []}
    if manifest_path and manifest_path.exists():
        try:
            manifest_data = _read_json(manifest_path)
        except Exception as exc:
            notes.append(str(exc))
    jobs = manifest_data.get("jobs", [])
    total_jobs = len(jobs)
    completed_jobs = len(
        [job for job in jobs if str(job.get("status", "")).upper() in FINAL_SUCCESS_STATUSES]
    )
    failed_jobs = len(
        [job for job in jobs if str(job.get("status", "")).upper() in FINAL_FAILURE_STATUSES]
    )
    skipped_jobs = len([job for job in jobs if bool(job.get("cache_hit"))])
    remaining_jobs = max(0, total_jobs - completed_jobs - failed_jobs)

    cache_entries = 0
    if cache_index_path and cache_index_path.exists():
        try:
            cache_entries = len(_read_json(cache_index_path).get("entries", {}))
        except Exception as exc:
            notes.append(str(exc))

    return ResumeSummary(
        manifest_path=manifest_path if manifest_path and manifest_path.exists() else None,
        cache_index_path=cache_index_path if cache_index_path and cache_index_path.exists() else None,
        total_jobs=total_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        remaining_jobs=remaining_jobs,
        skipped_jobs=skipped_jobs,
        cache_entries=cache_entries,
        notes=notes,
    )


def _build_launch_config(
    repo_root: Path,
    discovery: ProjectDiscovery,
    drive_credentials: DriveCredentialDiscovery,
    *,
    working_root: Path = Path("/kaggle/working"),
) -> tuple[Config, ResumeSummary]:
    run_root = working_root / "ltx_runs" / discovery.project_name
    output_dir = run_root / "outputs"
    log_dir = run_root / "logs"
    temp_dir = run_root / "temp"
    manifest_path = output_dir / "manifest.json"
    cache_index_path = output_dir / "cache" / "render_cache.json"

    restored_manifest = _restore_seed_file(discovery.manifest_seed_path, manifest_path)
    restored_cache = _restore_seed_file(discovery.cache_seed_path, cache_index_path)
    resume_summary = _summarize_resume(restored_manifest, restored_cache)

    runtime_overrides: dict[str, Any] = {
        "jobs_csv_path": discovery.jobs_csv_path,
        "reference_images_dir": discovery.reference_images_dir,
        "output_dir": output_dir,
        "log_dir": log_dir,
        "temp_dir": temp_dir,
        "manifest_path": manifest_path,
        "report_path": output_dir / "report.json",
        "model_cache_dir": working_root / "models",
        "hf_cache_dir": working_root / "hf_cache",
        "asset_temp_dir": working_root / "asset_temp",
        "asset_report_dir": output_dir / "reports" / "assets",
        "asset_download_manifest_path": output_dir / "manifests" / "download_manifest.json",
        "drive_project_name": discovery.project_name,
        "heartbeat_interval_seconds": 15,
        "health_poll_interval_seconds": 10,
        "sync_heartbeat_to_drive": drive_credentials.enabled,
        "wan2gp_dir": working_root / "Wan2GP",
    }
    if not drive_credentials.enabled:
        runtime_overrides["enable_drive_upload"] = False
    elif drive_credentials.raw_json:
        runtime_overrides["extra"] = {"drive_credentials_json": drive_credentials.raw_json}

    config = load_config(
        default_config_path=repo_root / "config" / "default.yaml",
        project_config_path=discovery.project_config_path,
        runtime_overrides=runtime_overrides,
    )
    return config, resume_summary


def _render_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except Exception:
        print(text)


def _clear_output(wait: bool = True) -> None:
    try:
        from IPython.display import clear_output

        clear_output(wait=wait)
    except Exception:
        pass


class KaggleNotebookLauncher:
    def __init__(
        self,
        repo_root: Path,
        *,
        input_root: Path = Path("/kaggle/input"),
        working_root: Path = Path("/kaggle/working"),
    ) -> None:
        self.repo_root = repo_root.resolve(strict=False)
        self.input_root = input_root
        self.working_root = working_root
        self.source_root = detect_source_root(self.repo_root)
        self.context: Optional[NotebookLaunchContext] = None
        self.runner: Optional[ApplicationRunner] = None
        self.state = LauncherExecutionState()
        self._dashboard_errors: dict[str, str] = {}

    def _reset_state(self) -> None:
        self.context = None
        self.runner = None
        self.state = LauncherExecutionState()
        self._dashboard_errors = {}

    def _stage_suggestions(self, stage_name: str, exc: Optional[Exception] = None) -> list[str]:
        suggestions = {
            "bootstrap_context": [
                "Confirm the repository was cloned successfully and contains the expected source tree.",
                "Verify /kaggle/input contains a valid jobs.csv and referenced assets.",
                "Check Drive credentials only if Google Drive upload is required.",
            ],
            "prepare_runtime": [
                "Review the dependency and runtime sections in the bootstrap report.",
                "Verify CUDA, Torch, FFmpeg, and model assets are compatible with the Kaggle runtime.",
                "Disable optional features or use a lighter execution profile if a non-critical dependency failed.",
            ],
            "run_preflight": [
                "Inspect validation_report.json and diagnostics.json for blocking issues.",
                "Resolve CSV, image, disk, or configuration failures before retrying.",
            ],
            "display_preparation": [
                "Re-run the notebook cell after resolving earlier bootstrap failures.",
                "Inspect the bootstrap report object directly if notebook rendering is degraded.",
            ],
            "launch_pipeline": [
                "Inspect logs, manifest, diagnostics, and validation reports in the output directory.",
                "Retry after resolving the reported runtime or upload failure.",
            ],
        }.get(stage_name, ["Inspect the structured bootstrap report and generated diagnostics."])
        if exc is not None and isinstance(exc, FileNotFoundError):
            suggestions = [
                "Confirm the expected file or dataset is attached and accessible.",
                *suggestions,
            ]
        return suggestions

    def _upsert_stage(self, stage: StageExecutionReport) -> None:
        stages = [
            existing
            for existing in self.state.bootstrap_report.stages
            if existing.stage_name != stage.stage_name
        ]
        stages.append(stage)
        self.state.bootstrap_report.stages = stages

    def _execute_stage(
        self,
        stage_name: str,
        callback: Callable[[], Any],
        *,
        success_message: str,
        skip_reason: Optional[str] = None,
    ) -> Any:
        started_at = datetime.now().isoformat()
        started_monotonic = time.monotonic()
        if skip_reason is not None:
            stage = StageExecutionReport(
                stage_name=stage_name,
                status="SKIPPED",
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                duration_seconds=0.0,
                message=skip_reason,
                suggestions=self._stage_suggestions(stage_name),
            )
            self._upsert_stage(stage)
            self._refresh_bootstrap_report()
            return None
        try:
            result = callback()
        except Exception as exc:
            stage = StageExecutionReport(
                stage_name=stage_name,
                status="FAILED",
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                duration_seconds=round(time.monotonic() - started_monotonic, 2),
                message=str(exc),
                details={
                    "exception_type": exc.__class__.__name__,
                    "traceback": traceback.format_exc(limit=8),
                },
                suggestions=self._stage_suggestions(stage_name, exc),
                error_type=exc.__class__.__name__,
            )
            self._upsert_stage(stage)
            self._refresh_bootstrap_report()
            return None
        stage = StageExecutionReport(
            stage_name=stage_name,
            status="PASS",
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            duration_seconds=round(time.monotonic() - started_monotonic, 2),
            message=success_message,
        )
        self._upsert_stage(stage)
        self._refresh_bootstrap_report()
        return result

    def _make_section(
        self,
        name: str,
        status: str,
        message: str,
        *,
        details: Optional[list[str]] = None,
        suggestions: Optional[list[str]] = None,
    ) -> BootstrapSection:
        return BootstrapSection(
            name=name,
            status=status,
            message=message,
            details=details or [],
            suggestions=suggestions or [],
        )

    def _make_launcher_failure_result(
        self,
        *,
        category: str,
        summary: str,
        recommendation: str,
    ) -> ApplicationRunResult:
        config = self.context.config if self.context is not None else Config()
        diagnostics = (
            self.state.preparation.diagnostics
            if self.state.preparation is not None
            else {}
        )
        preflight = (
            self.state.preparation.preflight
            if self.state.preparation is not None
            else {}
        )
        return ApplicationRunResult(
            success=False,
            exit_code=1,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            config=config,
            diagnostics=diagnostics,
            preflight=preflight,
            summary={},
            failure={
                "category": category,
                "summary": summary,
                "recommendation": recommendation,
            },
        )

    def _refresh_bootstrap_report(self) -> BootstrapReport:
        sections: dict[str, BootstrapSection] = {}
        stage_map = self.state.bootstrap_report.stage_by_name()
        context = self.state.context
        runtime_preparation = self.state.runtime_preparation
        preparation = self.state.preparation

        bootstrap_stage = stage_map.get("bootstrap_context")
        if context is not None:
            repo_status = "PASS" if context.repo_validation.ready else "FAILED"
            if context.repo_validation.optional_missing:
                repo_status = "WARNING" if repo_status == "PASS" else repo_status
            sections["Repository"] = self._make_section(
                "Repository",
                repo_status,
                "Repository validation complete.",
                details=[
                    f"Source Root: {context.source_root}",
                    "Optional Missing: "
                    + (", ".join(context.repo_validation.optional_missing) or "None"),
                ],
                suggestions=self._stage_suggestions("bootstrap_context"),
            )
        elif bootstrap_stage is not None and bootstrap_stage.status == "FAILED":
            sections["Repository"] = self._make_section(
                "Repository",
                "FAILED",
                bootstrap_stage.message,
                details=[bootstrap_stage.details.get("exception_type", "")],
                suggestions=bootstrap_stage.suggestions,
            )
        else:
            sections["Repository"] = self._make_section(
                "Repository",
                "SKIPPED",
                "Repository validation has not completed.",
            )

        dep_profile = runtime_preparation.get("dependency_profile") or {}
        prepare_stage = stage_map.get("prepare_runtime")
        failed_optional = dep_profile.get("failed_optional", [])
        feature_statuses = dep_profile.get("feature_statuses", [])
        verification_checks = dep_profile.get("verification", [])
        required_feature_failures = [
            status
            for status in feature_statuses
            if status.get("enabled") and status.get("required") and status.get("state") == "FAILED"
        ]
        runtime_failures = [check for check in verification_checks if not check.get("success")]

        if dep_profile:
            dep_status = "PASS"
            if required_feature_failures:
                dep_status = "FAILED"
            elif failed_optional or dep_profile.get("skipped_packages"):
                dep_status = "WARNING"
            sections["Dependencies"] = self._make_section(
                "Dependencies",
                dep_status,
                f"Execution profile `{dep_profile.get('profile_name', 'unknown')}` resolved.",
                details=[
                    "Enabled Features: "
                    + (", ".join(dep_profile.get("enabled_features", [])) or "None"),
                    f"Packages Installed This Run: {len(dep_profile.get('installed_packages', []))}",
                    f"Optional Dependency Failures: {len(failed_optional)}",
                ],
                suggestions=self._stage_suggestions("prepare_runtime"),
            )
        elif prepare_stage is not None and prepare_stage.status == "FAILED":
            sections["Dependencies"] = self._make_section(
                "Dependencies",
                "FAILED",
                prepare_stage.message,
                suggestions=prepare_stage.suggestions,
            )
        else:
            sections["Dependencies"] = self._make_section(
                "Dependencies",
                "SKIPPED",
                "Runtime dependency preparation has not completed.",
            )
        if dep_profile:
            runtime_status = "PASS"
            if required_feature_failures:
                runtime_status = "FAILED"
            elif runtime_failures:
                runtime_status = "WARNING"
            sections["Runtime"] = self._make_section(
                "Runtime",
                runtime_status,
                f"Runtime verification completed for `{dep_profile.get('renderer_backend', 'unknown')}`.",
                details=[
                    f"Verification Checks: {len(verification_checks)}",
                    f"Verification Failures: {len(runtime_failures)}",
                    "Failed Checks: "
                    + (
                        ", ".join(check.get("name", "unknown") for check in runtime_failures[:5])
                        or "None"
                    ),
                ],
                suggestions=self._stage_suggestions("prepare_runtime"),
            )
        elif prepare_stage is not None and prepare_stage.status == "FAILED":
            sections["Runtime"] = self._make_section(
                "Runtime",
                "FAILED",
                prepare_stage.message,
                suggestions=prepare_stage.suggestions,
            )
        else:
            sections["Runtime"] = self._make_section(
                "Runtime",
                "SKIPPED",
                "Runtime validation has not completed.",
            )

        if context is not None:
            dataset_status = "PASS" if context.discovery.jobs_csv_path.exists() else "FAILED"
            if dataset_status == "PASS" and context.discovery.missing_image_count:
                dataset_status = "WARNING"
            sections["Dataset"] = self._make_section(
                "Dataset",
                dataset_status,
                f"Selected jobs.csv: {context.discovery.jobs_csv_path}",
                details=(context.discovery.notes or [])
                + [
                    f"Reference Images Root: {context.discovery.reference_images_dir}",
                    f"Referenced Images: {context.discovery.referenced_image_count}",
                    f"Matched Images: {context.discovery.image_match_count}",
                    f"Missing Images: {context.discovery.missing_image_count}",
                ]
                + (
                    [f"First Missing Image: {context.discovery.missing_image_refs[0]}"]
                    if context.discovery.missing_image_refs
                    else []
                ),
                suggestions=[
                    "Attach the intended Kaggle dataset and ensure jobs.csv contains valid image references.",
                    "Resolve missing images before launch if the dataset section reports warnings.",
                ],
            )
            drive_status = "PASS" if context.drive_credentials.enabled else "WARNING"
            sections["Drive"] = self._make_section(
                "Drive",
                drive_status,
                f"Drive mode: {context.drive_credentials.source}",
                details=context.drive_credentials.notes,
                suggestions=[
                    "Add Kaggle secrets or credential JSON only if Drive upload is required."
                ],
            )
            resume_status = "WARNING" if context.resume_summary.notes else "PASS"
            sections["Resume"] = self._make_section(
                "Resume",
                resume_status,
                (
                    f"Manifest present={bool(context.resume_summary.manifest_path)} "
                    f"cache_entries={context.resume_summary.cache_entries}"
                ),
                details=context.resume_summary.notes,
                suggestions=[
                    "Remove stale manifest/cache files if resume state looks inconsistent."
                ],
            )
            sections["Configuration"] = self._make_section(
                "Configuration",
                "PASS",
                f"Execution profile `{detect_execution_profile(context.config)}` loaded.",
                details=[
                    f"Renderer backend: {context.config.renderer_backend}",
                    f"Project config: {context.discovery.project_config_path or 'Not found'}",
                ],
            )
            env = context.environment_info
            gpu = env.get("gpu", {})
            disk = env.get("disk", {})
            ram = env.get("ram", {})
            sections["Environment"] = self._make_section(
                "Environment",
                "PASS",
                f"Kaggle detected={env.get('is_kaggle', False)} platform={env.get('platform', 'Unknown')}",
            )
            sections["GPU"] = self._make_section(
                "GPU",
                "PASS" if gpu.get("available") else "WARNING",
                gpu.get("name", "GPU unavailable"),
            )
            sections["CUDA"] = self._make_section(
                "CUDA",
                "PASS" if context.dependency_inspection.cuda_available else "WARNING",
                context.dependency_inspection.cuda_version,
            )
            sections["Torch"] = self._make_section(
                "Torch",
                "PASS" if dep_profile.get("torch_version") else "SKIPPED",
                dep_profile.get("torch_version", "Unavailable"),
            )
            sections["Python"] = self._make_section(
                "Python",
                "PASS",
                env.get("python_version", sys.version.split()[0]),
            )
            sections["Disk"] = self._make_section(
                "Disk",
                "PASS" if disk.get("free_bytes") is not None else "WARNING",
                _format_bytes(disk.get("free_bytes")),
            )
            sections["RAM"] = self._make_section(
                "RAM",
                "PASS" if ram.get("total_bytes") is not None else "WARNING",
                _format_bytes(ram.get("total_bytes")),
            )
        else:
            for section_name in ("Dataset", "Drive", "Resume", "Configuration", "Environment", "GPU", "CUDA", "Torch", "Python", "Disk", "RAM"):
                sections[section_name] = self._make_section(
                    section_name,
                    "SKIPPED",
                    "Context unavailable because bootstrap did not complete.",
                )

        if detect_renderer_dependency_profile(context.config) == "wan2gp" if context else False:
            model_status = "PASS" if runtime_preparation.get("wan2gp_assets") else "WARNING"
            if required_feature_failures:
                model_status = "FAILED"
            sections["Models"] = self._make_section(
                "Models",
                model_status,
                (
                    f"Wan2GP assets ready at "
                    f"{(runtime_preparation.get('wan2gp_assets') or {}).get('model_dir', 'unknown')}"
                ),
                details=[
                    f"Downloaded Files: {len((runtime_preparation.get('wan2gp_assets') or {}).get('downloaded_files', []))}",
                    f"Existing Files: {len((runtime_preparation.get('wan2gp_assets') or {}).get('existing_files', []))}",
                ],
            )
        else:
            sections["Models"] = self._make_section(
                "Models",
                "SKIPPED" if context is None else "PASS",
                "Wan2GP model preparation not required for the active renderer."
                if context is not None
                else "Model preparation unavailable because bootstrap did not complete.",
            )

        preflight_stage = stage_map.get("run_preflight")
        if preparation is not None:
            validation_status = "PASS" if preparation.ready else "FAILED"
            if preparation.preflight.get("warnings") and preparation.ready:
                validation_status = "WARNING"
            sections["Validation"] = self._make_section(
                "Validation",
                validation_status,
                f"Preflight ready={preparation.ready}",
                details=[
                    f"Diagnostics Status: {preparation.diagnostics.get('status', 'UNKNOWN')}",
                    f"Blocking Failures: {len(preparation.preflight.get('blocking_failures', []))}",
                    f"Warnings: {len(preparation.preflight.get('warnings', []))}",
                ],
                suggestions=self._stage_suggestions("run_preflight"),
            )
        elif preflight_stage is not None and preflight_stage.status == "FAILED":
            sections["Validation"] = self._make_section(
                "Validation",
                "FAILED",
                preflight_stage.message,
                suggestions=preflight_stage.suggestions,
            )
        else:
            sections["Validation"] = self._make_section(
                "Validation",
                "SKIPPED",
                "Preflight has not completed.",
            )

        ready_to_launch = (
            context is not None
            and preparation is not None
            and preparation.ready
            and not required_feature_failures
        )
        self.state.bootstrap_report.sections = sections
        self.state.bootstrap_report.ready_to_launch = ready_to_launch
        self.state.bootstrap_report.summary_lines = [
            "==================================",
            "KAGGLE LAUNCH SUMMARY",
            "==================================",
            *[
                f"{name}: {section.status} - {section.message}"
                for name, section in sections.items()
            ],
            f"Ready To Launch: {'YES' if ready_to_launch else 'NO'}",
            "==================================",
        ]
        return self.state.bootstrap_report

    def get_execution_state(self) -> LauncherExecutionState:
        return self.state

    def can_launch_pipeline(self) -> bool:
        return bool(self.state.bootstrap_report.ready_to_launch)

    def bootstrap_context(self) -> NotebookLaunchContext:
        dependency_inspection = inspect_runtime_dependencies()
        repo_validation = validate_repository_layout(self.repo_root)
        if not repo_validation.ready:
            raise RuntimeError(
                "Critical repository files are missing: "
                + ", ".join(repo_validation.critical_missing)
            )
        discovery = discover_kaggle_project(self.input_root)
        drive_credentials = discover_drive_credentials(
            self.repo_root,
            input_root=self.input_root,
            working_root=self.working_root,
        )
        config, resume_summary = _build_launch_config(
            self.source_root,
            discovery,
            drive_credentials,
            working_root=self.working_root,
        )
        environment_info = _bootstrap_module.collect_environment_info(config)
        self.runner = ApplicationRunner(config)
        self.context = NotebookLaunchContext(
            repo_validation=repo_validation,
            dependency_inspection=dependency_inspection,
            discovery=discovery,
            drive_credentials=drive_credentials,
            config=config,
            environment_info=environment_info,
            resume_summary=resume_summary,
            source_root=self.source_root,
        )
        return self.context

    def prepare_runtime(self) -> dict[str, Any]:
        if self.context is None:
            self.bootstrap_context()
        if self.context is None:
            raise RuntimeError("Launcher context not prepared")

        reports: dict[str, Any] = {
            "source_root": str(self.context.source_root),
            "dependency_profile": None,
            "wan2gp_runtime": None,
            "wan2gp_assets": None,
        }

        # The Wan2GP runtime must be cloned BEFORE dependency verification because the
        # verification includes a check that the Wan2GP directory exists. Cloning first
        # lets that check pass and lets model asset preparation run afterwards.
        wan2gp_runtime_report = None
        if detect_renderer_dependency_profile(self.context.config) == "wan2gp":
            wan2gp_runtime_report = ensure_wan2gp_runtime(self.context.config)

        dependency_report = ensure_runtime_dependency_profile(self.context.config)
        reports["dependency_profile"] = {
            "profile_name": dependency_report.profile_name,
            "renderer_backend": dependency_report.renderer_backend,
            "python_version": dependency_report.python_version,
            "torch_version": dependency_report.torch_version,
            "cuda_version": dependency_report.cuda_version,
            "enabled_features": dependency_report.enabled_features,
            "disabled_features": dependency_report.disabled_features,
            "selected_requirements": dependency_report.selected_requirements,
            "installed_packages": [
                _serialize_pip_result(item) for item in dependency_report.installed_packages
            ],
            "skipped_packages": [
                _serialize_pip_result(item)
                for item in dependency_report.skipped_packages
            ],
            "failed_optional": [
                _serialize_pip_result(item)
                for item in dependency_report.failed_optional
            ],
            "verification": [
                {
                    "name": check.name,
                    "success": check.success,
                    "details": check.details,
                    "feature_key": check.feature_key,
                }
                for check in dependency_report.verification_checks
            ],
            "feature_statuses": [
                {
                    "feature_key": status.feature_key,
                    "enabled": status.enabled,
                    "required": status.required,
                    "state": status.state,
                    "reason": status.reason,
                }
                for status in dependency_report.feature_statuses
            ],
        }
        if dependency_report.failed_required:
            failure = dependency_report.failed_required[0]
            raise RuntimeError(
                f"Required dependency install failed for {failure.package_name} "
                f"(requested={getattr(failure, 'requested_version', 'n/a') or 'n/a'}, "
                f"installed_before={getattr(failure, 'installed_version_before', 'n/a') or 'n/a'}, "
                f"python={getattr(failure, 'python_version', 'n/a') or 'n/a'}, "
                f"cuda={getattr(failure, 'cuda_version', 'n/a') or 'n/a'}, "
                f"exit_code={failure.exit_code}).\n"
                f"stdout:\n{failure.stdout or '<empty>'}\n"
                f"stderr:\n{failure.stderr or '<empty>'}\n"
                f"suggested_resolution:\n{getattr(failure, 'suggested_resolution', 'Review pip diagnostics and retry.')}"
            )
        failed_checks = [
            status
            for status in dependency_report.feature_statuses
            if status.enabled and status.required and status.state == "FAILED"
        ]
        if failed_checks:
            raise RuntimeError(
                "Runtime dependency verification failed: "
                + "; ".join(f"{status.feature_key}={status.reason}" for status in failed_checks)
            )

        if detect_renderer_dependency_profile(self.context.config) == "wan2gp":
            runtime_report = wan2gp_runtime_report
            drive_client = None
            if (
                self.context.drive_credentials.enabled
                and self.context.config.enable_drive_model_cache
            ):
                try:
                    from drive.gdrive import GoogleDriveClient

                    drive_client = GoogleDriveClient(self.context.config)
                    drive_client.connect()
                except Exception as exc:
                    reports["drive_model_cache"] = {"status": "FAILED", "reason": str(exc)}

            model_registry = ensure_model_registry(self.context.config)
            reports["model_registry"] = model_registry.to_dict()
            asset_report = ensure_wan2gp_model_assets(self.context.config, drive_client=drive_client)
            reports["wan2gp_runtime"] = {
                "destination": str(runtime_report.destination),
                "repo_url": runtime_report.repo_url,
                "ref": runtime_report.ref,
                "cloned": runtime_report.cloned,
                "updated": runtime_report.updated,
            }
            reports["wan2gp_assets"] = asset_report
            reports["wan2gp_assets"]["wan2gp_runtime"] = reports["wan2gp_runtime"]

        self.context.runtime_preparation = reports
        self.context.dependency_inspection = inspect_runtime_dependencies(self.context.config)
        self.context.config.extra["runtime_preparation"] = reports
        self.state.runtime_preparation = reports
        return reports

    def run_preflight(self) -> PreparationResult:
        if self.context is None:
            self.bootstrap_context()
        if self.runner is None:
            raise RuntimeError("Application runner not initialized")
        preparation = self.runner.prepare()
        if self.context is not None:
            self.context.preparation = preparation
        self.state.preparation = preparation
        return preparation

    def prepare(self) -> NotebookLaunchContext:
        state = self.execute_bootstrap_flow()
        if state.context is None:
            raise RuntimeError("Launcher context could not be prepared")
        return state.context

    def execute_bootstrap_flow(self) -> LauncherExecutionState:
        self._reset_state()
        context = self._execute_stage(
            "bootstrap_context",
            self.bootstrap_context,
            success_message="Bootstrap context created successfully.",
        )
        self.state.context = context
        runtime = self._execute_stage(
            "prepare_runtime",
            self.prepare_runtime,
            success_message="Runtime preparation completed.",
            skip_reason=None if context is not None else "Skipped because bootstrap context failed.",
        )
        if isinstance(runtime, dict):
            self.state.runtime_preparation = runtime
        preparation = self._execute_stage(
            "run_preflight",
            self.run_preflight,
            success_message="Preflight completed.",
            skip_reason=None if context is not None and runtime is not None else "Skipped because runtime preparation failed.",
        )
        self.state.preparation = preparation if isinstance(preparation, PreparationResult) else None
        self._execute_stage(
            "display_preparation",
            self.display_preparation,
            success_message="Preparation report displayed.",
        )
        self._refresh_bootstrap_report()
        return self.state

    def render_startup_banner(self) -> str:
        context = self.state.context
        env = (context.environment_info if context is not None else {}) or {}
        gpu = env.get("gpu", {})
        ram = env.get("ram", {})
        disk = env.get("disk", {})
        cuda_version = (
            context.dependency_inspection.cuda_version
            if context is not None
            else "Unavailable"
        )
        return "\n".join(
            [
                f"# {APP_NAME}",
                "",
                f"- Version: `{APP_VERSION}`",
                f"- Renderer Backend: `{context.config.renderer_backend if context is not None else 'Unavailable'}`",
                f"- Timestamp: `{datetime.now().isoformat()}`",
                f"- Python Version: `{env.get('python_version', sys.version.split()[0])}`",
                f"- CUDA Version: `{cuda_version}`",
                f"- GPU Model: `{gpu.get('name', 'Unavailable')}`",
                f"- Available VRAM: `{_format_bytes(gpu.get('free_vram_bytes'))}`",
                f"- RAM: `{_format_bytes(ram.get('total_bytes'))}`",
                f"- Disk Free: `{_format_bytes(disk.get('free_bytes'))}`",
                f"- Kaggle Runtime Detected: `{env.get('is_kaggle', False)}`",
            ]
        )

    def render_preparation_summary(self) -> str:
        report = self._refresh_bootstrap_report()
        stage_lines = []
        for stage in report.stages:
            stage_lines.append(
                f"- {stage.stage_name}: `{stage.status}` in `{stage.duration_seconds:.2f}s`"
                + (f" - {stage.message}" if stage.message else "")
            )
        section_lines = []
        for section_name, section in report.sections.items():
            section_lines.append(f"- {section_name}: `{section.status}` - {section.message}")
        return "\n".join(
            [
                "## Stage Status",
                *stage_lines,
                "",
                "## Bootstrap Report",
                *section_lines,
                "",
                "## Ready To Launch",
                f"- Ready: `{report.ready_to_launch}`",
            ]
        )

    def display_preparation(self) -> None:
        _render_markdown(self.render_startup_banner())
        _render_markdown(self.render_preparation_summary())

    def render_bootstrap_report(self) -> str:
        report = self._refresh_bootstrap_report()
        return "\n".join(report.summary_lines)

    def _load_runtime_json(self, path: Optional[Path], *, label: str) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        try:
            self._dashboard_errors.pop(label, None)
            return _read_json(path)
        except Exception as exc:
            self._dashboard_errors[label] = str(exc)
            return {}

    def _build_dashboard_markdown(self, started_monotonic: float) -> str:
        context = self.state.context
        snapshot = self._load_runtime_json(
            context.config.heartbeat_path if context is not None else None,
            label="heartbeat",
        )
        manifest = self._load_runtime_json(
            context.config.manifest_path if context is not None else None,
            label="manifest",
        )
        completed_jobs = int(snapshot.get("completed_jobs", 0))
        failed_jobs = int(snapshot.get("failed_jobs", 0))
        remaining_jobs = int(snapshot.get("remaining_jobs", 0))
        total_jobs = int(snapshot.get("total_jobs", 0) or len(manifest.get("jobs", [])))
        elapsed_seconds = snapshot.get("elapsed_seconds")
        if elapsed_seconds is None:
            elapsed_seconds = time.monotonic() - started_monotonic
        uploaded_jobs = len(
            [
                job
                for job in manifest.get("jobs", [])
                if str(job.get("uploaded_at") or "").strip()
            ]
        )
        stitched_output = context.config.stitched_output_path if context else None
        dashboard_notes = [
            f"{label}: {message}" for label, message in sorted(self._dashboard_errors.items())
        ]
        return "\n".join(
            [
                "# Runtime Dashboard",
                "",
                f"- Current Job: `{snapshot.get('current_job', 'N/A')}`",
                f"- Status: `{snapshot.get('status', 'INITIALIZING')}`",
                f"- Completed Jobs: `{completed_jobs}`",
                f"- Failed Jobs: `{failed_jobs}`",
                f"- Uploaded Jobs: `{uploaded_jobs}`",
                f"- Total Jobs: `{total_jobs}`",
                f"- Remaining Jobs: `{remaining_jobs}`",
                f"- Upload Queue: `{snapshot.get('upload_queue', 0)}`",
                f"- Elapsed Runtime: `{_format_duration(elapsed_seconds)}`",
                f"- Average Render Time: `{snapshot.get('average_render_time_seconds', 0.0):.2f}s`",
                f"- Average Upload Time: `{snapshot.get('average_upload_time_seconds', 0.0):.2f}s`",
                f"- Average Throughput: `{snapshot.get('throughput_jobs_per_hour', 0.0):.2f} jobs/hour`",
                f"- ETA: `{_format_duration(snapshot.get('eta_seconds'))}`",
                f"- GPU Utilization: `{snapshot.get('gpu_utilization_percent', 'N/A')}`",
                f"- VRAM Usage: `{snapshot.get('vram_used_mb', 'N/A')} / {snapshot.get('vram_total_mb', 'N/A')} MB`",
                f"- CPU Utilization: `{snapshot.get('cpu_percent', 'N/A')}`",
                f"- RAM Usage: `{snapshot.get('ram_used_mb', 'N/A')} / {snapshot.get('ram_total_mb', 'N/A')} MB`",
                f"- Disk Free: `{_format_bytes((context.environment_info if context else {}).get('disk', {}).get('free_bytes'))}`",
                f"- Google Drive Status: `{context.drive_credentials.source if context else 'N/A'}`",
                f"- Resume Status: `completed={context.resume_summary.completed_jobs if context else 0} remaining={context.resume_summary.remaining_jobs if context else 0}`",
                f"- Heartbeat Path: `{context.config.heartbeat_path if context else 'N/A'}`",
                f"- Manifest Path: `{context.config.manifest_path if context else 'N/A'}`",
                f"- Stitched Output Target: `{stitched_output if stitched_output else 'N/A'}`",
                f"- Dashboard Parse Errors: `{'; '.join(dashboard_notes) or 'None'}`",
            ]
        )

    def _render_completion_summary(self, result: ApplicationRunResult) -> str:
        summary = result.summary.get("summary", {}) if result.summary else {}
        benchmark = result.summary.get("benchmark", {}) if result.summary else {}
        stitching = result.summary.get("stitching", {}) if result.summary else {}
        return "\n".join(
            [
                "# Completion Summary",
                "",
                f"- Success: `{result.success}`",
                f"- Total Runtime: `{summary.get('total_processing_time', 'N/A')}`",
                f"- Rendered Clips: `{summary.get('completed_jobs', 0)}`",
                f"- Skipped Clips: `{self.context.resume_summary.skipped_jobs if self.context else 0}`",
                f"- Uploaded Clips: `{summary.get('uploaded_jobs', 0)}`",
                f"- Failed Clips: `{summary.get('failed_jobs', 0)}`",
                f"- Stitched Output: `{stitching.get('output_path', 'N/A')}`",
                f"- Report JSON: `{result.config.report_path}`",
                f"- Summary TXT: `{result.config.summary_path}`",
                f"- Diagnostics: `{result.config.diagnostics_path}`",
                f"- Validation Report: `{result.config.validation_report_path}`",
                f"- Google Drive Destination: `{self.context.drive_credentials.source if self.context else 'N/A'}`",
                f"- Benchmark Average Job Time: `{benchmark.get('current_run', {}).get('average_total_job_seconds', 0.0):.2f}s`",
            ]
        )

    def _render_failure_summary(self, result: ApplicationRunResult) -> str:
        failure = result.failure
        return "\n".join(
            [
                "# Execution Failed",
                "",
                f"- Category: `{failure.get('category', 'unknown')}`",
                f"- Summary: `{failure.get('summary', 'Unknown error')}`",
                f"- Recommendation: `{failure.get('recommendation', 'Inspect the generated reports and logs.')}`",
                f"- Manifest: `{result.config.manifest_path}`",
                f"- Logs: `{result.config.log_dir}`",
                f"- Reports: `{result.config.output_dir}`",
                "",
                "Recovery Instructions:",
                "1. Inspect `validation_report.json`, `diagnostics.json`, and the log files.",
                "2. Fix the reported issue or provide missing credentials/assets.",
                "3. Re-run the notebook without changing paths manually.",
            ]
        )

    def get_artifact_report(self) -> dict[str, dict[str, Any]]:
        context = self.state.context
        if context is None:
            return {}
        artifact_paths = {
            "report_json": context.config.report_path,
            "summary_txt": context.config.summary_path,
            "diagnostics_json": context.config.diagnostics_path,
            "validation_report_json": context.config.validation_report_path,
            "performance_json": context.config.performance_report_path,
            "benchmark_json": context.config.benchmark_json_path,
            "manifest_json": context.config.manifest_path,
            "heartbeat_json": context.config.heartbeat_path,
            "stitched_output": context.config.stitched_output_path,
            "thumbnail": context.config.thumbnail_path,
            "preview_480p": context.config.preview_480p_path,
            "preview_720p": context.config.preview_720p_path,
            "log_dir": context.config.log_dir,
        }
        return {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in artifact_paths.items()
        }

    def execute_pipeline_stage(self, *, refresh_seconds: int = 5) -> ApplicationRunResult:
        if self.state.context is None or self.state.preparation is None:
            self.execute_bootstrap_flow()
        if not self.can_launch_pipeline():
            result = self._make_launcher_failure_result(
                category="bootstrap",
                summary="Launch pipeline skipped because bootstrap did not complete successfully.",
                recommendation="Review the bootstrap report above and resolve the failed stage before retrying.",
            )
            self.state.result = result
            self._execute_stage(
                "launch_pipeline",
                lambda: result,
                success_message="Launch skipped.",
                skip_reason="Skipped because the launcher is not ready to launch.",
            )
            return result
        return self.run_with_dashboard(refresh_seconds=refresh_seconds)

    def run_with_dashboard(self, *, refresh_seconds: int = 5) -> ApplicationRunResult:
        if self.context is None or self.runner is None or self.state.preparation is None:
            self.execute_bootstrap_flow()
        if self.context is None or self.runner is None:
            result = self._make_launcher_failure_result(
                category="bootstrap",
                summary="Launcher context could not be prepared.",
                recommendation="Resolve repository, dataset, or dependency issues reported in the bootstrap report.",
            )
            self.state.result = result
            return result
        if self.state.preparation is None:
            result = self._make_launcher_failure_result(
                category="preflight",
                summary="Preflight did not execute.",
                recommendation="Review the staged bootstrap report and rerun after fixing the failed stage.",
            )
            self.state.result = result
            return result
        if not self.state.preparation.ready:
            result = self.runner.run(preflight_only=True)
            self.state.result = result
            _render_markdown(self._render_failure_summary(result))
            return result

        started_at = datetime.now().isoformat()
        started_monotonic = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1)
        future: Future[ApplicationRunResult] = executor.submit(self.runner.run_pipeline_only)
        result: ApplicationRunResult
        try:
            while not future.done():
                _clear_output(wait=True)
                self.display_preparation()
                _render_markdown(self._build_dashboard_markdown(started_monotonic))
                time.sleep(max(1, refresh_seconds))
            result = future.result()
        except Exception as exc:
            result = self._make_launcher_failure_result(
                category="launch",
                summary=str(exc),
                recommendation="Inspect logs, diagnostics, and manifest artifacts for the failing runtime stage.",
            )
        finally:
            executor.shutdown(wait=True)

        self.state.result = result
        launch_stage = StageExecutionReport(
            stage_name="launch_pipeline",
            status="PASS" if result.success else "FAILED",
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            duration_seconds=round(time.monotonic() - started_monotonic, 2),
            message="Pipeline completed successfully." if result.success else result.failure.get("summary", "Pipeline execution failed."),
            details={"failure": result.failure} if not result.success else {},
            suggestions=self._stage_suggestions("launch_pipeline"),
            error_type=result.failure.get("exception_type", "") if not result.success else "",
        )
        self._upsert_stage(launch_stage)
        self._refresh_bootstrap_report()
        _clear_output(wait=True)
        self.display_preparation()
        if result.success:
            _render_markdown(self._render_completion_summary(result))
        else:
            _render_markdown(self._render_failure_summary(result))
        return result
