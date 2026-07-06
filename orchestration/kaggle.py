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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bootstrap import collect_environment_info
from config import Config, load_config
from core import APP_NAME, APP_VERSION
from orchestration.runner import ApplicationRunResult, ApplicationRunner, PreparationResult

NOTEBOOK_PACKAGE_SPECS = {
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "numpy": "numpy",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "psutil": "psutil",
    "googleapiclient": "google-api-python-client",
    "google.auth": "google-auth",
    "google_auth_oauthlib": "google-auth-oauthlib",
    "torch": "torch",
    "torchvision": "torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
}
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
    project_name: str
    image_match_count: int
    notes: list[str] = field(default_factory=list)


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


@dataclass
class NotebookLaunchContext:
    repo_validation: RepositoryValidation
    dependency_inspection: DependencyInspection
    discovery: ProjectDiscovery
    drive_credentials: DriveCredentialDiscovery
    config: Config
    environment_info: dict[str, Any]
    resume_summary: ResumeSummary
    preparation: PreparationResult


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
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def inspect_runtime_dependencies() -> DependencyInspection:
    packages: list[PackageInspection] = []
    missing_packages: list[str] = []
    for module_name, package_name in NOTEBOOK_PACKAGE_SPECS.items():
        installed = importlib.util.find_spec(module_name) is not None
        version = _safe_version(package_name) if installed else ""
        packages.append(
            PackageInspection(
                module_name=module_name,
                package_name=package_name,
                installed=installed,
                version=version,
            )
        )
        if not installed:
            missing_packages.append(package_name)

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
    inspection = inspect_runtime_dependencies()
    if inspection.missing_packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *inspection.missing_packages]
        )
    return inspect_runtime_dependencies()


def validate_repository_layout(repo_root: Path) -> RepositoryValidation:
    critical_missing = [
        str(relative_path)
        for relative_path in CRITICAL_REPOSITORY_FILES
        if not (repo_root / relative_path).exists()
    ]
    optional_missing = [
        str(relative_path)
        for relative_path in OPTIONAL_REPOSITORY_FILES
        if not (repo_root / relative_path).exists()
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


def _score_reference_root(root: Path, image_refs: list[str]) -> int:
    score = 0
    for ref in image_refs:
        if not ref:
            continue
        if (root / ref).exists():
            score += 1
    return score


def _discover_reference_root(jobs_csv_path: Path, input_root: Path) -> tuple[Path, int]:
    rows = _read_csv_rows(jobs_csv_path)
    image_refs: list[str] = []
    for row in rows:
        for key in ("start_image", "end_image"):
            value = (row.get(key) or "").strip()
            if value:
                image_refs.append(value)

    if not image_refs:
        return jobs_csv_path.parent, 0

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
    return best_root, max(0, best_score)


def discover_kaggle_project(input_root: Path = Path("/kaggle/input")) -> ProjectDiscovery:
    if not input_root.exists():
        raise FileNotFoundError(f"Kaggle input root not found: {input_root}")

    dataset_roots = sorted([path for path in input_root.iterdir() if path.is_dir()])
    jobs_candidates = _candidate_files(input_root, file_name="jobs.csv")
    if not jobs_candidates:
        raise FileNotFoundError("No jobs.csv file was found under /kaggle/input")

    best_candidate: Optional[ProjectDiscovery] = None
    for jobs_csv_path in jobs_candidates:
        reference_root, image_match_count = _discover_reference_root(jobs_csv_path, input_root)
        project_config_path = _find_nearest_config(jobs_csv_path, input_root)
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
            project_name=_slugify(dataset_name),
            image_match_count=image_match_count,
            notes=[],
        )
        if project_config_path is not None:
            candidate.notes.append(f"Using project config: {project_config_path}")
        if manifest_seed_path is not None:
            candidate.notes.append(f"Found resume manifest seed: {manifest_seed_path}")
        if cache_seed_path is not None:
            candidate.notes.append(f"Found cache seed: {cache_seed_path}")

        if best_candidate is None:
            best_candidate = candidate
            continue

        current_score = (
            candidate.image_match_count,
            1 if candidate.project_config_path else 0,
            1 if candidate.manifest_seed_path else 0,
            -len(candidate.jobs_csv_path.parts),
        )
        best_score = (
            best_candidate.image_match_count,
            1 if best_candidate.project_config_path else 0,
            1 if best_candidate.manifest_seed_path else 0,
            -len(best_candidate.jobs_csv_path.parts),
        )
        if current_score > best_score:
            best_candidate = candidate

    if best_candidate is None:
        raise FileNotFoundError("Unable to determine a project candidate from /kaggle/input")
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
    manifest_data = {"jobs": []}
    if manifest_path and manifest_path.exists():
        manifest_data = _read_json(manifest_path)
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
        cache_entries = len(_read_json(cache_index_path).get("entries", {}))

    return ResumeSummary(
        manifest_path=manifest_path if manifest_path and manifest_path.exists() else None,
        cache_index_path=cache_index_path if cache_index_path and cache_index_path.exists() else None,
        total_jobs=total_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        remaining_jobs=remaining_jobs,
        skipped_jobs=skipped_jobs,
        cache_entries=cache_entries,
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
        "drive_project_name": discovery.project_name,
        "heartbeat_interval_seconds": 15,
        "health_poll_interval_seconds": 10,
        "sync_heartbeat_to_drive": drive_credentials.enabled,
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
        self.context: Optional[NotebookLaunchContext] = None
        self.runner: Optional[ApplicationRunner] = None

    def prepare(self) -> NotebookLaunchContext:
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
            self.repo_root,
            discovery,
            drive_credentials,
            working_root=self.working_root,
        )
        environment_info = collect_environment_info(config)
        self.runner = ApplicationRunner(config)
        preparation = self.runner.prepare()
        self.context = NotebookLaunchContext(
            repo_validation=repo_validation,
            dependency_inspection=dependency_inspection,
            discovery=discovery,
            drive_credentials=drive_credentials,
            config=config,
            environment_info=environment_info,
            resume_summary=resume_summary,
            preparation=preparation,
        )
        return self.context

    def render_startup_banner(self) -> str:
        if self.context is None:
            raise RuntimeError("Launcher context not prepared")
        env = self.context.environment_info
        gpu = env.get("gpu", {})
        ram = env.get("ram", {})
        disk = env.get("disk", {})
        cuda_version = self.context.dependency_inspection.cuda_version
        return "\n".join(
            [
                f"# {APP_NAME}",
                "",
                f"- Version: `{APP_VERSION}`",
                f"- Renderer Backend: `{self.context.config.renderer_backend}`",
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
        if self.context is None:
            raise RuntimeError("Launcher context not prepared")
        preflight = self.context.preparation.preflight
        return "\n".join(
            [
                "## Automatic Discovery",
                f"- Jobs CSV: `{self.context.discovery.jobs_csv_path}`",
                f"- Reference Images Root: `{self.context.discovery.reference_images_dir}`",
                f"- Project Config: `{self.context.discovery.project_config_path or 'Not found'}`",
                f"- Resume Manifest: `{self.context.resume_summary.manifest_path or 'Not found'}`",
                f"- Cache Index: `{self.context.resume_summary.cache_index_path or 'Not found'}`",
                f"- Drive Mode: `{self.context.drive_credentials.source}`",
                "",
                "## Repository Validation",
                f"- Critical Components Ready: `{self.context.repo_validation.ready}`",
                f"- Optional Missing: `{', '.join(self.context.repo_validation.optional_missing) or 'None'}`",
                "",
                "## Resume Detection",
                f"- Previous Manifest: `{bool(self.context.resume_summary.manifest_path)}`",
                f"- Completed Jobs: `{self.context.resume_summary.completed_jobs}`",
                f"- Remaining Jobs: `{self.context.resume_summary.remaining_jobs}`",
                f"- Skipped Jobs: `{self.context.resume_summary.skipped_jobs}`",
                f"- Cache Entries: `{self.context.resume_summary.cache_entries}`",
                "",
                "## Preflight",
                f"- Diagnostics Status: `{self.context.preparation.diagnostics.get('status', 'UNKNOWN')}`",
                f"- Preflight Ready: `{preflight.get('ready', False)}`",
                f"- Expected Clip Count: `{preflight.get('expected_clip_count', 0)}`",
                f"- Estimated Runtime: `{_format_duration(preflight.get('estimated_runtime_seconds'))}`",
                f"- Estimated Storage: `{_format_bytes(preflight.get('estimated_storage_bytes'))}`",
                f"- Blocking Failures: `{len(preflight.get('blocking_failures', []))}`",
                f"- Warnings: `{len(preflight.get('warnings', []))}`",
            ]
        )

    def display_preparation(self) -> None:
        _render_markdown(self.render_startup_banner())
        _render_markdown(self.render_preparation_summary())

    def _load_snapshot(self) -> dict[str, Any]:
        if self.context is None:
            return {}
        heartbeat_path = self.context.config.heartbeat_path
        if heartbeat_path.exists():
            try:
                return _read_json(heartbeat_path)
            except Exception:
                return {}
        return {}

    def _load_manifest(self) -> dict[str, Any]:
        if self.context is None:
            return {}
        manifest_path = self.context.config.manifest_path
        if manifest_path.exists():
            try:
                return _read_json(manifest_path)
            except Exception:
                return {}
        return {}

    def _build_dashboard_markdown(self, started_monotonic: float) -> str:
        snapshot = self._load_snapshot()
        manifest = self._load_manifest()
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
        stitched_output = self.context.config.stitched_output_path if self.context else None
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
                f"- Average Throughput: `{snapshot.get('throughput_jobs_per_hour', 0.0):.2f} jobs/hour`",
                f"- ETA: `{_format_duration(snapshot.get('eta_seconds'))}`",
                f"- GPU Utilization: `{snapshot.get('gpu_utilization_percent', 'N/A')}`",
                f"- VRAM Usage: `{snapshot.get('vram_used_mb', 'N/A')} / {snapshot.get('vram_total_mb', 'N/A')} MB`",
                f"- CPU Utilization: `{snapshot.get('cpu_percent', 'N/A')}`",
                f"- RAM Usage: `{snapshot.get('ram_used_mb', 'N/A')} / {snapshot.get('ram_total_mb', 'N/A')} MB`",
                f"- Heartbeat Path: `{self.context.config.heartbeat_path if self.context else 'N/A'}`",
                f"- Manifest Path: `{self.context.config.manifest_path if self.context else 'N/A'}`",
                f"- Stitched Output Target: `{stitched_output if stitched_output else 'N/A'}`",
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

    def run_with_dashboard(self, *, refresh_seconds: int = 5) -> ApplicationRunResult:
        if self.context is None or self.runner is None:
            self.prepare()
        if self.context is None or self.runner is None:
            raise RuntimeError("Launcher context could not be prepared")
        if not self.context.preparation.ready:
            result = self.runner.run(preflight_only=True)
            _render_markdown(self._render_failure_summary(result))
            return result

        started_monotonic = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=1)
        future: Future[ApplicationRunResult] = executor.submit(self.runner.run_pipeline_only)
        try:
            while not future.done():
                _clear_output(wait=True)
                self.display_preparation()
                _render_markdown(self._build_dashboard_markdown(started_monotonic))
                time.sleep(max(1, refresh_seconds))
            result = future.result()
        finally:
            executor.shutdown(wait=True)

        _clear_output(wait=True)
        self.display_preparation()
        if result.success:
            _render_markdown(self._render_completion_summary(result))
        else:
            _render_markdown(self._render_failure_summary(result))
        return result
