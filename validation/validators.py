import csv
import json
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image, UnidentifiedImageError

from config import Config
from config.profiles import CONFIG_PROFILES
from core import (
    AspectRatio,
    ConfigurationError,
    Duration,
    Resolution,
    SUPPORTED_IMAGE_FORMATS,
    ValidationError,
)
from logging_system import get_logger
from renderers.factory import get_available_renderer_backends
from validation.models import (
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
)

logger = get_logger("validation")

SUPPORTED_PRECISIONS = {"fp16", "bf16", "fp32"}
SUPPORTED_LOG_LEVELS = {
    "TRACE",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}
SUPPORTED_CLEANUP_POLICIES = {
    "keep_everything",
    "delete_uploaded_clips",
    "delete_temp_only",
    "delete_everything_except_logs",
}

REQUIRED_CSV_FIELDS = [
    "prompt",
    "start_image",
    "duration",
    "resolution",
    "aspect_ratio",
    "seed",
    "guide_scale",
    "steps",
]


class BaseValidator(ABC):
    name = "base"

    def run(self, config: Config) -> ValidationReport:
        started = datetime.now()
        start_time = time.perf_counter()
        findings = list(self.validate(config))
        report = ValidationReport(
            validator_name=self.name,
            findings=findings,
            started_at=started.isoformat(),
            completed_at=datetime.now().isoformat(),
            execution_time_seconds=time.perf_counter() - start_time,
        )
        return report

    @abstractmethod
    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        raise NotImplementedError

    def _finding(
        self,
        severity: ValidationSeverity,
        message: str,
        *,
        recommendation: str = "",
        blocking: bool = False,
        details: Dict[str, Any] | None = None,
    ) -> ValidationFinding:
        return ValidationFinding(
            validator_name=self.name,
            severity=severity,
            message=message,
            recommendation=recommendation,
            blocking=blocking,
            details=details or {},
        )


def _resolve_image_path(image_path: Path, reference_dir: Path) -> Path:
    full_path = reference_dir / image_path
    if full_path.exists():
        return full_path.resolve(strict=False)
    return image_path.resolve(strict=False)


def _load_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


class ConfigValidator(BaseValidator):
    name = "ConfigValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        if config.config_version != "1.0":
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Unsupported config_version: {config.config_version}",
                recommendation="Migrate the configuration file to version 1.0 before running.",
                blocking=True,
            )
        if config.profile not in CONFIG_PROFILES:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Unknown profile: {config.profile}",
                recommendation="Select one of the predefined configuration profiles.",
                blocking=True,
            )
        if config.renderer_backend not in {"auto", *get_available_renderer_backends()}:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Unknown renderer backend: {config.renderer_backend}",
                recommendation="Select a registered renderer backend or use the auto profile.",
                blocking=True,
            )
        if not config.jobs_csv_path.exists():
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Jobs CSV not found: {config.jobs_csv_path}",
                recommendation="Upload or mount the jobs.csv file and update jobs_csv_path.",
                blocking=True,
            )
        if not config.reference_images_dir.exists():
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Reference image directory not found: {config.reference_images_dir}",
                recommendation="Mount the image dataset and update reference_images_dir.",
                blocking=True,
            )
        if not config.model_name:
            yield self._finding(
                ValidationSeverity.FAIL,
                "model_name must not be empty",
                recommendation="Provide a valid model_name or select a backend-specific profile.",
                blocking=True,
            )
        if config.precision.lower() not in SUPPORTED_PRECISIONS:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Unsupported precision: {config.precision}",
                recommendation="Use fp16, bf16, or fp32 precision.",
                blocking=True,
            )
        if config.use_cuda and config.use_mps:
            yield self._finding(
                ValidationSeverity.FAIL,
                "use_cuda and use_mps cannot both be enabled",
                recommendation="Select a single accelerator mode in the configuration.",
                blocking=True,
            )
        if config.frame_rate <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "frame_rate must be greater than zero",
                recommendation="Set frame_rate to a positive integer.",
                blocking=True,
            )
        if config.output_quality < 0 or config.output_quality > 10:
            yield self._finding(
                ValidationSeverity.FAIL,
                "output_quality must be between 0 and 10",
                recommendation="Use an output_quality value in the supported encoder range.",
                blocking=True,
            )
        if not config.output_codec:
            yield self._finding(
                ValidationSeverity.FAIL,
                "output_codec must not be empty",
                recommendation="Select a valid ffmpeg codec such as libx264.",
                blocking=True,
            )
        if not config.output_container:
            yield self._finding(
                ValidationSeverity.FAIL,
                "output_container must not be empty",
                recommendation="Select a valid output container such as mp4.",
                blocking=True,
            )
        if config.guidance_scale < 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "guidance_scale must be zero or greater",
                recommendation="Use a non-negative guidance scale.",
                blocking=True,
            )
        if config.num_inference_steps <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "num_inference_steps must be greater than zero",
                recommendation="Use a positive num_inference_steps value.",
                blocking=True,
            )
        if config.resolution_alignment <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "resolution_alignment must be greater than zero",
                recommendation="Use a positive output alignment value.",
                blocking=True,
            )
        if config.expected_duration_tolerance_seconds < 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "expected_duration_tolerance_seconds must be zero or greater",
                recommendation="Use a non-negative duration tolerance.",
                blocking=True,
            )
        if config.min_output_size_bytes < 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "min_output_size_bytes must be zero or greater",
                recommendation="Use a non-negative minimum output size threshold.",
                blocking=True,
            )
        if config.warmup_steps < 0 or config.warmup_num_frames < 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "Warmup settings must be zero or greater",
                recommendation="Use non-negative warmup_steps and warmup_num_frames values.",
                blocking=True,
            )
        if config.empty_cache_interval < 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "empty_cache_interval must be zero or greater",
                recommendation="Use a non-negative cache eviction interval.",
                blocking=True,
            )
        if config.stitching_thumbnail_timestamp_seconds < 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "stitching_thumbnail_timestamp_seconds must be zero or greater",
                recommendation="Use a non-negative thumbnail timestamp.",
                blocking=True,
            )
        if config.drive_max_parallel_uploads <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "drive_max_parallel_uploads must be greater than zero",
                recommendation="Set drive_max_parallel_uploads to at least 1.",
                blocking=True,
            )
        if config.max_retries < 0 or config.drive_upload_retries < 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "Retry counts must be zero or greater",
                recommendation="Use non-negative retry values for pipeline and Drive retries.",
                blocking=True,
            )
        if config.drive_retry_base_seconds <= 0 or config.drive_retry_max_seconds <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "Drive retry delays must be greater than zero",
                recommendation="Use positive values for drive_retry_base_seconds and drive_retry_max_seconds.",
                blocking=True,
            )
        if config.drive_retry_max_seconds < config.drive_retry_base_seconds:
            yield self._finding(
                ValidationSeverity.FAIL,
                "drive_retry_max_seconds must be greater than or equal to drive_retry_base_seconds",
                recommendation="Increase the maximum Drive retry delay.",
                blocking=True,
            )
        if config.drive_request_spacing_seconds < 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "drive_request_spacing_seconds must be zero or greater",
                recommendation="Use a non-negative Drive API request spacing value.",
                blocking=True,
            )
        if config.drive_cleanup_policy not in SUPPORTED_CLEANUP_POLICIES:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Unsupported drive_cleanup_policy: {config.drive_cleanup_policy}",
                recommendation="Use one of the supported Drive cleanup policies.",
                blocking=True,
            )
        if not config.drive_project_name:
            yield self._finding(
                ValidationSeverity.FAIL,
                "drive_project_name must not be empty",
                recommendation="Provide a stable Drive project folder name.",
                blocking=True,
            )
        if not config.drive_required_subfolders:
            yield self._finding(
                ValidationSeverity.FAIL,
                "drive_required_subfolders must not be empty",
                recommendation="Configure the required Drive project folder structure.",
                blocking=True,
            )
        if config.heartbeat_interval_seconds <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "heartbeat_interval_seconds must be greater than zero",
                recommendation="Set heartbeat_interval_seconds to a positive integer.",
                blocking=True,
            )
        if config.health_poll_interval_seconds <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "health_poll_interval_seconds must be greater than zero",
                recommendation="Set health_poll_interval_seconds to a positive integer.",
                blocking=True,
            )
        if config.log_rotation_max_bytes <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "log_rotation_max_bytes must be greater than zero",
                recommendation="Set a positive maximum log file size.",
                blocking=True,
            )
        if config.log_rotation_backup_count <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "log_rotation_backup_count must be greater than zero",
                recommendation="Keep at least one rotated log backup.",
                blocking=True,
            )
        if config.log_rotation_max_age_days <= 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "log_rotation_max_age_days must be greater than zero",
                recommendation="Set a positive log retention age in days.",
                blocking=True,
            )
        if config.benchmark_max_jobs < 0:
            yield self._finding(
                ValidationSeverity.FAIL,
                "benchmark_max_jobs must be zero or greater",
                recommendation="Use zero to disable limits or a positive number to sample jobs.",
                blocking=True,
            )
        if not config.app_version:
            yield self._finding(
                ValidationSeverity.FAIL,
                "app_version must not be empty",
                recommendation="Set a semantic application version for manifests and reports.",
                blocking=True,
            )
        if config.benchmark_mode and config.benchmark_max_jobs <= 0:
            yield self._finding(
                ValidationSeverity.WARNING,
                "Benchmark mode is enabled but benchmark_max_jobs is not set",
                recommendation="Set benchmark_max_jobs to a small representative sample size.",
                blocking=False,
            )
        if config.log_level.upper() not in SUPPORTED_LOG_LEVELS:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Unsupported log level: {config.log_level}",
                recommendation="Use TRACE, DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
                blocking=True,
            )
        if config.enable_drive_upload and not (
            config.drive_credentials_path or config.extra.get("drive_credentials_json")
        ):
            yield self._finding(
                ValidationSeverity.FAIL,
                "Google Drive upload is enabled but no credentials were provided",
                recommendation="Provide drive_credentials_path or LTX_DRIVE_CREDENTIALS_JSON.",
                blocking=True,
            )
        yield self._finding(
            ValidationSeverity.PASS,
            "Configuration validation completed",
            details={"profile": config.profile, "config_version": config.config_version},
        )


class EnvironmentValidator(BaseValidator):
    name = "EnvironmentValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        if sys.version_info < (3, 11):
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Python 3.11+ is required; found {sys.version.split()[0]}",
                recommendation="Use Python 3.11 or newer for unattended production runs.",
                blocking=True,
            )
        else:
            yield self._finding(
                ValidationSeverity.PASS,
                f"Python version is supported: {sys.version.split()[0]}",
                details={"python_version": sys.version.split()[0]},
            )

        csv_parent = config.jobs_csv_path.resolve(strict=False).parent
        if not csv_parent.exists():
            yield self._finding(
                ValidationSeverity.FAIL,
                f"CSV parent directory does not exist: {csv_parent}",
                recommendation="Fix jobs_csv_path or create the project input directory.",
                blocking=True,
            )
        else:
            yield self._finding(
                ValidationSeverity.PASS,
                f"Input directory is accessible: {csv_parent}",
            )


class DependencyValidator(BaseValidator):
    name = "DependencyValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        required = ["torch", "PIL", "numpy"]
        if config.renderer_backend == "diffusers":
            required.extend(["diffusers", "transformers", "accelerate"])
        missing: list[str] = []
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        if missing:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Missing required dependencies: {', '.join(missing)}",
                recommendation="Install the missing Python packages before starting the render.",
                blocking=True,
                details={"missing_packages": missing},
            )
        else:
            yield self._finding(
                ValidationSeverity.PASS,
                "Python dependencies are available",
                details={"package_count": len(required)},
            )

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            yield self._finding(
                ValidationSeverity.FAIL,
                "ffmpeg is not available on PATH",
                recommendation="Install ffmpeg or add it to PATH before rendering.",
                blocking=True,
            )
        else:
            yield self._finding(
                ValidationSeverity.PASS,
                "ffmpeg is available",
                details={"ffmpeg_path": ffmpeg_path},
            )


class GPUValidator(BaseValidator):
    name = "GPUValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        try:
            import torch
        except ImportError:
            yield self._finding(
                ValidationSeverity.FAIL,
                "PyTorch is not installed",
                recommendation="Install torch before rendering.",
                blocking=True,
            )
            return

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            yield self._finding(
                ValidationSeverity.PASS,
                f"CUDA GPU detected: {props.name}",
                details={
                    "gpu_name": props.name,
                    "total_vram_bytes": total_bytes,
                    "free_vram_bytes": free_bytes,
                    "device_count": torch.cuda.device_count(),
                },
            )
            return

        severity = ValidationSeverity.FAIL if config.use_cuda else ValidationSeverity.WARNING
        yield self._finding(
            severity,
            "CUDA GPU is not available",
            recommendation=(
                "Attach a Kaggle GPU accelerator or disable use_cuda for CPU diagnostics only."
            ),
            blocking=config.use_cuda,
        )


class DiskValidator(BaseValidator):
    name = "DiskValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        output_root = config.output_dir.resolve(strict=False)
        output_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(output_root)
        free_gb = usage.free / (1024**3)
        severity = ValidationSeverity.PASS if free_gb >= 10 else ValidationSeverity.FAIL
        yield self._finding(
            severity,
            f"Disk space available: {free_gb:.2f} GB",
            recommendation=(
                "Free additional space or reduce benchmark size before rendering."
                if free_gb < 10
                else ""
            ),
            blocking=free_gb < 10,
            details={"free_bytes": usage.free, "total_bytes": usage.total},
        )


class ManifestValidator(BaseValidator):
    name = "ManifestValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        if not config.manifest_path.exists():
            yield self._finding(
                ValidationSeverity.PASS,
                "Manifest does not exist yet; a new manifest will be created",
            )
            return
        try:
            with config.manifest_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            jobs = data.get("jobs", [])
            yield self._finding(
                ValidationSeverity.PASS,
                "Manifest is readable",
                details={"job_count": len(jobs)},
            )
        except Exception as exc:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Manifest is not readable: {exc}",
                recommendation="Repair or remove the manifest before resuming.",
                blocking=True,
            )


class OutputValidator(BaseValidator):
    name = "OutputValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        for path in [config.output_dir, config.log_dir, config.temp_dir]:
            try:
                path.mkdir(parents=True, exist_ok=True)
                probe = path / ".write_test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                yield self._finding(
                    ValidationSeverity.PASS,
                    f"Writable path verified: {path}",
                )
            except Exception as exc:
                yield self._finding(
                    ValidationSeverity.FAIL,
                    f"Path is not writable: {path} ({exc})",
                    recommendation="Fix filesystem permissions or select a writable working directory.",
                    blocking=True,
                )


class CSVValidator(BaseValidator):
    name = "CSVValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        try:
            rows = _load_csv_rows(config.jobs_csv_path)
        except UnicodeDecodeError as exc:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"CSV encoding is invalid: {exc}",
                recommendation="Save jobs.csv as UTF-8 or UTF-8 with BOM.",
                blocking=True,
            )
            return
        except Exception as exc:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Unable to read CSV: {exc}",
                recommendation="Verify that jobs.csv exists and is readable.",
                blocking=True,
            )
            return

        if not rows:
            yield self._finding(
                ValidationSeverity.FAIL,
                "CSV contains no jobs",
                recommendation="Add at least one valid render row to jobs.csv.",
                blocking=True,
            )
            return

        row_keys = set(rows[0].keys())
        missing_fields = [field for field in REQUIRED_CSV_FIELDS if field not in row_keys]
        if missing_fields:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"CSV is missing required columns: {', '.join(missing_fields)}",
                recommendation="Update jobs.csv to include the required schema.",
                blocking=True,
            )

        duplicate_prompts = 0
        duplicate_render_keys = 0
        render_keys: set[tuple[str, str, str, str]] = set()
        for row in rows:
            prompt = (row.get("prompt") or "").strip()
            render_key = (
                prompt,
                (row.get("start_image") or "").strip(),
                (row.get("duration") or "").strip(),
                (row.get("resolution") or "").strip(),
            )
            if render_key in render_keys:
                duplicate_render_keys += 1
            render_keys.add(render_key)
            if not prompt:
                yield self._finding(
                    ValidationSeverity.FAIL,
                    "CSV contains an empty prompt",
                    recommendation="Populate every prompt cell with a non-empty prompt.",
                    blocking=True,
                )
        prompt_values = [
            (row.get("prompt") or "").strip() for row in rows if (row.get("prompt") or "").strip()
        ]
        duplicate_prompts = len(prompt_values) - len(set(prompt_values))

        if duplicate_prompts:
            yield self._finding(
                ValidationSeverity.WARNING,
                f"Duplicate prompts detected: {duplicate_prompts}",
                recommendation="Review duplicates to avoid unintentionally rendering repeated scenes.",
                details={"duplicate_prompts": duplicate_prompts},
            )
        if duplicate_render_keys:
            yield self._finding(
                ValidationSeverity.WARNING,
                f"Potential duplicate clip definitions detected: {duplicate_render_keys}",
                recommendation="Review repeated rows to avoid duplicate output names or repeated rendering.",
                details={"duplicate_render_keys": duplicate_render_keys},
            )

        invalid_rows = 0
        for row in rows:
            try:
                Duration.from_string(row["duration"])
                Resolution.from_string(row["resolution"])
                AspectRatio.from_string(row["aspect_ratio"])
                int(row["seed"])
                float(row["guide_scale"])
                int(row["steps"])
            except Exception:
                invalid_rows += 1

        if invalid_rows:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"CSV contains invalid numeric or enum values in {invalid_rows} rows",
                recommendation="Fix duration, resolution, aspect_ratio, seed, guide_scale, or steps values.",
                blocking=True,
                details={"invalid_rows": invalid_rows},
            )
        yield self._finding(
            ValidationSeverity.PASS,
            f"CSV loaded successfully with {len(rows)} jobs",
            details={"job_count": len(rows)},
        )


class ImageValidator(BaseValidator):
    name = "ImageValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        try:
            rows = _load_csv_rows(config.jobs_csv_path)
        except Exception:
            return

        missing_images = 0
        corrupted_images = 0
        validated_images = 0

        for row in rows:
            for field in ("start_image", "end_image"):
                raw_value = (row.get(field) or "").strip()
                if not raw_value:
                    if field == "end_image":
                        continue
                    yield self._finding(
                        ValidationSeverity.FAIL,
                        "A CSV row is missing start_image",
                        recommendation="Populate start_image for every job.",
                        blocking=True,
                    )
                    continue

                image_path = _resolve_image_path(Path(raw_value), config.reference_images_dir)
                if not image_path.exists():
                    missing_images += 1
                    continue
                if image_path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
                    yield self._finding(
                        ValidationSeverity.FAIL,
                        f"Unsupported image format: {image_path.name}",
                        recommendation="Use PNG, JPG, JPEG, or WEBP reference images.",
                        blocking=True,
                    )
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        width, height = image.size
                    validated_images += 1
                    if width < 64 or height < 64:
                        yield self._finding(
                            ValidationSeverity.WARNING,
                            f"Image is unusually small: {image_path.name} ({width}x{height})",
                            recommendation="Use larger source images for better video quality.",
                        )
                except (UnidentifiedImageError, OSError):
                    corrupted_images += 1

        if missing_images:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Missing images detected: {missing_images}",
                recommendation="Upload the missing reference images before rendering.",
                blocking=True,
                details={"missing_images": missing_images},
            )
        if corrupted_images:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Corrupted images detected: {corrupted_images}",
                recommendation="Replace the corrupted image files with valid PNG/JPG inputs.",
                blocking=True,
                details={"corrupted_images": corrupted_images},
            )
        yield self._finding(
            ValidationSeverity.PASS,
            f"Validated {validated_images} image references",
            details={"validated_images": validated_images},
        )


class DriveValidator(BaseValidator):
    name = "DriveValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        if not config.enable_drive_upload:
            yield self._finding(
                ValidationSeverity.WARNING,
                "Google Drive uploads are disabled",
                recommendation="Enable Drive uploads for unattended persistent storage.",
            )
            return

        try:
            from drive.gdrive import GoogleDriveClient

            client = GoogleDriveClient(config)
            client.connect()
            yield self._finding(
                ValidationSeverity.PASS,
                "Google Drive authentication succeeded",
                details={"project_name": config.drive_project_name},
            )
        except Exception as exc:
            yield self._finding(
                ValidationSeverity.FAIL,
                f"Google Drive validation failed: {exc}",
                recommendation="Verify Drive credentials, folder access, and network connectivity.",
                blocking=True,
            )


class ModelAssetValidator(BaseValidator):
    name = "ModelAssetValidator"

    def validate(self, config: Config) -> Iterable[ValidationFinding]:
        backend = config.renderer_backend
        if backend == "auto":
            backend = "wan2gp" if config.wan2gp_dir.exists() else "diffusers"

        if backend == "wan2gp":
            model_dir = (
                config.wan2gp_model_dir.resolve(strict=False)
                if config.wan2gp_model_dir
                else (config.wan2gp_dir / "models").resolve(strict=False)
            )
            transformer_path = model_dir / config.wan2gp_transformer_filename
            text_encoder_dir = model_dir / config.wan2gp_text_encoder_dirname
            if not transformer_path.exists():
                yield self._finding(
                    ValidationSeverity.FAIL,
                    f"Wan2GP transformer checkpoint not found: {transformer_path}",
                    recommendation="Mount the Wan2GP model assets dataset before rendering.",
                    blocking=True,
                )
            if not text_encoder_dir.exists():
                yield self._finding(
                    ValidationSeverity.FAIL,
                    f"Wan2GP text encoder directory not found: {text_encoder_dir}",
                    recommendation="Mount the Gemma text encoder assets before rendering.",
                    blocking=True,
                )
            yield self._finding(
                ValidationSeverity.PASS,
                "Wan2GP model asset paths validated",
                details={"model_dir": str(model_dir)},
            )
            return

        yield self._finding(
            ValidationSeverity.PASS,
            "Diffusers backend selected; model will be resolved at runtime",
            details={"model_name": config.model_name},
        )


def validate_config(config: Config) -> bool:
    report = ConfigValidator().run(config)
    failures = [finding.message for finding in report.findings if finding.blocking]
    if failures:
        raise ConfigurationError("; ".join(failures))
    return True


def validate_csv_schema(row: Dict[str, Any]) -> bool:
    missing_fields = [field for field in REQUIRED_CSV_FIELDS if field not in row]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
    return True


def validate_image(image_path: Path, reference_dir: Path) -> bool:
    full_path = _resolve_image_path(image_path, reference_dir)
    if not full_path.exists():
        raise ValidationError(f"Image not found: {image_path}")
    if not full_path.is_file():
        raise ValidationError(f"Not a file: {image_path}")
    if full_path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise ValidationError(f"Unsupported image format: {image_path}")
    try:
        with Image.open(full_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValidationError(f"Corrupted image file: {image_path}") from exc
    return True


def validate_job_data(job_data: Dict[str, Any], reference_dir: Path) -> bool:
    validate_csv_schema(job_data)
    validate_image(Path(job_data["start_image"]), reference_dir)
    if job_data.get("end_image"):
        validate_image(Path(job_data["end_image"]), reference_dir)
    try:
        Duration.from_string(job_data["duration"])
        Resolution.from_string(job_data["resolution"])
        AspectRatio.from_string(job_data["aspect_ratio"])
        int(job_data["seed"])
        float(job_data["guide_scale"])
        int(job_data["steps"])
    except Exception as exc:
        raise ValidationError("Invalid numeric or enum field in job data") from exc
    return True
