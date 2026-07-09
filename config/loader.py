import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from config.config import Config
from config.profiles import CONFIG_PROFILES
from core import ConfigurationError

ENV_PREFIX = "LTX_"
CURRENT_CONFIG_VERSION = "1.0"
PATH_FIELD_NAMES = {
    "jobs_csv_path",
    "reference_images_dir",
    "output_dir",
    "log_dir",
    "temp_dir",
    "manifest_path",
    "report_path",
    "heartbeat_path",
    "diagnostics_path",
    "validation_report_path",
    "performance_report_path",
    "summary_path",
    "project_report_csv_path",
    "benchmark_history_path",
    "benchmark_json_path",
    "benchmark_csv_path",
    "performance_summary_path",
    "cache_dir",
    "cache_index_path",
    "stitched_output_path",
    "thumbnail_path",
    "preview_480p_path",
    "preview_720p_path",
    "project_config_path",
    "drive_credentials_path",
    "wan2gp_dir",
    "wan2gp_model_dir",
}


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigurationError(
            "PyYAML is required to load configuration files"
        ) from exc

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration file must contain a mapping: {path}")
    return data


def _flatten_config_sections(data: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict) and key in {
            "paths",
            "model",
            "repository",
            "drive",
            "pipeline",
            "logging",
            "observability",
            "reports",
            "execution",
        }:
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
            and key in {"extra", "features"}
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_env_value(value: str) -> Any:
    normalized = value.strip()
    if normalized.lower() in {"true", "false"}:
        return normalized.lower() == "true"
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        pass
    try:
        if "." in normalized:
            return float(normalized)
        return int(normalized)
    except ValueError:
        return normalized


def _load_env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        config_key = key[len(ENV_PREFIX) :].lower()
        overrides[config_key] = _parse_env_value(value)
    return overrides


def _normalize_paths(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    for key in PATH_FIELD_NAMES:
        if key in normalized and normalized[key] is not None:
            normalized[key] = Path(normalized[key])
    return normalized


def _migrate_config(data: dict[str, Any]) -> dict[str, Any]:
    version = str(data.get("config_version") or CURRENT_CONFIG_VERSION)
    if version != CURRENT_CONFIG_VERSION:
        raise ConfigurationError(
            f"Unsupported config version: {version}. Expected {CURRENT_CONFIG_VERSION}"
        )
    data["config_version"] = CURRENT_CONFIG_VERSION
    return data


def _apply_profile(base: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profile_key = (profile_name or "balanced").lower()
    if profile_key not in CONFIG_PROFILES:
        available = ", ".join(sorted(CONFIG_PROFILES))
        raise ConfigurationError(
            f"Unknown config profile '{profile_name}'. Available profiles: {available}"
        )
    profiled = _deep_merge(base, CONFIG_PROFILES[profile_key])
    profiled["profile"] = profile_key
    return profiled


def _non_none_values(data: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not data:
        return {}
    return {key: value for key, value in data.items() if value is not None}


def _finalize_runtime_fields(config: Config) -> Config:
    if not config.project_id:
        config.project_id = uuid.uuid4().hex[:12]
    if not config.run_id:
        config.run_id = uuid.uuid4().hex[:12]
    if not config.correlation_id:
        config.correlation_id = config.run_id

    config.output_dir = config.output_dir.resolve(strict=False)
    config.log_dir = config.log_dir.resolve(strict=False)
    config.temp_dir = config.temp_dir.resolve(strict=False)
    config.jobs_csv_path = config.jobs_csv_path.resolve(strict=False)
    config.reference_images_dir = config.reference_images_dir.resolve(strict=False)
    config.manifest_path = config.manifest_path.resolve(strict=False)
    config.report_path = config.report_path.resolve(strict=False)

    artifact_defaults = {
        "heartbeat_path": config.output_dir / "heartbeat.json",
        "diagnostics_path": config.output_dir / "diagnostics.json",
        "validation_report_path": config.output_dir / "validation_report.json",
        "performance_report_path": config.output_dir / "performance.json",
        "summary_path": config.output_dir / "summary.txt",
        "project_report_csv_path": config.output_dir / "project_report.csv",
        "benchmark_history_path": config.output_dir / "benchmark_history.json",
        "benchmark_json_path": config.output_dir / "benchmark.json",
        "benchmark_csv_path": config.output_dir / "benchmark.csv",
        "performance_summary_path": config.output_dir / "performance_summary.txt",
        "cache_dir": config.output_dir / "cache",
        "cache_index_path": config.output_dir / "cache" / "render_cache.json",
        "stitched_output_path": config.output_dir / "stitched" / "final_movie.mp4",
        "thumbnail_path": config.output_dir / "previews" / "thumbnail.jpg",
        "preview_480p_path": config.output_dir / "previews" / "preview_480p.mp4",
        "preview_720p_path": config.output_dir / "previews" / "preview_720p.mp4",
    }
    for field_name, default_value in artifact_defaults.items():
        current_value = getattr(config, field_name)
        if not current_value:
            setattr(config, field_name, default_value)
        else:
            setattr(
                config,
                field_name,
                Path(current_value).resolve(strict=False),
            )
    return config


def load_config(
    *,
    default_config_path: Optional[Path] = None,
    project_config_path: Optional[Path] = None,
    env_overrides: Optional[dict[str, Any]] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
    runtime_overrides: Optional[dict[str, Any]] = None,
) -> Config:
    builtin_defaults = Config().to_dict()
    default_config = _flatten_config_sections(
        _load_yaml_file(default_config_path or Path(__file__).with_name("default.yaml"))
    )
    project_config = _flatten_config_sections(
        _load_yaml_file(project_config_path) if project_config_path else {}
    )
    env_config = _non_none_values(env_overrides or _load_env_overrides())
    cli_config = _non_none_values(cli_overrides)
    runtime_config = _non_none_values(runtime_overrides)

    final_profile = (
        runtime_config.get("profile")
        or cli_config.get("profile")
        or env_config.get("profile")
        or project_config.get("profile")
        or default_config.get("profile")
        or builtin_defaults.get("profile")
        or "balanced"
    )

    merged = dict(builtin_defaults)
    merged = _apply_profile(merged, str(final_profile))
    merged = _deep_merge(merged, default_config)
    merged = _deep_merge(merged, project_config)
    merged = _deep_merge(merged, env_config)
    merged = _deep_merge(merged, cli_config)
    merged = _deep_merge(merged, runtime_config)

    if project_config_path is not None:
        merged["project_config_path"] = project_config_path

    merged = _normalize_paths(_migrate_config(merged))
    config = Config.from_dict(merged)
    return _finalize_runtime_fields(config)
