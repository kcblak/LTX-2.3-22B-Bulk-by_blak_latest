#!/usr/bin/env python3
"""
LTX Video Bulk Renderer - Main Entry Point
Production-grade unattended bulk image-to-video rendering
"""

import argparse
import json
import sys
from typing import Any
from pathlib import Path

from config import load_config
from core import APP_VERSION
from orchestration.runner import ApplicationRunner
from renderers.factory import get_available_renderer_backends


def _parse_override_value(value: str) -> Any:
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


def _parse_runtime_overrides(values: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid runtime override '{item}'. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        overrides[key.strip()] = _parse_override_value(raw_value)
    return overrides


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "jobs_csv_path": args.jobs_csv,
        "reference_images_dir": args.reference_images_dir,
        "output_dir": args.output_dir,
        "log_dir": args.log_dir,
        "temp_dir": args.temp_dir,
        "manifest_path": args.manifest_path,
        "report_path": args.report_path,
        "renderer_backend": args.renderer_backend,
        "model_name": args.model_name,
        "wan2gp_dir": args.wan2gp_dir,
        "wan2gp_model_dir": args.wan2gp_model_dir,
        "enable_drive_upload": args.enable_drive_upload,
        "drive_credentials_path": args.drive_credentials_path,
        "drive_folder_id": args.drive_folder_id,
        "drive_project_name": args.drive_project_name,
        "drive_max_parallel_uploads": args.drive_max_parallel_uploads,
        "drive_cleanup_policy": args.drive_cleanup_policy,
        "resume_enabled": args.resume_enabled,
        "log_level": args.log_level,
        "profile": args.profile,
        "heartbeat_interval_seconds": args.heartbeat_interval_seconds,
        "health_poll_interval_seconds": args.health_poll_interval_seconds,
        "benchmark_mode": args.benchmark_mode,
        "benchmark_max_jobs": args.benchmark_max_jobs,
        "enable_stitching": args.enable_stitching,
        "cache_enabled": args.cache_enabled,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    renderer_backend_choices = ["auto", *get_available_renderer_backends()]
    parser = argparse.ArgumentParser(
        description="LTX Video Bulk Renderer - Production-grade unattended bulk image-to-video rendering"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APP_VERSION}",
    )
    parser.add_argument(
        "--jobs-csv",
        type=Path,
        default=None,
        help="Path to jobs.csv file"
    )
    parser.add_argument(
        "--reference-images-dir",
        type=Path,
        default=None,
        help="Directory containing reference images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for logs"
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Directory for temporary files"
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Path to manifest.json file (default: output_dir/manifest.json)"
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path to report.json file (default: output_dir/report.json)"
    )
    parser.add_argument(
        "--renderer-backend",
        type=str,
        default=None,
        choices=renderer_backend_choices,
        help="Renderer backend to use"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the model to use"
    )
    parser.add_argument(
        "--wan2gp-dir",
        type=Path,
        default=None,
        help="Path to the Wan2GP runtime directory"
    )
    parser.add_argument(
        "--wan2gp-model-dir",
        type=Path,
        default=None,
        help="Path to the Wan2GP models directory"
    )
    parser.add_argument(
        "--enable-drive-upload",
        action="store_true",
        default=None,
        help="Enable Google Drive upload"
    )
    parser.add_argument(
        "--disable-drive-upload",
        action="store_false",
        dest="enable_drive_upload",
        help="Disable Google Drive upload"
    )
    parser.add_argument(
        "--drive-credentials-path",
        type=Path,
        default=None,
        help="Path to Google Drive credentials JSON file"
    )
    parser.add_argument(
        "--drive-folder-id",
        type=str,
        default=None,
        help="Google Drive folder ID to upload files to"
    )
    parser.add_argument(
        "--drive-project-name",
        type=str,
        default=None,
        help="Google Drive project folder name"
    )
    parser.add_argument(
        "--drive-max-parallel-uploads",
        type=int,
        default=None,
        help="Maximum number of concurrent upload workers"
    )
    parser.add_argument(
        "--drive-cleanup-policy",
        type=str,
        default=None,
        choices=[
            "keep_everything",
            "delete_uploaded_clips",
            "delete_temp_only",
            "delete_everything_except_logs",
        ],
        help="Cleanup policy to apply after successful upload"
    )
    parser.add_argument(
        "--resume-enabled",
        action="store_true",
        default=None,
        help="Enable resuming from manifest"
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume_enabled",
        help="Disable resuming from manifest"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=None,
        help="Optional project-specific YAML configuration file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Named configuration profile to apply",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=int,
        default=None,
        help="Seconds between heartbeat file updates",
    )
    parser.add_argument(
        "--health-poll-interval-seconds",
        type=int,
        default=None,
        help="Seconds between runtime health samples",
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        default=None,
        help="Enable benchmark mode",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_false",
        dest="benchmark_mode",
        help="Disable benchmark mode",
    )
    parser.add_argument(
        "--benchmark-max-jobs",
        type=int,
        default=None,
        help="Limit benchmark mode to the first N jobs",
    )
    parser.add_argument(
        "--enable-stitching",
        action="store_true",
        default=None,
        help="Enable post-render FFmpeg stitching",
    )
    parser.add_argument(
        "--disable-stitching",
        action="store_false",
        dest="enable_stitching",
        help="Disable post-render FFmpeg stitching",
    )
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        default=None,
        help="Enable persistent verified render cache reuse",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_false",
        dest="cache_enabled",
        help="Disable persistent verified render cache reuse",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run diagnostics and preflight only, then exit",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply a runtime override using KEY=VALUE",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        runtime_overrides = _parse_runtime_overrides(args.set)
        config = load_config(
            project_config_path=args.project_config,
            cli_overrides=_build_cli_overrides(args),
            runtime_overrides=runtime_overrides,
        )
    except Exception as exc:
        print(f"Configuration failed: {exc}", file=sys.stderr)
        return 1

    runner = ApplicationRunner(config)
    result = runner.run(preflight_only=args.preflight_only)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
