#!/usr/bin/env python3
"""Standalone diagnostics entry point."""

import argparse
import sys
from pathlib import Path

from config import load_config
from diagnostics import DiagnosticsRunner
from logging_system import get_logger, setup_logging


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone environment and project diagnostics for LTX Bulk Renderer"
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=None,
        help="Optional project-specific YAML configuration file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for diagnostics.json",
    )
    parser.add_argument(
        "--network-check",
        action="store_true",
        help="Enable optional internet connectivity verification",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = load_config(
        project_config_path=args.project_config,
        runtime_overrides={
            "diagnostics_network_check": args.network_check,
            "diagnostics_path": args.output_path,
        },
    )
    setup_logging(config)
    logger = get_logger("diagnostics")

    runner = DiagnosticsRunner(config)
    result = runner.run()
    runner.save(result, config.diagnostics_path)

    logger.info("Diagnostics status: %s", result.status, extra={"job_id": "N/A"})
    for report in result.reports:
        report_status = "FAIL" if report.has_blocking_failures else "WARNING" if report.has_warnings else "PASS"
        logger.info(
            "%s -> %s",
            report.validator_name,
            report_status,
            extra={"job_id": "N/A"},
        )
    return 0 if result.status != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
