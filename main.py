#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.core.models import Config
from src.workers.pipeline_orchestrator import PipelineOrchestrator
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="LTX Video Bulk Renderer")
    parser.add_argument(
        "--jobs-csv",
        type=Path,
        default=Path("jobs.csv"),
        help="Path to jobs.csv file",
    )
    parser.add_argument(
        "--reference-images",
        type=Path,
        default=Path("reference_images"),
        help="Directory containing reference images",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Directory for outputs",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path("logs"),
        help="Directory for logs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/manifest.json"),
        help="Path to manifest.json file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/report.json"),
        help="Path to report.json file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Lightricks/LTX-Video-2.3-22B-Ref-Distilled-1.1",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--enable-gdrive",
        action="store_true",
        help="Enable Google Drive upload",
    )
    parser.add_argument(
        "--gdrive-credentials",
        type=Path,
        default=Path("gdrive_credentials.json"),
        help="Path to Google Drive credentials JSON file",
    )
    parser.add_argument(
        "--gdrive-folder-id",
        type=str,
        default=None,
        help="Google Drive folder ID to upload files to",
    )

    args = parser.parse_args()

    config = Config(
        jobs_csv_path=args.jobs_csv,
        reference_images_dir=args.reference_images,
        outputs_dir=args.outputs,
        logs_dir=args.logs,
        manifest_path=args.manifest,
        report_path=args.report,
        model_name=args.model_name,
        enable_gdrive_upload=args.enable_gdrive,
        gdrive_credentials_path=args.gdrive_credentials,
        gdrive_folder_id=args.gdrive_folder_id,
    )

    logger = setup_logger("main", config.logs_dir)
    logger.info("Starting LTX Video Bulk Renderer")

    orchestrator = PipelineOrchestrator(config)
    report = orchestrator.run()

    logger.info("Rendering complete!")
    print("Report summary:")
    print(report["summary"])


if __name__ == "__main__":
    main()
