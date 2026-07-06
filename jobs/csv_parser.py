import csv
from pathlib import Path
from typing import List
from core import Duration, Resolution, AspectRatio, JobStatus
from jobs.job import Job
from utils import generate_job_id
from validation import validate_job_data
from logging_system import get_logger

logger = get_logger("jobs.csv_parser")


def parse_jobs_from_csv(csv_path: Path, reference_images_dir: Path) -> List[Job]:
    """Parse jobs from a CSV file."""
    jobs = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            try:
                validate_job_data(row, reference_images_dir)
                # Resolve paths
                start_image = Path(row["start_image"])
                full_start = reference_images_dir / start_image
                if not full_start.exists():
                    start_image = full_start

                end_image = None
                if row.get("end_image"):
                    end_image = Path(row["end_image"])
                    full_end = reference_images_dir / end_image
                    if full_end.exists():
                        end_image = full_end

                job_data = {k: v for k, v in row.items()}
                job_id = generate_job_id(job_data)

                job = Job(
                    job_id=job_id,
                    sequence_index=idx + 1,
                    prompt=row["prompt"],
                    start_image=start_image,
                    end_image=end_image,
                    duration=Duration.from_string(row["duration"]),
                    resolution=Resolution.from_string(row["resolution"]),
                    aspect_ratio=AspectRatio.from_string(row["aspect_ratio"]),
                    seed=int(row["seed"]),
                    guidance_scale=float(row["guide_scale"]),
                    num_inference_steps=int(row["steps"]),
                )
                jobs.append(job)
                logger.debug(f"Parsed job {job_id}", extra={"job_id": job_id})
            except Exception as e:
                logger.error(f"Failed to parse row {idx}: {e}", extra={"job_id": "N/A"})
                raise
    return jobs
