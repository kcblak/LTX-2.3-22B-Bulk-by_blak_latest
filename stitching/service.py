from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import Config
from core import JobStatus, StorageError
from jobs.job import Job
from jobs.job_queue import JobQueue
from logging_system import get_logger
from stitching.ffmpeg_wrapper import FFmpegService

logger = get_logger("stitching")

SUCCESS_STATUSES = {JobStatus.COMPLETED, JobStatus.UPLOADED}


@dataclass
class StitchingResult:
    success: bool
    message: str
    stitched_output_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    previews: dict[str, str] = field(default_factory=dict)
    clip_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "stitched_output_path": str(self.stitched_output_path) if self.stitched_output_path else None,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "previews": self.previews,
            "clip_count": self.clip_count,
            "details": self.details,
        }


class VideoStitcher:
    def __init__(
        self,
        config: Config,
        job_queue: JobQueue,
        ffmpeg: Optional[FFmpegService] = None,
        drive_client=None,
        drive_project_paths=None,
    ):
        self.config = config
        self.job_queue = job_queue
        self.ffmpeg = ffmpeg or FFmpegService(config)
        self.drive_client = drive_client
        self.drive_project_paths = drive_project_paths

    def run(self) -> StitchingResult:
        self.ffmpeg.VerifyFFmpeg()
        jobs = self._discover_jobs()
        if not jobs:
            return StitchingResult(
                success=False,
                message="No verified clips available for stitching",
            )
        discovery = self._verify_jobs(jobs)
        if discovery["blocking_failures"]:
            return StitchingResult(
                success=False,
                message="Clip integrity checks failed before stitching",
                clip_count=len(jobs),
                details=discovery,
            )

        clip_paths = [Path(item["local_path"]) for item in discovery["ordered_clips"]]
        concat_list_path = self.config.temp_dir / "stitch_concat.txt"
        self.ffmpeg.BuildConcatList(clip_paths, concat_list_path)
        stitched_output = self.ffmpeg.Stitch(concat_list_path, self.config.stitched_output_path)
        stitched_metadata = self.ffmpeg.ProbeVideo(stitched_output)
        self._validate_final_movie(stitched_output, stitched_metadata, jobs)

        thumbnail = self.ffmpeg.ExtractThumbnail(
            stitched_output,
            self.config.thumbnail_path,
            self.config.stitching_thumbnail_timestamp_seconds,
        )
        previews: dict[str, str] = {}
        if self.config.generate_preview_480p:
            previews["480p"] = str(
                self.ffmpeg.GeneratePreview(stitched_output, self.config.preview_480p_path, 480)
            )
        if self.config.generate_preview_720p:
            previews["720p"] = str(
                self.ffmpeg.GeneratePreview(stitched_output, self.config.preview_720p_path, 720)
            )

        result = StitchingResult(
            success=True,
            message="Final movie stitched successfully",
            stitched_output_path=stitched_output,
            thumbnail_path=thumbnail,
            previews=previews,
            clip_count=len(clip_paths),
            details={
                **discovery,
                "stitched_metadata": stitched_metadata,
                "completed_at": datetime.now().isoformat(),
            },
        )
        self.config.extra["stitching"] = result.to_dict()
        return result

    def _discover_jobs(self) -> list[Job]:
        jobs = [
            job
            for job in sorted(self.job_queue.jobs, key=lambda item: item.sequence_index)
            if job.status in SUCCESS_STATUSES and job.output_path is not None
        ]
        return jobs

    def _verify_jobs(self, jobs: list[Job]) -> dict[str, Any]:
        ordered_clips: list[dict[str, Any]] = []
        blocking_failures: list[str] = []
        warnings: list[str] = []

        expected_indices = list(range(1, len(jobs) + 1))
        actual_indices = [job.sequence_index for job in jobs]
        if self.config.stitch_require_contiguous_success and actual_indices != expected_indices:
            blocking_failures.append(
                "Completed clips are not contiguous from the start of the manifest"
            )

        seen_paths: set[str] = set()
        reference_resolution = None
        reference_frame_rate = None
        remote_clip_names = self._list_remote_clip_names()

        for job in jobs:
            if job.output_path is None:
                blocking_failures.append(f"Missing local output path for job {job.job_id}")
                continue
            output_path = job.output_path.resolve(strict=False)
            if str(output_path) in seen_paths:
                blocking_failures.append(f"Duplicate clip path detected: {output_path.name}")
                continue
            seen_paths.add(str(output_path))
            if not output_path.exists():
                if output_path.name in remote_clip_names:
                    blocking_failures.append(
                        f"Clip {output_path.name} exists remotely but is missing locally"
                    )
                else:
                    blocking_failures.append(f"Clip {output_path.name} is missing")
                continue

            metadata = self.ffmpeg.ProbeVideo(output_path)
            if reference_resolution is None:
                reference_resolution = (metadata["width"], metadata["height"])
                reference_frame_rate = metadata["frame_rate"]
            else:
                if reference_resolution != (metadata["width"], metadata["height"]):
                    blocking_failures.append(
                        f"Resolution mismatch for {output_path.name}: "
                        f"{metadata['width']}x{metadata['height']}"
                    )
                if abs(reference_frame_rate - metadata["frame_rate"]) > 0.01:
                    blocking_failures.append(
                        f"Frame rate mismatch for {output_path.name}: {metadata['frame_rate']}"
                    )
            expected_duration = float(job.duration.seconds)
            if abs(metadata["duration_seconds"] - expected_duration) > 1.0:
                warnings.append(
                    f"Duration deviation for {output_path.name}: expected {expected_duration}s "
                    f"got {metadata['duration_seconds']:.2f}s"
                )

            ordered_clips.append(
                {
                    "job_id": job.job_id,
                    "sequence_index": job.sequence_index,
                    "local_path": str(output_path),
                    "remote_present": output_path.name in remote_clip_names,
                    "duration_seconds": metadata["duration_seconds"],
                    "frame_rate": metadata["frame_rate"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                }
            )

        return {
            "ordered_clips": ordered_clips,
            "blocking_failures": blocking_failures,
            "warnings": warnings,
        }

    def _list_remote_clip_names(self) -> set[str]:
        if self.drive_client is None or self.drive_project_paths is None:
            return set()
        try:
            folder_id = self.drive_project_paths.folders.get("clips")
            if not folder_id:
                return set()
            return {item.name for item in self.drive_client.list_directory(folder_id)}
        except Exception as exc:
            logger.warning(
                f"Remote clip discovery skipped: {exc}",
                extra={"job_id": "N/A"},
            )
            return set()

    def _validate_final_movie(
        self,
        output_path: Path,
        metadata: dict[str, Any],
        jobs: list[Job],
    ) -> None:
        if not output_path.exists():
            raise StorageError("Final stitched movie was not created")
        if metadata["size_bytes"] <= 0:
            raise StorageError("Final stitched movie has an invalid file size")
        if metadata["width"] <= 0 or metadata["height"] <= 0:
            raise StorageError("Final stitched movie has invalid dimensions")
        expected_duration = float(sum(job.duration.seconds for job in jobs))
        if metadata["duration_seconds"] <= 0:
            raise StorageError("Final stitched movie is not playable")
        if abs(metadata["duration_seconds"] - expected_duration) > max(2.0, len(jobs) * 0.5):
            raise StorageError(
                "Final stitched movie duration does not match the expected clip total"
            )
