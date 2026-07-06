import tempfile
import unittest
from pathlib import Path

from cache_system import RenderCacheStore
from config import Config
from core import AspectRatio, Duration, JobStatus, Resolution
from jobs.job import Job
from stitching.service import VideoStitcher


class _FakeJobQueue:
    def __init__(self, jobs):
        self.jobs = jobs


class _FakeFFmpegService:
    def __init__(self):
        self.probe_results = {}

    def VerifyFFmpeg(self) -> None:
        return None

    def BuildConcatList(self, clip_paths, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(str(path) for path in clip_paths), encoding="utf-8")
        return destination

    def Stitch(self, concat_list_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"stitched-video")
        self.probe_results[str(output_path)] = {
            "duration_seconds": 10.0,
            "size_bytes": output_path.stat().st_size,
            "width": 640,
            "height": 480,
            "frame_rate": 24.0,
            "codec": "h264",
        }
        return output_path

    def ExtractThumbnail(self, input_path, output_path, timestamp_seconds):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"thumbnail")
        return output_path

    def GeneratePreview(self, input_path, output_path, height):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"preview")
        return output_path

    def ProbeVideo(self, input_path):
        return self.probe_results[str(input_path)]


class CacheAndStitchingTests(unittest.TestCase):
    def _make_job(self, root: Path, sequence_index: int, status: JobStatus = JobStatus.COMPLETED) -> Job:
        output_path = root / f"clip_{sequence_index:06d}.mp4"
        output_path.write_bytes(b"clip-data")
        return Job(
            job_id=f"job-{sequence_index}",
            sequence_index=sequence_index,
            prompt=f"prompt-{sequence_index}",
            start_image=root / f"start-{sequence_index}.png",
            end_image=None,
            duration=Duration.D5,
            resolution=Resolution.R480,
            aspect_ratio=AspectRatio.AR_1_1,
            seed=sequence_index,
            guidance_scale=3.0,
            num_inference_steps=8,
            status=status,
            output_path=output_path,
            output_metadata={"file_size_bytes": output_path.stat().st_size},
        )

    def test_render_cache_registers_and_validates_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            start_image = root / "start.png"
            start_image.write_bytes(b"start")
            output_path = root / "outputs" / "clip.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"video-bytes")

            config = Config(
                output_dir=root / "outputs",
                log_dir=root / "logs",
                temp_dir=root / "temp",
                cache_dir=root / "outputs" / "cache",
                cache_index_path=root / "outputs" / "cache" / "render_cache.json",
                enable_drive_upload=False,
            )
            job = Job(
                job_id="job-1",
                sequence_index=1,
                prompt="prompt",
                start_image=start_image,
                end_image=None,
                duration=Duration.D5,
                resolution=Resolution.R480,
                aspect_ratio=AspectRatio.AR_1_1,
                seed=1,
                guidance_scale=3.0,
                num_inference_steps=8,
            )
            store = RenderCacheStore(config)
            cache_key = store.build_cache_key(job)
            store.register(
                cache_key,
                output_path,
                {"checksum_sha256": "", "file_size_bytes": output_path.stat().st_size},
            )

            self.assertEqual(store.lookup(cache_key), output_path.resolve(strict=False))

    def test_video_stitcher_uses_manifest_order_and_generates_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = Config(
                output_dir=root / "outputs",
                log_dir=root / "logs",
                temp_dir=root / "temp",
                stitched_output_path=root / "outputs" / "stitched" / "final_movie.mp4",
                thumbnail_path=root / "outputs" / "previews" / "thumbnail.jpg",
                preview_480p_path=root / "outputs" / "previews" / "preview_480.mp4",
                enable_drive_upload=False,
                enable_stitching=True,
            )
            jobs = [self._make_job(root, 1), self._make_job(root, 2)]
            ffmpeg = _FakeFFmpegService()
            for job in jobs:
                ffmpeg.probe_results[str(job.output_path)] = {
                    "duration_seconds": 5.0,
                    "size_bytes": job.output_path.stat().st_size,
                    "width": 640,
                    "height": 480,
                    "frame_rate": 24.0,
                    "codec": "h264",
                }

            stitcher = VideoStitcher(config, _FakeJobQueue(jobs), ffmpeg=ffmpeg)
            result = stitcher.run()

            self.assertTrue(result.success)
            self.assertTrue(config.stitched_output_path.exists())
            self.assertTrue(config.thumbnail_path.exists())
            self.assertTrue(config.preview_480p_path.exists())


if __name__ == "__main__":
    unittest.main()
