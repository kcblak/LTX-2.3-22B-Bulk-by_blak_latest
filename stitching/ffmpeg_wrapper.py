import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from config import Config
from core import StorageError


class FFmpegService:
    def __init__(self, config: Config):
        self.config = config

    def VerifyFFmpeg(self) -> None:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise StorageError("ffmpeg and ffprobe must be available on PATH for stitching")

    def BuildConcatList(self, clip_paths: list[Path], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for clip_path in clip_paths:
            escaped = str(clip_path.resolve(strict=False)).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination

    def Stitch(self, concat_list_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise StorageError(f"FFmpeg stitching failed: {result.stderr.strip()}")
        return output_path

    def ExtractThumbnail(self, input_path: Path, output_path: Path, timestamp_seconds: int) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(max(0, timestamp_seconds)),
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise StorageError(f"Thumbnail extraction failed: {result.stderr.strip()}")
        return output_path

    def GeneratePreview(self, input_path: Path, output_path: Path, height: int) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            f"scale=-2:{height}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "28",
            "-an",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise StorageError(f"Preview generation failed: {result.stderr.strip()}")
        return output_path

    def ProbeVideo(self, input_path: Path) -> dict[str, Any]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(input_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise StorageError(f"ffprobe failed: {result.stderr.strip()}")
        data = json.loads(result.stdout or "{}")
        video_stream = next(
            (stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"),
            {},
        )
        frame_rate = video_stream.get("avg_frame_rate", "0/1")
        if "/" in frame_rate:
            numerator, denominator = frame_rate.split("/", 1)
            try:
                frame_rate_value = float(numerator) / max(float(denominator), 1.0)
            except ValueError:
                frame_rate_value = 0.0
        else:
            try:
                frame_rate_value = float(frame_rate)
            except ValueError:
                frame_rate_value = 0.0
        return {
            "duration_seconds": float(data.get("format", {}).get("duration", 0.0) or 0.0),
            "size_bytes": int(data.get("format", {}).get("size", 0) or 0),
            "width": int(video_stream.get("width", 0) or 0),
            "height": int(video_stream.get("height", 0) or 0),
            "frame_rate": frame_rate_value,
            "codec": video_stream.get("codec_name"),
        }
