import tempfile
import unittest
from pathlib import Path

from config import Config
from core import DriveProjectPaths, Duration, RemoteFileMetadata, Resolution, AspectRatio
from drive.sync_engine import DriveSyncEngine
from jobs.job import Job


class _FakeJobQueue:
    def __init__(self, jobs):
        self.jobs = jobs

    def get_job(self, job_id):
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None

    def update_job(self, _job):
        return None


class _FakeDriveClient:
    def __init__(self):
        self.remote_by_name = {}
        self.uploaded = []
        self.project = DriveProjectPaths(
            root_folder_id="root",
            project_folder_id="project",
            folders={"clips": "clips", "reports": "reports", "manifests": "manifests", "logs": "logs", "input": "input", "config": "config"},
        )

    def connect(self):
        return True

    def ensure_project_structure(self, _project_name):
        return self.project

    def find_file_by_name(self, name, folder_id):
        return self.remote_by_name.get((folder_id, name))

    def upload_file(self, local_path, remote_name, folder_id=None, mime_type=None):
        metadata = RemoteFileMetadata(
            file_id="remote-1",
            name=remote_name,
            folder_id=folder_id,
            size_bytes=local_path.stat().st_size,
            md5_checksum="md5",
        )
        self.remote_by_name[(folder_id, remote_name)] = metadata
        self.uploaded.append((local_path, remote_name, folder_id))
        return metadata

    def verify_upload(self, file_id, local_size_bytes=None, local_md5=None):
        return RemoteFileMetadata(
            file_id=file_id,
            name="clip.mp4",
            folder_id="clips",
            size_bytes=local_size_bytes or 0,
            md5_checksum=local_md5,
        )

    def list_directory(self, folder_id):
        return []

    def get_metadata(self, file_id):
        return None

    def create_folder(self, name, parent_id=None):
        return f"{parent_id or 'root'}:{name}"

    def download_file(self, file_id, destination_path):
        return destination_path

    def delete_remote_file(self, file_id):
        return None


class DriveSyncEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_path = self.root / "clip.mp4"
        self.output_path.write_bytes(b"x" * 2048)
        self.job = Job(
            job_id="job-1",
            prompt="prompt",
            start_image=self.root / "img.png",
            end_image=None,
            duration=Duration.D5,
            resolution=Resolution.R480,
            aspect_ratio=AspectRatio.AR_1_1,
            seed=1,
            guidance_scale=3.0,
            num_inference_steps=4,
            output_path=self.output_path,
            output_metadata={"file_size_bytes": 2048},
        )
        self.config = Config(
            output_dir=self.root / "outputs",
            temp_dir=self.root / "temp",
            log_dir=self.root / "logs",
            drive_cleanup_policy="keep_everything",
        )
        self.queue = _FakeJobQueue([self.job])
        self.client = _FakeDriveClient()
        self.engine = DriveSyncEngine(self.config, self.client, self.queue)
        self.engine.connect()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_duplicate_upload_is_skipped(self):
        task_md5 = "md5"
        self.client.remote_by_name[("clips", self.output_path.name)] = RemoteFileMetadata(
            file_id="existing",
            name=self.output_path.name,
            folder_id="clips",
            size_bytes=self.output_path.stat().st_size,
            md5_checksum=task_md5,
        )
        self.job.remote_metadata["local_md5"] = task_md5

        self.engine.enqueue_job(self.job)
        task = self.engine.upload_queue.get_nowait()
        task.local_md5 = task_md5
        self.engine._process_upload_task(task)

        self.assertEqual(self.job.status.name, "COMPLETED")
        self.assertEqual(len(self.client.uploaded), 0)

    def test_successful_upload_marks_job_complete(self):
        self.engine.enqueue_job(self.job)
        task = self.engine.upload_queue.get_nowait()
        task.local_md5 = "md5"
        self.engine._process_upload_task(task)

        self.assertEqual(self.job.status.name, "COMPLETED")
        self.assertEqual(self.job.remote_metadata["file_id"], "remote-1")
        self.assertEqual(len(self.client.uploaded), 1)

    def test_cleanup_is_deferred_when_stitching_is_enabled(self):
        self.config.enable_stitching = True
        self.config.drive_cleanup_policy = "delete_uploaded_clips"
        self.engine.enqueue_job(self.job)
        task = self.engine.upload_queue.get_nowait()
        task.local_md5 = "md5"
        self.engine._process_upload_task(task)

        self.assertTrue(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
