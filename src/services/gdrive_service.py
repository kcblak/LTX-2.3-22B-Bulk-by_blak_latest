from pathlib import Path
from typing import Optional

from ..core.models import Config, Job
from ..utils.logger import setup_logger


class GDriveService:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__, config.logs_dir)
        self.service = None

    def authenticate(self):
        if not self.config.enable_gdrive_upload:
            self.logger.info("Google Drive upload disabled")
            return

        self.logger.info("Authenticating with Google Drive...")
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google.auth.transport.requests import Request
            import os

            creds = None
            token_path = self.config.gdrive_credentials_path
            if token_path and token_path.exists():
                creds = Credentials.from_authorized_user_file(str(token_path))

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    raise Exception("No valid credentials available")

            self.service = build("drive", "v3", credentials=creds)
            self.logger.info("Google Drive authentication successful")
        except Exception as e:
            self.logger.error(f"Failed to authenticate with Google Drive: {e}")
            raise

    def upload_file(self, file_path: Path, file_name: Optional[str] = None) -> Optional[str]:
        if not self.config.enable_gdrive_upload:
            return None

        if not self.service:
            self.authenticate()

        if not file_name:
            file_name = file_path.name

        self.logger.info(f"Uploading {file_path} to Google Drive...")
        try:
            from googleapiclient.http import MediaFileUpload

            file_metadata = {"name": file_name}
            if self.config.gdrive_folder_id:
                file_metadata["parents"] = [self.config.gdrive_folder_id]

            media = MediaFileUpload(
                str(file_path),
                mimetype="video/mp4",
                resumable=True,
            )

            file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )

            file_id = file.get("id")
            self.logger.info(f"File uploaded successfully! ID: {file_id}")
            return file_id
        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            return None

    def upload_job_output(self, job: Job) -> bool:
        if not job.output_path or not job.output_path.exists():
            self.logger.error(f"No valid output path for job {job.job_id}")
            return False

        file_id = self.upload_file(job.output_path, f"{job.job_id}.mp4")
        return file_id is not None
