import json
import mimetypes
import threading
import time
from pathlib import Path
from typing import Optional

from config import Config
from core import DriveError, DriveProjectPaths, IDriveClient, RemoteFileMetadata
from logging_system import get_logger

logger = get_logger("upload")


class GoogleDriveClient(IDriveClient):
    """Google Drive storage adapter with project discovery and metadata caching."""

    def __init__(self, config: Config):
        self.config = config
        self.service = None
        self._folder_cache: dict[tuple[str, str], str] = {}
        self._directory_cache: dict[str, list[RemoteFileMetadata]] = {}
        self._project_paths: Optional[DriveProjectPaths] = None
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        with self._request_lock:
            delay = max(0.0, self.config.drive_request_spacing_seconds)
            elapsed = time.perf_counter() - self._last_request_at
            if elapsed < delay:
                time.sleep(delay - elapsed)
            self._last_request_at = time.perf_counter()

    def _execute(self, request):
        self._throttle()
        return request.execute()

    def _load_credentials(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google.oauth2.service_account import Credentials as ServiceAccountCredentials

        raw_json = None
        if self.config.extra.get("drive_credentials_json"):
            raw_json = self.config.extra["drive_credentials_json"]
        elif self.config.drive_credentials_path and self.config.drive_credentials_path.exists():
            raw_json = self.config.drive_credentials_path.read_text(encoding="utf-8")

        if not raw_json:
            raise DriveError("No Google Drive credentials were provided")

        info = json.loads(raw_json)
        if info.get("type") == "service_account":
            return ServiceAccountCredentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/drive"],
            )

        creds = Credentials.from_authorized_user_info(info)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                raise DriveError("Google Drive credentials are invalid or expired")
        return creds

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _to_metadata(item: dict) -> RemoteFileMetadata:
        size_raw = item.get("size")
        try:
            size_bytes = int(size_raw) if size_raw is not None else 0
        except (TypeError, ValueError):
            size_bytes = 0
        parents = item.get("parents") or []
        return RemoteFileMetadata(
            file_id=item["id"],
            name=item["name"],
            folder_id=parents[0] if parents else None,
            size_bytes=size_bytes,
            md5_checksum=item.get("md5Checksum"),
            mime_type=item.get("mimeType"),
            web_view_link=item.get("webViewLink"),
        )

    def connect(self) -> bool:
        if self.service is not None:
            return True

        try:
            from googleapiclient.discovery import build

            self.service = build(
                "drive",
                "v3",
                credentials=self._load_credentials(),
                cache_discovery=False,
            )
            logger.info("Connected to Google Drive", extra={"job_id": "N/A"})
            return True
        except Exception as exc:
            raise DriveError(f"Authentication failed: {exc}") from exc

    def create_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        self.connect()
        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            metadata["parents"] = [parent_id]
        result = self._execute(self.service.files().create(body=metadata, fields="id"))
        folder_id = result["id"]
        self._folder_cache[(parent_id or "root", name)] = folder_id
        self._directory_cache.pop(parent_id or "root", None)
        return folder_id

    def _find_folder_by_name(self, name: str, parent_id: Optional[str]) -> Optional[str]:
        cache_key = (parent_id or "root", name)
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        self.connect()
        query = [
            f"name = '{self._escape_query_value(name)}'",
            "mimeType = 'application/vnd.google-apps.folder'",
            "trashed = false",
        ]
        if parent_id:
            query.append(f"'{parent_id}' in parents")
        result = self._execute(
            self.service.files().list(
                q=" and ".join(query),
                spaces="drive",
                pageSize=1,
                fields="files(id,name,parents)",
            )
        )
        files = result.get("files", [])
        if not files:
            return None
        folder_id = files[0]["id"]
        self._folder_cache[cache_key] = folder_id
        return folder_id

    def _ensure_folder(self, name: str, parent_id: Optional[str]) -> str:
        existing = self._find_folder_by_name(name, parent_id)
        if existing:
            return existing
        return self.create_folder(name, parent_id)

    def ensure_project_structure(self, project_name: str) -> DriveProjectPaths:
        self.connect()
        if self.config.drive_folder_id:
            root_folder_id = self.config.drive_folder_id
            project_folder_id = self.config.drive_folder_id
        else:
            root_folder_id = self._ensure_folder(self.config.drive_root_folder_name, None)
            project_folder_id = self._ensure_folder(project_name, root_folder_id)

        folders: dict[str, str] = {}
        required = list(self.config.drive_required_subfolders)
        if self.config.enable_drive_model_cache and self.config.drive_model_cache_folder_name not in required:
            required.append(self.config.drive_model_cache_folder_name)
        for folder_name in required:
            folders[folder_name] = self._ensure_folder(folder_name, project_folder_id)

        self._project_paths = DriveProjectPaths(
            root_folder_id=root_folder_id,
            project_folder_id=project_folder_id,
            folders=folders,
        )
        return self._project_paths

    def list_directory(self, folder_id: str) -> list[RemoteFileMetadata]:
        if folder_id in self._directory_cache:
            return list(self._directory_cache[folder_id])

        self.connect()
        result = self._execute(
            self.service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                pageSize=1000,
                fields="files(id,name,size,md5Checksum,mimeType,parents,webViewLink)",
            )
        )
        entries = [self._to_metadata(item) for item in result.get("files", [])]
        self._directory_cache[folder_id] = entries
        return list(entries)

    def find_file_by_name(
        self, name: str, folder_id: str
    ) -> Optional[RemoteFileMetadata]:
        for item in self.list_directory(folder_id):
            if item.name == name:
                return item
        return None

    def get_metadata(self, file_id: str) -> Optional[RemoteFileMetadata]:
        self.connect()
        try:
            result = self._execute(
                self.service.files().get(
                    fileId=file_id,
                    fields="id,name,size,md5Checksum,mimeType,parents,webViewLink",
                )
            )
            return self._to_metadata(result)
        except Exception:
            return None

    def upload_file(
        self,
        local_path: Path,
        remote_name: str,
        folder_id: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Optional[RemoteFileMetadata]:
        self.connect()
        from googleapiclient.http import MediaFileUpload

        mime_type = mime_type or mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        metadata = {"name": remote_name}
        if folder_id:
            metadata["parents"] = [folder_id]
        result = self._execute(
            self.service.files().create(
                body=metadata,
                media_body=MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True),
                fields="id,name,size,md5Checksum,mimeType,parents,webViewLink",
            )
        )
        self._directory_cache.pop(folder_id or "root", None)
        return self._to_metadata(result)

    def verify_upload(
        self,
        file_id: str,
        local_size_bytes: Optional[int] = None,
        local_md5: Optional[str] = None,
    ) -> Optional[RemoteFileMetadata]:
        metadata = self.get_metadata(file_id)
        if metadata is None:
            return None
        if local_size_bytes is not None and metadata.size_bytes != local_size_bytes:
            return None
        if local_md5 is not None and metadata.md5_checksum and metadata.md5_checksum != local_md5:
            return None
        return metadata

    def download_file(self, file_id: str, destination_path: Path) -> Path:
        self.connect()
        from googleapiclient.http import MediaIoBaseDownload

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        request = self.service.files().get_media(fileId=file_id)
        with destination_path.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                self._throttle()
                _status, done = downloader.next_chunk()
        return destination_path

    def delete_remote_file(self, file_id: str) -> None:
        self.connect()
        self._execute(self.service.files().delete(fileId=file_id))
