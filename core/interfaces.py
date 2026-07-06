from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RenderParams:
    job_id: str
    prompt: str
    start_image: Path
    end_image: Optional[Path]
    duration: Any
    resolution: Any
    aspect_ratio: Any
    seed: int
    guidance_scale: float
    num_inference_steps: int
    frame_rate: int
    output_codec: str
    output_container: str
    output_quality: int


@dataclass
class RenderMetrics:
    image_loading_seconds: float = 0.0
    prompt_preparation_seconds: float = 0.0
    inference_seconds: float = 0.0
    encoding_seconds: float = 0.0
    validation_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class RenderedClip:
    path: Path
    checksum_sha256: str
    file_size_bytes: int
    width: int
    height: int
    frame_rate: int
    frame_count: int
    duration_seconds: float
    codec: str


@dataclass
class RenderResult:
    success: bool
    job_id: str
    clip: Optional[RenderedClip] = None
    metrics: RenderMetrics = field(default_factory=RenderMetrics)
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class RendererSessionInfo:
    renderer_name: str
    device: str
    precision: str
    total_vram_bytes: int
    available_vram_bytes: int
    scheduler_name: Optional[str] = None
    warmup_performed: bool = False
    initialization_seconds: float = 0.0


@dataclass
class RemoteFileMetadata:
    file_id: str
    name: str
    folder_id: Optional[str]
    size_bytes: int = 0
    md5_checksum: Optional[str] = None
    mime_type: Optional[str] = None
    web_view_link: Optional[str] = None


@dataclass
class DriveProjectPaths:
    root_folder_id: str
    project_folder_id: str
    folders: Dict[str, str]


@dataclass
class UploadMetrics:
    upload_seconds: float = 0.0
    verification_seconds: float = 0.0
    total_seconds: float = 0.0
    average_upload_mbps: float = 0.0
    retry_count: int = 0
    queue_depth_at_submit: int = 0
    duplicate_skipped: bool = False


@dataclass
class UploadTask:
    job_id: str
    local_path: Path
    remote_name: str
    remote_folder_key: str
    local_size_bytes: int
    local_md5: str
    cleanup_policy: str


@dataclass
class UploadResult:
    success: bool
    job_id: str
    remote_metadata: Optional[RemoteFileMetadata] = None
    metrics: UploadMetrics = field(default_factory=UploadMetrics)
    error_message: Optional[str] = None


class IRenderer(ABC):
    """Interface for video renderers."""

    @abstractmethod
    def initialize(self) -> RendererSessionInfo:
        """Initialize the renderer (load model, etc.)."""
        pass

    @abstractmethod
    def validate_parameters(self, params: RenderParams) -> bool:
        """Validate rendering parameters."""
        pass

    @abstractmethod
    def generate_clip(self, params: RenderParams, output_path: Path) -> RenderResult:
        """Generate a video clip."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass


class IStorage(ABC):
    """Interface for storage operations."""

    @abstractmethod
    def get_temp_path(self, job_id: str, suffix: str = "") -> Path:
        """Get a temporary file path."""
        pass

    @abstractmethod
    def get_output_path(self, job_id: str, extension: str = "mp4") -> Path:
        """Get an output file path."""
        pass

    @abstractmethod
    def verify_file(self, path: Path, min_size_bytes: int = 1024) -> bool:
        """Verify a file exists and is valid."""
        pass

    @abstractmethod
    def cleanup_temp(self, job_id: Optional[str] = None) -> None:
        """Clean up temporary files."""
        pass


class IDriveClient(ABC):
    """Interface for cloud drive clients."""

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate with the drive service."""
        pass

    @abstractmethod
    def create_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """Create a folder and return its ID."""
        pass

    @abstractmethod
    def ensure_project_structure(self, project_name: str) -> DriveProjectPaths:
        """Locate or create the standard project folder structure."""
        pass

    @abstractmethod
    def list_directory(self, folder_id: str) -> list[RemoteFileMetadata]:
        """List files in a remote directory."""
        pass

    @abstractmethod
    def find_file_by_name(
        self, name: str, folder_id: str
    ) -> Optional[RemoteFileMetadata]:
        """Find a single file by name within a folder."""
        pass

    @abstractmethod
    def get_metadata(self, file_id: str) -> Optional[RemoteFileMetadata]:
        """Fetch remote metadata for a file."""
        pass

    @abstractmethod
    def upload_file(
        self,
        local_path: Path,
        remote_name: str,
        folder_id: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Optional[RemoteFileMetadata]:
        """Upload a file to the drive."""
        pass

    @abstractmethod
    def verify_upload(
        self,
        file_id: str,
        local_size_bytes: Optional[int] = None,
        local_md5: Optional[str] = None,
    ) -> Optional[RemoteFileMetadata]:
        """Verify a file was uploaded successfully."""
        pass

    @abstractmethod
    def download_file(self, file_id: str, destination_path: Path) -> Path:
        """Download a remote file."""
        pass

    @abstractmethod
    def delete_remote_file(self, file_id: str) -> None:
        """Delete a remote file."""
        pass


class IReporter(ABC):
    """Interface for report generators."""

    @abstractmethod
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary report."""
        pass

    @abstractmethod
    def save_json_report(self, path: Path) -> None:
        """Save report as JSON."""
        pass

    @abstractmethod
    def save_text_report(self, path: Path) -> None:
        """Save report as plain text."""
        pass
