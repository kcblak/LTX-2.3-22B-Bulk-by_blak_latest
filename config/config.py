from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, get_args, get_origin

from core import (
    APP_VERSION,
    DEFAULT_CSV_FILENAME,
    DEFAULT_EMPTY_CACHE_INTERVAL,
    DEFAULT_ENABLE_ATTENTION_SLICING,
    DEFAULT_ENABLE_MEMORY_EFFICIENT_ATTENTION,
    DEFAULT_ENABLE_MODEL_CPU_OFFLOAD,
    DEFAULT_ENABLE_SEQUENTIAL_CPU_OFFLOAD,
    DEFAULT_ENABLE_VAE_SLICING,
    DEFAULT_ENABLE_VAE_TILING,
    DEFAULT_EXPECTED_DURATION_TOLERANCE_SECONDS,
    DEFAULT_FRAME_RATE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_LOG_DIR,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_MIN_OUTPUT_SIZE_BYTES,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_OUTPUT_CODEC,
    DEFAULT_OUTPUT_CONTAINER,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_FILENAME,
    DEFAULT_RESOLUTION_ALIGNMENT,
    DEFAULT_SEED,
    DEFAULT_TEMP_DIR,
    DEFAULT_VIDEO_QUALITY,
    DEFAULT_WARMUP_ENABLED,
    DEFAULT_WARMUP_NUM_FRAMES,
    DEFAULT_WARMUP_STEPS,
)


@dataclass
class Config:
    config_version: str = "1.0"
    app_version: str = APP_VERSION
    profile: str = "balanced"
    project_id: str = ""
    run_id: str = ""
    correlation_id: str = ""
    project_config_path: Optional[Path] = None

    # Paths
    jobs_csv_path: Path = field(default_factory=lambda: Path(DEFAULT_CSV_FILENAME))
    reference_images_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)
    temp_dir: Path = field(default_factory=lambda: DEFAULT_TEMP_DIR)
    manifest_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / DEFAULT_MANIFEST_FILENAME)
    report_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / DEFAULT_REPORT_FILENAME)
    heartbeat_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "heartbeat.json")
    diagnostics_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "diagnostics.json")
    validation_report_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "validation_report.json")
    performance_report_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "performance.json")
    summary_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "summary.txt")
    project_report_csv_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "project_report.csv")
    benchmark_history_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "benchmark_history.json")
    benchmark_json_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "benchmark.json")
    benchmark_csv_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "benchmark.csv")
    performance_summary_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "performance_summary.txt")
    cache_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "cache")
    cache_index_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "cache" / "render_cache.json")
    stitched_output_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "stitched" / "final_movie.mp4")
    thumbnail_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "previews" / "thumbnail.jpg")
    preview_480p_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "previews" / "preview_480p.mp4")
    preview_720p_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR / "previews" / "preview_720p.mp4")

    # Model & Rendering
    renderer_backend: str = "auto"
    model_name: str = DEFAULT_MODEL_NAME
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    seed: int = DEFAULT_SEED
    use_cuda: bool = True
    use_mps: bool = False
    precision: str = "bf16"
    frame_rate: int = DEFAULT_FRAME_RATE
    output_codec: str = DEFAULT_OUTPUT_CODEC
    output_container: str = DEFAULT_OUTPUT_CONTAINER
    output_quality: int = DEFAULT_VIDEO_QUALITY
    resolution_alignment: int = DEFAULT_RESOLUTION_ALIGNMENT
    expected_duration_tolerance_seconds: float = (
        DEFAULT_EXPECTED_DURATION_TOLERANCE_SECONDS
    )
    min_output_size_bytes: int = DEFAULT_MIN_OUTPUT_SIZE_BYTES
    warmup_enabled: bool = DEFAULT_WARMUP_ENABLED
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    warmup_num_frames: int = DEFAULT_WARMUP_NUM_FRAMES
    enable_attention_slicing: bool = DEFAULT_ENABLE_ATTENTION_SLICING
    enable_vae_slicing: bool = DEFAULT_ENABLE_VAE_SLICING
    enable_vae_tiling: bool = DEFAULT_ENABLE_VAE_TILING
    enable_model_cpu_offload: bool = DEFAULT_ENABLE_MODEL_CPU_OFFLOAD
    enable_sequential_cpu_offload: bool = DEFAULT_ENABLE_SEQUENTIAL_CPU_OFFLOAD
    enable_memory_efficient_attention: bool = DEFAULT_ENABLE_MEMORY_EFFICIENT_ATTENTION
    empty_cache_interval: int = DEFAULT_EMPTY_CACHE_INTERVAL
    wan2gp_dir: Path = field(default_factory=lambda: Path("Wan2GP"))
    wan2gp_model_dir: Optional[Path] = None
    wan2gp_transformer_filename: str = "ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"
    wan2gp_text_encoder_dirname: str = "gemma-3-12b-it-qat-q4_0-unquantized"
    wan2gp_text_encoder_filename: Optional[str] = None
    wan2gp_lora_filename: str = "LTX-2.3-Licon-MSR-V1.safetensors"
    wan2gp_base_model_type: str = "ltx2_22B_msr"
    wan2gp_pipeline_variant: str = "distilled"
    wan2gp_msr_enabled: bool = True
    wan2gp_msr_frame_count: int = 41
    wan2gp_mmgp_profile: int = 4
    wan2gp_vae_tile_size: int = 256
    wan2gp_guide_phases: int = 2
    wan2gp_negative_prompt: str = ""
    wan2gp_quantize_transformer: bool = False
    wan2gp_audio_enabled: bool = False
    wan2gp_mmgp_budgets: Dict[str, int] = field(
        default_factory=lambda: {
            "transformer": 6000,
            "text_encoder": 1500,
            "video_encoder": 2000,
            "video_decoder": 3000,
            "audio_encoder": 1000,
            "audio_decoder": 1000,
            "vocoder": 500,
            "spatial_upsampler": 1500,
            "vae": 1000,
            "*": 1000,
        }
    )

    # Drive
    enable_drive_upload: bool = True
    drive_credentials_path: Optional[Path] = None
    drive_folder_id: Optional[str] = None
    drive_project_name: str = "default_project"
    drive_root_folder_name: str = "LTX_PROJECTS"
    drive_verify_uploads: bool = True
    drive_upload_retries: int = 5
    drive_retry_base_seconds: float = 2.0
    drive_retry_max_seconds: float = 60.0
    drive_request_spacing_seconds: float = 0.25
    drive_max_parallel_uploads: int = 2
    drive_cleanup_policy: str = "delete_temp_only"
    drive_required_subfolders: list[str] = field(
        default_factory=lambda: [
            "config",
            "input",
            "images",
            "clips",
            "stitched",
            "logs",
            "reports",
            "manifests",
            "thumbnails",
            "previews",
            "cache",
        ]
    )

    # Pipeline
    resume_enabled: bool = True
    enable_stitching: bool = False
    stitch_require_contiguous_success: bool = True
    stitch_upload_outputs: bool = True
    stitching_thumbnail_timestamp_seconds: int = 1
    generate_preview_480p: bool = True
    generate_preview_720p: bool = False
    generate_compressed_preview: bool = False
    cache_enabled: bool = True
    max_retries: int = 3
    parallel_uploads: bool = True
    cleanup_temp: bool = True
    auto_verify: bool = True
    preflight_enabled: bool = True
    preflight_abort_on_warning: bool = False
    diagnostics_network_check: bool = False
    benchmark_mode: bool = False
    benchmark_max_jobs: int = 0
    benchmark_compare_previous: bool = True

    # Logging
    log_level: str = "INFO"
    log_performance: bool = True
    log_to_console: bool = True
    log_rotation_max_bytes: int = 10 * 1024 * 1024
    log_rotation_backup_count: int = 5
    log_rotation_max_age_days: int = 7
    log_rotation_compress: bool = True

    # Observability
    heartbeat_enabled: bool = True
    heartbeat_interval_seconds: int = 60
    health_poll_interval_seconds: int = 30
    sync_heartbeat_to_drive: bool = True

    # Extra
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from a dictionary."""
        normalized: Dict[str, Any] = dict(data)
        for config_field in fields(cls):
            field_name = config_field.name
            if field_name not in normalized or normalized[field_name] is None:
                continue
            if _is_path_annotation(config_field.type):
                normalized[field_name] = Path(normalized[field_name])
        return cls(**normalized)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to a dictionary."""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value
        return data


def _is_path_annotation(annotation: Any) -> bool:
    if annotation is Path:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(arg is Path for arg in get_args(annotation))
