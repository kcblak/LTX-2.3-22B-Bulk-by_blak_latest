from pathlib import Path

# Default filenames and directories
DEFAULT_CSV_FILENAME = "jobs.csv"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_REPORT_FILENAME = "report.json"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_TEMP_DIR = Path("temp")

# Supported formats
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv"}

# Model defaults
# The product is Kaggle-optimized and restricted to the exact model set shipped by
# ltx-2-3-22b-msr-ref-distilled-1-1-kaggle (Wan2GP + GGUF transformer + DeepBeepMeep/LTX-2
# companion files + Licon MSR LoRA + Gemma-3 text encoder). This identifier is the Kaggle
# transformer model repo; the full wan2gp asset set is enumerated in config/default.yaml and
# assets/asset_manager.py:_build_wan2gp_manifest.
DEFAULT_MODEL_NAME = "Abiray/LTX-2.3-22B-DISTILLED-1.1-GGUF"
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_NUM_INFERENCE_STEPS = 8
DEFAULT_SEED = -1  # random
DEFAULT_FRAME_RATE = 24
DEFAULT_OUTPUT_CODEC = "libx264"
DEFAULT_OUTPUT_CONTAINER = "mp4"
DEFAULT_VIDEO_QUALITY = 8
DEFAULT_EXPECTED_DURATION_TOLERANCE_SECONDS = 0.35
DEFAULT_MIN_OUTPUT_SIZE_BYTES = 1024
DEFAULT_RESOLUTION_ALIGNMENT = 32
DEFAULT_WARMUP_ENABLED = True
DEFAULT_WARMUP_STEPS = 1
DEFAULT_WARMUP_NUM_FRAMES = 9
DEFAULT_ENABLE_ATTENTION_SLICING = False
DEFAULT_ENABLE_VAE_SLICING = False
DEFAULT_ENABLE_VAE_TILING = False
DEFAULT_ENABLE_MODEL_CPU_OFFLOAD = False
DEFAULT_ENABLE_SEQUENTIAL_CPU_OFFLOAD = False
DEFAULT_ENABLE_MEMORY_EFFICIENT_ATTENTION = True
DEFAULT_EMPTY_CACHE_INTERVAL = 5
