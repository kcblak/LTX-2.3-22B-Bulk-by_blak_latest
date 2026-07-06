from typing import Any


CONFIG_PROFILES: dict[str, dict[str, Any]] = {
    "balanced": {},
    "maximum_quality": {
        "num_inference_steps": 12,
        "output_quality": 9,
        "enable_attention_slicing": False,
        "enable_vae_tiling": False,
    },
    "maximum_throughput": {
        "num_inference_steps": 6,
        "output_quality": 7,
        "warmup_enabled": True,
        "enable_memory_efficient_attention": True,
        "drive_max_parallel_uploads": 3,
    },
    "low_vram": {
        "precision": "fp16",
        "enable_attention_slicing": True,
        "enable_vae_slicing": True,
        "enable_vae_tiling": True,
        "enable_model_cpu_offload": True,
        "enable_sequential_cpu_offload": False,
    },
    "debug": {
        "log_level": "DEBUG",
        "heartbeat_interval_seconds": 15,
        "health_poll_interval_seconds": 10,
        "drive_max_parallel_uploads": 1,
        "benchmark_mode": True,
        "benchmark_max_jobs": 1,
    },
    "benchmark": {
        "benchmark_mode": True,
        "benchmark_max_jobs": 3,
        "log_level": "INFO",
        "heartbeat_interval_seconds": 10,
        "health_poll_interval_seconds": 10,
        "preflight_enabled": True,
    },
}
