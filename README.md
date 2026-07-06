# LTX Bulk Renderer

Production-grade unattended batch image-to-video rendering for Kaggle GPU workers with Google Drive as persistent storage.

## Highlights
- Layered configuration with defaults, YAML, environment variables, CLI overrides, and runtime overrides.
- Structured validation, diagnostics, preflight analysis, runtime health monitoring, and benchmark reporting.
- Plugin-based renderer backends for `diffusers` and `wan2gp`.
- Background Google Drive synchronization with verification, retries, and resumable manifests.
- Manifest-driven FFmpeg stitching, thumbnail extraction, preview generation, and benchmark artifact export.
- Persistent verified render cache for duplicate work reuse.

## Quick Start
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python main.py --project-config examples/config/project.yaml --preflight-only
python main.py --project-config examples/config/project.yaml
```

## Standalone Diagnostics
```bash
python diagnose.py --project-config examples/config/project.yaml --network-check
```
