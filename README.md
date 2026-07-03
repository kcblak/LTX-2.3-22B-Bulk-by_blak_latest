# LTX Video Bulk Renderer

A production-grade, unattended bulk image-to-video rendering system optimized for Kaggle GPUs.

## Features

- **Unattended Execution**: Run and leave, no manual supervision needed
- **Automatic Resumption**: Picks up where it left off if interrupted
- **Google Drive Integration**: Automatically uploads generated clips
- **Parallel Uploads**: Keep GPU busy while uploads happen in background
- **Manifest & Reports**: Tracks all jobs with detailed reports
- **Modular Architecture**: Clean, maintainable, extensible design

## Project Structure

```
.
├── src/
│   ├── core/               # Core data models and job management
│   ├── services/           # Model service and Google Drive service
│   ├── workers/            # Pipeline orchestrator
│   ├── storage/            # Local storage management
│   ├── reporting/          # Reporting and manifest generation
│   └── utils/              # Logging and utilities
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
└── jobs.csv                # Jobs configuration
```

## Usage

### On Kaggle

1. Upload your `jobs.csv` and reference images as a dataset
2. Use the provided Kaggle notebook to run the renderer
3. Come back later to find all videos uploaded to your Google Drive

### Local Execution

```bash
pip install -r requirements.txt
python main.py --jobs-csv jobs.csv --reference-images path/to/images
```

## Configuration

All configuration is done via command-line arguments to `main.py`.
