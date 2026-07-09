# Kaggle Guide

## Philosophy

The Kaggle notebook is a launcher, not the application.

The notebook is responsible for:

- detecting the Kaggle runtime
- configuring the environment
- cloning the repository into `/kaggle/working`
- validating the repository before import
- locating the source root
- installing only missing Python requirements
- delegating runtime preparation, model preparation, dataset discovery, preflight, execution, monitoring, and shutdown to repository modules

The notebook is not responsible for:

- rendering logic
- queue management
- Google Drive business logic
- manifest updates
- CSV parsing
- retry logic
- stitching implementation
- reporting implementation

## Execution Flow

The current launcher executes the following phases:

1. Detect Kaggle runtime and print an environment summary.
2. Configure CUDA-related environment variables.
3. Clone or update the GitHub repository in `/kaggle/working`.
4. Detect whether the source root is the repository root or `src/`.
5. Validate critical repository files before importing any repository modules.
6. Install only missing packages from the repository `requirements.txt`.
7. Import repository launcher helpers.
8. Produce a repository-aware dependency report.
9. Bootstrap the launcher context.
10. Prepare runtime dependencies, Wan2GP runtime checkout, and model assets when needed.
11. Run diagnostics and preflight.
12. Launch the production application and display the live dashboard.
13. Print final artifact paths and fail explicitly if the run did not succeed.

## Configuration

The notebook requires no path edits for normal use.

External configuration is supported through environment variables:

- `LTX_REPO_URL`: overrides the GitHub repository URL
- `LTX_REPO_REF`: overrides the Git ref to check out
- `LTX_REPO_DIRNAME`: overrides the clone directory name inside `/kaggle/working`
- Google Drive credential environment variables supported by the repository launcher

## Runtime Preparation

When the selected renderer backend is `auto` or `wan2gp`, the repository launcher prepares:

- the external Wan2GP runtime checkout
- Wan2GP runtime Python requirements
- GGUF transformer asset
- Gemma text encoder directory and tokenizer files
- VAE and companion assets
- MSR LoRA asset when MSR is enabled

The notebook does not download model assets directly. It delegates that work to repository code.

## Resume

Resume support is automatic when the launcher discovers:

- a manifest seed
- a render cache index
- completed local artifacts

The notebook surfaces the detected resume state before launch and the repository decides what to skip or resume.
