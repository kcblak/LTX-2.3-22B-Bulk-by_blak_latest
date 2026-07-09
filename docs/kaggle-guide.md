# Kaggle Guide

## Philosophy

The Kaggle notebook is a launcher, not the application.

The notebook is responsible for:

- detecting the Kaggle runtime
- configuring the environment
- cloning the repository into `/kaggle/working`
- validating the repository before import
- locating the source root
- delegating dependency resolution to the repository's feature-aware dependency manager
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
6. Import repository launcher helpers and resolve the bootstrap dependency profile.
7. Import repository launcher helpers.
8. Produce a repository-aware dependency report.
9. Bootstrap the launcher context.
10. Resolve the execution profile, enabled feature graph, active renderer runtime, and runtime validation report.
11. Prepare Wan2GP runtime/model assets when needed.
12. Run diagnostics and preflight.
13. Launch the production application and display the live dashboard.
14. Print final artifact paths and fail explicitly if the run did not succeed.

## Configuration

The notebook requires no path edits for normal use.

External configuration is supported through environment variables:

- `LTX_REPO_URL`: overrides the GitHub repository URL
- `LTX_REPO_REF`: overrides the Git ref to check out
- `LTX_REPO_DIRNAME`: overrides the clone directory name inside `/kaggle/working`
- Google Drive credential environment variables supported by the repository launcher

Repository configuration now also supports feature-aware dependency control:

- `execution_profile`: selects an execution profile such as `kaggle_bulk`, `kaggle_interactive`, `local_development`, `workstation`, or `production_server`
- `features`: enables or disables optional feature groups without editing installer code
- `dependency_allow_experimental`: allows experimental package groups when explicitly enabled
- `dependency_allow_development_wheels`: allows development-wheel packages when explicitly enabled

Feature groups currently modeled by the repository include:

- core renderer
- Wan2GP runtime
- Diffusers backend
- GGUF runtime
- MSR models
- Google Drive integration
- CSV batch rendering
- resume engine
- video stitching
- reporting
- Gradio UI
- Whisper
- audio processing
- speech recognition
- face restoration
- background removal
- image editing
- development tools
- testing tools

## Runtime Preparation

The dependency manager does not install upstream `requirements.txt` files blindly.

Instead it:

- resolves the execution profile
- resolves the enabled feature graph
- filters packages by platform, Kaggle compatibility, and Python compatibility
- installs only missing packages required by enabled features
- records optional-package failures without aborting the render path
- verifies enabled features independently

When the selected renderer backend is `auto` or `wan2gp`, the repository launcher prepares:

- the external Wan2GP runtime checkout
- the Wan2GP feature set and dependency graph
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
