# Kaggle Guide

## Philosophy

The Kaggle notebook is a launcher, not the application.

The notebook is responsible for:

- detecting the Kaggle runtime
- configuring the environment
- resolving the repository source from GitHub or a Kaggle Dataset
- self-healing the checkout in `/kaggle/working` when GitHub is the source
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
3. Resolve the repository source and update policy.
4. Clone, repair, reuse, or fall back to a local repository copy as needed.
5. Detect whether the source root is the repository root or `src/`.
6. Validate critical repository files before importing any repository modules.
7. Import repository launcher helpers and resolve the bootstrap dependency profile.
8. Produce a repository-aware dependency report.
9. Execute the staged launcher flow: `bootstrap_context` -> `prepare_runtime` -> `run_preflight` -> `display_preparation`.
10. Build a structured `BootstrapReport` with stage timing, PASS/WARNING/FAILED/SKIPPED statuses, explanations, and recovery suggestions.
11. Resolve the execution profile, enabled feature graph, active renderer runtime, dataset discovery report, and runtime validation report.
12. Prepare Wan2GP runtime/model assets when needed.
13. Launch the production application through the staged pipeline entry point and display the live dashboard.
14. Print final artifact paths and fail explicitly only after structured diagnostics have been rendered.

## Configuration

The notebook requires no path edits for normal use.

External configuration is supported through environment variables:

- `LTX_REPO_URL`: overrides the GitHub repository URL
- `LTX_REPO_REF`: overrides the Git ref to check out
- `LTX_REPO_DIRNAME`: overrides the clone directory name inside `/kaggle/working`
- `LTX_REPOSITORY_SOURCE`: selects `github` or `dataset`
- `LTX_REPOSITORY_UPDATE_POLICY`: selects `auto`, `never`, or `force`
- `LTX_REPO_PATH`: explicitly points to a repository copy, useful for dataset-based launches
- Google Drive credential environment variables supported by the repository launcher

Repository configuration now also supports feature-aware dependency control:

- `execution_profile`: selects an execution profile such as `kaggle_bulk`, `kaggle_interactive`, `local_development`, `production`, or `testing`
- `repository_source`: selects `github` or `dataset`
- `repository_update_policy`: selects `auto`, `never`, or `force`
- `features`: enables or disables optional feature groups without editing installer code
- `dependency_allow_experimental`: allows experimental package groups when explicitly enabled
- `dependency_allow_development_wheels`: allows development-wheel packages when explicitly enabled

Legacy aliases remain accepted for compatibility:

- `workstation` -> `local_development`
- `production_server` -> `production`

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
- captures package install duration, stdout, stderr, platform, Python, CUDA, and suggested resolution for every attempted install
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

## Bootstrap Reporting

The launcher now produces a structured bootstrap report before launch.

Each stage records:

- PASS, FAILED, or SKIPPED status
- started/completed timestamps
- elapsed time
- exception type and traceback snippet when a stage fails
- recovery suggestions tailored to the stage

The report sections summarize repository, runtime, dependencies, dataset discovery, Drive, resume state, configuration, environment, GPU, CUDA, Torch, Python, disk, RAM, models, and validation readiness.

## Repository Bootstrap

The repository bootstrap is designed for unattended Kaggle execution and treats every launch as potentially starting from an unexpected state.

Supported states include:

- missing checkout
- healthy Git checkout
- detached HEAD
- missing or incorrect `origin` remote
- missing target branch
- corrupted Git metadata
- source tree copied from a Kaggle Dataset without `.git`
- offline Kaggle sessions where GitHub cannot be reached

Recovery order:

1. validate the destination and Git metadata
2. repair or recreate `origin`
3. detect remote reachability and default branch
4. fall back to the remote default branch if the configured branch is missing
5. fetch and check out the target branch
6. reclone the repository if recovery fails
7. use a valid local source tree when offline or dataset-backed
8. fail only after all recovery paths are exhausted

## Resume

Resume support is automatic when the launcher discovers:

- a manifest seed
- a render cache index
- completed local artifacts

The notebook surfaces the detected resume state before launch and the repository decides what to skip or resume.
