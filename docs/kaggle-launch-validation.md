# Kaggle Launch Validation

## Intended Flow

The launcher is designed to follow this orchestration sequence:

1. Notebook starts.
2. Kaggle runtime is detected.
3. Environment summary is printed.
4. CUDA-related environment variables are configured.
5. GitHub repository is cloned or updated in `/kaggle/working`.
6. Repository structure is validated before any repository imports occur.
7. The source root is detected dynamically from either the repository root or `src/`.
8. The bootstrap dependency profile is resolved before repository startup.
9. Repository launcher helpers are imported.
10. Repository dependency status is reported.
11. Dataset discovery runs against `/kaggle/input`.
12. Google Drive credentials are detected automatically when available.
13. Execution-profile and feature-graph resolution run inside the repository.
14. Platform, Kaggle, and Python compatibility filtering run inside the repository.
15. Optional feature failures are downgraded to feature disablement when the feature is not required.
16. Diagnostics and preflight run inside the repository.
17. The staged bootstrap report is rendered even when an earlier stage failed.
18. The production application is launched through the repository runner only when the bootstrap report is ready to launch.
19. The live dashboard is updated from repository heartbeat and manifest artifacts.
20. Final artifacts are printed and the notebook exits cleanly after structured diagnostics are available.

## Validation Summary

The current launcher satisfies the following design constraints:

- No hardcoded dataset paths are required for normal use.
- No hardcoded repository filesystem paths are required for normal use.
- Repository clone location is dynamic and configurable through environment variables.
- Repository source root discovery supports both repository-root and `src/` layouts.
- Repository validation occurs before repository module imports.
- Every notebook stage is wrapped independently and captured in the staged bootstrap report.
- Execution profiles can change enabled feature sets without code edits.
- Execution profiles support `kaggle_bulk`, `kaggle_interactive`, `local_development`, `production`, and `testing`.
- Repository bootstrap supports `github` and `dataset` repository sources.
- Repository bootstrap supports `auto`, `never`, and `force` update policies.
- Feature manifests preserve optional upstream capabilities without installing them unnecessarily.
- Dependency installation records stdout, stderr, exit code, duration, Python, CUDA, and suggested resolution per package.
- Dataset discovery reports referenced image counts and missing image references before launch.
- Rendering logic remains in repository modules.
- Upload logic remains in repository modules.
- Resume logic remains in repository modules.
- Reporting remains in repository modules.
- Stitching remains in repository modules.
- Google Drive integration remains in repository modules.
- Wan2GP runtime and model asset preparation remain in repository modules.

## Known Boundaries

- The notebook still contains bootstrap orchestration logic because it must clone and validate the repository before importing repository modules.
- The notebook does not implement renderer internals, queueing, uploads, or stitching.
- Runtime verification still depends on the actual Kaggle environment for package availability, CUDA state, and network access.
- Real Kaggle execution and live Google Drive credentials still require runtime verification in Kaggle.
- Offline recovery depends on a valid local source tree already being present in `/kaggle/working` or `/kaggle/input`.
