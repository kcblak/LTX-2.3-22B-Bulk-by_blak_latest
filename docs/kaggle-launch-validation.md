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
8. Missing Python requirements are installed from the repository requirement file.
9. Repository launcher helpers are imported.
10. Repository dependency status is reported.
11. Dataset discovery runs against `/kaggle/input`.
12. Google Drive credentials are detected automatically when available.
13. Runtime preparation runs inside the repository.
14. Diagnostics and preflight run inside the repository.
15. The production application is launched through the repository runner.
16. The live dashboard is updated from repository heartbeat and manifest artifacts.
17. Final artifacts are printed and the notebook exits cleanly.

## Validation Summary

The current launcher satisfies the following design constraints:

- No hardcoded dataset paths are required for normal use.
- No hardcoded repository filesystem paths are required for normal use.
- Repository clone location is dynamic and configurable through environment variables.
- Repository source root discovery supports both repository-root and `src/` layouts.
- Repository validation occurs before repository module imports.
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
- Real Kaggle execution and live Google Drive credentials still require runtime verification in Kaggle.
