# API Reference

## Public Entry Points
- `main.py`: primary render orchestration CLI.
- `diagnose.py`: standalone diagnostics CLI.

## Public Modules
- `config`: `Config`, `load_config()`.
- `diagnostics`: `DiagnosticsRunner`.
- `preflight`: `PreflightAnalyzer`.
- `observability`: `EventBus`, `RuntimeMonitor`.
- `reports`: `ReportGenerator`.
- `stitching`: `FFmpegService`, `VideoStitcher`.
- `cache_system`: `RenderCacheStore`.
- `benchmarking`: `BenchmarkComparator`.
