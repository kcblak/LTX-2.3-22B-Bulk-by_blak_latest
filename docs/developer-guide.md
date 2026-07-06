# Developer Guide

- Prefer dependency injection through `Config`.
- Keep rendering, uploads, stitching, and reporting independent.
- Run `python diagnose.py`, `python main.py --preflight-only`, and `python -m unittest discover -s tests` before shipping changes.
