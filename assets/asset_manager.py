from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from assets.asset_cache import AssetCache
from assets.asset_cleanup import AssetCleanup
from assets.asset_downloader import AssetDownloader
from assets.asset_manifest import AssetManifest, AssetSpec, AssetStateEntry, DownloadManifest
from assets.asset_reports import AssetReport, AssetReportWriter
from assets.asset_sources import (
    AssetCandidate,
    SOURCE_DATASET,
    SOURCE_DRIVE_CACHE,
    SOURCE_DRIVE_DOWNLOAD,
    SOURCE_HF_CACHE,
    SOURCE_HF_DOWNLOAD,
    SOURCE_LOCAL_CACHE,
    SOURCE_USER_DIR,
)
from assets.asset_validator import AssetValidator
from assets.disk_manager import DiskManager
from config import Config
from logging_system import get_logger


@dataclass(frozen=True)
class AssetManagerPaths:
    cache_root: Path
    temp_root: Path
    report_dir: Path
    download_manifest_path: Path


class AssetManager:
    def __init__(self, config: Config, *, drive_client: Optional[object] = None) -> None:
        self.config = config
        self.drive_client = drive_client
        self.logger = get_logger("assets")
        self.validator = AssetValidator()
        self.disk = DiskManager(safety_margin_gb=config.asset_disk_safety_margin_gb)
        self.paths = self._resolve_paths()
        self.cache = AssetCache(self.paths.cache_root)
        self.cleanup = AssetCleanup(self.paths.temp_root)
        self.report_writer = AssetReportWriter(self.paths.report_dir)

    def _resolve_paths(self) -> AssetManagerPaths:
        cache_root = self.config.model_cache_dir.resolve(strict=False)
        temp_root = self.config.asset_temp_dir.resolve(strict=False)
        report_dir = self.config.asset_report_dir.resolve(strict=False)
        download_manifest_path = self.config.asset_download_manifest_path.resolve(strict=False)
        return AssetManagerPaths(
            cache_root=cache_root,
            temp_root=temp_root,
            report_dir=report_dir,
            download_manifest_path=download_manifest_path,
        )

    def _read_download_manifest(self) -> DownloadManifest:
        path = self.paths.download_manifest_path
        if not path.exists():
            return DownloadManifest()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return DownloadManifest()
        if not isinstance(payload, dict):
            return DownloadManifest()
        return DownloadManifest.from_dict(payload)

    def _write_download_manifest(self, manifest: DownloadManifest) -> None:
        path = self.paths.download_manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def resolve_backend(self) -> str:
        backend = (self.config.renderer_backend or "auto").strip().lower()
        if backend != "auto":
            return backend
        if self.config.features.get("wan2gp_runtime", False):
            return "wan2gp"
        if self.config.features.get("diffusers_backend", False):
            return "diffusers"
        return "wan2gp"

    def build_manifest(self, backend: Optional[str] = None) -> AssetManifest:
        backend = backend or self.resolve_backend()
        assets: list[AssetSpec] = []
        if backend == "wan2gp":
            assets.extend(self._build_wan2gp_manifest())
        if backend == "diffusers":
            assets.extend(self._build_diffusers_manifest())
        return AssetManifest(backend=backend, assets=assets)

    def _build_wan2gp_manifest(self) -> list[AssetSpec]:
        if not self.config.features.get("wan2gp_runtime", False):
            return []
        backend = "wan2gp"
        specs: list[AssetSpec] = []
        specs.append(
            AssetSpec(
                key="wan2gp.transformer.gguf",
                filename=self.config.wan2gp_transformer_filename,
                backend=backend,
                required=True,
                priority=10,
                hf_repo_id=self.config.wan2gp_transformer_repo_id,
                hf_filename=self.config.wan2gp_transformer_source_filename,
            )
        )
        for filename in self.config.wan2gp_required_companion_files:
            specs.append(
                AssetSpec(
                    key=f"wan2gp.companion.{filename}",
                    filename=filename,
                    backend=backend,
                    required=True,
                    priority=20,
                    hf_repo_id=self.config.wan2gp_companion_repo_id,
                    hf_filename=filename,
                )
            )
        for filename in self.config.wan2gp_required_text_encoder_files:
            relative = f"{self.config.wan2gp_text_encoder_dirname}/{filename}"
            specs.append(
                AssetSpec(
                    key=f"wan2gp.text_encoder.{filename}",
                    filename=relative,
                    backend=backend,
                    required=True,
                    priority=30,
                    hf_repo_id=self.config.wan2gp_companion_repo_id,
                    hf_filename=relative,
                )
            )
        if self.config.wan2gp_msr_enabled:
            specs.append(
                AssetSpec(
                    key="wan2gp.lora.msr",
                    filename=self.config.wan2gp_lora_filename,
                    backend=backend,
                    required=False,
                    priority=40,
                    hf_repo_id=self.config.wan2gp_companion_repo_id,
                    hf_filename=self.config.wan2gp_lora_source_path,
                )
            )
        return specs

    def _build_diffusers_manifest(self) -> list[AssetSpec]:
        if not self.config.features.get("diffusers_backend", False):
            return []
        backend = "diffusers"
        model_id = self.config.model_name
        return [
            AssetSpec(
                key="diffusers.model",
                filename=model_id.replace("/", "__"),
                backend=backend,
                required=True,
                priority=10,
                hf_repo_id=model_id,
                hf_filename=None,
            )
        ]

    def _candidate_roots(self) -> list[tuple[str, Path]]:
        roots: list[tuple[str, Path]] = []
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            roots.append((SOURCE_DATASET, kaggle_input))
            for child in kaggle_input.iterdir():
                if child.is_dir():
                    roots.append((SOURCE_DATASET, child))
        if self.config.user_model_dir is not None:
            roots.append((SOURCE_USER_DIR, self.config.user_model_dir))
        roots.append((SOURCE_LOCAL_CACHE, self.config.model_cache_dir))
        if self.config.hf_cache_dir is not None:
            roots.append((SOURCE_HF_CACHE, self.config.hf_cache_dir))
        return [(source, root.resolve(strict=False)) for source, root in roots]

    def _find_complete_source(
        self, manifest: AssetManifest, cache_subdir: Path
    ) -> Optional[tuple[str, Path, list[str]]]:
        notes: list[str] = []
        for source, root in self._candidate_roots():
            missing: list[str] = []
            for spec in manifest.required_assets():
                candidate = root / cache_subdir / spec.filename
                if not candidate.exists():
                    candidate = root / spec.filename
                if not candidate.exists():
                    missing.append(spec.filename)
                    continue
                result = self.validator.validate(spec, candidate)
                if not result.ok:
                    missing.append(spec.filename)
            if not missing:
                return source, root, notes
            notes.append(f"source={source} root={root} missing={len(missing)}")
        return None

    def _link_from_root(
        self, manifest: AssetManifest, source: str, root: Path, cache_subdir: Path
    ) -> list[AssetStateEntry]:
        entries: list[AssetStateEntry] = []
        for spec in manifest.assets:
            started_at = datetime.now().isoformat()
            destination = cache_subdir / spec.filename
            destination_path = destination
            if not destination_path.is_absolute():
                destination_path = self.paths.cache_root / destination
            candidate_path = root / destination
            if not candidate_path.exists():
                candidate_path = root / spec.filename
            validation = self.validator.validate(spec, candidate_path)
            if not validation.ok:
                status = "FAILED" if spec.required else "SKIPPED"
                entries.append(
                    AssetStateEntry(
                        asset_key=spec.key,
                        status=status,
                        source=source,
                        path=str(destination_path),
                        size_bytes=validation.size_bytes,
                        sha256=validation.sha256,
                        started_at=started_at,
                        completed_at=datetime.now().isoformat(),
                        notes=[validation.reason],
                    )
                )
                continue
            decision = self.cache.link_or_copy(
                AssetCandidate(source=source, path=candidate_path),
                destination_path,
            )
            entries.append(
                AssetStateEntry(
                    asset_key=spec.key,
                    status="READY",
                    source=source,
                    path=decision.destination_path,
                    size_bytes=validation.size_bytes,
                    sha256=validation.sha256 or (spec.sha256 or ""),
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    notes=[decision.action],
                )
            )
        return entries

    def _link_from_registry(
        self, manifest: AssetManifest, registry: dict[str, Any]
    ) -> list[AssetStateEntry]:
        entries: list[AssetStateEntry] = []
        registry_entries = {e["asset_key"]: e for e in registry.get("entries", [])}
        for spec in manifest.assets:
            started_at = datetime.now().isoformat()
            destination = cache_subdir / spec.filename
            destination_path = destination
            if not destination_path.is_absolute():
                destination_path = self.paths.cache_root / destination
            reg_entry = registry_entries.get(spec.key)
            if reg_entry is None or reg_entry.get("status") != "found":
                status = "FAILED" if spec.required else "SKIPPED"
                entries.append(
                    AssetStateEntry(
                        asset_key=spec.key,
                        status=status,
                        source="registry",
                        path=str(destination_path),
                        size_bytes=0,
                        sha256="",
                        started_at=started_at,
                        completed_at=datetime.now().isoformat(),
                        notes=["not_found_in_registry"],
                    )
                )
                continue
            candidate_path = Path(reg_entry["actual_path"])
            validation = self.validator.validate(spec, candidate_path)
            if not validation.ok:
                status = "FAILED" if spec.required else "SKIPPED"
                entries.append(
                    AssetStateEntry(
                        asset_key=spec.key,
                        status=status,
                        source=reg_entry.get("source", "registry"),
                        path=str(destination_path),
                        size_bytes=validation.size_bytes,
                        sha256=validation.sha256,
                        started_at=started_at,
                        completed_at=datetime.now().isoformat(),
                        notes=[validation.reason],
                    )
                )
                continue
            decision = self.cache.link_or_copy(
                AssetCandidate(
                    source=reg_entry.get("source", "registry"),
                    path=candidate_path,
                ),
                destination_path,
            )
            entries.append(
                AssetStateEntry(
                    asset_key=spec.key,
                    status="READY",
                    source=reg_entry.get("source", "registry"),
                    path=decision.destination_path,
                    size_bytes=validation.size_bytes,
                    sha256=validation.sha256 or (spec.sha256 or ""),
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    notes=[decision.action],
                )
            )
        return entries

    def ensure_assets(self, *, backend: Optional[str] = None) -> AssetReport:
        started_at = datetime.now().isoformat()
        backend = backend or self.resolve_backend()
        manifest = self.build_manifest(backend)
        cache_subdir = Path(backend)
        cache_root = self.paths.cache_root / cache_subdir
        cache_root.mkdir(parents=True, exist_ok=True)
        download_manifest = self._read_download_manifest()
        report_entries: list[AssetStateEntry] = []
        report_notes: list[str] = []
        self.cleanup.ensure_temp_root()

        registry = None
        raw_registry = (self.config.extra or {}).get("model_registry")
        if isinstance(raw_registry, dict):
            registry = raw_registry

        complete_source = self._find_complete_source(manifest, cache_subdir)
        if complete_source is not None:
            source, root, notes = complete_source
            report_notes.extend(notes)
            report_entries.extend(self._link_from_root(manifest, source, root, cache_subdir))
            ready = all(entry.status == "READY" for entry in report_entries if entry.asset_key in manifest.keys())
            report = AssetReport(
                backend=backend,
                ready=ready,
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                entries=report_entries,
                notes=report_notes,
            )
            self._write_reports(report)
            self._apply_config_paths(backend)
            return report

        if registry is not None:
            registry_missing = [
                entry["asset_key"]
                for entry in registry.get("entries", [])
                if entry.get("status") != "found"
            ]
            if not registry_missing:
                report_entries.extend(self._link_from_registry(manifest, registry))
                ready = all(entry.status == "READY" for entry in report_entries if entry.asset_key in manifest.keys())
                report = AssetReport(
                    backend=backend,
                    ready=ready,
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    entries=report_entries,
                    notes=report_notes + ["assembled_from_model_registry"],
                )
                self._write_reports(report)
                self._apply_config_paths(backend)
                return report

        drive_models_folder_id = None
        if self.drive_client is not None and self.config.enable_drive_model_cache:
            try:
                paths = self.drive_client.ensure_project_structure(self.config.drive_project_name)
                drive_models_folder_id = paths.folders.get(self.config.drive_model_cache_folder_name)
            except Exception as exc:
                report_notes.append(str(exc))

        downloader = AssetDownloader(
            hf_cache_dir=self.config.hf_cache_dir,
            temp_root=self.paths.temp_root,
            drive_client=self.drive_client if self.config.enable_drive_model_cache else None,
            drive_models_folder_id=drive_models_folder_id,
        )

        for spec in manifest.assets:
            started = datetime.now().isoformat()
            destination = cache_root / spec.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if spec.key == "diffusers.model":
                marker = destination / "model_index.json"
                if marker.exists():
                    sha256 = self.validator.compute_sha256(marker)
                    report_entries.append(
                        AssetStateEntry(
                            asset_key=spec.key,
                            status="READY",
                            source=SOURCE_LOCAL_CACHE,
                            path=str(destination),
                            size_bytes=marker.stat().st_size,
                            sha256=sha256,
                            started_at=started,
                            completed_at=datetime.now().isoformat(),
                            notes=["cached_snapshot"],
                        )
                    )
                    continue
                if self.config.hf_cache_dir is None:
                    status = "FAILED" if spec.required else "SKIPPED"
                    report_entries.append(
                        AssetStateEntry(
                            asset_key=spec.key,
                            status=status,
                            source=SOURCE_HF_DOWNLOAD,
                            path=str(destination),
                            size_bytes=0,
                            sha256="",
                            started_at=started,
                            completed_at=datetime.now().isoformat(),
                            notes=["hf_cache_dir_not_configured"],
                        )
                    )
                    if spec.required:
                        break
                    continue
                try:
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        repo_id=spec.hf_repo_id or "",
                        cache_dir=str(self.config.hf_cache_dir),
                        local_dir=str(destination),
                        local_dir_use_symlinks=True,
                        resume_download=True,
                    )
                except Exception as exc:
                    status = "FAILED" if spec.required else "SKIPPED"
                    report_entries.append(
                        AssetStateEntry(
                            asset_key=spec.key,
                            status=status,
                            source=SOURCE_HF_DOWNLOAD,
                            path=str(destination),
                            size_bytes=0,
                            sha256="",
                            started_at=started,
                            completed_at=datetime.now().isoformat(),
                            notes=[str(exc)],
                        )
                    )
                    if spec.required:
                        break
                    continue
                marker = destination / "model_index.json"
                if not marker.exists():
                    status = "FAILED" if spec.required else "SKIPPED"
                    report_entries.append(
                        AssetStateEntry(
                            asset_key=spec.key,
                            status=status,
                            source=SOURCE_HF_DOWNLOAD,
                            path=str(destination),
                            size_bytes=0,
                            sha256="",
                            started_at=started,
                            completed_at=datetime.now().isoformat(),
                            notes=["snapshot_missing_model_index"],
                        )
                    )
                    if spec.required:
                        break
                    continue
                sha256 = self.validator.compute_sha256(marker)
                report_entries.append(
                    AssetStateEntry(
                        asset_key=spec.key,
                        status="READY",
                        source=SOURCE_HF_DOWNLOAD,
                        path=str(destination),
                        size_bytes=marker.stat().st_size,
                        sha256=sha256,
                        started_at=started,
                        completed_at=datetime.now().isoformat(),
                        notes=["snapshot_downloaded"],
                    )
                )
                download_manifest.mark(report_entries[-1])
                continue
            previous = download_manifest.get(spec.key)
            expected_size = (
                previous.size_bytes
                if previous is not None and previous.size_bytes
                else spec.expected_size_bytes
            )
            expected_sha = (
                previous.sha256
                if previous is not None and previous.sha256
                else spec.sha256
            )
            existing = self.validator.validate(
                spec,
                destination,
                expected_size_bytes=expected_size,
                expected_sha256=expected_sha,
            )
            if existing.ok:
                sha256 = existing.sha256
                if not sha256:
                    sha256 = self.validator.compute_sha256(destination)
                report_entries.append(
                    AssetStateEntry(
                        asset_key=spec.key,
                        status="READY",
                        source=SOURCE_LOCAL_CACHE,
                        path=str(destination),
                        size_bytes=existing.size_bytes,
                        sha256=sha256,
                        started_at=started,
                        completed_at=datetime.now().isoformat(),
                        notes=["cached"],
                    )
                )
                continue

            download_size = int(expected_size or 0)
            plan = self.disk.plan_download(
                destination_root=self.paths.cache_root,
                download_bytes=download_size,
                temp_overhead_bytes=int(max(0, download_size * 0.1)),
            )
            if download_size and not plan.ok:
                status = "FAILED" if spec.required else "SKIPPED"
                report_entries.append(
                    AssetStateEntry(
                        asset_key=spec.key,
                        status=status,
                        source="disk_manager",
                        path=str(destination),
                        size_bytes=0,
                        sha256="",
                        started_at=started,
                        completed_at=datetime.now().isoformat(),
                        notes=[
                            "insufficient_disk_space",
                            json.dumps(plan.to_dict(), sort_keys=True),
                        ],
                    )
                )
                if spec.required:
                    break
                continue

            if self.config.enable_drive_model_cache:
                drive_download = downloader.download_from_drive(spec)
                if drive_download.ok and drive_download.path is not None:
                    validation = self.validator.validate(
                        spec,
                        drive_download.path,
                        expected_size_bytes=expected_size,
                        expected_sha256=expected_sha,
                    )
                    if validation.ok:
                        sha256 = validation.sha256
                        if not sha256:
                            sha256 = self.validator.compute_sha256(drive_download.path)
                        decision = self.cache.link_or_copy(
                            AssetCandidate(source=SOURCE_DRIVE_DOWNLOAD, path=drive_download.path),
                            destination,
                        )
                        entry = AssetStateEntry(
                            asset_key=spec.key,
                            status="READY",
                            source=SOURCE_DRIVE_DOWNLOAD,
                            path=decision.destination_path,
                            size_bytes=validation.size_bytes,
                            sha256=sha256,
                            started_at=started,
                            completed_at=datetime.now().isoformat(),
                            download_seconds=drive_download.duration_seconds,
                            notes=[decision.action],
                        )
                        download_manifest.mark(entry)
                        report_entries.append(entry)
                        continue
                    self.cleanup.delete_path(drive_download.path)

            hf_download = downloader.download_from_huggingface(spec)
            if not hf_download.ok or hf_download.path is None:
                status = "FAILED" if spec.required else "SKIPPED"
                entry = AssetStateEntry(
                    asset_key=spec.key,
                    status=status,
                    source=SOURCE_HF_DOWNLOAD,
                    path=str(destination),
                    size_bytes=0,
                    sha256="",
                    started_at=started,
                    completed_at=datetime.now().isoformat(),
                    download_seconds=hf_download.duration_seconds,
                    notes=[hf_download.error or "hf_download_failed"],
                )
                download_manifest.mark(entry)
                report_entries.append(entry)
                if spec.required:
                    break
                continue

            validation = self.validator.validate(
                spec,
                hf_download.path,
                expected_size_bytes=expected_size,
                expected_sha256=expected_sha,
            )
            if not validation.ok:
                status = "FAILED" if spec.required else "SKIPPED"
                entry = AssetStateEntry(
                    asset_key=spec.key,
                    status=status,
                    source=SOURCE_HF_DOWNLOAD,
                    path=str(destination),
                    size_bytes=validation.size_bytes,
                    sha256=validation.sha256,
                    started_at=started,
                    completed_at=datetime.now().isoformat(),
                    download_seconds=hf_download.duration_seconds,
                    notes=[validation.reason],
                )
                download_manifest.mark(entry)
                report_entries.append(entry)
                if spec.required:
                    break
                continue

            sha256 = validation.sha256
            if not sha256:
                sha256 = self.validator.compute_sha256(hf_download.path)
            decision = self.cache.link_or_copy(
                AssetCandidate(source=SOURCE_HF_DOWNLOAD, path=hf_download.path),
                destination,
            )
            entry = AssetStateEntry(
                asset_key=spec.key,
                status="READY",
                source=SOURCE_HF_DOWNLOAD,
                path=decision.destination_path,
                size_bytes=validation.size_bytes,
                sha256=sha256,
                started_at=started,
                completed_at=datetime.now().isoformat(),
                download_seconds=hf_download.duration_seconds,
                notes=[decision.action],
            )
            download_manifest.mark(entry)
            report_entries.append(entry)

            if self.drive_client is not None and self.config.enable_drive_model_cache and drive_models_folder_id:
                try:
                    self.drive_client.upload_file(
                        local_path=Path(entry.path),
                        remote_name=spec.filename.split("/")[-1],
                        folder_id=drive_models_folder_id,
                    )
                except Exception as exc:
                    report_notes.append(str(exc))

        self._write_download_manifest(download_manifest)
        self.cleanup.cleanup_temp_root()
        ready = all(
            self.validator.validate(spec, cache_root / spec.filename).ok
            for spec in manifest.required_assets()
        )
        report = AssetReport(
            backend=backend,
            ready=ready,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            entries=report_entries,
            notes=report_notes,
        )
        self._write_reports(report)
        self._apply_config_paths(backend)
        return report

    def _apply_config_paths(self, backend: str) -> None:
        if backend == "wan2gp":
            self.config.wan2gp_model_dir = (self.paths.cache_root / backend).resolve(strict=False)
            self.config.extra["wan2gp_model_dir"] = str(self.config.wan2gp_model_dir)
        if backend == "diffusers":
            self.config.extra["diffusers_model_dir"] = str(
                (self.paths.cache_root / backend / self.config.model_name.replace("/", "__")).resolve(strict=False)
            )

    def _write_reports(self, report: AssetReport) -> None:
        self.report_writer.write_json(report)
        self.report_writer.write_text(report)
        self.report_writer.write_markdown(report)
        self.config.extra["asset_report"] = report.to_dict()
