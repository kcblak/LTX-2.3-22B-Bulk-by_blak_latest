from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from config import Config


_REQUIREMENT_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


@dataclass
class RequirementStatus:
    requirement: str
    package_name: str
    installed: bool
    version: str = ""


@dataclass
class RequirementFileReport:
    path: Path
    inspected: list[RequirementStatus] = field(default_factory=list)
    installed_requirements: list[str] = field(default_factory=list)
    skipped_requirements: list[str] = field(default_factory=list)


@dataclass
class GitCheckoutReport:
    destination: Path
    repo_url: str
    ref: str
    cloned: bool
    updated: bool


@dataclass
class ModelAssetReport:
    model_dir: Path
    downloaded_files: list[str] = field(default_factory=list)
    existing_files: list[str] = field(default_factory=list)
    runtime_report: Optional[GitCheckoutReport] = None
    requirement_reports: list[RequirementFileReport] = field(default_factory=list)


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _strip_wrapping_delimiters(value: str) -> str:
    cleaned = value.strip()
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'", "`"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _sanitize_git_text(value: str, *, field_name: str) -> str:
    cleaned = _strip_wrapping_delimiters(value)
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _installed_distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        normalized = _normalize_package_name(distribution.metadata["Name"])
        versions[normalized] = distribution.version
    return versions


def _iter_requirement_entries(requirements_path: Path) -> Iterable[tuple[str, str]]:
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement = raw_line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        if requirement.startswith(("-", "git+", "http://", "https://")):
            continue
        requirement = requirement.split(" #", 1)[0].strip()
        match = _REQUIREMENT_NAME_PATTERN.match(requirement)
        if not match:
            continue
        package_name = _normalize_package_name(match.group(1))
        yield requirement, package_name


def inspect_requirement_file(requirements_path: Path) -> RequirementFileReport:
    installed_versions = _installed_distribution_versions()
    report = RequirementFileReport(path=requirements_path)
    if not requirements_path.exists():
        return report

    for requirement, package_name in _iter_requirement_entries(requirements_path):
        version = installed_versions.get(package_name, "")
        report.inspected.append(
            RequirementStatus(
                requirement=requirement,
                package_name=package_name,
                installed=bool(version),
                version=version,
            )
        )
    return report


def ensure_requirement_file(requirements_path: Path) -> RequirementFileReport:
    report = inspect_requirement_file(requirements_path)
    missing_requirements = [
        status.requirement for status in report.inspected if not status.installed
    ]
    if missing_requirements:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing_requirements]
        )
        report.installed_requirements.extend(missing_requirements)
    report.skipped_requirements.extend(
        [status.requirement for status in report.inspected if status.installed]
    )
    return report


def ensure_git_checkout(destination: Path, repo_url: str, ref: str = "main") -> GitCheckoutReport:
    destination = destination.resolve(strict=False)
    repo_url = _sanitize_git_text(repo_url, field_name="repo_url")
    ref = _sanitize_git_text(ref, field_name="ref")
    cloned = False
    updated = False

    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(destination)]
        )
        cloned = True
    elif (destination / ".git").exists():
        try:
            subprocess.check_call(["git", "-C", str(destination), "rev-parse", "--verify", ref])
            subprocess.check_call(["git", "-C", str(destination), "checkout", ref])
        except subprocess.CalledProcessError:
            subprocess.check_call(["git", "-C", str(destination), "checkout", "-b", ref, f"origin/{ref}"])
        updated = True

    return GitCheckoutReport(
        destination=destination,
        repo_url=repo_url,
        ref=ref,
        cloned=cloned,
        updated=updated,
    )


def ensure_wan2gp_runtime(config: Config) -> GitCheckoutReport:
    return ensure_git_checkout(
        destination=config.wan2gp_dir,
        repo_url=config.wan2gp_repo_url,
        ref=config.wan2gp_repo_ref,
    )


def _download_file(
    *,
    repo_id: str,
    filename: str,
    local_dir: Path,
    target_path: Path,
    report: ModelAssetReport,
) -> None:
    if target_path.exists():
        report.existing_files.append(str(target_path))
        return

    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    downloaded_path = local_dir / filename
    downloaded_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if downloaded_path != target_path and downloaded_path.exists():
        downloaded_path.replace(target_path)
    report.downloaded_files.append(str(target_path))


def ensure_wan2gp_model_assets(config: Config) -> ModelAssetReport:
    model_dir = (
        config.wan2gp_model_dir.resolve(strict=False)
        if config.wan2gp_model_dir is not None
        else (config.wan2gp_dir / "models").resolve(strict=False)
    )
    report = ModelAssetReport(model_dir=model_dir)

    runtime_requirements = config.wan2gp_dir / "requirements.txt"
    if runtime_requirements.exists():
        report.requirement_reports.append(ensure_requirement_file(runtime_requirements))

    transformer_target = model_dir / config.wan2gp_transformer_filename
    _download_file(
        repo_id=config.wan2gp_transformer_repo_id,
        filename=config.wan2gp_transformer_source_filename,
        local_dir=model_dir,
        target_path=transformer_target,
        report=report,
    )

    for filename in config.wan2gp_required_companion_files:
        target = model_dir / filename
        _download_file(
            repo_id=config.wan2gp_companion_repo_id,
            filename=filename,
            local_dir=model_dir,
            target_path=target,
            report=report,
        )

    text_encoder_dir = model_dir / config.wan2gp_text_encoder_dirname
    for filename in config.wan2gp_required_text_encoder_files:
        relative_name = f"{config.wan2gp_text_encoder_dirname}/{filename}"
        target = text_encoder_dir / filename
        _download_file(
            repo_id=config.wan2gp_companion_repo_id,
            filename=relative_name,
            local_dir=model_dir,
            target_path=target,
            report=report,
        )

    if config.wan2gp_msr_enabled:
        lora_target = model_dir / config.wan2gp_lora_filename
        _download_file(
            repo_id=config.wan2gp_companion_repo_id,
            filename=config.wan2gp_lora_source_path,
            local_dir=model_dir,
            target_path=lora_target,
            report=report,
        )

    return report
