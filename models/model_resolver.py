from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from models.model_registry import ModelEntry, ModelRegistry, QUANTIZATION_PATTERNS


_QUANTIZATION_KEYWORDS = {
    "GPTQ": ("gptq",),
    "AWQ": ("awq",),
    "FP16": ("fp16", "float16"),
    "BF16": ("bf16",),
    "GGUF": ("q8", "q6", "q5", "q4", "q3", "q2", "gguf"),
}


def _detect_quantization(name: str) -> tuple[str, str]:
    lower = name.lower()
    for qtype, keywords in _QUANTIZATION_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                q_match = re.search(r'Q[0-9]+(?:_[A-Z0-9]+)*', name, re.IGNORECASE)
                if q_match:
                    return qtype, q_match.group(0).upper()
                prefix_end = lower.find(kw) + len(kw)
                precision = name[prefix_end:].lstrip("-_.")
                if not precision:
                    precision = kw.upper()
                return qtype, precision
    if "bf16" in lower:
        return "BF16", "bf16"
    if "fp16" in lower or "float16" in lower:
        return "FP16", "fp16"
    return "UNKNOWN", "unknown"


def _sha256_for(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()
    except Exception:
        return ""


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return 0


@dataclass(frozen=True)
class _DiscoveryRoot:
    source: str
    path: Path
    dataset_name: Optional[str] = None


class ModelResolver:
    def __init__(
        self,
        *,
        backend: str = "wan2gp",
        search_roots: Optional[list[Path]] = None,
    ) -> None:
        self.backend = backend
        self.search_roots = list(search_roots or [])

    def _default_roots(self) -> list[_DiscoveryRoot]:
        roots: list[_DiscoveryRoot] = []
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            roots.append(_DiscoveryRoot(source="dataset", path=kaggle_input, dataset_name=None))
            try:
                for child in kaggle_input.iterdir():
                    if child.is_dir():
                        roots.append(
                            _DiscoveryRoot(
                                source="dataset",
                                path=child,
                                dataset_name=child.name,
                            )
                        )
            except Exception:
                pass
        for root in self.search_roots:
            roots.append(_DiscoveryRoot(source="configured", path=root, dataset_name=root.name))
        return roots

    def discover(
        self,
        required_types: list[str],
        candidates: dict[str, list[str]],
    ) -> ModelRegistry:
        registry = ModelRegistry()
        roots = self._default_roots()
        matched_paths: dict[str, list[Path]] = {rtype: [] for rtype in required_types}
        root_map: dict[Path, _DiscoveryRoot] = {r.path: r for r in roots}

        for root in roots:
            for rtype, names in candidates.items():
                for name in names:
                    candidate = root.path / name
                    if candidate.exists():
                        matched_paths[rtype].append(candidate)

        for rtype in required_types:
            paths = matched_paths.get(rtype, [])
            chosen: Optional[Path] = None
            if paths:
                quantized = [p for p in paths if any(kw in p.name.lower() for kw in ("q4", "q3", "q5", "q6", "q8", "gguf"))]
                if quantized:
                    chosen = sorted(quantized, key=lambda p: p.stat().st_size)[0]
                else:
                    chosen = sorted(paths, key=lambda p: p.stat().st_size)[0]
            if chosen is None:
                continue
            root_info = root_map.get(chosen.parent.parent, root_map.get(chosen.parent, _DiscoveryRoot(source="unknown", path=chosen.parent)))
            quantization, precision = _detect_quantization(chosen.name)
            registry.add(
                ModelEntry(
                    logical_name=rtype,
                    actual_path=chosen.resolve(),
                    dataset_name=root_info.dataset_name,
                    backend=self.backend,
                    model_type=rtype,
                    precision=precision,
                    quantization=quantization,
                    size=_file_size(chosen),
                    checksum=_sha256_for(chosen),
                    status="found",
                )
            )
        return registry

    def build_wan2gp_registry(self, config: Any) -> ModelRegistry:
        candidates = {
            "transformer": [config.wan2gp_transformer_filename],
            "text_encoder": [config.wan2gp_text_encoder_dirname],
            "vae": config.wan2gp_required_companion_files[2:4],
            "lora": [config.wan2gp_lora_filename],
            "tokenizer": [f for f in config.wan2gp_required_text_encoder_files if "tokenizer" in f],
        }
        required = ["transformer", "text_encoder", "vae", "lora", "tokenizer"]
        return self.discover(required, {k: v for k, v in candidates.items() if k in required})

    def build_diffusers_registry(self, config: Any) -> ModelRegistry:
        model_id = getattr(config, "model_name", "")
        if not model_id:
            return ModelRegistry()
        candidates = {
            "diffusers_model": [model_id.replace("/", "__")],
        }
        return self.discover(["diffusers_model"], candidates)
