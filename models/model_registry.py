import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


QUANTIZATION_PATTERNS = {
    "GGUF": re.compile(r"Q[0-9]+_?[A-Z]?", re.IGNORECASE),
    "GPTQ": re.compile(r"gptq", re.IGNORECASE),
    "AWQ": re.compile(r"awq", re.IGNORECASE),
    "FP16": re.compile(r"fp16", re.IGNORECASE),
    "BF16": re.compile(r"bf16", re.IGNORECASE),
}


@dataclass(frozen=True)
class ModelEntry:
    logical_name: str
    actual_path: Path
    dataset_name: Optional[str]
    backend: str
    model_type: str
    precision: str
    quantization: str
    size: int
    checksum: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        actual_path_str = str(self.actual_path) if self.actual_path and self.actual_path != Path() else ""
        return {
            "asset_key": self.logical_name,
            "logical_name": self.logical_name,
            "actual_path": actual_path_str,
            "dataset_name": self.dataset_name,
            "backend": self.backend,
            "model_type": self.model_type,
            "precision": self.precision,
            "quantization": self.quantization,
            "size": self.size,
            "checksum": self.checksum,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}

    def add(self, entry: ModelEntry) -> None:
        self._entries[entry.logical_name] = entry

    def get(self, logical_name: str) -> Optional[ModelEntry]:
        return self._entries.get(logical_name)

    def entries(self) -> dict[str, ModelEntry]:
        return dict(self._entries)

    def status(self) -> dict[str, Any]:
        found = {k: v for k, v in self._entries.items() if v.status == "found"}
        missing = [k for k, v in self._entries.items() if v.status != "found"]
        return {
            "found_count": len(found),
            "missing": missing,
            "entries": [v.to_dict() for v in self._entries.values()],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.status()

    def write_reports(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        json_path = output_dir / "model_registry.json"
        json_path.write_text(
            __import__("json").dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths["json"] = json_path

        md_lines = ["# Model Registry", ""]
        md_lines.append(f"- found_count: {self.status()['found_count']}")
        md_lines.append(f"- missing: {', '.join(self.status()['missing']) or 'none'}")
        md_lines.append("")
        md_lines.append("| Logical Name | Backend | Model Type | Precision | Quantization | Size | Status |")
        md_lines.append("|---|---|---|---|---|---|---|")
        for entry in self._entries.values():
            md_lines.append(
                f"| {entry.logical_name} | {entry.backend} | {entry.model_type} "
                f"| {entry.precision} | {entry.quantization} | {entry.size} | {entry.status} |"
            )
        md_path = output_dir / "model_registry.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        paths["markdown"] = md_path

        txt_lines = ["=" * 80, "MODEL REGISTRY", "=" * 80]
        txt_lines.append(f"found_count: {self.status()['found_count']}")
        txt_lines.append(f"missing: {', '.join(self.status()['missing']) or 'none'}")
        txt_lines.append("")
        for entry in self._entries.values():
            txt_lines.append(f"- {entry.logical_name}:")
            txt_lines.append(f"    backend: {entry.backend}")
            txt_lines.append(f"    model_type: {entry.model_type}")
            txt_lines.append(f"    precision: {entry.precision}")
            txt_lines.append(f"    quantization: {entry.quantization}")
            txt_lines.append(f"    size: {entry.size}")
            txt_lines.append(f"    path: {entry.actual_path}")
            txt_lines.append(f"    dataset: {entry.dataset_name}")
            txt_lines.append(f"    status: {entry.status}")
            txt_lines.append("")
        txt_path = output_dir / "model_registry.txt"
        txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
        paths["text"] = txt_path
        return paths
