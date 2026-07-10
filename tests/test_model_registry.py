import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from models.model_registry import ModelEntry, ModelRegistry
from models.model_resolver import ModelResolver
from orchestration.runtime_assets import ensure_model_registry


class ModelRegistryTests(unittest.TestCase):
    def test_model_entry_serializes(self):
        entry = ModelEntry(
            logical_name="transformer",
            actual_path=Path("/kaggle/input/ltx-models/ltx-2.3-22b.gguf"),
            dataset_name="ltx-models",
            backend="wan2gp",
            model_type="transformer",
            precision="Q3_K_M",
            quantization="GGUF",
            size=1024,
            checksum="abc123",
            status="found",
        )
        registry = ModelRegistry()
        registry.add(entry)
        self.assertEqual(registry.get("transformer").logical_name, "transformer")
        self.assertEqual(registry.status()["found_count"], 1)

    def test_model_resolver_discovers_quantized_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            q4_model = root / "ltx-2.3-22b_distilled-1.1-Q4_K_M.gguf"
            q4_model.write_bytes(b"gguf")
            fp16_model = root / "ltx-2.3-22b_distilled-1.1-fp16.gguf"
            fp16_model.write_bytes(b"gguf")

            resolver = ModelResolver(backend="wan2gp", search_roots=[root])
            registry = resolver.discover(
                required_types=["transformer"],
                candidates={"transformer": ["ltx-2.3-22b_distilled-1.1-Q4_K_M.gguf"]},
            )

            entry = registry.get("transformer")
            self.assertIsNotNone(entry)
            self.assertEqual(entry.quantization, "GGUF")
            self.assertEqual(entry.precision, "Q4_K_M")
            self.assertEqual(entry.status, "found")

    def test_ensure_model_registry_populates_config_extra(self):
        config = Config(wan2gp_transformer_filename="missing.gguf")
        registry = ensure_model_registry(config)
        self.assertIn("model_registry", config.extra)
        self.assertIsInstance(registry.status(), dict)
        self.assertIn("entries", registry.status())
        self.assertIn("missing", registry.status())

    def test_model_entry_missing_status(self):
        registry = ModelRegistry()
        registry.add(
            ModelEntry(
                logical_name="vae",
                actual_path=Path("/missing/path"),
                dataset_name=None,
                backend="wan2gp",
                model_type="vae",
                precision="unknown",
                quantization="UNKNOWN",
                size=0,
                checksum="",
                status="missing",
            )
        )
        self.assertEqual(registry.status()["found_count"], 0)
        self.assertIn("vae", registry.status()["missing"])


if __name__ == "__main__":
    unittest.main()
