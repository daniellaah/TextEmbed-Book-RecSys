from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src/embedding/generate_item_embeddings.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_item_embeddings", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generate_item_embeddings module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_module()


class GenerateItemEmbeddingsTests(unittest.TestCase):
    @staticmethod
    def _minimal_experiment_config(model_name: str) -> dict:
        return {
            "experiment_id": "exp_test",
            "model": {
                "name": model_name,
                "max_length": 16,
                "batch_size": 2,
                "normalize_embeddings": True,
            },
            "text_views": {
                "views": [
                    {
                        "view_id": "view_title",
                        "fields": ["title"],
                        "template": "Title: {title}",
                    }
                ]
            },
            "fusion": {
                "method": "identity",
                "input_views": ["view_title"],
                "normalization": False,
            },
        }

    def test_adapt_text_for_e5(self) -> None:
        self.assertEqual(mod.adapt_text_for_model("intfloat/e5-base-v2", "hello"), "passage: hello")
        self.assertEqual(
            mod.adapt_text_for_model("intfloat/e5-base-v2", "passage: keep me"),
            "passage: keep me",
        )
        self.assertEqual(mod.adapt_text_for_model("BAAI/bge-m3", "hello"), "hello")

    def test_render_view_text_missing_field(self) -> None:
        item = {"title": "T1"}
        view_cfg = {
            "view_id": "v",
            "fields": ["title", "author"],
            "template": "Title: {title}\nAuthor: {author}",
        }
        rendered = mod.render_view_text(item, view_cfg)
        self.assertEqual(rendered, "Title: T1\nAuthor:")

    def test_fuse_identity(self) -> None:
        view_embeddings = {
            "view_a": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        }
        fusion_cfg = {"method": "identity", "input_views": ["view_a"], "normalization": False}
        fused = mod.fuse_batch_embeddings(view_embeddings, fusion_cfg)
        self.assertTrue(torch.equal(fused, view_embeddings["view_a"]))

    def test_fuse_weighted_mean(self) -> None:
        view_embeddings = {
            "v1": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "v2": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        }
        fusion_cfg = {
            "method": "weighted_mean",
            "input_views": ["v1", "v2"],
            "weights": {"v1": 0.75, "v2": 0.25},
            "normalization": False,
        }
        fused = mod.fuse_batch_embeddings(view_embeddings, fusion_cfg)
        self.assertTrue(torch.allclose(fused, torch.tensor([[0.75, 0.25]], dtype=torch.float32)))

    def test_config_hash_stable(self) -> None:
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        self.assertEqual(mod.compute_config_hash(a), mod.compute_config_hash(b))

    def test_validate_model_name_rejects_path_like_values(self) -> None:
        path_like_values = ["/tmp/model", "./local-model", "../local-model", "~/local-model"]
        for model_name in path_like_values:
            with self.subTest(model_name=model_name):
                cfg = self._minimal_experiment_config(model_name=model_name)
                with self.assertRaisesRegex(ValueError, "path-like values are not allowed"):
                    mod.validate_experiment_config(cfg)

    def test_resolve_local_model_ref_uses_fixed_hf_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_home = tmp_path / "home"
            fixed_root = fake_home / ".cache" / "huggingface" / "hub"
            snapshot = fixed_root / "models--BAAI--bge-m3" / "snapshots" / "snap_ok"
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_text("", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")

            fake_hf_home = tmp_path / "custom-hf-home"
            fake_hf_home.mkdir(parents=True, exist_ok=True)
            with mock.patch.dict(os.environ, {"HF_HOME": str(fake_hf_home)}, clear=False):
                with mock.patch.object(mod.Path, "home", return_value=fake_home):
                    resolved = mod.resolve_local_model_ref("BAAI/bge-m3")
            self.assertEqual(resolved, str(snapshot))

    def test_resolve_local_model_ref_not_found_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / ".cache" / "huggingface" / "hub"
            with mock.patch.object(mod, "get_hf_cache_root", return_value=cache_root):
                with self.assertRaises(FileNotFoundError) as ctx:
                    mod.resolve_local_model_ref("BAAI/bge-m3")
            message = str(ctx.exception)
            self.assertIn("Local model snapshot not found", message)
            self.assertIn("Please pre-download this model", message)
            self.assertIn(str(cache_root), message)


if __name__ == "__main__":
    unittest.main()
