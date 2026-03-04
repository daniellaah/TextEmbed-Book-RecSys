from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
