"""Validate the staged nomic-embed-text-v1.5 mirror against the upstream PyTorch model.

Compares last_hidden_state and the pooled embeddings (full 768 and Matryoshka-truncated) between the staged
ONNX file and the upstream PyTorch weights, and writes reference embeddings for inference4j model tests.

    python validation/nomic_embed_validate.py <staged model dir> [--out validation/reference/nomic-embed-text-v1.5.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

UPSTREAM = "nomic-ai/nomic-embed-text-v1.5"
REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
MATRYOSHKA_DIMS = (512, 256, 128, 64)

QUERIES = (
    "search_query: What is the capital of France?",
    "search_query: How do vaccines train the immune system?",
)
DOCUMENTS = (
    "search_document: Paris is the capital and largest city of France.",
    "search_document: Vaccines expose the immune system to a harmless piece of a pathogen so it can build a response.",
    "search_document: The Great Barrier Reef is the world's largest coral reef system.",
)
LONG_PARAGRAPH = (
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. "
    "She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two sciences. "
)


def _mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None].astype(np.float32)
    return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)


def _l2(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _matryoshka(pooled: np.ndarray, dim: int, eps: float = 1e-5) -> np.ndarray:
    # Model card recipe: layer_norm over the full vector (no affine), truncate, then L2-normalize.
    mu = pooled.mean(axis=-1, keepdims=True)
    var = pooled.var(axis=-1, keepdims=True)
    normed = (pooled - mu) / np.sqrt(var + eps)
    return _l2(normed[:, :dim])


def _run_onnx(session, enc) -> np.ndarray:
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "token_type_ids": np.zeros_like(enc["input_ids"], dtype=np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    return session.run(["last_hidden_state"], feeds)[0]


def _run_torch(model, enc) -> np.ndarray:
    import torch

    with torch.no_grad():
        out = model(
            input_ids=torch.from_numpy(enc["input_ids"]),
            attention_mask=torch.from_numpy(enc["attention_mask"]),
            token_type_ids=torch.zeros_like(torch.from_numpy(enc["input_ids"])),
        )
    return out[0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import onnxruntime as ort
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(UPSTREAM, revision=REVISION)
    model = AutoModel.from_pretrained(UPSTREAM, revision=REVISION, trust_remote_code=True)
    model.eval()
    session = ort.InferenceSession(str(args.model_dir / "model.onnx"), providers=["CPUExecutionProvider"])

    texts = list(QUERIES + DOCUMENTS)
    long_text = "search_document: " + LONG_PARAGRAPH * 80
    report: dict = {"upstream": UPSTREAM, "revision": REVISION}

    # Short batch (padded) and a long single input past the 2048 trained positions.
    worst = 0.0
    pooled = {}
    for label, batch in (("short", texts), ("long", [long_text])):
        enc = tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors="np")
        h_onnx = _run_onnx(session, enc)
        h_torch = _run_torch(model, enc)
        mask = enc["attention_mask"]
        diff = float(np.abs((h_onnx - h_torch) * mask[..., None]).max())
        p_onnx, p_torch = _mean_pool(h_onnx, mask), _mean_pool(h_torch, mask)
        cos = float((_l2(p_onnx) * _l2(p_torch)).sum(axis=-1).min())
        print(f"{label:5s} tokens={enc['input_ids'].shape[1]:5d}  max|Δ hidden|={diff:.2e}  min cos(onnx, torch)={cos:.8f}")
        report[f"{label}_tokens"] = int(enc["input_ids"].shape[1])
        report[f"{label}_max_abs_diff_hidden"] = diff
        report[f"{label}_min_cosine"] = cos
        worst = max(worst, diff)
        pooled[label] = p_torch

    full = _l2(pooled["short"])
    sims = full[: len(QUERIES)] @ full[len(QUERIES):].T
    print("query x document cosine (768):")
    for q, row in zip(QUERIES, sims):
        print(f"  {q[:50]:50s} " + "  ".join(f"{s:.4f}" for s in row))

    report["texts"] = texts
    report["embeddings_768"] = full.round(6).tolist()
    report["similarities_768"] = sims.round(6).tolist()
    report["long_text"] = {"paragraph": LONG_PARAGRAPH, "repeat": 80, "prefix": "search_document: "}
    report["long_embedding_768"] = _l2(pooled["long"]).round(6)[0].tolist()
    for dim in MATRYOSHKA_DIMS:
        report[f"embeddings_{dim}"] = _matryoshka(pooled["short"], dim).round(6).tolist()

    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    print(f"worst max|Δ hidden| = {worst:.2e}")


if __name__ == "__main__":
    main()
