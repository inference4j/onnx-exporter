"""Record GLiNER ONNX inputs and verify parity with pinned PyTorch weights.

Requires ``gliner==0.2.29``. This is intentionally a validation utility rather
than an exporter dependency: the published artifacts are byte-for-byte mirrors.

Examples:
    python validation/gliner_validate.py gliner-small-v2.1 \
        --candidate-root /tmp/gliner-candidate --output /tmp/gliner-small.json
    python validation/gliner_validate.py gliner-pii-base-v1.0 \
        --candidate-root /tmp/gliner-candidate --output /tmp/gliner-pii.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from gliner import GLiNER


GLINER_VERSION = "0.2.29"
LABELS = ["person", "organization", "invoice id", "date"]
WORKED_TEXT = "Alice Johnson paid invoice INV-2024-0042 to Acme Corp on 3 March."
TEST_TEXTS = [
    WORKED_TEXT,
    "Maria Gomez at Globex Corporation approved invoice Q4-778 on 14 July 2025.",
    "Alice Johnson joined Acme Corp on 3 March with invoice INV-2024-0042. "
    + " ".join(f"filler{i}" for i in range(385)),
]
INPUT_NAMES = (
    "input_ids",
    "attention_mask",
    "words_mask",
    "text_lengths",
    "span_idx",
    "span_mask",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    pytorch_repo: str
    pytorch_revision: str
    onnx_revision: str
    thresholds: tuple[float, ...]


SPECS = {
    "gliner-small-v2.1": ModelSpec(
        name="gliner-small-v2.1",
        pytorch_repo="urchade/gliner_small-v2.1",
        pytorch_revision="4e091416cf7c3481db542c2a3d26156916f3a47f",
        onnx_revision="8142fb00740ccea973e64b1272949ff48653df5e",
        thresholds=(0.5,),
    ),
    "gliner-pii-base-v1.0": ModelSpec(
        name="gliner-pii-base-v1.0",
        pytorch_repo="knowledgator/gliner-pii-base-v1.0",
        pytorch_revision="61726e0ad791dcab3e29339bbec3ad42ded65641",
        onnx_revision="61726e0ad791dcab3e29339bbec3ad42ded65641",
        thresholds=(0.5, 0.3),
    ),
}


def _batch(model: GLiNER, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = model.prepare_batch(text, LABELS)
    batch = model.collate_batch(
        prepared["input_x"],
        prepared["entity_types"],
        label_names=prepared["label_names"],
    )
    return prepared, batch


def _tensor_record(value: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "values": value.detach().cpu().tolist(),
    }


def _contract_item(value: Any) -> dict[str, Any]:
    return {"name": value.name, "dtype": value.type, "shape": list(value.shape)}


def _entity_identity(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: entity[key] for key in ("start", "end", "text", "label")}
        for entity in entities
    ]


def _max_score_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    if not left:
        return 0.0
    return max(abs(float(a["score"]) - float(b["score"])) for a, b in zip(left, right, strict=True))


def _predict_and_capture(model: GLiNER, text: str, threshold: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    captured: list[np.ndarray] = []
    original = model.run_batch

    def run_and_capture(*args, **kwargs):
        output = original(*args, **kwargs)
        logits = output[0]
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        captured.append(np.asarray(logits))
        return output

    model.run_batch = run_and_capture
    try:
        entities = model.predict_entities(
            text,
            LABELS,
            flat_ner=True,
            threshold=threshold,
            multi_label=False,
        )
    finally:
        model.run_batch = original
    assert len(captured) == 1
    return entities, captured[0]


def _top_valid_spans(logits: np.ndarray, span_mask: torch.Tensor, limit: int = 8) -> list[dict[str, Any]]:
    valid = span_mask.detach().cpu().numpy().reshape(logits.shape[:3])
    masked = np.where(valid[..., None], logits, -np.inf)
    flat_indices = np.argsort(masked.reshape(-1), kind="stable")[::-1][:limit]
    result = []
    for flat_index in flat_indices:
        batch, start, width, label_index = np.unravel_index(int(flat_index), masked.shape)
        logit = float(masked.reshape(-1)[flat_index])
        result.append(
            {
                "batch": int(batch),
                "word_start": int(start),
                "width_index": int(width),
                "word_count": int(width) + 1,
                "label_index": int(label_index),
                "label": LABELS[int(label_index)],
                "logit": logit,
                "score": float(1.0 / (1.0 + np.exp(-logit))),
            }
        )
    return result


def validate(spec: ModelSpec, candidate_root: Path) -> dict[str, Any]:
    installed = importlib.metadata.version("gliner")
    if installed != GLINER_VERSION:
        raise RuntimeError(f"Expected gliner=={GLINER_VERSION}, found {installed}")

    model_dir = candidate_root / "inference4j" / spec.name
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_model = GLiNER.from_pretrained(str(model_dir), load_onnx_model=True)
    pytorch_model = GLiNER.from_pretrained(
        spec.pytorch_repo,
        revision=spec.pytorch_revision,
        map_location="cpu",
    )

    prepared, onnx_batch = _batch(onnx_model, WORKED_TEXT)
    _, pytorch_batch = _batch(pytorch_model, WORKED_TEXT)
    for name in INPUT_NAMES:
        if not torch.equal(onnx_batch[name], pytorch_batch[name]):
            raise AssertionError(f"Worked-example tensor mismatch for {name}")

    parity = []
    worked_logits = None
    for text_index, text in enumerate(TEST_TEXTS):
        per_threshold = []
        for threshold in spec.thresholds:
            onnx_entities, onnx_logits = _predict_and_capture(onnx_model, text, threshold)
            pytorch_entities, pytorch_logits = _predict_and_capture(pytorch_model, text, threshold)
            if onnx_logits.shape != pytorch_logits.shape:
                raise AssertionError(
                    f"Logit shape mismatch for text {text_index}: "
                    f"{onnx_logits.shape} != {pytorch_logits.shape}"
                )
            max_abs_diff = float(np.max(np.abs(onnx_logits - pytorch_logits)))
            identities_match = _entity_identity(onnx_entities) == _entity_identity(pytorch_entities)
            if not identities_match:
                raise AssertionError(f"Entity identity mismatch for text {text_index} at {threshold}")
            if text_index == 0 and threshold == spec.thresholds[0]:
                worked_logits = onnx_logits
            per_threshold.append(
                {
                    "threshold": threshold,
                    "logits_shape": list(onnx_logits.shape),
                    "max_abs_logit_diff": max_abs_diff,
                    "entity_identities_identical": identities_match,
                    "max_entity_score_diff": _max_score_delta(onnx_entities, pytorch_entities),
                    "onnx_entities": onnx_entities,
                    "pytorch_entities": pytorch_entities,
                }
            )
        parity.append(
            {
                "text_index": text_index,
                "word_count_before_truncation": len(prepared["tokens"][0])
                if text_index == 0
                else len(onnx_model.prepare_batch(text, LABELS)["tokens"][0]),
                "thresholds": per_threshold,
            }
        )

    assert worked_logits is not None
    tokens = prepared["tokens"][0]
    worked = {
        "text": WORKED_TEXT,
        "labels_in_prompt_order": LABELS,
        "words": [
            {
                "index": index,
                "text": token,
                "char_start": prepared["start_token_map"][0][index],
                "char_end_exclusive": prepared["end_token_map"][0][index],
            }
            for index, token in enumerate(tokens)
        ],
        "inputs": {name: _tensor_record(onnx_batch[name]) for name in INPUT_NAMES},
        "raw_logits_shape": list(worked_logits.shape),
        "raw_logits_range": [float(worked_logits.min()), float(worked_logits.max())],
        "top_valid_spans": _top_valid_spans(worked_logits, onnx_batch["span_mask"]),
    }

    return {
        "model": spec.name,
        "gliner_package_version": installed,
        "onnx_revision": spec.onnx_revision,
        "pytorch_source": spec.pytorch_repo,
        "pytorch_revision": spec.pytorch_revision,
        "onnx_contract": {
            "inputs": [_contract_item(value) for value in session.get_inputs()],
            "outputs": [_contract_item(value) for value in session.get_outputs()],
        },
        "worked_example": worked,
        "parity": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(SPECS))
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = validate(SPECS[args.model], args.candidate_root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
        print(f"Wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
