"""Validate a staged DeBERTa-v3 zero-shot export against PyTorch and the HF pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TEXTS = (
    "My invoice shows the same charge twice.",
    "The service has been unavailable since this morning.",
    "Please add dark mode to the mobile application.",
)
LABELS = ("billing", "outage", "feature request", "account access")
HYPOTHESIS_TEMPLATE = "This text is about {}."


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _ordered_pipeline_scores(result: dict, labels: tuple[str, ...]) -> np.ndarray:
    by_label = dict(zip(result["labels"], result["scores"]))
    return np.asarray([by_label[label] for label in labels], dtype=np.float32)


def validate(model_dir: Path, source_repo: str, revision: str) -> dict:
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(source_repo, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(source_repo, revision=revision)
    model.eval()

    premises = [text for text in TEXTS for _ in LABELS]
    hypotheses = [HYPOTHESIS_TEMPLATE.format(label) for _ in TEXTS for label in LABELS]
    encoded = tokenizer(premises, hypotheses, padding=True, return_tensors="pt")

    with torch.no_grad():
        pytorch_logits = model(**encoded).logits.cpu().numpy()

    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    input_names = {value.name for value in session.get_inputs()}
    ort_inputs = {
        name: value.cpu().numpy().astype(np.int64)
        for name, value in encoded.items()
        if name in input_names
    }
    onnx_logits = session.run(["logits"], ort_inputs)[0]

    shape = (len(TEXTS), len(LABELS), 2)
    onnx_by_pair = onnx_logits.reshape(shape)
    single_label_scores = _softmax(onnx_by_pair[..., 0], axis=1)
    multi_label_scores = _softmax(onnx_by_pair, axis=2)[..., 0]

    classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
    )
    pipeline_single = []
    pipeline_multi = []
    for text in TEXTS:
        common = {
            "sequences": text,
            "candidate_labels": list(LABELS),
            "hypothesis_template": HYPOTHESIS_TEMPLATE,
        }
        pipeline_single.append(_ordered_pipeline_scores(classifier(**common), LABELS))
        pipeline_multi.append(
            _ordered_pipeline_scores(classifier(**common, multi_label=True), LABELS)
        )

    pipeline_single = np.stack(pipeline_single)
    pipeline_multi = np.stack(pipeline_multi)
    pair = tokenizer(TEXTS[0], HYPOTHESIS_TEMPLATE.format(LABELS[0]))

    details = []
    for text_index, text in enumerate(TEXTS):
        details.append({
            "text": text,
            "scores": {
                label: {
                    "single_onnx": float(single_label_scores[text_index, label_index]),
                    "single_pipeline": float(pipeline_single[text_index, label_index]),
                    "multi_onnx": float(multi_label_scores[text_index, label_index]),
                    "multi_pipeline": float(pipeline_multi[text_index, label_index]),
                }
                for label_index, label in enumerate(LABELS)
            },
        })

    return {
        "source_repo": source_repo,
        "revision": revision,
        "onnx_inputs": [
            {"name": value.name, "shape": value.shape, "type": value.type}
            for value in session.get_inputs()
        ],
        "onnx_outputs": [
            {"name": value.name, "shape": value.shape, "type": value.type}
            for value in session.get_outputs()
        ],
        "id2label": model.config.id2label,
        "pair_input_ids": pair["input_ids"],
        "pair_tokens": tokenizer.convert_ids_to_tokens(pair["input_ids"]),
        "tokenizer_produced_token_type_ids": "token_type_ids" in pair,
        "pytorch_onnx_max_abs_diff": float(np.max(np.abs(pytorch_logits - onnx_logits))),
        "single_label_pipeline_max_abs_diff": float(
            np.max(np.abs(single_label_scores - pipeline_single))
        ),
        "multi_label_pipeline_max_abs_diff": float(
            np.max(np.abs(multi_label_scores - pipeline_multi))
        ),
        "cases": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("source_repo")
    parser.add_argument("revision")
    args = parser.parse_args()
    print(json.dumps(validate(args.model_dir, args.source_repo, args.revision), indent=2))


if __name__ == "__main__":
    main()
