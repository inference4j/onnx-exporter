"""Validate SAM 2.1 ONNX parity and generate Java reference fixtures.

The fixture images are deterministic drawings created by this script, so they carry no
third-party image licensing requirements.

Example:
    python validation/sam2_validate.py \
        /tmp/sam21-candidate/inference4j/sam2.1-hiera-tiny
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from transformers import Sam2Model, Sam2Processor


ONNX_SOURCE = "onnx-community/sam2.1-hiera-tiny-ONNX"
ONNX_REVISION = "814a066640debee5a91e70aa401fb8e17e030503"
PYTORCH_SOURCE = "facebook/sam2.1-hiera-tiny"
PYTORCH_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sam2.1-hiera-tiny"
REFERENCE_PATH = Path(__file__).parent / "reference" / "sam2.1-hiera-tiny.json"


@dataclass(frozen=True)
class FixtureCase:
    name: str
    image: str
    points: list[list[float]] | None = None
    labels: list[int] | None = None
    boxes: list[list[float]] | None = None


CASES = (
    FixtureCase(
        name="one-point",
        image="non-square.png",
        points=[[250.0, 220.0]],
        labels=[1],
    ),
    FixtureCase(
        name="box-plus-point",
        image="non-square.png",
        points=[[250.0, 220.0]],
        labels=[1],
        boxes=[[105.0, 65.0, 405.0, 375.0]],
    ),
    FixtureCase(
        name="two-points-fg-bg",
        image="square.png",
        points=[[180.0, 260.0], [390.0, 120.0]],
        labels=[1, 0],
    ),
    FixtureCase(
        name="one-box",
        image="square.png",
        boxes=[[70.0, 115.0, 295.0, 405.0]],
    ),
)


def _create_fixture_images(images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)

    non_square = Image.new("RGB", (640, 480), (220, 225, 235))
    draw = ImageDraw.Draw(non_square)
    draw.rectangle((120, 80, 390, 360), fill=(220, 45, 45), outline=(80, 0, 0), width=8)
    draw.ellipse((420, 160, 590, 330), fill=(40, 120, 220), outline=(0, 40, 100), width=8)
    draw.line((0, 430, 640, 390), fill=(70, 145, 75), width=24)
    non_square.save(images_dir / "non-square.png")

    square = Image.new("RGB", (512, 512), (242, 226, 190))
    draw = ImageDraw.Draw(square)
    draw.ellipse((80, 125, 285, 400), fill=(45, 165, 95), outline=(10, 75, 35), width=8)
    draw.polygon(
        ((340, 55), (465, 220), (315, 245)),
        fill=(150, 75, 195),
        outline=(65, 20, 100),
    )
    draw.rectangle((325, 330, 470, 455), fill=(235, 160, 35), outline=(110, 65, 5), width=8)
    square.save(images_dir / "square.png")


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(path),
        options,
        providers=["CPUExecutionProvider"],
    )


def _contract(values: list[Any]) -> list[dict[str, Any]]:
    return [
        {"name": value.name, "dtype": value.type, "shape": list(value.shape)}
        for value in values
    ]


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return 1.0 if union == 0 else float(intersection / union)


def _prompt_kwargs(case: FixtureCase) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if case.points is not None:
        result["input_points"] = [[case.points]]
        result["input_labels"] = [[case.labels]]
    if case.boxes is not None:
        result["input_boxes"] = [case.boxes]
    return result


def _onnx_prompts(inputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = inputs.get("input_points")
    labels = inputs.get("input_labels")
    boxes = inputs.get("input_boxes")
    if points is None:
        points_array = np.empty((1, 1, 0, 2), dtype=np.float32)
        labels_array = np.empty((1, 1, 0), dtype=np.int64)
    else:
        points_array = points.cpu().numpy().astype(np.float32)
        labels_array = labels.cpu().numpy().astype(np.int64)
    if boxes is None:
        boxes_array = np.empty((1, 0, 4), dtype=np.float32)
    else:
        boxes_array = boxes.cpu().numpy().astype(np.float32)
    return points_array, labels_array, boxes_array


def _postprocess(
    processor: Sam2Processor,
    masks: torch.Tensor,
    original_sizes: torch.Tensor,
) -> np.ndarray:
    return processor.post_process_masks(masks, original_sizes)[0].cpu().numpy()


def validate(model_dir: Path, fixture_root: Path, reference_path: Path) -> dict[str, Any]:
    images_dir = fixture_root / "images"
    masks_dir = fixture_root / "masks"
    cases_dir = fixture_root / "cases"
    masks_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    _create_fixture_images(images_dir)

    processor = Sam2Processor.from_pretrained(PYTORCH_SOURCE, revision=PYTORCH_REVISION)
    pytorch = Sam2Model.from_pretrained(PYTORCH_SOURCE, revision=PYTORCH_REVISION).eval()
    encoder = _session(model_dir / "vision_encoder.onnx")
    decoder = _session(model_dir / "prompt_encoder_mask_decoder.onnx")

    decoder_input_names = [value.name for value in decoder.get_inputs()]
    if "mask_input" in decoder_input_names or "input_masks" in decoder_input_names:
        raise AssertionError("This export unexpectedly acquired a previous-mask input")

    image_cache: dict[str, dict[str, Any]] = {}
    encoder_reports = []
    case_reports = []

    for case in CASES:
        if case.image not in image_cache:
            image = Image.open(images_dir / case.image).convert("RGB")
            base_inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                pytorch_embeddings = pytorch.get_image_embeddings(base_inputs["pixel_values"])
            onnx_embeddings = encoder.run(
                None,
                {"pixel_values": base_inputs["pixel_values"].cpu().numpy()},
            )
            embedding_diffs = [
                float(np.max(np.abs(pt.cpu().numpy() - onnx)))
                for pt, onnx in zip(pytorch_embeddings, onnx_embeddings, strict=True)
            ]
            if max(embedding_diffs) > 1e-3:
                raise AssertionError(
                    f"Encoder parity failed for {case.image}: {max(embedding_diffs)}"
                )
            encoder_reports.append(
                {
                    "image": f"images/{case.image}",
                    "original_size": [image.width, image.height],
                    "embedding_shapes": [list(value.shape) for value in onnx_embeddings],
                    "embedding_max_abs_diff": embedding_diffs,
                }
            )
            image_cache[case.image] = {
                "image": image,
                "pytorch_embeddings": pytorch_embeddings,
                "onnx_embeddings": onnx_embeddings,
            }

        cached = image_cache[case.image]
        image = cached["image"]
        inputs = processor(images=image, return_tensors="pt", **_prompt_kwargs(case))
        pytorch_kwargs = {
            name: inputs[name]
            for name in ("input_points", "input_labels", "input_boxes")
            if name in inputs
        }
        with torch.no_grad():
            pytorch_canonical = pytorch(
                image_embeddings=cached["pytorch_embeddings"],
                **pytorch_kwargs,
            )

        points, labels, boxes = _onnx_prompts(inputs)
        contract_kwargs = dict(pytorch_kwargs)
        if "input_points" not in contract_kwargs:
            contract_kwargs["input_points"] = torch.from_numpy(points)
            contract_kwargs["input_labels"] = torch.from_numpy(labels)
        with torch.no_grad():
            pytorch_contract = pytorch(
                image_embeddings=cached["pytorch_embeddings"],
                **contract_kwargs,
            )

        embedding_feed = {
            value.name: embedding
            for value, embedding in zip(
                encoder.get_outputs(),
                cached["onnx_embeddings"],
                strict=True,
            )
        }
        onnx_iou, onnx_masks, onnx_object = decoder.run(
            None,
            {
                **embedding_feed,
                "input_points": points,
                "input_labels": labels,
                "input_boxes": boxes,
            },
        )

        contract_iou_diff = float(
            np.max(np.abs(pytorch_contract.iou_scores.cpu().numpy() - onnx_iou))
        )
        if contract_iou_diff > 1e-3:
            raise AssertionError(
                f"Decoder IoU score parity failed for {case.name}: {contract_iou_diff}"
            )

        onnx_post = _postprocess(
            processor,
            torch.from_numpy(onnx_masks),
            inputs["original_sizes"],
        )
        canonical_post = _postprocess(
            processor,
            pytorch_canonical.pred_masks,
            inputs["original_sizes"],
        )
        contract_post = _postprocess(
            processor,
            pytorch_contract.pred_masks,
            inputs["original_sizes"],
        )
        selected = int(np.argmax(onnx_iou[0, 0]))
        canonical_mask_iou = _mask_iou(onnx_post[0, selected], canonical_post[0, selected])
        contract_mask_iou = _mask_iou(onnx_post[0, selected], contract_post[0, selected])
        if canonical_mask_iou < 0.99:
            raise AssertionError(
                f"Post-processed mask parity failed for {case.name}: {canonical_mask_iou}"
            )

        mask_name = f"{case.name}.png"
        Image.fromarray(
            onnx_post[0, selected].astype(np.uint8) * 255,
            mode="L",
        ).save(masks_dir / mask_name)

        case_report = {
            "name": case.name,
            "image": f"images/{case.image}",
            "mask": f"masks/{mask_name}",
            "original_image_size": [image.width, image.height],
            "prompt_original_coordinates": {
                "points": case.points or [],
                "labels": case.labels or [],
                "boxes": case.boxes or [],
            },
            "onnx_prompt_shapes": {
                "input_points": list(points.shape),
                "input_labels": list(labels.shape),
                "input_boxes": list(boxes.shape),
            },
            "iou_scores": onnx_iou[0, 0].tolist(),
            "selected_mask_index": selected,
            "object_score_logits": onnx_object[0, 0].tolist(),
            "parity": {
                "contract_pytorch_iou_scores_max_abs_diff": contract_iou_diff,
                "contract_pytorch_mask_logits_max_abs_diff": float(
                    np.max(np.abs(pytorch_contract.pred_masks.cpu().numpy() - onnx_masks))
                ),
                "contract_pytorch_object_score_max_abs_diff": float(
                    np.max(
                        np.abs(
                            pytorch_contract.object_score_logits.cpu().numpy() - onnx_object
                        )
                    )
                ),
                "contract_pytorch_binary_mask_iou": contract_mask_iou,
                "canonical_processor_pytorch_iou_scores_max_abs_diff": float(
                    np.max(np.abs(pytorch_canonical.iou_scores.cpu().numpy() - onnx_iou))
                ),
                "canonical_processor_pytorch_binary_mask_iou": canonical_mask_iou,
            },
        }
        (cases_dir / f"{case.name}.json").write_text(
            json.dumps(case_report, indent=2) + "\n"
        )
        case_reports.append(case_report)

    report = {
        "model": "sam2.1-hiera-tiny",
        "onnx_source": ONNX_SOURCE,
        "onnx_revision": ONNX_REVISION,
        "pytorch_source": PYTORCH_SOURCE,
        "pytorch_revision": PYTORCH_REVISION,
        "contracts": {
            "vision_encoder": {
                "inputs": _contract(encoder.get_inputs()),
                "outputs": _contract(encoder.get_outputs()),
            },
            "prompt_encoder_mask_decoder": {
                "inputs": _contract(decoder.get_inputs()),
                "outputs": _contract(decoder.get_outputs()),
            },
        },
        "prompt_behavior": {
            "points_only_accepts_zero_boxes": True,
            "box_only_accepts_zero_points": True,
            "box_only_requires_padding_point": False,
            "point_and_box_together": True,
            "multiple_foreground_background_points": True,
            "has_previous_mask_input": False,
            "box_only_note": (
                "The mandatory ONNX point inputs use zero-length axes. They match "
                "Sam2Model with explicit empty point tensors. Sam2Processor omits point "
                "arguments for box-only input; that changes raw scores slightly but not "
                "the selected post-processed binary mask."
            ),
        },
        "encoder_parity": encoder_reports,
        "cases": case_reports,
    }
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--fixture-root", type=Path, default=FIXTURE_ROOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    args = parser.parse_args()

    report = validate(args.model_dir, args.fixture_root, args.reference)
    max_encoder_diff = max(
        max(item["embedding_max_abs_diff"])
        for item in report["encoder_parity"]
    )
    min_mask_iou = min(
        item["parity"]["canonical_processor_pytorch_binary_mask_iou"]
        for item in report["cases"]
    )
    max_contract_iou_diff = max(
        item["parity"]["contract_pytorch_iou_scores_max_abs_diff"]
        for item in report["cases"]
    )
    print(f"Wrote fixtures to {args.fixture_root}")
    print(f"Wrote parity report to {args.reference}")
    print(f"Max encoder embedding abs diff: {max_encoder_diff:.8g}")
    print(f"Max decoder IoU-score abs diff (contract tensors): {max_contract_iou_diff:.8g}")
    print(f"Min binary-mask IoU vs canonical processor/model: {min_mask_iou:.8g}")


if __name__ == "__main__":
    main()
