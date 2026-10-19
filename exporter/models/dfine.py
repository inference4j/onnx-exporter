from pathlib import Path

from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register

_FILES = [
    FileMapping(src="onnx/model.onnx", dst="model.onnx"),
    FileMapping(src="config.json", dst="config.json"),
    FileMapping(src="preprocessor_config.json", dst="preprocessor_config.json"),
]


class DFineNanoCoco(MirroredModel):
    name = "dfine-nano-coco"
    repo_id = "inference4j/dfine-nano-coco"
    source_repo = "onnx-community/dfine-nano-coco-ONNX"
    source_type = "hf"
    revision = "89df95b0e8575ff9879f13351bafe69bbe8e6823"
    files = list(_FILES)

    def stage(self, staging_dir: Path) -> None:
        super().stage(staging_dir)
        # Apply static shapes for CoreML compatibility (fixing dynamic GridSample coordinates)
        print("  Simplifying graph with static shapes [1, 3, 640, 640] via onnxsim...")
        model_path = staging_dir / "model.onnx"
        import onnx
        from onnx import shape_inference
        import onnxsim

        model = onnx.load(str(model_path))
        simplified_model, check = onnxsim.simplify(
            model,
            overwrite_input_shapes={"pixel_values": [1, 3, 640, 640]},
        )
        if not check:
            raise RuntimeError("onnxsim check failed on dfine-nano-coco")
        simplified_model = shape_inference.infer_shapes(simplified_model)
        onnx.save(simplified_model, str(model_path))
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"    -> Static model saved to {model_path} ({size_mb:.1f} MB)")

    card = ModelCard(
        title="D-FINE Nano (COCO) — ONNX",
        description=(
            "ONNX export of [ustc-community/dfine-nano-coco](https://huggingface.co/ustc-community/dfine-nano-coco), "
            "the nano variant of D-FINE (Redefine Bounding Box Regression for DETR-based Object Detection) "
            "trained on COCO (80 classes) with static shapes for full ONNX Runtime and CoreML compatibility. "
            "Licensed under Apache-2.0, providing a commercially permissive, non-copyleft alternative to "
            "AGPL-licensed YOLOv8 and YOLO26 models."
        ),
        license="apache-2.0",
        pipeline_tag="object-detection",
        tags=["dfine", "object-detection", "coco", "computer-vision", "apache-2.0"],
        original_source_url="https://huggingface.co/ustc-community/dfine-nano-coco",
        original_author="USTC / onnx-community (D-FINE)",
        java_usage="""\
try (DFine model = DFine.fromPretrained("inference4j/dfine-nano-coco")) {
    List<Detection> detections = model.detect(Path.of("street.jpg"));
    detections.forEach(d -> System.out.printf("%s (%.0f%%) at [%.0f, %.0f, %.0f, %.0f]%n",
        d.label(), d.confidence() * 100,
        d.box().x1(), d.box().y1(), d.box().x2(), d.box().y2()));
}""",
        model_details={
            "Architecture": "D-FINE Nano (transformer object detector)",
            "Task": "Object detection (COCO 80 classes)",
            "License note": "Apache-2.0 (permissive non-AGPL alternative to YOLOv8/YOLO26)",
            "Input": "`pixel_values`: `[1, 3, 640, 640]` — RGB, normalized 0-1 (bilinear resize, no letterbox, static shapes for CoreML)",
            "Output": "`logits`: `[1, 300, 80]` (raw logits, apply sigmoid) and `pred_boxes`: `[1, 300, 4]` (normalized [cx, cy, w, h])",
            "Post-processing": "Convert [cx, cy, w, h] to [x1, y1, x2, y2] scaled to image dimensions; filter by score threshold (> 0.5). No NMS required.",
            "Hardware support": "Verified on CPU and Apple Silicon CoreML (NeuralNetwork & MLProgram formats).",
            "Original framework": "PyTorch (USTC D-FINE)",
        },
        license_text=(
            "This model is licensed under the [Apache License 2.0]"
            "(https://www.apache.org/licenses/LICENSE-2.0). Original model by "
            "[USTC / Peng et al.](https://github.com/Peterande/D-FINE), "
            "hosted by [ustc-community](https://huggingface.co/ustc-community/dfine-nano-coco) "
            "and converted by [onnx-community](https://huggingface.co/onnx-community/dfine-nano-coco-ONNX)."
        ),
    )


class DFineSmallCoco(MirroredModel):
    name = "dfine-small-coco"
    repo_id = "inference4j/dfine-small-coco"
    source_repo = "onnx-community/dfine_s_coco-ONNX"
    source_type = "hf"
    revision = "a3cf03147a9b86c78475139115c8ac142577352d"
    files = list(_FILES)
    card = ModelCard(
        title="D-FINE Small (COCO) — ONNX",
        description=(
            "ONNX export of [ustc-community/dfine-small-coco](https://huggingface.co/ustc-community/dfine-small-coco), "
            "the small variant of D-FINE (Redefine Bounding Box Regression for DETR-based Object Detection) "
            "trained on COCO (80 classes). "
            "Licensed under Apache-2.0, providing a commercially permissive, non-copyleft alternative to "
            "AGPL-licensed YOLOv8 and YOLO26 models."
        ),
        license="apache-2.0",
        pipeline_tag="object-detection",
        tags=["dfine", "object-detection", "coco", "computer-vision", "apache-2.0"],
        original_source_url="https://huggingface.co/ustc-community/dfine-small-coco",
        original_author="USTC / onnx-community (D-FINE)",
        java_usage="""\
try (DFine model = DFine.fromPretrained("inference4j/dfine-small-coco")) {
    List<Detection> detections = model.detect(Path.of("street.jpg"));
    detections.forEach(d -> System.out.printf("%s (%.0f%%) at [%.0f, %.0f, %.0f, %.0f]%n",
        d.label(), d.confidence() * 100,
        d.box().x1(), d.box().y1(), d.box().x2(), d.box().y2()));
}""",
        model_details={
            "Architecture": "D-FINE Small (transformer object detector)",
            "Task": "Object detection (COCO 80 classes)",
            "License note": "Apache-2.0 (permissive non-AGPL alternative to YOLOv8/YOLO26)",
            "Input": "`pixel_values`: `[batch, 3, 640, 640]` — RGB, normalized 0-1 (bilinear resize, no letterbox)",
            "Output": "`logits`: `[batch, 300, 80]` (raw logits, apply sigmoid) and `pred_boxes`: `[batch, 300, 4]` (normalized [cx, cy, w, h])",
            "Post-processing": "Convert [cx, cy, w, h] to [x1, y1, x2, y2] scaled to image dimensions; filter by score threshold (> 0.5). No NMS required.",
            "Original framework": "PyTorch (USTC D-FINE)",
        },
        license_text=(
            "This model is licensed under the [Apache License 2.0]"
            "(https://www.apache.org/licenses/LICENSE-2.0). Original model by "
            "[USTC / Peng et al.](https://github.com/Peterande/D-FINE), "
            "hosted by [ustc-community](https://huggingface.co/ustc-community/dfine-small-coco) "
            "and converted by [onnx-community](https://huggingface.co/onnx-community/dfine_s_coco-ONNX)."
        ),
    )


register(DFineNanoCoco())
register(DFineSmallCoco())
