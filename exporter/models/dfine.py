from pathlib import Path

from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register
from exporter.models._detection import COCO, DFINE_CODE, UNCHANGED, java_usage, provenance

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
        java_usage=java_usage("DetrDetector.dfineNano()"),
        model_details={
            "Architecture": "D-FINE Nano (transformer object detector)",
            "Task": "Object detection (COCO 80 classes)",
            "COCO AP": "42.8 (val2017, as reported by the D-FINE authors)",
            "License note": "Apache-2.0 (permissive non-AGPL alternative to YOLOv8/YOLO26)",
            "Input": "`pixel_values`: `[1, 3, 640, 640]` — RGB, normalized 0-1 (bilinear resize, no letterbox, static shapes for CoreML)",
            "Output": "`logits`: `[1, 300, 80]` (raw logits, apply sigmoid) and `pred_boxes`: `[1, 300, 4]` (normalized [cx, cy, w, h])",
            "Post-processing": "Convert [cx, cy, w, h] to [x1, y1, x2, y2] scaled to image dimensions; filter by score threshold (> 0.5). No NMS required.",
            "Hardware support": "Verified on CPU and Apple Silicon CoreML (NeuralNetwork & MLProgram formats).",
            "Original framework": "PyTorch (USTC D-FINE)",
        },
        extra_sections=provenance(
            code=DFINE_CODE,
            weights="[ustc-community/dfine-nano-coco](https://huggingface.co/ustc-community/dfine-nano-coco), Apache-2.0",
            onnx="[onnx-community/dfine-nano-coco-ONNX](https://huggingface.co/onnx-community/dfine-nano-coco-ONNX)",
            training_data=COCO,
            conversion="Graph simplified with `onnxsim` to a static `[1, 3, 640, 640]` input",
        ),
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
        java_usage=java_usage("DetrDetector.dfineSmall()"),
        model_details={
            "Architecture": "D-FINE Small (transformer object detector)",
            "Task": "Object detection (COCO 80 classes)",
            "COCO AP": "48.5 (val2017, as reported by the D-FINE authors)",
            "License note": "Apache-2.0 (permissive non-AGPL alternative to YOLOv8/YOLO26)",
            "Input": "`pixel_values`: `[batch, 3, 640, 640]` — RGB, normalized 0-1 (bilinear resize, no letterbox)",
            "Output": "`logits`: `[batch, 300, 80]` (raw logits, apply sigmoid) and `pred_boxes`: `[batch, 300, 4]` (normalized [cx, cy, w, h])",
            "Post-processing": "Convert [cx, cy, w, h] to [x1, y1, x2, y2] scaled to image dimensions; filter by score threshold (> 0.5). No NMS required.",
            "Original framework": "PyTorch (USTC D-FINE)",
        },
        extra_sections=provenance(
            code=DFINE_CODE,
            weights="[ustc-community/dfine-small-coco](https://huggingface.co/ustc-community/dfine-small-coco), Apache-2.0",
            onnx="[onnx-community/dfine_s_coco-ONNX](https://huggingface.co/onnx-community/dfine_s_coco-ONNX)",
            training_data=COCO,
            conversion=UNCHANGED,
        ),
        license_text=(
            "This model is licensed under the [Apache License 2.0]"
            "(https://www.apache.org/licenses/LICENSE-2.0). Original model by "
            "[USTC / Peng et al.](https://github.com/Peterande/D-FINE), "
            "hosted by [ustc-community](https://huggingface.co/ustc-community/dfine-small-coco) "
            "and converted by [onnx-community](https://huggingface.co/onnx-community/dfine_s_coco-ONNX)."
        ),
    )


class DFineMediumCoco(MirroredModel):
    name = "dfine-medium-coco"
    repo_id = "inference4j/dfine-medium-coco"
    source_repo = "onnx-community/dfine_m_coco-ONNX"
    source_type = "hf"
    revision = "489756db2825cb068a588fe930af239a656d1fe1"
    files = list(_FILES)
    card = ModelCard(
        title="D-FINE Medium (COCO) — ONNX",
        description=(
            "ONNX export of [ustc-community/dfine-medium-coco](https://huggingface.co/ustc-community/dfine-medium-coco), "
            "the medium variant of D-FINE (Redefine Bounding Box Regression for DETR-based Object Detection) "
            "trained on COCO (80 classes). "
            "Licensed under Apache-2.0, providing a commercially permissive, non-copyleft alternative to "
            "AGPL-licensed YOLOv8 and YOLO26 models."
        ),
        license="apache-2.0",
        pipeline_tag="object-detection",
        tags=["dfine", "object-detection", "coco", "computer-vision", "apache-2.0"],
        original_source_url="https://huggingface.co/ustc-community/dfine-medium-coco",
        original_author="USTC / onnx-community (D-FINE)",
        java_usage=java_usage("DetrDetector.dfineMedium()"),
        model_details={
            "Architecture": "D-FINE Medium (transformer object detector)",
            "Task": "Object detection (COCO 80 classes)",
            "COCO AP": "52.3 (val2017, as reported by the D-FINE authors)",
            "License note": "Apache-2.0 (permissive non-AGPL alternative to YOLOv8/YOLO26)",
            "Input": "`pixel_values`: `[batch, 3, 640, 640]` — RGB, normalized 0-1 (bilinear resize, no letterbox)",
            "Output": "`logits`: `[batch, 300, 80]` (raw logits, apply sigmoid) and `pred_boxes`: `[batch, 300, 4]` (normalized [cx, cy, w, h])",
            "Post-processing": "Convert [cx, cy, w, h] to [x1, y1, x2, y2] scaled to image dimensions; filter by score threshold (> 0.5). No NMS required.",
            "Original framework": "PyTorch (USTC D-FINE)",
        },
        extra_sections=provenance(
            code=DFINE_CODE,
            weights="[ustc-community/dfine-medium-coco](https://huggingface.co/ustc-community/dfine-medium-coco), Apache-2.0",
            onnx="[onnx-community/dfine_m_coco-ONNX](https://huggingface.co/onnx-community/dfine_m_coco-ONNX)",
            training_data=COCO,
            conversion=UNCHANGED,
        ),
        license_text=(
            "This model is licensed under the [Apache License 2.0]"
            "(https://www.apache.org/licenses/LICENSE-2.0). Original model by "
            "[USTC / Peng et al.](https://github.com/Peterande/D-FINE), "
            "hosted by [ustc-community](https://huggingface.co/ustc-community/dfine-medium-coco) "
            "and converted by [onnx-community](https://huggingface.co/onnx-community/dfine_m_coco-ONNX)."
        ),
    )


class DFineLargeCoco(MirroredModel):
    name = "dfine-large-coco"
    repo_id = "inference4j/dfine-large-coco"
    source_repo = "onnx-community/dfine_l_coco-ONNX"
    source_type = "hf"
    revision = "e09b218185400510b68d067bbd0c9379d905ffcd"
    files = list(_FILES)
    card = ModelCard(
        title="D-FINE Large (COCO) — ONNX",
        description=(
            "ONNX export of [ustc-community/dfine-large-coco](https://huggingface.co/ustc-community/dfine-large-coco), "
            "the large variant of D-FINE (Redefine Bounding Box Regression for DETR-based Object Detection) "
            "trained on COCO (80 classes). "
            "Licensed under Apache-2.0, providing a commercially permissive, non-copyleft alternative to "
            "AGPL-licensed YOLOv8 and YOLO26 models."
        ),
        license="apache-2.0",
        pipeline_tag="object-detection",
        tags=["dfine", "object-detection", "coco", "computer-vision", "apache-2.0"],
        original_source_url="https://huggingface.co/ustc-community/dfine-large-coco",
        original_author="USTC / onnx-community (D-FINE)",
        java_usage=java_usage("DetrDetector.dfineLarge()"),
        model_details={
            "Architecture": "D-FINE Large (transformer object detector)",
            "Task": "Object detection (COCO 80 classes)",
            "COCO AP": "54.0 (val2017, as reported by the D-FINE authors)",
            "License note": "Apache-2.0 (permissive non-AGPL alternative to YOLOv8/YOLO26)",
            "Input": "`pixel_values`: `[batch, 3, 640, 640]` — RGB, normalized 0-1 (bilinear resize, no letterbox)",
            "Output": "`logits`: `[batch, 300, 80]` (raw logits, apply sigmoid) and `pred_boxes`: `[batch, 300, 4]` (normalized [cx, cy, w, h])",
            "Post-processing": "Convert [cx, cy, w, h] to [x1, y1, x2, y2] scaled to image dimensions; filter by score threshold (> 0.5). No NMS required.",
            "Original framework": "PyTorch (USTC D-FINE)",
        },
        extra_sections=provenance(
            code=DFINE_CODE,
            weights="[ustc-community/dfine-large-coco](https://huggingface.co/ustc-community/dfine-large-coco), Apache-2.0",
            onnx="[onnx-community/dfine_l_coco-ONNX](https://huggingface.co/onnx-community/dfine_l_coco-ONNX)",
            training_data=COCO,
            conversion=UNCHANGED,
        ),
        license_text=(
            "This model is licensed under the [Apache License 2.0]"
            "(https://www.apache.org/licenses/LICENSE-2.0). Original model by "
            "[USTC / Peng et al.](https://github.com/Peterande/D-FINE), "
            "hosted by [ustc-community](https://huggingface.co/ustc-community/dfine-large-coco) "
            "and converted by [onnx-community](https://huggingface.co/onnx-community/dfine_l_coco-ONNX)."
        ),
    )


register(DFineNanoCoco())
register(DFineSmallCoco())
register(DFineMediumCoco())
register(DFineLargeCoco())
