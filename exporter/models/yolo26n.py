from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register
from exporter.models._detection import COCO, ULTRALYTICS_CODE, UNCHANGED, java_usage, provenance


class Yolo26n(MirroredModel):
    name = "yolo26n"
    repo_id = "inference4j/yolo26n"
    source_repo = "onnx-community/yolo26n-ONNX"
    source_type = "hf"
    license_file = "AGPL-3.0"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
    ]
    card = ModelCard(
        title="YOLO26n — ONNX",
        description="ONNX export of [YOLO26n](https://huggingface.co/onnx-community/yolo26n-ONNX), the nano variant of Ultralytics YOLO26 object detection model. Features built-in NMS (no post-processing required). Trained on COCO with 80-class output.",
        license="agpl-3.0",
        pipeline_tag="object-detection",
        tags=["yolo26", "object-detection", "coco", "computer-vision", "ultralytics"],
        original_source_url="https://huggingface.co/onnx-community/yolo26n-ONNX",
        original_author="onnx-community (Ultralytics YOLO26)",
        java_usage=java_usage("Yolo26Detector.builder()"),
        model_details={
            "Architecture": "YOLO26 Nano (NMS-free single-shot detector)",
            "Task": "Object detection (COCO 80 classes)",
            "COCO AP": "40.9 (val2017, as reported by Ultralytics)",
            "Input": "`[1, 3, 640, 640]` — RGB, normalized 0-1",
            "Output": "NMS-free — outputs filtered detections directly",
            "Post-processing": "None required (built-in NMS)",
            "Original framework": "PyTorch (Ultralytics)",
        },
        extra_sections=provenance(
            code=ULTRALYTICS_CODE,
            weights="Ultralytics YOLO26n, AGPL-3.0",
            onnx="[onnx-community/yolo26n-ONNX](https://huggingface.co/onnx-community/yolo26n-ONNX)",
            training_data=COCO,
            conversion=UNCHANGED,
        ),
        license_text="This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html); the full text is in the `LICENSE` file of this repository. Original model by [Ultralytics](https://ultralytics.com/), exported by [onnx-community](https://huggingface.co/onnx-community).",
    )


register(Yolo26n())
