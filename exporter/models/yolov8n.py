from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register
from exporter.models._detection import COCO, ULTRALYTICS_CODE, UNCHANGED, java_usage, provenance


class Yolov8n(MirroredModel):
    name = "yolov8n"
    repo_id = "inference4j/yolov8n"
    source_repo = "Kalray/yolov8"
    source_type = "hf"
    license_file = "AGPL-3.0"
    files = [
        FileMapping(src="yolov8n.onnx", dst="model.onnx"),
    ]
    card = ModelCard(
        title="YOLOv8n — ONNX",
        description="ONNX export of [YOLOv8n](https://huggingface.co/Kalray/yolov8), the nano variant of Ultralytics YOLOv8 object detection model. Trained on COCO with 80-class output. Optimized for real-time inference.",
        license="agpl-3.0",
        pipeline_tag="object-detection",
        tags=["yolov8", "object-detection", "coco", "computer-vision", "ultralytics"],
        original_source_url="https://huggingface.co/Kalray/yolov8",
        original_author="Kalray (Ultralytics YOLOv8)",
        java_usage=java_usage("YoloV8Detector.builder()"),
        model_details={
            "Architecture": "YOLOv8 Nano (single-shot detector)",
            "Task": "Object detection (COCO 80 classes)",
            "Input": "`[1, 3, 640, 640]` — RGB, normalized 0-1",
            "Output": "`[1, 84, 8400]` — bounding boxes + class scores",
            "Post-processing": "Non-Maximum Suppression (NMS) required",
            "Original framework": "PyTorch (Ultralytics)",
        },
        extra_sections=provenance(
            code=ULTRALYTICS_CODE,
            weights="Ultralytics YOLOv8n, AGPL-3.0",
            onnx="[Kalray/yolov8](https://huggingface.co/Kalray/yolov8)",
            training_data=COCO,
            conversion=UNCHANGED,
        ),
        license_text="This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html); the full text is in the `LICENSE` file of this repository. Original model by [Ultralytics](https://ultralytics.com/), hosted by [Kalray](https://huggingface.co/Kalray).",
    )


register(Yolov8n())
