from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class Yolov8n(MirroredModel):
    name = "yolov8n"
    repo_id = "inference4j/yolov8n"
    source_repo = "Kalray/yolov8"
    source_type = "hf"
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
        java_usage="""\
try (YoloV8 model = YoloV8.fromPretrained("models/yolov8n")) {
    List<Detection> detections = model.detect(Path.of("street.jpg"));
    detections.forEach(d -> System.out.printf("%s (%.0f%%) at [%.0f, %.0f, %.0f, %.0f]%n",
        d.label(), d.confidence() * 100,
        d.box().x1(), d.box().y1(), d.box().x2(), d.box().y2()));
}""",
        model_details={
            "Architecture": "YOLOv8 Nano (single-shot detector)",
            "Task": "Object detection (COCO 80 classes)",
            "Input": "`[1, 3, 640, 640]` — RGB, normalized 0-1",
            "Output": "`[1, 84, 8400]` — bounding boxes + class scores",
            "Post-processing": "Non-Maximum Suppression (NMS) required",
            "Original framework": "PyTorch (Ultralytics)",
        },
        license_text="This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html). Original model by [Ultralytics](https://ultralytics.com/), hosted by [Kalray](https://huggingface.co/Kalray).",
    )


register(Yolov8n())
