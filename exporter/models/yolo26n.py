from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class Yolo26n(MirroredModel):
    name = "yolo26n"
    repo_id = "inference4j/yolo26n"
    source_repo = "onnx-community/yolo26n-ONNX"
    source_type = "hf"
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
        java_usage="""\
try (Yolo26 model = Yolo26.fromPretrained("models/yolo26n")) {
    List<Detection> detections = model.detect(Path.of("street.jpg"));
    detections.forEach(d -> System.out.printf("%s (%.0f%%) at [%.0f, %.0f, %.0f, %.0f]%n",
        d.label(), d.confidence() * 100,
        d.box().x1(), d.box().y1(), d.box().x2(), d.box().y2()));
}""",
        model_details={
            "Architecture": "YOLO26 Nano (NMS-free single-shot detector)",
            "Task": "Object detection (COCO 80 classes)",
            "Input": "`[1, 3, 640, 640]` — RGB, normalized 0-1",
            "Output": "NMS-free — outputs filtered detections directly",
            "Post-processing": "None required (built-in NMS)",
            "Original framework": "PyTorch (Ultralytics)",
        },
        license_text="This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html). Original model by [Ultralytics](https://ultralytics.com/), exported by [onnx-community](https://huggingface.co/onnx-community).",
    )


register(Yolo26n())
