"""Shared card content for the object detection mirrors."""


def java_usage(builder: str) -> str:
    """Card snippet for an inference4j ObjectDetector built from the given builder expression."""
    return f"""\
try (ObjectDetector detector = {builder}.build()) {{
    List<Detection> detections = detector.detect(Path.of("street.jpg"));
    detections.forEach(d -> System.out.printf("%s (%.0f%%) at [%.0f, %.0f, %.0f, %.0f]%n",
        d.label(), d.confidence() * 100,
        d.box().x1(), d.box().y1(), d.box().x2(), d.box().y2()));
}}"""


def provenance(*, code: str, weights: str, onnx: str, training_data: str, conversion: str) -> str:
    """A Provenance section: where each part of the published model comes from, and under which license."""
    return f"""\
## Provenance

| Part | Origin |
|------|--------|
| Code | {code} |
| Weights | {weights} |
| ONNX export | {onnx} |
| Training data | {training_data} |
| Changes in this mirror | {conversion} |"""


DFINE_CODE = "[Peterande/D-FINE](https://github.com/Peterande/D-FINE), Apache-2.0"
RTDETR_CODE = "[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR), Apache-2.0"
ULTRALYTICS_CODE = "[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0"
COCO = "COCO 2017 (80 classes)"
UNCHANGED = "None: files copied unchanged"
YOLOX_CODE = "[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), Apache-2.0"
