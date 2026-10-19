from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register

_JAVA_USAGE = """\
try (RtDetr model = RtDetr.fromPretrained("inference4j/rtdetr-v2-r18vd")) {
    List<Detection> detections = model.detect(Path.of("street.jpg"));
    detections.forEach(d -> System.out.printf("%s (%.0f%%) at [%.0f, %.0f, %.0f, %.0f]%n",
        d.label(), d.confidence() * 100,
        d.box().x1(), d.box().y1(), d.box().x2(), d.box().y2()));
}"""

_LICENSE_TEXT = (
    "This model is licensed under the [Apache License 2.0]"
    "(https://www.apache.org/licenses/LICENSE-2.0). Original model by "
    "[Peking University / Baidu](https://github.com/lyuwenyu/RT-DETR), "
    "hosted by [PekingU](https://huggingface.co/PekingU/rtdetr_v2_r18vd) "
    "and converted by [onnx-community](https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX)."
)

_FILES = [
    FileMapping(src="onnx/model.onnx", dst="model.onnx"),
    FileMapping(src="config.json", dst="config.json"),
    FileMapping(src="preprocessor_config.json", dst="preprocessor_config.json"),
]


class RtDetrV2R18vd(MirroredModel):
    name = "rtdetr-v2-r18vd"
    repo_id = "inference4j/rtdetr-v2-r18vd"
    source_repo = "onnx-community/rtdetr_v2_r18vd-ONNX"
    source_type = "hf"
    revision = "936f90b6a476c6da4dfe053fc521af55285976ba"
    files = list(_FILES)
    card = ModelCard(
        title="RT-DETRv2-R18-vd — ONNX",
        description=(
            "ONNX export of [PekingU/rtdetr_v2_r18vd](https://huggingface.co/PekingU/rtdetr_v2_r18vd), "
            "the ResNet-18 vd variant of RT-DETRv2 (Real-Time DEtection TRansformer v2) trained on COCO (80 classes). "
            "Licensed under Apache-2.0, providing a commercially permissive, non-copyleft alternative to "
            "AGPL-licensed YOLOv8 and YOLO26 models."
        ),
        license="apache-2.0",
        pipeline_tag="object-detection",
        tags=["rtdetr", "rtdetr-v2", "object-detection", "coco", "computer-vision", "apache-2.0"],
        original_source_url="https://huggingface.co/PekingU/rtdetr_v2_r18vd",
        original_author="Peking University (RT-DETRv2)",
        java_usage=_JAVA_USAGE,
        model_details={
            "Architecture": "RT-DETRv2 ResNet-18 vd (transformer object detector)",
            "Task": "Object detection (COCO 80 classes)",
            "License note": "Apache-2.0 (permissive non-AGPL alternative to YOLOv8/YOLO26)",
            "Input": "`pixel_values`: `[batch, 3, 640, 640]` — RGB, normalized 0-1 (bilinear resize, no letterbox)",
            "Output": "`logits`: `[batch, 300, 80]` (raw logits, apply sigmoid) and `pred_boxes`: `[batch, 300, 4]` (normalized [cx, cy, w, h])",
            "Post-processing": "Convert [cx, cy, w, h] to [x1, y1, x2, y2] scaled to image dimensions; filter by score threshold (> 0.5). No NMS required.",
            "Original framework": "PyTorch (PekingU / Baidu RT-DETR)",
        },
        license_text=_LICENSE_TEXT,
    )


register(RtDetrV2R18vd())
