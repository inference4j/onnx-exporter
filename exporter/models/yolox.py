"""YOLOX nano, tiny and s: the official Megvii ONNX exports, mirrored unchanged."""

from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register
from exporter.models._detection import UNCHANGED, YOLOX_CODE, java_usage, provenance

_RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
_TRAINING_DATA = (
    "COCO 2017 train (80 classes), trained from scratch: the YOLOX paper drops ImageNet pre-training"
)
_PRE = (
    "Letterbox to the top-left corner: scale `r = min({s} / h, {s} / w)`, bilinear resize to "
    "`(int(w * r), int(h * r))`, pad right and bottom with 114. Raw 0-255 **BGR**, no mean/std, NCHW float32"
)
_POST = (
    "Decode per anchor point over strides 8, 16, 32 (row-major grid, stride 8 first): "
    "`cx = (x + grid_x) * stride`, `cy = (y + grid_y) * stride`, `w = exp(w) * stride`, "
    "`h = exp(h) * stride`. Score = objectness × class score (both already sigmoid), take the "
    "best class, divide boxes by `r`. NMS is not in the graph: the reference demo uses "
    "class-agnostic NMS at IoU 0.45"
)


def _yolox(size: str, title: str, input_size: int, anchors: int, sha256: str,
           ap: str, params: str, builder: str) -> MirroredModel:
    url = f"{_RELEASE}/yolox_{size}.onnx"

    class YoloX(MirroredModel):
        name = f"yolox-{size}"
        repo_id = f"inference4j/yolox-{size}"
        source_repo = url
        source_type = "url"
        files = [FileMapping(src=url, dst="model.onnx", sha256=sha256)]
        card = ModelCard(
            title=f"YOLOX-{title} — ONNX",
            description=(
                f"The official ONNX export of YOLOX-{title} from Megvii's "
                f"[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) 0.1.1rc0 release, an anchor-free "
                f"YOLO detector trained on COCO (80 classes). Apache-2.0 code and weights, a permissive "
                f"alternative to AGPL-licensed YOLOv8 and YOLO26."
            ),
            license="apache-2.0",
            pipeline_tag="object-detection",
            tags=["yolox", "object-detection", "coco", "computer-vision", "apache-2.0"],
            original_source_url="https://github.com/Megvii-BaseDetection/YOLOX",
            original_author="Megvii (YOLOX)",
            java_usage=java_usage(builder),
            model_details={
                "Architecture": f"YOLOX-{title} (anchor-free, decoupled head), {params} parameters",
                "Task": "Object detection (COCO 80 classes)",
                "COCO AP": f"{ap} (val2017, {input_size}x{input_size}, as reported by the YOLOX authors)",
                "Input": f"`images`: `[1, 3, {input_size}, {input_size}]` float32, static",
                "Pre-processing": _PRE.format(s=input_size),
                "Output": (
                    f"`output`: `[1, {anchors}, 85]`: per anchor point `[x, y, w, h, objectness, 80 class scores]`, "
                    "box not yet decoded"
                ),
                "Post-processing": _POST,
                "Labels": "COCO-80 in the usual order (0 = person ... 79 = toothbrush)",
                "Original framework": "PyTorch 1.7, ONNX opset 11",
            },
            extra_sections=provenance(
                code=YOLOX_CODE,
                weights=f"[yolox_{size}.pth]({_RELEASE}/yolox_{size}.pth), YOLOX 0.1.1rc0 release, Apache-2.0",
                onnx=f"[yolox_{size}.onnx]({url}), the official export from the same release",
                training_data=_TRAINING_DATA,
                conversion=UNCHANGED,
            ),
            license_text=(
                "This model is licensed under the [Apache License 2.0]"
                "(https://www.apache.org/licenses/LICENSE-2.0). Original model by "
                "[Megvii](https://github.com/Megvii-BaseDetection/YOLOX) "
                "(Ge et al., *YOLOX: Exceeding YOLO Series in 2021*, arXiv:2107.08430)."
            ),
        )

    YoloX.__name__ = YoloX.__qualname__ = f"YoloX{title}"
    return YoloX()


register(_yolox("nano", "Nano", 416, 3549,
                "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
                "25.8", "0.91M", "YoloXDetector.nano()"))
register(_yolox("tiny", "Tiny", 416, 3549,
                "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
                "32.8", "5.06M", "YoloXDetector.tiny()"))
register(_yolox("s", "S", 640, 8400,
                "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
                "40.5", "9.0M", "YoloXDetector.s()"))
