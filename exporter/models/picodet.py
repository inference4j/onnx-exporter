"""PP-PicoDet S and M (416, LCNet): the official PaddleDetection ONNX exports without post-processing."""

from pathlib import Path

from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register
from exporter.models._detection import PADDLEDETECTION_CODE, java_usage, provenance

_BASE = "https://paddledet.bj.bcebos.com/deploy/third_engine"
_TRAINING_DATA = (
    "COCO 2017 train (80 classes); the LCNet backbone starts from PaddleClas ImageNet-1k weights"
)
_PRE = (
    "Resize to 416 x 416 without keeping the aspect ratio (bicubic, `cv2.INTER_CUBIC`, as in the model's "
    "`TestReader`), **RGB**, `/ 255`, then ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`, NCHW float32"
)
_OUTPUTS = (
    "Eight float32 tensors, in this order: class scores for strides 8, 16, 32, 64 "
    "(`transpose_0.tmp_0` `[1, 2704, 80]`, `transpose_2.tmp_0` `[1, 676, 80]`, `transpose_4.tmp_0` `[1, 169, 80]`, "
    "`transpose_6.tmp_0` `[1, 49, 80]`), then box distributions for the same strides "
    "(`transpose_1.tmp_0` `[1, 2704, 32]`, `transpose_3.tmp_0`, `transpose_5.tmp_0`, `transpose_7.tmp_0`)"
)
_POST = (
    "Scores are final (no sigmoid needed). Anchor points per stride `s` on a `ceil(416 / s)` grid, row-major: "
    "`((x + 0.5) * s, (y + 0.5) * s)`. Box: the 32 values are 4 sides (left, top, right, bottom) x 8 bins; "
    "softmax over the bins, distance = sum(p_i * i) * s; box = `(cx - l, cy - t, cx + r, cy + b)`, then divide "
    "x by `416 / w` and y by `416 / h`. NMS is not in the graph: PaddleDetection uses class-aware NMS, IoU 0.6, "
    "score > 0.025, top 1000 per class, 100 kept"
)


def _picodet(size: str, title: str, src: str, sha256: str, ap: str, params: str, builder: str):
    url = f"{_BASE}/{src}"

    class PicoDet(MirroredModel):
        name = f"picodet-{size}"
        repo_id = f"inference4j/picodet-{size}"
        source_repo = url
        source_type = "url"
        files = [FileMapping(src=url, dst="model.onnx", sha256=sha256)]

        def stage(self, staging_dir: Path) -> None:
            super().stage(staging_dir)
            # Paddle2ONNX lists every weight as a graph input as well as an initializer, so
            # ONNX Runtime treats the weights as overridable inputs: it warns once per weight
            # and skips constant folding. Dropping them from the inputs changes no output.
            import onnx

            model_path = staging_dir / "model.onnx"
            model = onnx.load(str(model_path))
            weights = {init.name for init in model.graph.initializer}
            inputs = [i for i in model.graph.input if i.name not in weights]
            print(f"  Removing {len(model.graph.input) - len(inputs)} initializers from the graph inputs")
            del model.graph.input[:]
            model.graph.input.extend(inputs)
            onnx.checker.check_model(model)
            onnx.save(model, str(model_path))

        card = ModelCard(
            title=f"PP-PicoDet-{title} (416, COCO) — ONNX",
            description=(
                f"The official ONNX export of PP-PicoDet-{title} (LCNet backbone, 416 x 416) from "
                "[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/configs/picodet), "
                "a lightweight anchor-free detector trained on COCO (80 classes). This is the export without "
                "post-processing: box decoding and NMS run outside the graph. Apache-2.0."
            ),
            license="apache-2.0",
            pipeline_tag="object-detection",
            tags=["picodet", "paddledetection", "object-detection", "coco", "computer-vision", "apache-2.0"],
            original_source_url="https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/configs/picodet",
            original_author="PaddlePaddle (PaddleDetection)",
            java_usage=java_usage(builder),
            model_details={
                "Architecture": f"PP-PicoDet-{title} (LCNet backbone, LCPAN neck, PicoHeadV2), {params} parameters",
                "Task": "Object detection (COCO 80 classes)",
                "COCO AP": f"{ap} (val2017, 416 x 416, as reported by the PaddleDetection authors)",
                "Input": "`image`: `[1, 3, 416, 416]` float32, static",
                "Pre-processing": _PRE,
                "Output": _OUTPUTS,
                "Post-processing": _POST,
                "Labels": "COCO-80 in the usual order (0 = person ... 79 = toothbrush)",
                "Original framework": "PaddlePaddle, converted with Paddle2ONNX (opset 11)",
            },
            extra_sections=provenance(
                code=PADDLEDETECTION_CODE,
                weights=(
                    f"[picodet_{size}_416_coco_lcnet.pdparams](https://paddledet.bj.bcebos.com/models/"
                    f"picodet_{size}_416_coco_lcnet.pdparams), PaddleDetection model zoo, Apache-2.0"
                ),
                onnx=f"[{src}]({url}), the official export linked from the PicoDet README",
                training_data=_TRAINING_DATA,
                conversion="Weights removed from the graph inputs (they stay initializers); outputs unchanged",
            ),
            license_text=(
                "This model is licensed under the [Apache License 2.0]"
                "(https://www.apache.org/licenses/LICENSE-2.0). Original model by "
                "[PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection) "
                "(Yu et al., *PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices*, arXiv:2111.00902)."
            ),
        )

    PicoDet.__name__ = PicoDet.__qualname__ = f"PicoDet{title}"
    return PicoDet()


register(_picodet("s", "S", "picodet_s_416_coco_lcnet.onnx",
                  "d080f8d9931bf9164934a7863e989605c7b3aa7751e35be406e20aa6b68256a7",
                  "32.5", "1.18M", "PicoDetDetector.s()"))
register(_picodet("m", "M", "picodet_m_416_coco_lcnet.onnx",
                  "73175223e66d458a341be0443cb469d8ae1d83a86c1a0fe4d29b85e0770830d6",
                  "37.5", "3.46M", "PicoDetDetector.m()"))
