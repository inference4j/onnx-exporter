"""RTMO-s, -m and -l (COCO-only checkpoints): one-stage multi-person pose estimation, exported with MMDeploy.

Only the checkpoints trained on COCO alone are exported. The "body7" checkpoints, which MMPose also ships as
ONNX, add AI Challenger, CrowdPose, Halpe, MPII, PoseTrack18 and sub-JHMDB, several of which restrict
commercial use. The export runs in the pinned `.venv-rtmo` (see `exporter/rtmo_builder.py`) and produces the
graph layout MMPose publishes for body7 (NMS and keypoint decoding inside, `dets` and `keypoints` out), except
that MMDeploy's padding row scores 0 (see `exporter/rtmo_export_script.py`).
"""

from pathlib import Path

from exporter.base import ExportedModel, FileMapping, ModelCard
from exporter.card import render
from exporter.registry import register
from exporter.models._detection import provenance

_ZOO = "https://download.openmmlab.com/mmpose/v1/projects/rtmo"
_PROJECT = "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo"
_CONFIGS = "https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/body_2d_keypoint/rtmo/coco"
MMPOSE_CODE = "[open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) 1.3.2 (RTMO project), Apache-2.0"
_TRAINING_DATA = (
    "COCO 2017 train person keypoints only (annotations CC BY 4.0, images under Flickr terms). The CSPDarknet "
    "backbone is initialised from MMDetection's COCO-trained YOLOX-{size} weights ({yolox})"
)
_CONVERSION = (
    "Exported with MMDeploy 1.3.1 (`pose-detection_rtmo_onnxruntime_dynamic.py`, opset 11) in a pinned "
    "environment: PyTorch 2.1.2, MMPose 1.3.2, mmcv 2.1.0, mmengine 0.10.4. Same graph layout as MMPose's "
    "official body7 `onnx_sdk` exports, except that the padding row scores 0 instead of repeating the last "
    "anchor's prediction. Weights unchanged apart from MMPose's `switch_to_deploy` (the neck's RepVGG blocks fused, anchor grid precomputed)"
)
_PRE = (
    "Letterbox to the top-left corner: `r = min(640 / h, 640 / w)`, bilinear resize to `(int(w * r), int(h * r))`, "
    "pad right and bottom with 114. Raw 0-255 **BGR**, no mean/std, NCHW float32"
)
_OUTPUTS = (
    "`dets`: `[batch, N, 5]` = `x1, y1, x2, y2, score` per person; `keypoints`: `[batch, N, 17, 3]` = "
    "`x, y, score` per COCO-17 joint. Coordinates in 640 x 640 input pixels, rows sorted by score"
)
_POST = (
    "NMS is in the graph (score > 0.1, class-agnostic IoU 0.65, at most 50 people). When fewer than 50 people "
    "are found, the graph appends one padding row with score 0 (box and keypoint scores), so keep rows with "
    "score > 0.1. Divide coordinates by `r` and clamp boxes to the image"
)
_JOINTS = (
    "COCO-17: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, "
    "right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle"
)
_LICENSE = (
    "Code and weights: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), from "
    "[MMPose](https://github.com/open-mmlab/mmpose) by OpenMMLab (Lu et al., *RTMO: Towards High-Performance "
    "One-Stage Real-Time Multi-Person Pose Estimation*, arXiv:2312.07526, 2023). Training data: COCO "
    "(annotations CC BY 4.0; images remain under their Flickr terms)."
)


def _java_usage(builder: str) -> str:
    return f"""\
try (PoseDetector detector = {builder}.build()) {{
    List<Pose> poses = detector.detect(Path.of("people.jpg"));
    poses.forEach(p -> System.out.printf("person %.2f, left wrist at %s%n",
        p.person().confidence(), p.keypoint(CocoJoint.LEFT_WRIST)));
}}"""


def _rtmo(size: str, title: str, config: str, checkpoint: str, sha256: str, ap: str,
          params: str, yolox: str, builder: str) -> ExportedModel:
    url = f"{_ZOO}/{checkpoint}"

    class Rtmo(ExportedModel):
        name = f"rtmo-{size}-coco"
        repo_id = f"inference4j/rtmo-{size}-coco"
        source_repo = url
        source_type = "url"
        # The checkpoint is the only upstream input; its sha256 is its version (the zoo has no revisions).
        files = [FileMapping(src=url, dst="model.onnx", sha256=sha256)]
        mmpose_config = f"body_2d_keypoint/rtmo/coco/{config}"
        weights_cache = Path.home() / ".cache" / "inference4j" / "rtmo" / checkpoint

        card = ModelCard(
            title=f"RTMO-{title} (COCO, 640) — ONNX",
            description=(
                f"[RTMO-{title}]({_PROJECT}), a one-stage real-time multi-person pose estimator from MMPose, "
                f"trained on COCO only. One run returns every person's box and 17 body keypoints. "
                f"Apache-2.0 code and weights, exported to ONNX with MMDeploy."
            ),
            license="apache-2.0",
            pipeline_tag="keypoint-detection",
            tags=["rtmo", "mmpose", "pose-estimation", "keypoint-detection", "coco", "apache-2.0"],
            original_source_url=_PROJECT,
            original_author="OpenMMLab (MMPose)",
            java_usage=_java_usage(builder),
            model_details={
                "Architecture": f"RTMO-{title} (CSPDarknet + hybrid encoder, dynamic coordinate classifier), {params} parameters",
                "Task": "Multi-person 2D pose estimation, 17 COCO keypoints",
                "COCO AP": f"{ap} (val2017, 640x640, single pass with MMPose's centred letterbox, as reported by the RTMO authors)",
                "Input": "`input`: `[batch, 3, 640, 640]` float32 (batch dynamic)",
                "Pre-processing": _PRE,
                "Output": _OUTPUTS,
                "Post-processing": _POST,
                "Keypoint order": _JOINTS,
                "ONNX opset": "11",
            },
            datasets=["coco"],
            extra_sections=provenance(
                code=MMPOSE_CODE,
                weights=f"[{checkpoint}]({url}) from the [RTMO model zoo]({_PROJECT}), config "
                        f"[{config}]({_CONFIGS}/{config}), Apache-2.0",
                onnx="Exported here (no upstream ONNX exists for the COCO-only checkpoints)",
                training_data=_TRAINING_DATA.format(size=size.upper(), yolox=yolox),
                conversion=_CONVERSION,
            ),
            license_text=_LICENSE,
        )

        def stage(self, staging_dir: Path) -> None:
            from exporter import rtmo_builder

            ckpt = rtmo_builder.download(url, sha256, self.weights_cache)
            output = staging_dir / "model.onnx"
            rtmo_builder.export(self.mmpose_config, ckpt, output)
            _check_contract(output)
            print(f"  Exported model.onnx ({output.stat().st_size / 1024 / 1024:.1f} MB)")

        def render_card(self) -> str:
            return render(self.card).replace("Mirrored for use with", "Exported for use with")

    Rtmo.__name__ = Rtmo.__qualname__ = f"Rtmo{title}Coco"
    return Rtmo()


def _check_contract(path: Path) -> None:
    """Fail the export if the graph's names, ranks or fixed dimensions are not what inference4j reads."""
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    def dims(value):
        return [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]

    inputs = {v.name: dims(v) for v in model.graph.input}
    outputs = {v.name: dims(v) for v in model.graph.output}
    assert list(inputs) == ["input"] and inputs["input"][1:] == [3, 640, 640], inputs
    assert list(outputs) == ["dets", "keypoints"], outputs
    assert len(outputs["dets"]) == 3 and len(outputs["keypoints"]) == 4, outputs
    assert outputs["keypoints"][2:] == [17, 3], outputs


register(_rtmo("s", "S", "rtmo-s_8xb32-600e_coco-640x640.py",
               "rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth",
               "8db55a5974d4d31164cc327805c57932cc7b4f79874b5e5bfbbfa32a7d4e5b05",
               "67.7", "9.9M", "`yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth`", "RtmoPoseDetector.s()"))
register(_rtmo("m", "M", "rtmo-m_16xb16-600e_coco-640x640.py",
               "rtmo-m_16xb16-600e_coco-640x640-6f4e0306_20231211.pth",
               "6f4e03062fb24263b93bf3f916284643dd76e93665d3cac2a645b619873f3053",
               "70.9", "22.6M", "`pretrained_models/yolox_m_8x8_300e_coco_20230829.pth`, a local path in the config (no public URL)",
               "RtmoPoseDetector.m()"))
register(_rtmo("l", "L", "rtmo-l_16xb16-600e_coco-640x640.py",
               "rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth",
               "516a421f8717548300c3ee6356a3444ac539083d4a9912f8ca1619ee63d0986d",
               "72.4", "44.8M", "`yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth`", "RtmoPoseDetector.l()"))
