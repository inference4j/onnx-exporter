"""Validate the RTMO COCO ONNX exports against MMPose and write Java reference outputs.

Runs inside .venv-rtmo (MMPose is not installed in the main venv):

    .venv-rtmo/bin/python -I validation/rtmo_validate.py output/rtmo-candidate [--models s m l]

For each model and each COCO val2017 image below it checks two things:

1. **Export parity.** The same letterboxed tensor goes through the PyTorch model (MMPose `RTMOHead.predict`,
   not re-parameterised) and through the ONNX graph in ONNX Runtime. Boxes, box scores, keypoints and keypoint
   scores are compared person by person (both lists are sorted by score). This isolates the export itself.
2. **Pipeline agreement.** The ONNX path with inference4j's pre-processing (top-left letterbox) is compared
   with MMPose's own `inference_bottomup` (centred letterbox through `warpAffine`), in original-image pixels.
   Differences here come from the letterbox placement, not from the export; they are reported, not asserted.

The ONNX outputs, mapped back to original-image pixels, are written to `validation/reference/<model>.json`
for the Java parity tests. Threads are capped at 4.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

IMAGES = (
    "000000000785",  # one skier
    "000000040083",  # two people
    "000000196141",  # small group
    "000000197388",  # several players
    "000000252219",  # street, three people
    "000000397133",  # kitchen, one main person plus background
    "000000001000",  # crowded group (~20 people)
)
COCO_URL = "http://images.cocodataset.org/val2017/{}.jpg"
IMAGE_CACHE = Path.home() / ".cache" / "inference4j" / "rtmo" / "images"
CHECKPOINT_CACHE = Path.home() / ".cache" / "inference4j" / "rtmo"
REFERENCE_DIR = Path(__file__).parent / "reference"
CONFIGS = {
    "s": ("rtmo-s_8xb32-600e_coco-640x640.py", "rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth"),
    "m": ("rtmo-m_16xb16-600e_coco-640x640.py", "rtmo-m_16xb16-600e_coco-640x640-6f4e0306_20231211.pth"),
    "l": ("rtmo-l_16xb16-600e_coco-640x640.py", "rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth"),
}
SIZE = 640
SCORE_THR = 0.1
THREADS = 4


def image_path(image_id: str) -> Path:
    path = IMAGE_CACHE / f"{image_id}.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COCO_URL.format(image_id), path)
    return path


def letterbox(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Top-left letterbox: OpenCV bilinear resize, pad 114, raw BGR, NCHW float32."""
    h, w = img.shape[:2]
    r = min(SIZE / h, SIZE / w)
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    return canvas.transpose(2, 0, 1)[None].astype(np.float32), r


def onnx_people(session: ort.InferenceSession, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the graph and keep rows scoring above the threshold. Returns (dets [n,5], keypoints [n,17,3], raw N).

    The graph appends one padding row with score 0 when fewer than 50 people are found; the score filter
    removes it. Asserts that every dropped row is such a padding row.
    """
    dets, kpts = session.run(None, {"input": x})
    dets, kpts = dets[0], kpts[0]
    raw = dets.shape[0]
    keep = dets[:, 4] > SCORE_THR
    dropped = dets[~keep]
    assert len(dropped) <= 1 and np.all(dropped[:, 4] == 0), f"unexpected low-score rows: {dropped}"
    assert np.all(kpts[~keep][:, :, 2] == 0), "padding row keypoint scores are not zero"
    assert np.all(np.diff(dets[:, 4]) <= 0), "rows are not sorted by score"
    dets, kpts = dets[keep].copy(), kpts[keep].copy()
    dets[:, 0:4:2] = dets[:, 0:4:2].clip(0, SIZE)
    dets[:, 1:4:2] = dets[:, 1:4:2].clip(0, SIZE)
    return dets, kpts, raw


def torch_people(model, x: np.ndarray):
    from mmpose.structures import PoseDataSample

    sample = PoseDataSample(metainfo=dict(input_size=(SIZE, SIZE)))
    with torch.no_grad():
        feats = model.extract_feat(torch.from_numpy(x))
        result = model.head.predict(feats, [sample], test_cfg=model.test_cfg)[0]
    return result


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def validate(size: str, candidate: Path) -> dict:
    from mmpose.apis import inference_bottomup, init_model
    import mmpose

    config, checkpoint = CONFIGS[size]
    config_path = Path(mmpose.__file__).parent / ".mim" / "configs" / "body_2d_keypoint" / "rtmo" / "coco" / config
    model = init_model(str(config_path), str(CHECKPOINT_CACHE / checkpoint), device="cpu")
    model.eval()

    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    onnx_path = candidate / "inference4j" / f"rtmo-{size}-coco" / "model.onnx"
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])

    worst = dict(box=0.0, box_score=0.0, keypoint=0.0, keypoint_score=0.0)
    pipeline_diffs = []
    reference = []
    count_mismatches = []
    for image_id in IMAGES:
        img = cv2.imread(str(image_path(image_id)))
        x, r = letterbox(img)

        dets, kpts, raw = onnx_people(session, x)
        ref = torch_people(model, x)
        n = len(ref.scores)
        if n != len(dets):
            count_mismatches.append((image_id, n, len(dets)))
        m = min(n, len(dets))
        if m:
            worst["box"] = max(worst["box"], float(np.abs(dets[:m, :4] - ref.bboxes[:m]).max()))
            worst["box_score"] = max(worst["box_score"], float(np.abs(dets[:m, 4] - ref.scores[:m]).max()))
            worst["keypoint"] = max(worst["keypoint"], float(np.abs(kpts[:m, :, :2] - ref.keypoints[:m]).max()))
            worst["keypoint_score"] = max(worst["keypoint_score"],
                                          float(np.abs(kpts[:m, :, 2] - ref.keypoint_scores[:m]).max()))

        # Original-image pixels: undo the top-left letterbox.
        h, w = img.shape[:2]
        boxes = dets[:, :4] / r
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
        points = kpts.copy()
        points[:, :, :2] /= r

        # MMPose's own pipeline (centred letterbox), matched by box IoU, confident people and joints only.
        mm = inference_bottomup(model, str(image_path(image_id)))[0].pred_instances
        for i in np.where(dets[:, 4] > 0.5)[0]:
            ious = [iou(boxes[i], b) for b in mm.bboxes]
            if not ious or max(ious) < 0.5:
                continue
            j = int(np.argmax(ious))
            visible = (points[i, :, 2] > 0.5) & (mm.keypoint_scores[j] > 0.5)
            if visible.any():
                d = np.linalg.norm(points[i, visible, :2] - mm.keypoints[j][visible], axis=1)
                pipeline_diffs.extend(d.tolist())

        reference.append({
            "image": f"{image_id}.jpg",
            "url": COCO_URL.format(image_id),
            "width": w,
            "height": h,
            "ratio": r,
            "raw_rows": raw,
            "people": [
                {
                    "box": [round(float(v), 2) for v in boxes[i]],
                    "score": round(float(dets[i, 4]), 4),
                    "keypoints": [[round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 4)]
                                  for p in points[i]],
                }
                for i in range(len(dets))
            ],
        })
        print(f"  {image_id}: {len(dets)} people (torch {n}), raw rows {raw}")

    diffs = np.array(pipeline_diffs) if pipeline_diffs else np.zeros(1)
    summary = {
        "export_max_abs_diff": {k: round(v, 5) for k, v in worst.items()},
        "count_mismatches": count_mismatches,
        "pipeline_topleft_vs_mmpose_centred_px": {
            "joints": int(len(pipeline_diffs)),
            "median": round(float(np.median(diffs)), 2),
            "p90": round(float(np.percentile(diffs, 90)), 2),
            "max": round(float(diffs.max()), 2),
        },
    }
    out = {
        "model": f"rtmo-{size}-coco",
        "checkpoint": checkpoint,
        "preprocessing": "top-left letterbox, cv2.INTER_LINEAR, pad 114, raw BGR 0-255",
        "postprocessing": f"keep rows with score > {SCORE_THR} (drops the score-0 padding row), "
                          "divide by ratio, clamp boxes to the image",
        "onnxruntime": ort.__version__,
        "summary": summary,
        "images": reference,
    }
    REFERENCE_DIR.mkdir(exist_ok=True)
    (REFERENCE_DIR / f"rtmo-{size}-coco.json").write_text(json.dumps(out, indent=1) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path, help="--stage-to directory holding inference4j/rtmo-*-coco/model.onnx")
    parser.add_argument("--models", nargs="+", default=["s", "m", "l"], choices=sorted(CONFIGS))
    args = parser.parse_args()
    torch.set_num_threads(THREADS)
    for size in args.models:
        print(f"rtmo-{size}-coco")
        print(json.dumps(validate(size, args.candidate), indent=1))


if __name__ == "__main__":
    main()
