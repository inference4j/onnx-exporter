"""Export one RTMO checkpoint to ONNX with MMDeploy. Runs inside .venv-rtmo, not the main venv.

`exporter.rtmo_builder` invokes this file as a script with `python -I`, so neither this
directory nor PYTHONPATH is on sys.path; the MMDeploy source tree is passed explicitly.

    python -I exporter/rtmo_export_script.py --mmdeploy-root DIR --config CFG \
        --checkpoint CKPT --output model.onnx

The graph is MMDeploy's RTMO graph with one change (see `_register_rtmo_rewriter`): its padding row gets
score 0 instead of the last anchor's real score, so "score > threshold" alone separates people from padding.

MMDeploy traces the model with one image. The graph does not depend on its content (the input is always
letterboxed to 640x640), so a synthetic grey image is used unless --image is given.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def _register_rtmo_rewriter() -> None:
    """Override MMDeploy 1.3.1's RTMOHead.forward rewriter for the ONNX Runtime backend.

    MMDeploy's `multiclass_nms` pads its result with one zero row so the graph never yields an empty tensor,
    and returns the kept anchor indices with -1 for that row. The stock rewriter then re-gathers boxes and
    scores by those indices, so the padding row becomes a copy of the *last anchor's* real prediction:
    usually a score near 0, but a real person score whenever someone stands in the bottom-right corner,
    and indistinguishable from a 50th person when exactly 49 are found. Here the score column comes from
    the NMS result instead (exactly the same value for real rows, 0 for padding), and the padding row's
    keypoint scores are zeroed too. Everything else is the stock rewriter, line for line.
    """
    import torch
    from mmdeploy.codebase.mmdet import get_post_processing_params
    from mmdeploy.core import FUNCTION_REWRITER
    from mmdeploy.mmcv.ops.nms import multiclass_nms
    from mmpose.structures.bbox import bbox_xyxy2cs

    # A backend-specific rewriter takes precedence over MMDeploy's default one for the same function.
    @FUNCTION_REWRITER.register_rewriter(
        func_name="mmpose.models.heads.hybrid_heads.rtmo_head.RTMOHead.forward", backend="onnxruntime")
    def predict(self, x, batch_data_samples=[], test_cfg=None):
        ctx = FUNCTION_REWRITER.get_context()
        deploy_cfg = ctx.cfg
        cfg = self.test_cfg if test_cfg is None else test_cfg

        cls_scores, bbox_preds, _, kpt_vis, pose_vecs = self.head_module(x)[:5]
        num_imgs = cls_scores[0].shape[0]

        scores = self._flatten_predictions(cls_scores).sigmoid()
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_pose_vecs = self._flatten_predictions(pose_vecs)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis).sigmoid()
        bboxes = self.decode_bbox(flatten_bbox_preds, self.flatten_priors, self.flatten_stride)

        post_params = get_post_processing_params(deploy_cfg)
        nms_dets, _, nms_indices = multiclass_nms(
            bboxes,
            scores,
            post_params.max_output_boxes_per_class,
            cfg.get("nms_thr", post_params.iou_threshold),
            cfg.get("score_thr", post_params.score_threshold),
            pre_top_k=post_params.get("pre_top_k", -1),
            keep_top_k=cfg.get("max_per_img", post_params.keep_top_k),
            output_index=True)

        batch_inds = torch.arange(num_imgs, device=scores.device).view(-1, 1)
        dets = torch.cat([bboxes, scores], dim=2)
        dets = dets[batch_inds, nms_indices, ...]
        pose_vecs = flatten_pose_vecs[batch_inds, nms_indices, ...]
        kpt_vis = flatten_kpt_vis[batch_inds, nms_indices, ...]
        grids = self.flatten_priors[nms_indices, ...]

        bbox_cs = torch.cat(bbox_xyxy2cs(dets[..., :4], self.bbox_padding), dim=-1)
        keypoints = self.dcc.forward_test(pose_vecs, bbox_cs, grids)

        # inference4j change: padding rows carry score 0 (box and keypoint scores).
        nms_scores = nms_dets[..., 4:5]
        dets = torch.cat([dets[..., :4], nms_scores], dim=-1)
        kpt_vis = kpt_vis * (nms_scores > 0).to(kpt_vis.dtype)

        pred_kpts = torch.cat([keypoints, kpt_vis.unsqueeze(-1)], dim=-1)
        return dets, pred_kpts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdeploy-root", required=True, type=Path)
    parser.add_argument("--config", required=True, help="MMPose model config (file or path under mmpose/.mim/configs)")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", type=Path, help="Image MMDeploy traces with (default: synthetic grey)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.mmdeploy_root))

    import mmpose
    from mmdeploy.apis import torch2onnx

    _register_rtmo_rewriter()

    config = Path(args.config)
    if not config.is_file():
        config = Path(mmpose.__file__).parent / ".mim" / "configs" / args.config
    deploy_cfg = args.mmdeploy_root / "configs" / "mmpose" / "pose-detection_rtmo_onnxruntime_dynamic.py"

    with tempfile.TemporaryDirectory() as work_dir:
        image = args.image
        if image is None:
            import cv2
            import numpy as np

            image = Path(work_dir) / "trace.png"
            cv2.imwrite(str(image), np.full((480, 640, 3), 114, np.uint8))
        torch2onnx(
            str(image),
            work_dir,
            "end2end.onnx",
            deploy_cfg=str(deploy_cfg),
            model_cfg=str(config),
            model_checkpoint=str(args.checkpoint),
            device="cpu",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(Path(work_dir) / "end2end.onnx"), args.output)
    print(f"exported {args.output}")


if __name__ == "__main__":
    main()
