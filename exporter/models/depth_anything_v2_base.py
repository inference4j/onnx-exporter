from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class DepthAnythingV2Base(MirroredModel):
    name = "depth-anything-v2-base"
    repo_id = "inference4j/depth-anything-v2-base"
    source_repo = "onnx-community/depth-anything-v2-base"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
    ]
    card = ModelCard(
        title="Depth Anything V2 Base — ONNX",
        description="ONNX export of [depth-anything/Depth-Anything-V2-Base-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf), a monocular depth estimation model. Predicts relative inverse depth from a single RGB image — larger values are closer to the camera.",
        license="cc-by-nc-4.0",
        pipeline_tag="depth-estimation",
        tags=["depth-anything", "depth-estimation", "monocular-depth-estimation", "vision", "dpt"],
        original_source_url="https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf",
        original_author="Depth Anything (ONNX by onnx-community)",
        java_usage="""\
try (DepthAnythingEstimator estimator = DepthAnythingEstimator.builder()
        .modelId("inference4j/depth-anything-v2-base")
        .build()) {
    DepthMap depth = estimator.estimate(Path.of("photo.jpg"));
    System.out.println("Depth at centre: " + depth.at(depth.width() / 2, depth.height() / 2));
}""",
        model_details={
            "Architecture": "DINOv2 ViT-Base backbone + DPT decoder",
            "Task": "Monocular depth estimation",
            "Parameters": "~98M",
            "Input": "`[1, 3, height, width]` — NCHW float32, RGB",
            "Normalization": "ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`",
            "Input sizing": "518 on the longer edge, aspect preserved, both dimensions rounded up to a multiple of 14 (ViT patch size), bicubic resampling",
            "Output name": "`predicted_depth`",
            "Output": "`[1, height, width]` — rank 3, no channel dimension",
            "Output semantics": "Relative inverse depth; larger is closer. Scale is per-image, not metric.",
            "Original framework": "PyTorch (HuggingFace Transformers)",
        },
        license_text="**Non-commercial use only.** This model is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), unlike the Small variant, which is Apache 2.0. Original model by [Depth Anything](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf), ONNX export by [onnx-community](https://huggingface.co/onnx-community/depth-anything-v2-base).",
        extra_sections="""\
## Preprocessing

1. Resize so the longer edge is 518, preserving aspect ratio (bicubic)
2. Round both dimensions **up** to a multiple of 14 (the ViT patch size)
3. ImageNet normalization: `(pixel / 255 - mean) / std`
   - mean = `[0.485, 0.456, 0.406]`
   - std = `[0.229, 0.224, 0.225]`
4. NCHW layout: `[1, 3, H, W]`

## Postprocessing

The output is relative inverse depth, so values are only comparable within a single image.
To render a depth map, normalize to `[0, 1]` against the per-image min and max, then resize
back to the original image dimensions.

## Original Paper

> Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024).
> Depth Anything V2.
> NeurIPS 2024. [arXiv:2406.09414](https://arxiv.org/abs/2406.09414)""",
    )


register(DepthAnythingV2Base())
