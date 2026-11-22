from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class Sam21HieraTiny(MirroredModel):
    name = "sam2.1-hiera-tiny"
    repo_id = "inference4j/sam2.1-hiera-tiny"
    source_repo = "onnx-community/sam2.1-hiera-tiny-ONNX"
    source_type = "hf"
    revision = "814a066640debee5a91e70aa401fb8e17e030503"
    files = [
        FileMapping(src="onnx/vision_encoder.onnx", dst="vision_encoder.onnx"),
        FileMapping(src="onnx/vision_encoder.onnx_data", dst="vision_encoder.onnx_data"),
        FileMapping(
            src="onnx/prompt_encoder_mask_decoder.onnx",
            dst="prompt_encoder_mask_decoder.onnx",
        ),
        FileMapping(
            src="onnx/prompt_encoder_mask_decoder.onnx_data",
            dst="prompt_encoder_mask_decoder.onnx_data",
        ),
        FileMapping(src="config.json", dst="config.json"),
        FileMapping(src="preprocessor_config.json", dst="preprocessor_config.json"),
        FileMapping(src="processor_config.json", dst="processor_config.json"),
    ]
    card = ModelCard(
        title="SAM 2.1 Hiera Tiny — ONNX",
        description=(
            "FP32 ONNX export of [facebook/sam2.1-hiera-tiny]"
            "(https://huggingface.co/facebook/sam2.1-hiera-tiny), split into an image "
            "encoder and a prompt encoder/mask decoder for point- and box-prompted "
            "image segmentation. The ONNX graphs are mirrored unchanged from the "
            "[onnx-community export]"
            "(https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX)."
        ),
        license="apache-2.0",
        pipeline_tag="mask-generation",
        tags=["sam2", "image-segmentation", "promptable-segmentation", "vision"],
        original_source_url="https://huggingface.co/facebook/sam2.1-hiera-tiny",
        original_author="Meta (ONNX export by onnx-community)",
        java_usage="""\
try (Sam2Segmenter segmenter = Sam2Segmenter.builder().build()) {
    SegmentationMask mask = segmenter.segment(
        Path.of("photo.jpg"),
        List.of(new PromptPoint(320, 240, true)));
    ImageIO.write(mask.toImage(), "png", Path.of("mask.png").toFile());
}""",
        model_details={
            "Architecture": "SAM 2.1 Hiera Tiny",
            "Task": "Promptable image segmentation",
            "Graphs": "`vision_encoder.onnx` and `prompt_encoder_mask_decoder.onnx` (opset 18, FP32)",
            "External data": "Each graph loads its matching `.onnx_data` sidecar from the same directory",
            "Image input": "`pixel_values`: `[batch, 3, 1024, 1024]` float32, RGB",
            "Image preprocessing": "Resize to 1024 x 1024 without preserving aspect ratio; rescale by 1/255; ImageNet mean/std normalization",
            "Prompts": "Point coordinates and boxes use the 1024 x 1024 input coordinate space",
            "Mask output": "`pred_masks`: `[batch, prompts, 3, 256, 256]` float32 low-resolution logits",
            "Mask selection": "Choose one of three candidate masks using `iou_scores`",
            "Refinement": "No previous-mask (`mask_input`) input; iterative mask refinement is not supported by this export",
        },
        license_text=(
            "The original SAM 2.1 model and weights are licensed under the "
            "[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). "
            "Original model by [Meta](https://huggingface.co/facebook/sam2.1-hiera-tiny); "
            "ONNX conversion by [onnx-community]"
            "(https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX)."
        ),
        extra_sections="""\
## Files

The FP32 model remains split into two ONNX graphs and their external-data sidecars:

- `vision_encoder.onnx` + `vision_encoder.onnx_data`
- `prompt_encoder_mask_decoder.onnx` + `prompt_encoder_mask_decoder.onnx_data`

All four files must remain in the same directory. The sidecars are deliberately not
folded into the graph files.

## Prompt Encoding

- Foreground point label: `1`
- Background point label: `0`
- Padding point label: `-10`
- Box layout: `(x1, y1, x2, y2)`

Coordinates are expressed in the resized 1024 x 1024 image space. See the source
`processor_config.json` for processor defaults and the inference4j wrapper for conversion
from original-image coordinates.

## Outputs

The decoder returns three candidate low-resolution mask logits per prompt, their
`iou_scores`, and `object_score_logits`. Resize the selected logits to the original image
size and threshold at zero to obtain a binary mask.
""",
    )


register(Sam21HieraTiny())
