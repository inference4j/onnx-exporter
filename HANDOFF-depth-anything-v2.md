# Handoff: export Depth Anything V2 Small

Written 2026-09-08 from the inference4j side. A depth estimation wrapper has landed in
inference4j but its model is not hosted yet, so the wrapper cannot resolve and its model
test is disabled. This note is the context needed to add the export here.

## What is needed

Publish **`inference4j/depth-anything-v2-small`** with `model.onnx` **at the repository
root**.

The flat layout is not cosmetic. `HuggingFaceModelSource.resolve(id, requiredFiles)` calls
`Files.createDirectories(repoDir)` for the repo directory only, then resolves each required
filename directly under it. A nested `onnx/model.onnx` has no parent directory created and
the download fails. Every other inference4j model is flat for this reason.

## Source

`onnx-community/depth-anything-v2-small` — Apache 2.0, already a working ONNX export.
The file is at `onnx/model.onnx`, ~94 MB.

This is a **mirror**, not a fresh export. `MirroredModel` already handles the flattening;
`exporter/models/bge_base_en_v1_5.py` is the closest existing example — it mirrors the same
`onnx/model.onnx` → `model.onnx` shape from a community ONNX repo.

The repo also carries `model_fp16`, `model_int8`, `model_q4`, `model_quantized`, `model_uint8`
and `model_bnb4` variants. Only the fp32 `model.onnx` is needed; the Java wrapper does not
reference the others.

## Model contract the Java side expects

Verified by running the upstream export end to end before writing this.

| | |
|---|---|
| Input | NCHW float32 `[1, 3, H, W]` |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Input sizing | 518 on the longer edge, aspect preserved, both dims rounded **up to a multiple of 14** (ViT patch size) |
| Resampling | bicubic |
| Output name | `predicted_depth` |
| Output shape | `[1, H, W]` — rank 3, no channel dimension |
| Output semantics | relative inverse depth; **larger is closer**; scale is per-image, not metric |

Sanity numbers from a 320×320 photo of a cat: inference 504 ms on CPU, output range
0.0 to 5.53, centre pixel (the cat) 4.55, corner (background) 0.0.

The wrapper accepts rank 3 or rank 4 output by squeezing all size-1 dimensions, so an export
emitting `[1, 1, H, W]` would also work — but `[1, H, W]` is what upstream produces.

## Also worth publishing, lower priority

The Base and Large variants share the same graph and are selectable via `.modelId(...)`.
The wrapper's docs already reference them:

- `inference4j/depth-anything-v2-base` — ~98M params
- `inference4j/depth-anything-v2-large` — ~335M params

Check whether `onnx-community` has equivalents before assuming a fresh export is needed.

## Follow-up in the inference4j repo once uploaded

`inference4j-core/src/modelTest/java/io/github/inference4j/vision/DepthAnythingEstimatorModelTest.java`
carries a class-level `@Disabled` whose message names this exact dependency. Remove the
annotation and run:

```bash
./gradlew :inference4j-core:modelTest --tests "io.github.inference4j.vision.DepthAnythingEstimatorModelTest"
```

Three assertions: output dimensions match the input image, values are finite and varying,
and the rendered image matches input size.

`DepthEstimationExample` in `inference4j-examples` will also start working — it currently
resolves the same unhosted model ID.
