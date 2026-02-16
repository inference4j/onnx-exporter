import os
from pathlib import Path

from exporter.base import ExportedModel
from exporter.registry import register


HF_MODEL_ID = "openai/clip-vit-base-patch32"


class _VisionEncoder:
    """Wraps vision_model + visual_projection for ONNX export."""

    @staticmethod
    def create(clip_model):
        import torch

        class VisionEncoder(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.vision_model = clip_model.vision_model
                self.visual_projection = clip_model.visual_projection

            def forward(self, pixel_values):
                vision_outputs = self.vision_model(pixel_values=pixel_values)
                image_embeds = self.visual_projection(vision_outputs.pooler_output)
                return image_embeds

        return VisionEncoder(clip_model)


class _TextEncoder:
    """Wraps text_model + text_projection for ONNX export."""

    @staticmethod
    def create(clip_model):
        import torch

        class TextEncoder(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.text_model = clip_model.text_model
                self.text_projection = clip_model.text_projection

            def forward(self, input_ids, attention_mask):
                text_outputs = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                pooled_output = text_outputs[1]
                text_embeds = self.text_projection(pooled_output)
                return text_embeds

        return TextEncoder(clip_model)


class ClipVitBasePatch32(ExportedModel):
    name = "clip-vit-base-patch32"
    repo_id = "inference4j/clip-vit-base-patch32"

    def stage(self, staging_dir: Path) -> None:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from transformers import CLIPModel, CLIPProcessor

        # Load model
        print(f"  Loading {HF_MODEL_ID}...")
        model = CLIPModel.from_pretrained(HF_MODEL_ID)
        processor = CLIPProcessor.from_pretrained(HF_MODEL_ID)
        model.eval()

        # Export vision encoder
        vision_encoder = _VisionEncoder.create(model)
        vision_encoder.eval()
        dummy_pixels = torch.randn(1, 3, 224, 224)
        vision_path = str(staging_dir / "vision_model.onnx")

        torch.onnx.export(
            vision_encoder,
            dummy_pixels,
            vision_path,
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "image_embeds": {0: "batch"},
            },
        )
        onnx_model = onnx.load(vision_path, load_external_data=True)
        onnx.save(onnx_model, vision_path)
        data_path = vision_path + ".data"
        if os.path.exists(data_path):
            os.remove(data_path)
        size_mb = os.path.getsize(vision_path) / (1024 * 1024)
        print(f"  Vision encoder exported ({size_mb:.1f} MB)")

        # Export text encoder
        text_encoder = _TextEncoder.create(model)
        text_encoder.eval()
        dummy_ids = torch.randint(0, 49408, (1, 77))
        dummy_mask = torch.ones(1, 77, dtype=torch.long)
        text_path = str(staging_dir / "text_model.onnx")

        torch.onnx.export(
            text_encoder,
            (dummy_ids, dummy_mask),
            text_path,
            opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "text_embeds": {0: "batch"},
            },
        )
        onnx_model = onnx.load(text_path, load_external_data=True)
        onnx.save(onnx_model, text_path)
        data_path = text_path + ".data"
        if os.path.exists(data_path):
            os.remove(data_path)
        size_mb = os.path.getsize(text_path) / (1024 * 1024)
        print(f"  Text encoder exported ({size_mb:.1f} MB)")

        # Copy tokenizer files
        tokenizer = processor.tokenizer
        saved = tokenizer.save_vocabulary(str(staging_dir))
        print(f"  Saved tokenizer vocabulary files: {saved}")

        vocab_path = staging_dir / "vocab.json"
        merges_path = staging_dir / "merges.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Expected {vocab_path} but it was not created")
        if not merges_path.exists():
            raise FileNotFoundError(f"Expected {merges_path} but it was not created")
        print(f"  Tokenizer files: vocab.json, merges.txt")

        # Validate
        print("  Validating exports...")
        vision_session = ort.InferenceSession(vision_path)
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        vision_result = vision_session.run(None, {"pixel_values": dummy})
        assert vision_result[0].shape == (1, 512), \
            f"Expected (1, 512), got {vision_result[0].shape}"
        print(f"    Vision: input (1,3,224,224) -> output {vision_result[0].shape}")

        text_session = ort.InferenceSession(text_path)
        dummy_ids = np.array([[49406, 320, 1125, 539, 320, 2368, 49407] +
                              [0] * 70], dtype=np.int64)
        dummy_mask = np.array([[1, 1, 1, 1, 1, 1, 1] + [0] * 70], dtype=np.int64)
        text_result = text_session.run(None, {
            "input_ids": dummy_ids,
            "attention_mask": dummy_mask,
        })
        assert text_result[0].shape == (1, 512), \
            f"Expected (1, 512), got {text_result[0].shape}"
        print(f"    Text: input (1,77) -> output {text_result[0].shape}")
        print("  Validation OK")

    def render_card(self) -> str:
        return """\
---
library_name: onnx
tags:
  - clip
  - multimodal
  - visual-search
  - zero-shot-classification
  - onnx
  - inference4j
license: mit
datasets:
  - openai/clip-training-data
---

# CLIP ViT-B/32 — ONNX (Vision + Text Encoders)

ONNX export of [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
split into separate vision and text encoder models for independent use.

Converted for use with [inference4j](https://github.com/inference4j/inference4j),
an inference-only AI library for Java.

## Usage with inference4j

### Visual search (image-text similarity)

```java
try (ClipImageEncoder imageEncoder = ClipImageEncoder.builder().build();
     ClipTextEncoder textEncoder = ClipTextEncoder.builder().build()) {

    float[] imageEmb = imageEncoder.encode(ImageIO.read(Path.of("photo.jpg").toFile()));
    float[] textEmb = textEncoder.encode("a photo of a cat");

    float similarity = dot(imageEmb, textEmb);
}
```

### Zero-shot classification

```java
float[] imageEmb = imageEncoder.encode(photo);
String[] labels = {"cat", "dog", "bird", "car"};

float bestScore = Float.NEGATIVE_INFINITY;
String bestLabel = null;
for (String label : labels) {
    float score = dot(imageEmb, textEncoder.encode("a photo of a " + label));
    if (score > bestScore) {
        bestScore = score;
        bestLabel = label;
    }
}
```

## Files

| File | Description | Size |
|------|-------------|------|
| `vision_model.onnx` | Vision encoder (ViT-B/32) | ~340 MB |
| `text_model.onnx` | Text encoder (Transformer) | ~255 MB |
| `vocab.json` | BPE vocabulary (49408 tokens) | ~1.6 MB |
| `merges.txt` | BPE merge rules (48894 merges) | ~1.7 MB |

## Model Details

| Property | Value |
|----------|-------|
| Architecture | ViT-B/32 (vision) + Transformer (text) |
| Embedding dim | 512 |
| Max text length | 77 tokens |
| Image input | `[batch, 3, 224, 224]` — CLIP-normalized |
| Text input | `input_ids` + `attention_mask` `[batch, 77]` |
| ONNX opset | 17 |

## Preprocessing

### Vision
1. Resize to 224×224 (bicubic)
2. CLIP normalization: mean=`[0.48145466, 0.4578275, 0.40821073]`,
   std=`[0.26862954, 0.26130258, 0.27577711]`
3. NCHW layout: `[1, 3, 224, 224]`

### Text
1. Byte-level BPE tokenization using `vocab.json` + `merges.txt`
2. Add `<|startoftext|>` (49406) and `<|endoftext|>` (49407)
3. Pad/truncate to 77 tokens

## Original Paper

> Radford, A., Kim, J. W., Hallacy, C., et al. (2021).
> Learning Transferable Visual Models From Natural Language Supervision.
> ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

## License

The original CLIP model is released under the MIT License by OpenAI.
"""


register(ClipVitBasePatch32())
