import os
from collections import OrderedDict
from pathlib import Path

from exporter.base import ExportedModel
from exporter.registry import register


WEIGHTS_URL = "https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"
WEIGHTS_CACHE = Path.home() / ".cache" / "inference4j" / "craft_mlt_25k.pth"


def _init_weights(modules):
    import torch.nn as nn

    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def _build_vgg16bn():
    import torch.nn as nn
    from torchvision import models

    class VGG16BN(nn.Module):
        def __init__(self):
            super().__init__()
            vgg_pretrained_features = models.vgg16_bn(pretrained=False).features

            self.slice1 = nn.Sequential()
            self.slice2 = nn.Sequential()
            self.slice3 = nn.Sequential()
            self.slice4 = nn.Sequential()
            self.slice5 = nn.Sequential()

            for x in range(12):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(12, 19):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(19, 29):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(29, 39):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])

            self.slice5 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1),
            )
            _init_weights(self.slice5.modules())

        def forward(self, x):
            h = self.slice1(x)
            h_relu2_2 = h
            h = self.slice2(h)
            h_relu3_2 = h
            h = self.slice3(h)
            h_relu4_3 = h
            h = self.slice4(h)
            h_relu5_3 = h
            h = self.slice5(h)
            h_fc7 = h
            return h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2

    return VGG16BN


def _build_craft():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    VGG16BN = _build_vgg16bn()

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class CRAFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.basenet = VGG16BN()

            self.upconv1 = DoubleConv(1024, 512, 256)
            self.upconv2 = DoubleConv(512, 256, 128)
            self.upconv3 = DoubleConv(256, 128, 64)
            self.upconv4 = DoubleConv(128, 64, 32)

            num_class = 2
            self.conv_cls = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, num_class, kernel_size=1),
            )

            _init_weights(self.upconv1.modules())
            _init_weights(self.upconv2.modules())
            _init_weights(self.upconv3.modules())
            _init_weights(self.upconv4.modules())
            _init_weights(self.conv_cls.modules())

        def forward(self, x):
            sources = self.basenet(x)

            y = torch.cat([sources[0], sources[1]], dim=1)
            y = self.upconv1(y)

            y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[2]], dim=1)
            y = self.upconv2(y)

            y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[3]], dim=1)
            y = self.upconv3(y)

            y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[4]], dim=1)
            feature = self.upconv4(y)

            y = self.conv_cls(feature)

            return y.permute(0, 2, 3, 1), feature

    return CRAFT


class CraftMlt25k(ExportedModel):
    name = "craft-mlt-25k"
    repo_id = "inference4j/craft-mlt-25k"

    def stage(self, staging_dir: Path) -> None:
        import gdown
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch

        # Download weights (cached)
        WEIGHTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        if not WEIGHTS_CACHE.exists():
            print(f"  Downloading weights to {WEIGHTS_CACHE}...")
            gdown.download(WEIGHTS_URL, str(WEIGHTS_CACHE), quiet=False)
        else:
            print(f"  Weights cached at {WEIGHTS_CACHE}")

        # Load model
        CRAFT = _build_craft()
        model = CRAFT()
        state_dict = torch.load(str(WEIGHTS_CACHE), map_location="cpu", weights_only=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.eval()

        # Export ONNX
        output_path = str(staging_dir / "model.onnx")
        dummy_input = torch.randn(1, 3, 640, 640)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["input"],
            output_names=["score_map", "feature_map"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "score_map": {0: "batch", 1: "height", 2: "width"},
                "feature_map": {0: "batch", 2: "height", 3: "width"},
            },
        )

        # Consolidate into single file
        onnx_model = onnx.load(output_path, load_external_data=True)
        onnx.save(onnx_model, output_path)
        data_path = output_path + ".data"
        if os.path.exists(data_path):
            os.remove(data_path)
            print("  Removed external data file (consolidated into single .onnx)")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Exported to model.onnx ({size_mb:.1f} MB, single file)")

        # Validate
        print("  Validating export...")
        session = ort.InferenceSession(output_path)
        dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
        results = session.run(None, {"input": dummy})
        print(f"    score_map shape: {results[0].shape}")
        print(f"    feature_map shape: {results[1].shape}")
        print("  Validation OK")

    def render_card(self) -> str:
        return """\
---
library_name: onnx
tags:
  - text-detection
  - craft
  - onnx
  - inference4j
license: mit
datasets:
  - SynthText
  - IC13
  - IC17
---

# CRAFT Text Detection (MLT 25K) — ONNX

ONNX export of [CRAFT](https://github.com/clovaai/CRAFT-pytorch) (Character Region Awareness for Text Detection) trained on SynthText + IC13/IC17 (MLT 25K variant).

Converted for use with [inference4j](https://github.com/inference4j/inference4j), an inference-only AI library for Java.

## Usage with inference4j

```java
try (CraftTextDetector detector = CraftTextDetector.builder().build()) {
    List<TextRegion> regions = detector.detect(Path.of("document.jpg"));
    for (TextRegion r : regions) {
        System.out.printf("Text at [%.0f, %.0f, %.0f, %.0f] (confidence=%.2f)%n",
            r.box().x1(), r.box().y1(), r.box().x2(), r.box().y2(),
            r.confidence());
    }
}
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | VGG16-BN backbone + U-Net decoder |
| Task | Text detection (character-level region + affinity maps) |
| Training data | SynthText + ICDAR 2013/2017 (MLT) |
| Weights | `craft_mlt_25k.pth` from [clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) |
| ONNX opset | 17 |
| Input | `[batch, 3, height, width]` — RGB, ImageNet-normalized, dimensions must be multiples of 32 |
| Output: score_map | `[batch, height/2, width/2, 2]` — channel 0: region score, channel 1: affinity score |
| Output: feature_map | `[batch, 32, height/2, width/2]` — intermediate features (optional, for refinement) |
| Dynamic axes | Batch, height, and width are dynamic |

## Preprocessing

1. Resize maintaining aspect ratio (long side to target size, e.g. 1280)
2. Round both dimensions to nearest multiple of 32
3. ImageNet normalization: `(pixel / 255 - mean) / std`
   - mean = `[0.485, 0.456, 0.406]`
   - std = `[0.229, 0.224, 0.225]`
4. NCHW layout: `[1, 3, H, W]`

## Postprocessing

1. Combine: `clip(region_score + affinity_score, 0, 1)`
2. Binary threshold at `low_text_threshold` (default 0.4)
3. Connected component labeling (4-connectivity)
4. For each component: compute mean region score, filter by `text_threshold` (default 0.7)
5. Extract axis-aligned bounding box, scale back to original image coordinates

## Original Paper

> Baek, Y., Lee, B., Han, D., Yun, S., & Lee, H. (2019).
> Character Region Awareness for Text Detection.
> CVPR 2019. [arXiv:1904.01941](https://arxiv.org/abs/1904.01941)

## License

The original CRAFT model weights are released under the MIT License by Clova AI Research (NAVER Corp).
"""


register(CraftMlt25k())
