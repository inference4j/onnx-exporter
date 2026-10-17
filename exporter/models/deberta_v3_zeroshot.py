import shutil
from pathlib import Path

from exporter.base import ExportedModel
from exporter.registry import register


_PUBLISHED_FILES = {
    "model.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}


class _DebertaV3ZeroShot(ExportedModel):
    parameters: str
    mean_f1: str
    size: str

    def stage(self, staging_dir: Path) -> None:
        from optimum.exporters.onnx import main_export

        print(f"  Exporting {self.source_repo} to ONNX (fp32, opset 18)...")
        main_export(
            model_name_or_path=self.source_repo,
            revision=self.revision,
            output=staging_dir,
            task="text-classification",
            opset=18,
            dtype="fp32",
        )

        sidecars = sorted(staging_dir.glob("*.onnx_data"))
        if sidecars:
            raise RuntimeError("Expected a single model.onnx file, but Optimum produced external data")

        for path in staging_dir.iterdir():
            if path.name in _PUBLISHED_FILES:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        missing = sorted(name for name in _PUBLISHED_FILES if not (staging_dir / name).is_file())
        if missing:
            raise RuntimeError(f"Export did not produce required files: {', '.join(missing)}")

    def render_card(self) -> str:
        return f"""\
---
library_name: onnx
tags:
  - nli
  - deberta-v3
  - zero-shot
  - commercially-friendly-data
  - onnx
  - inference4j
license: mit
pipeline_tag: zero-shot-classification
datasets:
  - MoritzLaurer/synthetic_zeroshot_mixtral_v0.1
  - nyu-mll/multi_nli
  - fever
---

# {self.name} — ONNX

FP32 ONNX export of [{self.source_repo}](https://huggingface.co/{self.source_repo}),
a DeBERTa-v3 natural-language-inference model for zero-shot text classification.

Converted for use with [inference4j](https://github.com/inference4j/inference4j),
an inference-only AI library for Java.

## Original Source

- **Repository:** [{self.source_repo}](https://huggingface.co/{self.source_repo})
- **Author:** [Moritz Laurer](https://huggingface.co/MoritzLaurer)
- **Revision:** `{self.revision}`
- **License:** MIT

The `-c` variant was trained only on commercially friendly data: synthetic data generated
with Mixtral-8x7B-Instruct-v0.1, MNLI, and FEVER-NLI. See the upstream model card's
[Training data section](https://huggingface.co/{self.source_repo}#training-data) for details.

## Usage with inference4j

```java
try (var classifier = ZeroShotTextClassifier.builder().build()) {{
    List<TextClassification> result = classifier.classify(
            text, List.of("billing", "outage", "feature request"));
}}
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | DeBERTa-v3 {self.size} (`DebertaV2ForSequenceClassification`) |
| Parameters | {self.parameters} |
| Task | Natural language inference / zero-shot text classification |
| Inputs | `input_ids`, `attention_mask` — int64 `[batch, sequence]` |
| Output | `logits` — float32 `[batch, 2]` |
| Labels | 0: `entailment`, 1: `not_entailment` |
| Maximum sequence length | 512 |
| Tokenizer | SentencePiece Unigram (`tokenizer.json`) |
| Mean F1 over 28 zero-shot tasks | {self.mean_f1} |
| Original framework | PyTorch (Transformers) |
| Export method | Hugging Face Optimum, fp32, opset 18 |

The logits are the original model output. No softmax or label reordering is included in the graph.

## License

Copyright (c) Moritz Laurer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""


class DebertaV3BaseZeroShotV20C(_DebertaV3ZeroShot):
    name = "deberta-v3-base-zeroshot-v2.0-c"
    repo_id = "inference4j/deberta-v3-base-zeroshot-v2.0-c"
    source_repo = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c"
    revision = "bddf8c5411c34ac3565e16e04384fd68b2618dda"
    parameters = "~184M"
    mean_f1 = "0.619"
    size = "base"


class DebertaV3LargeZeroShotV20C(_DebertaV3ZeroShot):
    name = "deberta-v3-large-zeroshot-v2.0-c"
    repo_id = "inference4j/deberta-v3-large-zeroshot-v2.0-c"
    source_repo = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c"
    revision = "b2730f16019076bb0009481121efbe4705e0e378"
    parameters = "~435M"
    mean_f1 = "0.676"
    size = "large"


register(DebertaV3BaseZeroShotV20C())
register(DebertaV3LargeZeroShotV20C())
