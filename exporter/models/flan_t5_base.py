import shutil
from pathlib import Path

from exporter.base import ExportedModel
from exporter.models.flan_t5_small import _validate_encoder_decoder
from exporter.registry import register


class FlanT5Base(ExportedModel):
    name = "flan-t5-base"
    repo_id = "inference4j/flan-t5-base"
    source_repo = "google/flan-t5-base"

    def stage(self, staging_dir: Path) -> None:
        from huggingface_hub import hf_hub_download
        from optimum.exporters.onnx import main_export

        model_id = "google/flan-t5-base"

        print("  Exporting Flan-T5 Base to ONNX (encoder-decoder with KV cache)...")
        main_export(
            model_name_or_path=model_id,
            output=staging_dir,
            task="text2text-generation-with-past",
            no_post_process=False,
        )

        # Clean up non-essential files
        for name in ("decoder_model_merged.onnx", "generation_config.json",
                     "added_tokens.json", "tokenizer_config.json",
                     "special_tokens_map.json", "sentencepiece.bpe.model"):
            p = staging_dir / name
            if p.exists():
                p.unlink()

        dwp = staging_dir / "decoder_with_past_model.onnx"
        if not dwp.exists():
            for candidate in ("decoder_with_past.onnx",):
                p = staging_dir / candidate
                if p.exists():
                    p.rename(dwp)
                    break

        config_dst = staging_dir / "config.json"
        if not config_dst.exists():
            src = hf_hub_download(repo_id=model_id, filename="config.json")
            shutil.copy2(src, config_dst)
            print("  Copied config.json")

        tok_dst = staging_dir / "tokenizer.json"
        if not tok_dst.exists():
            src = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
            shutil.copy2(src, tok_dst)
            print("  Copied tokenizer.json")

        _validate_encoder_decoder(staging_dir)

    def render_card(self) -> str:
        return """\
---
library_name: onnx
tags:
  - text2text-generation
  - t5
  - flan-t5
  - encoder-decoder
  - onnx
  - inference4j
license: apache-2.0
pipeline_tag: text2text-generation
---

# Flan-T5 Base — ONNX

ONNX export of [Flan-T5 Base](https://huggingface.co/google/flan-t5-base) \
(250M parameters) with encoder-decoder architecture and KV cache support.

Converted for use with [inference4j](https://github.com/inference4j/inference4j), \
an inference-only AI library for Java.

## Original Source

- **Repository:** [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
- **License:** Apache 2.0

## Usage with inference4j

```java
// Summarization
try (var gen = FlanT5TextGenerator.flanT5Base().build()) {
    System.out.println(gen.summarize("Long article text..."));
}

// Translation
try (var gen = FlanT5TextGenerator.flanT5Base().build()) {
    System.out.println(gen.translate("Hello!", Language.EN, Language.FR));
}
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | T5 encoder-decoder (250M parameters) |
| Tasks | Summarization, translation, grammar correction, text-to-SQL |
| Tokenizer | SentencePiece (32,128 tokens) |
| Original framework | PyTorch (transformers) |
| Export method | Hugging Face Optimum (encoder-decoder with KV cache) |

## License

This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). \
Original model by [Google](https://huggingface.co/google).
"""


register(FlanT5Base())
