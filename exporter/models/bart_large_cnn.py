import shutil
from pathlib import Path

from exporter.base import ExportedModel
from exporter.models.flan_t5_small import _validate_encoder_decoder
from exporter.registry import register


class BartLargeCnn(ExportedModel):
    name = "bart-large-cnn"
    repo_id = "inference4j/bart-large-cnn"
    source_repo = "facebook/bart-large-cnn"

    def stage(self, staging_dir: Path) -> None:
        from huggingface_hub import hf_hub_download
        from optimum.exporters.onnx import main_export

        model_id = "facebook/bart-large-cnn"

        print("  Exporting BART Large CNN to ONNX (encoder-decoder with KV cache)...")
        main_export(
            model_name_or_path=model_id,
            output=staging_dir,
            task="text2text-generation-with-past",
            no_post_process=False,
        )

        # Clean up non-essential files
        for name in ("decoder_model_merged.onnx", "generation_config.json",
                     "added_tokens.json", "tokenizer_config.json",
                     "special_tokens_map.json"):
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

        # BART uses BPE tokenizer (vocab.json + merges.txt)
        for filename in ("vocab.json", "merges.txt"):
            dst = staging_dir / filename
            if not dst.exists():
                src = hf_hub_download(repo_id=model_id, filename=filename)
                shutil.copy2(src, dst)
                print(f"  Copied {filename}")

        tok_json = staging_dir / "tokenizer.json"
        if tok_json.exists():
            tok_json.unlink()

        _validate_encoder_decoder(staging_dir)

    def render_card(self) -> str:
        return """\
---
library_name: onnx
tags:
  - text2text-generation
  - bart
  - summarization
  - encoder-decoder
  - onnx
  - inference4j
license: apache-2.0
pipeline_tag: summarization
---

# BART Large CNN — ONNX

ONNX export of [BART Large CNN](https://huggingface.co/facebook/bart-large-cnn) \
(406M parameters) with encoder-decoder architecture and KV cache support.

Fine-tuned for text summarization on the CNN/DailyMail dataset.

Converted for use with [inference4j](https://github.com/inference4j/inference4j), \
an inference-only AI library for Java.

## Original Source

- **Repository:** [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
- **License:** Apache 2.0

## Usage with inference4j

```java
try (var summarizer = BartSummarizer.bartLargeCnn().build()) {
    System.out.println(summarizer.summarize("Long article text..."));
}
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | BART encoder-decoder (406M parameters, 12 encoder + 12 decoder layers) |
| Task | Text summarization |
| Training data | CNN/DailyMail |
| Tokenizer | BPE (50,265 tokens) |
| Original framework | PyTorch (transformers) |
| Export method | Hugging Face Optimum (encoder-decoder with KV cache) |

## License

This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). \
Original model by [Facebook AI](https://huggingface.co/facebook).
"""


register(BartLargeCnn())
