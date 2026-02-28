import shutil
from pathlib import Path

from exporter.base import ExportedModel
from exporter.models.flan_t5_small import _validate_encoder_decoder
from exporter.registry import register


class CoeditBase(ExportedModel):
    name = "coedit-base"
    repo_id = "inference4j/coedit-base"
    source_repo = "jbochi/coedit-base"

    def stage(self, staging_dir: Path) -> None:
        from huggingface_hub import hf_hub_download
        from optimum.exporters.onnx import main_export

        model_id = "jbochi/coedit-base"

        print("  Exporting CoEdIT Base to ONNX (encoder-decoder with KV cache)...")
        main_export(
            model_name_or_path=model_id,
            output=staging_dir,
            task="text2text-generation-with-past",
            no_post_process=False,
        )

        # Clean up non-essential files
        for name in ("decoder_model_merged.onnx", "generation_config.json",
                     "added_tokens.json", "tokenizer_config.json",
                     "special_tokens_map.json", "sentencepiece.bpe.model",
                     "spiece.model"):
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

        # CoEdIT uses SentencePiece tokenizer (tokenizer.json)
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
  - coedit
  - grammar-correction
  - encoder-decoder
  - onnx
  - inference4j
license: apache-2.0
pipeline_tag: text2text-generation
---

# CoEdIT Base — ONNX

ONNX export of [jbochi/coedit-base](https://huggingface.co/jbochi/coedit-base) \
(250M parameters) with encoder-decoder architecture and KV cache support.

CoEdIT is a T5-based model fine-tuned on the [grammarly/coedit](https://huggingface.co/datasets/grammarly/coedit) \
dataset for text editing tasks including grammar correction, simplification, \
coherence, and paraphrasing. This base variant is fine-tuned from `google/flan-t5-base`.

Converted for use with [inference4j](https://github.com/inference4j/inference4j), \
an inference-only AI library for Java.

## Original Source

- **Repository:** [jbochi/coedit-base](https://huggingface.co/jbochi/coedit-base)
- **License:** Apache 2.0

## Usage with inference4j

```java
try (var corrector = CoeditGrammarCorrector.coeditBase().build()) {
    System.out.println(corrector.correct("She don't likes swimming."));
    // She doesn't like swimming.
}
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | T5 encoder-decoder (250M parameters) |
| Base model | google/flan-t5-base |
| Training data | grammarly/coedit |
| Task | Grammar correction, text editing |
| Tokenizer | SentencePiece (32,128 tokens) |
| Original framework | PyTorch (transformers) |
| Export method | Hugging Face Optimum (encoder-decoder with KV cache) |

## License

This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). \
Original model by [jbochi](https://huggingface.co/jbochi), trained on the \
[Grammarly CoEdIT dataset](https://huggingface.co/datasets/grammarly/coedit).
"""


register(CoeditBase())
