import shutil
from pathlib import Path

from exporter.base import ExportedModel
from exporter.models.flan_t5_small import _validate_encoder_decoder
from exporter.registry import register


class OpusMtEnEs(ExportedModel):
    name = "opus-mt-en-es"
    repo_id = "inference4j/opus-mt-en-es"
    source_repo = "Helsinki-NLP/opus-mt-en-es"

    def stage(self, staging_dir: Path) -> None:
        from huggingface_hub import hf_hub_download
        from optimum.exporters.onnx import main_export

        model_id = "Helsinki-NLP/opus-mt-en-es"

        print("  Exporting MarianMT en→es to ONNX (encoder-decoder with KV cache)...")
        main_export(
            model_name_or_path=model_id,
            output=staging_dir,
            task="text2text-generation-with-past",
            no_post_process=False,
        )

        # Convert SentencePiece tokenizer to tokenizer.json BEFORE cleanup
        # (MarianMT models ship source.spm + target.spm, not tokenizer.json)
        tok_dst = staging_dir / "tokenizer.json"
        if not tok_dst.exists():
            source_spm = staging_dir / "source.spm"
            if source_spm.exists():
                from transformers.convert_slow_tokenizer import SentencePieceExtractor
                from tokenizers import Tokenizer
                from tokenizers.models import BPE

                print("  Converting SentencePiece tokenizer to tokenizer.json...")
                extractor = SentencePieceExtractor(str(source_spm))
                vocab, merges = extractor.extract(None)
                tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))
                tokenizer.save(str(tok_dst))

        # Clean up non-essential files
        for name in ("decoder_model_merged.onnx", "generation_config.json",
                     "added_tokens.json", "tokenizer_config.json",
                     "special_tokens_map.json", "source.spm", "target.spm",
                     "vocab.json"):
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

        _validate_encoder_decoder(staging_dir)

    def render_card(self) -> str:
        return """\
---
library_name: onnx
tags:
  - translation
  - marian
  - encoder-decoder
  - en
  - es
  - onnx
  - inference4j
license: apache-2.0
pipeline_tag: translation
---

# MarianMT English→Spanish — ONNX

ONNX export of [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) \
with encoder-decoder architecture and KV cache support.

Trained on the OPUS parallel corpus for English to Spanish translation.

Converted for use with [inference4j](https://github.com/inference4j/inference4j), \
an inference-only AI library for Java.

## Original Source

- **Repository:** [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es)
- **License:** Apache 2.0

## Usage with inference4j

```java
try (var translator = MarianTranslator.builder()
        .modelId("inference4j/opus-mt-en-es").build()) {
    System.out.println(translator.translate("The weather is beautiful today."));
}
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | MarianMT encoder-decoder |
| Language pair | English → Spanish |
| Training data | OPUS |
| Tokenizer | SentencePiece |
| Original framework | PyTorch (transformers) |
| Export method | Hugging Face Optimum (encoder-decoder with KV cache) |

## License

This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). \
Original model by [Helsinki-NLP](https://huggingface.co/Helsinki-NLP).
"""


register(OpusMtEnEs())
