import shutil
from pathlib import Path

from exporter.base import ExportedModel
from exporter.models.flan_t5_small import _validate_encoder_decoder
from exporter.registry import register


class T5Text2SqlSpider(ExportedModel):
    name = "T5-LM-Large-text2sql-spider"
    repo_id = "inference4j/T5-LM-Large-text2sql-spider"
    source_repo = "gaussalgo/T5-LM-Large-text2sql-spider"

    def stage(self, staging_dir: Path) -> None:
        from huggingface_hub import hf_hub_download
        from optimum.exporters.onnx import main_export

        model_id = "gaussalgo/T5-LM-Large-text2sql-spider"

        print("  Exporting T5-LM-Large-text2sql-spider to ONNX (encoder-decoder with KV cache)...")
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
  - text-to-sql
  - sql
  - spider
  - encoder-decoder
  - onnx
  - inference4j
license: apache-2.0
pipeline_tag: text2text-generation
datasets:
  - spider
  - spider-syn
---

# T5-LM-Large text2sql-spider — ONNX

ONNX export of [T5-LM-Large-text2sql-spider](https://huggingface.co/gaussalgo/T5-LM-Large-text2sql-spider) \
(0.8B parameters) with encoder-decoder architecture and KV cache support.

This is a T5-large model fine-tuned on the Spider and Spider-Syn datasets for \
text-to-SQL generation. Given a natural language question and a database schema, \
it produces the corresponding SQL query.

Converted for use with [inference4j](https://github.com/inference4j/inference4j), \
an inference-only AI library for Java.

## Original Source

- **Repository:** [gaussalgo/T5-LM-Large-text2sql-spider](https://huggingface.co/gaussalgo/T5-LM-Large-text2sql-spider)
- **Base model:** [google/t5-large-lm-adapt](https://huggingface.co/google/t5-large-lm-adapt)
- **License:** Apache 2.0

## Usage with inference4j

```java
try (var sqlGen = T5SqlGenerator.t5LargeSpider().build()) {
    String sql = sqlGen.generateSql(
        "How many employees are in each department?",
        "\\"employees\\" \\"id\\" int, \\"name\\" varchar, \\"dept_id\\" int "
        + "[SEP] \\"departments\\" \\"id\\" int, \\"name\\" varchar");
    System.out.println(sql);
}
```

## Schema Format

The model expects the schema in the following format:

```
"table_name" "col1" type, "col2" type, foreign_key: "table"."col" = "other"."col" primary key: "col" [SEP] "table2" ...
```

- Table and column names are double-quoted
- Columns are comma-separated with types
- Tables are separated by `[SEP]`
- Foreign keys and primary keys are declared per table

## Model Details

| Property | Value |
|----------|-------|
| Architecture | T5 encoder-decoder (0.8B parameters) |
| Task | Text-to-SQL generation |
| Training data | Spider, Spider-Syn |
| Tokenizer | SentencePiece (32,128 tokens) |
| Original framework | PyTorch (transformers) |
| Export method | Hugging Face Optimum (encoder-decoder with KV cache) |

## License

This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). \
Original model by [Gaussalgo](https://huggingface.co/gaussalgo), \
base model by [Google](https://huggingface.co/google).
"""


register(T5Text2SqlSpider())
