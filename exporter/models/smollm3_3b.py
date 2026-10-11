import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

from exporter.base import ExportedModel
from exporter.genai_builder import TEXT_GENAI_SOURCE_PATTERNS, ensure_chat_template, get_builder_python
from exporter.registry import register


class SmolLM3_3B(ExportedModel):
    """SmolLM3-3B exported to onnxruntime-genai format (INT4, CPU) using genai's model builder."""

    name = "smollm3-3b"
    repo_id = "inference4j/smollm3-3b"
    source_repo = "HuggingFaceTB/SmolLM3-3B"
    tracked_patterns = TEXT_GENAI_SOURCE_PATTERNS

    def stage(self, staging_dir: Path) -> None:
        source_dir = staging_dir / "_source"
        output_dir = staging_dir / "_export"

        print(f"  Downloading {self.source_repo}@{(self.revision or 'HEAD')[:8]} ...")
        snapshot_download(
            repo_id=self.source_repo,
            revision=self.revision,
            allow_patterns=list(self.tracked_patterns),
            local_dir=source_dir,
        )

        cmd = [
            str(get_builder_python()), "-m", "onnxruntime_genai.models.builder",
            "-i", str(source_dir),
            "-o", str(output_dir),
            "-p", "int4",
            "-e", "cpu",
            "-c", str(staging_dir / "_cache"),
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  STDOUT:\n{result.stdout[-2000:]}")
            print(f"  STDERR:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"SmolLM3-3B export failed (exit code {result.returncode})")

        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                shutil.copy2(f, staging_dir / f.name)
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"    {f.name} ({size_mb:.1f} MB)")

        ensure_chat_template(source_dir, staging_dir)

        for temp in (source_dir, output_dir, staging_dir / "_cache"):
            shutil.rmtree(temp, ignore_errors=True)

        missing = [
            name for name in (
                "genai_config.json",
                "model.onnx",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
            if not (staging_dir / name).exists()
        ]
        if missing:
            raise RuntimeError(f"SmolLM3-3B export is missing {missing}")
        print("  Staging complete")

    def render_card(self) -> str:
        return f"""\
---
library_name: onnx
tags:
  - text-generation
  - smollm3
  - onnx
  - onnxruntime-genai
  - int4
  - cpu
  - inference4j
license: apache-2.0
pipeline_tag: text-generation
base_model: {self.source_repo}
---

# SmolLM3-3B — ONNX (INT4, onnxruntime-genai)

ONNX export of [{self.source_repo}](https://huggingface.co/{self.source_repo})
in [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) format for
efficient autoregressive text generation on CPU (INT4 RTN block-32 quantization).

Produced for use with [inference4j](https://github.com/inference4j/inference4j),
an inference-only AI library for Java.

## Usage with inference4j

```java
try (var generator = TextGenerator.builder()
        .model(ModelSources.smolLM3_3B())
        .build()) {{
    GenerationResult result = generator.generate("Explain quantum computing in simple terms.");
    System.out.println(result.text());
}}
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | SmolLM3 (3B parameters) |
| Task | Text generation / chat |
| Precision | INT4 |
| ONNX format | onnxruntime-genai |
| License | Apache-2.0 |
"""


register(SmolLM3_3B())
