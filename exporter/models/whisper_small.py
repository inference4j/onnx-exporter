import json
import shutil
import subprocess
import sys
from pathlib import Path

from exporter.base import ExportedModel
from exporter.registry import register


HF_MODEL_ID = "openai/whisper-small"


class WhisperSmall(ExportedModel):
    name = "whisper-small"
    repo_id = "inference4j/whisper-small-genai"

    def stage(self, staging_dir: Path) -> None:
        """Export Whisper Small to genai-compatible ONNX format.

        Uses onnxruntime's Whisper-specific converter with flags required
        for onnxruntime-genai compatibility (no beam search op, cross QK output).
        """
        print(f"  Exporting {HF_MODEL_ID} to genai format...")

        # Run the onnxruntime whisper converter
        export_dir = staging_dir / "_export"
        export_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, "-m",
            "onnxruntime.transformers.models.whisper.convert_to_onnx",
            "-m", HF_MODEL_ID,
            "--output", str(export_dir),
            "--use_external_data_format",
            "--precision", "fp32",
            "--optimize_onnx",
            "--no_beam_search_op",
            "--output_cross_qk",
        ]

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  STDOUT:\n{result.stdout}")
            print(f"  STDERR:\n{result.stderr}")
            raise RuntimeError(
                f"Whisper export failed (exit code {result.returncode}): "
                f"{result.stderr[:500]}"
            )

        print(f"  Export completed, staging files...")

        # The converter outputs files into a subdirectory structure.
        # Find and copy all relevant files to the staging root.
        _stage_exported_files(export_dir, staging_dir)

        # Clean up export temp dir
        shutil.rmtree(export_dir, ignore_errors=True)

        # List staged files
        for f in sorted(staging_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"    {f.name} ({size_mb:.1f} MB)")

        print("  Staging complete")

    def render_card(self) -> str:
        return """\
---
library_name: onnx
tags:
  - whisper
  - speech-to-text
  - automatic-speech-recognition
  - translation
  - onnx
  - inference4j
license: mit
pipeline_tag: automatic-speech-recognition
datasets:
  - librispeech_asr
---

# Whisper Small — ONNX (onnxruntime-genai)

ONNX export of [openai/whisper-small](https://huggingface.co/openai/whisper-small)
in onnxruntime-genai format for autoregressive speech-to-text and translation.

Converted for use with [inference4j](https://github.com/inference4j/inference4j),
an inference-only AI library for Java.

## Usage with inference4j

### Transcription

```java
try (var whisper = WhisperSpeechModel.builder()
        .modelId("inference4j/whisper-small-genai")
        .build()) {
    Transcription result = whisper.transcribe(Path.of("audio.wav"));
    System.out.println(result.text());
}
```

### Translation (any language to English)

```java
try (var whisper = WhisperSpeechModel.builder()
        .modelId("inference4j/whisper-small-genai")
        .language("fr")
        .task(WhisperTask.TRANSLATE)
        .build()) {
    Transcription result = whisper.transcribe(Path.of("french-audio.wav"));
    System.out.println(result.text());  // English text
}
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Whisper (encoder-decoder, 244M parameters) |
| Task | Speech-to-text / translation |
| Languages | 99 languages |
| Audio input | 16kHz mono WAV (auto-resampled) |
| Max audio chunk | 30 seconds (auto-chunked) |
| ONNX format | onnxruntime-genai compatible |

## Original Paper

> Radford, A., Kim, J. W., Xu, T., et al. (2022).
> Robust Speech Recognition via Large-Scale Weak Supervision.
> [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)

## License

The original Whisper model is released under the MIT License by OpenAI.
"""


def _stage_exported_files(export_dir: Path, staging_dir: Path) -> None:
    """Find and copy exported files to the staging directory.

    The converter may output files in subdirectories depending on
    the version. This searches for the key files and flattens them.
    """
    # Key files we expect from the whisper converter
    key_patterns = [
        "*.onnx",
        "*.onnx.data",
        "genai_config.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
    ]

    copied = set()
    for pattern in key_patterns:
        for f in export_dir.rglob(pattern):
            if f.is_file() and f.name not in copied:
                dst = staging_dir / f.name
                shutil.copy2(f, dst)
                copied.add(f.name)


register(WhisperSmall())
