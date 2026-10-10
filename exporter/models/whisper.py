import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from exporter.base import ExportedModel
from exporter.registry import register

# onnxruntime-genai's model builder needs transformers 5.x, while optimum-onnx (used by the other
# exports) pins transformers < 4.58. The builder therefore runs in its own pinned venv, created on
# first use next to the project's .venv.
_BUILDER_VENV = Path(__file__).resolve().parents[2] / ".venv-genai-builder"
_BUILDER_REQUIREMENTS = (
    "onnxruntime-genai==0.17.0",
    "onnx-ir==0.1.16",
    "onnx==1.23.2",
    "transformers==5.19.0",
    "torch==2.14.1",
)

# Files the genai model builder reads from an upstream Whisper repo. Downloading only these
# skips the duplicate PyTorch/TF/Flax weights the openai repos also carry.
_SOURCE_PATTERNS = (
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "normalizer.json",
    "preprocessor_config.json",
)


class WhisperGenai(ExportedModel):
    """Whisper exported to onnxruntime-genai format with genai's own model builder.

    The builder writes the encoder, the decoder with its self- and cross-attention KV cache,
    genai_config.json, audio_processor_config.json and the tokenizer files, which is
    everything `WhisperSpeechModel` loads.
    """

    tracked_patterns = _SOURCE_PATTERNS

    size: str
    parameters: str

    def stage(self, staging_dir: Path) -> None:
        source_dir = staging_dir / "_source"
        output_dir = staging_dir / "_export"

        print(f"  Downloading {self.source_repo}@{(self.revision or 'HEAD')[:8]} ...")
        snapshot_download(
            repo_id=self.source_repo,
            revision=self.revision,
            allow_patterns=list(_SOURCE_PATTERNS),
            local_dir=source_dir,
        )

        cmd = [
            str(_builder_python()), "-m", "onnxruntime_genai.models.builder",
            "-i", str(source_dir),
            "-o", str(output_dir),
            "-p", "fp32",
            "-e", "cpu",
            "-c", str(staging_dir / "_cache"),
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  STDOUT:\n{result.stdout[-2000:]}")
            print(f"  STDERR:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"Whisper export failed (exit code {result.returncode})")

        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                shutil.copy2(f, staging_dir / f.name)
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"    {f.name} ({size_mb:.1f} MB)")

        for temp in (source_dir, output_dir, staging_dir / "_cache"):
            shutil.rmtree(temp, ignore_errors=True)

        missing = [name for name in ("genai_config.json", "encoder.onnx", "decoder.onnx",
                                     "audio_processor_config.json", "tokenizer.json")
                   if not (staging_dir / name).exists()]
        if missing:
            raise RuntimeError(f"Whisper export is missing {missing}")
        print("  Staging complete")

    def render_card(self) -> str:
        return f"""\
---
library_name: onnx
tags:
  - whisper
  - speech-to-text
  - automatic-speech-recognition
  - translation
  - onnx
  - onnxruntime-genai
  - inference4j
license: mit
pipeline_tag: automatic-speech-recognition
base_model: {self.source_repo}
---

# Whisper {self.size.capitalize()} — ONNX (onnxruntime-genai)

ONNX export of [{self.source_repo}](https://huggingface.co/{self.source_repo})
in [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) format for
autoregressive speech-to-text and translation. Produced with onnxruntime-genai's model
builder (FP32, CPU).

Converted for use with [inference4j](https://github.com/inference4j/inference4j),
an inference-only AI library for Java.

## Usage with inference4j

### Transcription

```java
try (var whisper = WhisperSpeechModel.builder()
        .modelId("{self.repo_id}")
        .build()) {{
    Transcription result = whisper.transcribe(Path.of("audio.wav"));
    System.out.println(result.text());
}}
```

### Translation (any language to English)

```java
try (var whisper = WhisperSpeechModel.builder()
        .modelId("{self.repo_id}")
        .language("fr")
        .task(WhisperTask.TRANSLATE)
        .build()) {{
    Transcription result = whisper.transcribe(Path.of("french-audio.wav"));
    System.out.println(result.text());  // English text
}}
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Whisper {self.size} (encoder-decoder, {self.parameters} parameters) |
| Task | Speech-to-text / translation |
| Languages | 99 languages |
| Audio input | WAV, resampled to 16 kHz mono |
| Max audio chunk | 30 seconds (inference4j splits longer audio) |
| Precision | FP32 |
| ONNX format | onnxruntime-genai |

## Original Paper

> Radford, A., Kim, J. W., Xu, T., et al. (2022).
> Robust Speech Recognition via Large-Scale Weak Supervision.
> [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)

## License

The original Whisper model is released under the MIT License by OpenAI.
"""


def _builder_python() -> Path:
    """Return the builder venv's interpreter, creating or refreshing the venv when its pins change."""
    python = _BUILDER_VENV / "bin" / "python"
    stamp = _BUILDER_VENV / "requirements.txt"
    wanted = "\n".join(_BUILDER_REQUIREMENTS) + "\n"
    if python.exists() and stamp.exists() and stamp.read_text() == wanted:
        return python
    print(f"  Creating model builder venv at {_BUILDER_VENV} ...")
    subprocess.run([sys.executable, "-m", "venv", "--clear", str(_BUILDER_VENV)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-q", *_BUILDER_REQUIREMENTS], check=True)
    stamp.write_text(wanted)
    return python


class WhisperTiny(WhisperGenai):
    name = "whisper-tiny"
    repo_id = "inference4j/whisper-tiny-genai"
    source_repo = "openai/whisper-tiny"
    size = "tiny"
    parameters = "39M"


class WhisperSmall(WhisperGenai):
    name = "whisper-small"
    repo_id = "inference4j/whisper-small-genai"
    source_repo = "openai/whisper-small"
    size = "small"
    parameters = "244M"


register(WhisperTiny())
register(WhisperSmall())
