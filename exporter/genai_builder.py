"""Shared helpers for exports produced by onnxruntime-genai's model builder."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

# onnxruntime-genai's model builder needs transformers 5.x, while optimum-onnx (used by the other
# exports) pins transformers < 4.58. The builder therefore runs in its own pinned venv, created on
# first use next to the project's .venv.
BUILDER_VENV = Path(__file__).resolve().parents[1] / ".venv-genai-builder"
BUILDER_REQUIREMENTS = (
    "onnxruntime-genai==0.17.0",
    "onnx-ir==0.1.16",
    "onnx==1.23.2",
    "transformers==5.19.0",
    "torch==2.14.1",
)

# Files the genai model builder reads from an upstream HuggingFace text model repo.
TEXT_GENAI_SOURCE_PATTERNS = (
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def get_builder_python() -> Path:
    """Return the builder venv's interpreter, creating or refreshing the venv when its pins change."""
    python = BUILDER_VENV / "bin" / "python"
    stamp = BUILDER_VENV / "requirements.txt"
    wanted = "\n".join(BUILDER_REQUIREMENTS) + "\n"
    if python.exists() and stamp.exists() and stamp.read_text() == wanted:
        return python
    print(f"  Creating model builder venv at {BUILDER_VENV} ...")
    subprocess.run([sys.executable, "-m", "venv", "--clear", str(BUILDER_VENV)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-q", *BUILDER_REQUIREMENTS], check=True)
    stamp.write_text(wanted)
    return python


def ensure_chat_template(source_dir: Path | None, staging_dir: Path) -> None:
    """Ensure chat_template.jinja exists in staging_dir either by copying or extracting from tokenizer_config.json."""
    target = staging_dir / "chat_template.jinja"
    if target.exists():
        return

    # Check if upstream had a chat_template.jinja file
    if source_dir:
        upstream_jinja = source_dir / "chat_template.jinja"
        if upstream_jinja.exists():
            shutil.copy2(upstream_jinja, target)
            print("  Copied chat_template.jinja from source repo")
            return

    # Otherwise extract from tokenizer_config.json
    cfg_file = staging_dir / "tokenizer_config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file, encoding="utf-8") as f:
                cfg = json.load(f)
            template = cfg.get("chat_template")
            if template:
                target.write_text(template, encoding="utf-8")
                print("  Extracted chat_template.jinja from tokenizer_config.json")
        except Exception as e:
            print(f"  Warning: could not extract chat_template from tokenizer_config.json: {e}")
