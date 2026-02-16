# onnx-exporter

Export and mirror ONNX models for the [inference4j](https://github.com/inference4j/inference4j) HuggingFace organization.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                # Core deps (mirrored models only)
pip install -e ".[export]"      # + torch/transformers (for CLIP, CRAFT export)
```

## Usage

```bash
python -m exporter list                           # Show all registered models
python -m exporter run silero-vad --dry-run       # Stage + preview, no upload
python -m exporter run clip craft                 # Export + upload specific models
python -m exporter run --all                      # All models (explicit flag required)
python -m exporter card clip                      # Preview model card only
```

## Model types

- **Mirrored** — downloads existing ONNX from upstream HuggingFace/URL, adds model card, re-uploads
- **Exported** — custom PyTorch-to-ONNX conversion with model-specific code (CLIP, CRAFT)

## Authentication

Set `HF_TOKEN` environment variable or run `huggingface-cli login`.
