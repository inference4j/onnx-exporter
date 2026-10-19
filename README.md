# onnx-exporter

Export and mirror ONNX models for the [inference4j](https://github.com/inference4j/inference4j) HuggingFace organization.

## Models

| Name | Source | Repo |
|---|---|---|
| `all-MiniLM-L6-v2` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | [inference4j/all-MiniLM-L6-v2](https://huggingface.co/inference4j/all-MiniLM-L6-v2) |
| `all-MiniLM-L12-v2` | [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | [inference4j/all-MiniLM-L12-v2](https://huggingface.co/inference4j/all-MiniLM-L12-v2) |
| `all-mpnet-base-v2` | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | [inference4j/all-mpnet-base-v2](https://huggingface.co/inference4j/all-mpnet-base-v2) |
| `resnet50-v1-7` | [onnxmodelzoo/resnet50-v1-7](https://huggingface.co/onnxmodelzoo/resnet50-v1-7) | [inference4j/resnet50-v1-7](https://huggingface.co/inference4j/resnet50-v1-7) |
| `efficientnet-lite4` | [onnx/EfficientNet-Lite4](https://huggingface.co/onnx/EfficientNet-Lite4) | [inference4j/efficientnet-lite4](https://huggingface.co/inference4j/efficientnet-lite4) |
| `yolov8n` | [Kalray/yolov8](https://huggingface.co/Kalray/yolov8) | [inference4j/yolov8n](https://huggingface.co/inference4j/yolov8n) |
| `yolo26n` | [onnx-community/yolo26n-ONNX](https://huggingface.co/onnx-community/yolo26n-ONNX) | [inference4j/yolo26n](https://huggingface.co/inference4j/yolo26n) |
| `distilbert-base-uncased-finetuned-sst-2-english` | [Xenova/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english) | [inference4j/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/inference4j/distilbert-base-uncased-finetuned-sst-2-english) |
| `ms-marco-MiniLM-L-6-v2` | [Xenova/ms-marco-MiniLM-L-6-v2](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2) | [inference4j/ms-marco-MiniLM-L-6-v2](https://huggingface.co/inference4j/ms-marco-MiniLM-L-6-v2) |
| `wav2vec2-base-960h` | [Xenova/wav2vec2-base-960h](https://huggingface.co/Xenova/wav2vec2-base-960h) | [inference4j/wav2vec2-base-960h](https://huggingface.co/inference4j/wav2vec2-base-960h) |
| `silero-vad` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx) | [inference4j/silero-vad](https://huggingface.co/inference4j/silero-vad) |
| `clip-vit-base-patch32` | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | [inference4j/clip-vit-base-patch32](https://huggingface.co/inference4j/clip-vit-base-patch32) |
| `craft-mlt-25k` | [clovaai/CRAFT](https://huggingface.co/clovaai/CRAFT) | [inference4j/craft-mlt-25k](https://huggingface.co/inference4j/craft-mlt-25k) |
| `phi-3-mini` | [microsoft/Phi-3-mini-4k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) | [inference4j/phi-3-mini-4k-instruct](https://huggingface.co/inference4j/phi-3-mini-4k-instruct) |
| `deepseek-r1-distill-qwen-1.5b` | [onnxruntime/DeepSeek-R1-Distill-ONNX](https://huggingface.co/onnxruntime/DeepSeek-R1-Distill-ONNX) | [inference4j/deepseek-r1-distill-qwen-1.5b](https://huggingface.co/inference4j/deepseek-r1-distill-qwen-1.5b) |
| `whisper-tiny` | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | [inference4j/whisper-tiny-genai](https://huggingface.co/inference4j/whisper-tiny-genai) |
| `whisper-small` | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | [inference4j/whisper-small-genai](https://huggingface.co/inference4j/whisper-small-genai) |
| `phi-3.5-vision` | [microsoft/Phi-3.5-vision-instruct-onnx](https://huggingface.co/microsoft/Phi-3.5-vision-instruct-onnx) | [inference4j/phi-3.5-vision-instruct](https://huggingface.co/inference4j/phi-3.5-vision-instruct) |
| `gpt2` | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | [inference4j/gpt2](https://huggingface.co/inference4j/gpt2) |
| `smollm2-360m-instruct` | [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | [inference4j/smollm2-360m-instruct](https://huggingface.co/inference4j/smollm2-360m-instruct) |
| `smollm2-1.7b-instruct` | [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | [inference4j/smollm2-1.7b-instruct](https://huggingface.co/inference4j/smollm2-1.7b-instruct) |
| `qwen2.5-1.5b-instruct` | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | [inference4j/qwen2.5-1.5b-instruct](https://huggingface.co/inference4j/qwen2.5-1.5b-instruct) |
| `gemma-2-2b-it` | [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | [inference4j/gemma-2-2b-it](https://huggingface.co/inference4j/gemma-2-2b-it) |
| `tinyllama-1.1b-chat` | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | [inference4j/tinyllama-1.1b-chat](https://huggingface.co/inference4j/tinyllama-1.1b-chat) |
| `flan-t5-small` | [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) | [inference4j/flan-t5-small](https://huggingface.co/inference4j/flan-t5-small) |
| `flan-t5-base` | [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) | [inference4j/flan-t5-base](https://huggingface.co/inference4j/flan-t5-base) |
| `flan-t5-large` | [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) | [inference4j/flan-t5-large](https://huggingface.co/inference4j/flan-t5-large) |
| `distilbart-cnn-12-6` | [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) | [inference4j/distilbart-cnn-12-6](https://huggingface.co/inference4j/distilbart-cnn-12-6) |
| `bart-large-cnn` | [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) | [inference4j/bart-large-cnn](https://huggingface.co/inference4j/bart-large-cnn) |
| `opus-mt-en-fr` | [Helsinki-NLP/opus-mt-en-fr](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) | [inference4j/opus-mt-en-fr](https://huggingface.co/inference4j/opus-mt-en-fr) |
| `opus-mt-en-es` | [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) | [inference4j/opus-mt-en-es](https://huggingface.co/inference4j/opus-mt-en-es) |
| `opus-mt-en-de` | [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) | [inference4j/opus-mt-en-de](https://huggingface.co/inference4j/opus-mt-en-de) |
| `coedit-base` | [jbochi/coedit-base](https://huggingface.co/jbochi/coedit-base) | [inference4j/coedit-base](https://huggingface.co/inference4j/coedit-base) |
| `coedit-large` | [grammarly/coedit-large](https://huggingface.co/grammarly/coedit-large) | [inference4j/coedit-large](https://huggingface.co/inference4j/coedit-large) |
| `t5-small-awesome-text-to-sql` | [cssupport/t5-small-awesome-text-to-sql](https://huggingface.co/cssupport/t5-small-awesome-text-to-sql) | [inference4j/t5-small-awesome-text-to-sql](https://huggingface.co/inference4j/t5-small-awesome-text-to-sql) |
| `T5-LM-Large-text2sql-spider` | [gaussalgo/T5-LM-Large-text2sql-spider](https://huggingface.co/gaussalgo/T5-LM-Large-text2sql-spider) | [inference4j/T5-LM-Large-text2sql-spider](https://huggingface.co/inference4j/T5-LM-Large-text2sql-spider) |
| `distilbert-NER` | [onnx-community/distilbert-NER-ONNX](https://huggingface.co/onnx-community/distilbert-NER-ONNX) | [inference4j/distilbert-NER](https://huggingface.co/inference4j/distilbert-NER) |
| `bert-base-NER` | [Xenova/bert-base-NER](https://huggingface.co/Xenova/bert-base-NER) | [inference4j/bert-base-NER](https://huggingface.co/inference4j/bert-base-NER) |
| `bge-base-en-v1.5` | [Xenova/bge-base-en-v1.5](https://huggingface.co/Xenova/bge-base-en-v1.5) | [inference4j/bge-base-en-v1.5](https://huggingface.co/inference4j/bge-base-en-v1.5) |
| `gte-base` | [Xenova/gte-base](https://huggingface.co/Xenova/gte-base) | [inference4j/gte-base](https://huggingface.co/inference4j/gte-base) |
| `depth-anything-v2-small` | [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small) | [inference4j/depth-anything-v2-small](https://huggingface.co/inference4j/depth-anything-v2-small) |
| `depth-anything-v2-base` | [onnx-community/depth-anything-v2-base](https://huggingface.co/onnx-community/depth-anything-v2-base) | [inference4j/depth-anything-v2-base](https://huggingface.co/inference4j/depth-anything-v2-base) |
| `punctuation-fullstop-truecase-english` | [1-800-BAD-CODE/punctuation_fullstop_truecase_english](https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english) | [inference4j/punctuation-fullstop-truecase-english](https://huggingface.co/inference4j/punctuation-fullstop-truecase-english) |
| `phi-4-mini-instruct` | [microsoft/Phi-4-mini-instruct-onnx](https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx) | [inference4j/phi-4-mini-instruct](https://huggingface.co/inference4j/phi-4-mini-instruct) |
| `smollm3-3b` | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | [inference4j/smollm3-3b](https://huggingface.co/inference4j/smollm3-3b) |
| `qwen3-0.6b` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | [inference4j/qwen3-0.6b](https://huggingface.co/inference4j/qwen3-0.6b) |
| `moonshine-streaming-tiny-en` | [moonshine-ai/moonshine-voice-assets](https://huggingface.co/moonshine-ai/moonshine-voice-assets) | [inference4j/moonshine-streaming-tiny-en](https://huggingface.co/inference4j/moonshine-streaming-tiny-en) |
| `moonshine-streaming-small-en` | [moonshine-ai/moonshine-voice-assets](https://huggingface.co/moonshine-ai/moonshine-voice-assets) | [inference4j/moonshine-streaming-small-en](https://huggingface.co/inference4j/moonshine-streaming-small-en) |
| `moonshine-streaming-medium-en` | [moonshine-ai/moonshine-voice-assets](https://huggingface.co/moonshine-ai/moonshine-voice-assets) | [inference4j/moonshine-streaming-medium-en](https://huggingface.co/inference4j/moonshine-streaming-medium-en) |
| `deberta-v3-base-prompt-injection-v2` | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | [inference4j/deberta-v3-base-prompt-injection-v2](https://huggingface.co/inference4j/deberta-v3-base-prompt-injection-v2) |
| `pp-ocrv6-small-det` | [PaddlePaddle/PP-OCRv6_small_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx) | [inference4j/pp-ocrv6-small-det](https://huggingface.co/inference4j/pp-ocrv6-small-det) |
| `pp-ocrv6-small-rec` | [PaddlePaddle/PP-OCRv6_small_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx) | [inference4j/pp-ocrv6-small-rec](https://huggingface.co/inference4j/pp-ocrv6-small-rec) |
| `pp-ocrv6-tiny-det` | [PaddlePaddle/PP-OCRv6_tiny_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx) | [inference4j/pp-ocrv6-tiny-det](https://huggingface.co/inference4j/pp-ocrv6-tiny-det) |
| `pp-ocrv6-tiny-rec` | [PaddlePaddle/PP-OCRv6_tiny_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx) | [inference4j/pp-ocrv6-tiny-rec](https://huggingface.co/inference4j/pp-ocrv6-tiny-rec) |
| `deberta-v3-base-zeroshot-v2.0-c` | [MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c) | [inference4j/deberta-v3-base-zeroshot-v2.0-c](https://huggingface.co/inference4j/deberta-v3-base-zeroshot-v2.0-c) |
| `deberta-v3-large-zeroshot-v2.0-c` | [MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c) | [inference4j/deberta-v3-large-zeroshot-v2.0-c](https://huggingface.co/inference4j/deberta-v3-large-zeroshot-v2.0-c) |
| `gliner-small-v2.1` | [onnx-community/gliner_small-v2.1](https://huggingface.co/onnx-community/gliner_small-v2.1) | [inference4j/gliner-small-v2.1](https://huggingface.co/inference4j/gliner-small-v2.1) |
| `gliner-pii-base-v1.0` | [knowledgator/gliner-pii-base-v1.0](https://huggingface.co/knowledgator/gliner-pii-base-v1.0) | [inference4j/gliner-pii-base-v1.0](https://huggingface.co/inference4j/gliner-pii-base-v1.0) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                # Core deps (mirrored models only)
pip install -e ".[export]"      # + torch/transformers (for CLIP, CRAFT export)
```

Publishing an unchanged Hugging Face mirror copies its files inside the Hub instead of
downloading and re-uploading the model. On first use, the exporter creates the small
`.venv-hub-uploader` helper environment required for that operation. Keeping it separate
preserves compatibility with the Transformers 4.x ONNX export toolchain.

## Usage

```bash
python -m exporter list                           # Show all registered models
python -m exporter check                          # Which models have drifted upstream?
python -m exporter run silero-vad --dry-run       # Stage + preview, no upload
python -m exporter run clip craft                 # Export + upload specific models
python -m exporter run --all                      # All models (explicit flag required)
python -m exporter card clip                      # Preview model card only
```

## Model types

- **Mirrored** — copies unchanged Hugging Face artifacts server-side when possible; URL mirrors and models with custom staging still download and upload locally
- **Exported** — custom PyTorch-to-ONNX conversion with model-specific code (CLIP, CRAFT)

## Drift detection

`exports.lock.json` records which upstream revision each published model came from, so
exports are reproducible and staleness is detectable.

```bash
python -m exporter check                  # all models; no downloads, ~1s
python -m exporter check --json           # machine-readable, for scheduled jobs
python -m exporter check --refresh-lock   # accept upstream churn that didn't touch our files
python -m exporter check --deep           # download files when metadata can't decide
python -m exporter lock                   # show the current baseline
python -m exporter lock --bootstrap       # rebuild baselines for published models
```

Statuses: `ok`, `drift` (a tracked upstream file changed — re-export), `upstream-moved`
(the repo moved but nothing we consume did), `unpublished`, `unlocked`,
`published-mismatch` (the org holds something this repo didn't produce), `unverifiable`,
`error`. Exit codes: `0` clear, `1` drift, `2` error.

Only files an export actually consumes are compared, so an upstream README edit doesn't
raise an alarm. Mirrored models track their declared `FileMapping` sources; exported
models track weights, config and tokenizer files.

### Pinning

`run` builds from the locked revision, so re-running a model reproduces the same bytes.
To move to new upstream weights:

```bash
python -m exporter run gpt2 --update      # export from upstream HEAD, re-pin the lock
```

Models with no lock entry track upstream HEAD, as before.

## Successor detection

`check` compares the repo you pinned against itself, so it stays `ok` forever when an
author ships a *new* repo instead of updating the old one. `successors` covers that gap.

```bash
python -m exporter successors               # all models
python -m exporter successors --json        # machine-readable
```

It parses the generation out of the upstream repo name (`depth-anything-v2-small` → family
`depth-anything`, v2), searches the same author for higher generations, and ranks
candidates by variant match and adoption. These are name-matched **candidates**, not
verified drop-ins — always diff the contract before switching.

## Promotion: testing a new version before switching

`HuggingFaceModelSource` hardcodes `resolve/main/` and caches on file presence alone, so
re-uploading to an existing repo id reaches every consumer with a cold cache immediately,
with no version gate and no rollback. Never publish an update without staging it first:

```bash
python -m exporter run <model> --stage-to /tmp/candidate --update       # no upload
python -m exporter contract <model> --against /tmp/candidate/inference4j/<model>
cd ~/projects/java/inference4j
INFERENCE4J_CACHE_DIR=/tmp/candidate ./gradlew :inference4j-core:modelTest
python -m exporter run <model> --update                                 # publish + re-pin
```

**[RUNBOOK.md](RUNBOOK.md) has the full procedure** for every `check` status, plus the
path for adopting a new upstream generation alongside the current one.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

## Authentication

Set `HF_TOKEN` environment variable or run `huggingface-cli login`.
