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
python -m exporter check                          # Which models have drifted upstream?
python -m exporter run silero-vad --dry-run       # Stage + preview, no upload
python -m exporter run clip craft                 # Export + upload specific models
python -m exporter run --all                      # All models (explicit flag required)
python -m exporter card clip                      # Preview model card only
```

## Model types

- **Mirrored** — downloads existing ONNX from upstream HuggingFace/URL, adds model card, re-uploads
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

Models with no lock entry track upstream HEAD, as before. `whisper-small` cannot pin
(its converter takes a bare model id) and always builds from HEAD.

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
