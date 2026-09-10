# Runbook: responding to `exporter check`

What to do when the drift check reports something. Every command here is safe to run
locally; nothing publishes until the step that says it does.

```bash
python -m exporter check          # all models, ~1s, no downloads
```

Exit codes: `0` all clear · `1` drift found · `2` error.

Jump to your status:

| Status | Meaning | Go to |
|---|---|---|
| `ok` | tracked upstream files match the lock | nothing to do |
| `drift` | an upstream file we consume changed | [A](#a-drift) |
| `upstream-moved` | repo moved, our files untouched | [B](#b-upstream-moved) |
| `unpublished` | defined here, absent from the org | [C](#c-unpublished) |
| `published-mismatch` | the org holds something we didn't produce | [D](#d-published-mismatch) |
| `unlocked` | no baseline recorded | [E](#e-unlocked) |
| `unverifiable` | metadata can't settle it | [F](#f-unverifiable) |
| `error` | upstream gone, renamed, gated | [G](#g-error) |

A newer *generation* upstream (depth-anything v2 → v3) is not drift and never shows up
in `check`. That is [Path S](#path-s-a-successor-generation).

---

## A. `drift`

An upstream file the export consumes has changed. Work through these in order — each
step is cheaper than the next, so a failure early saves the later work.

### 1. See what actually changed

```bash
python -m exporter check --json <model> | jq '.[0].changed_files'
```

Weights changing is routine. A `config.json` or `tokenizer_config.json` change is the
one to read carefully — those alter behaviour without altering the graph, and no
downstream test will notice a changed chat template or `model_max_length`.

### 2. Stage both builds — nothing is published

```bash
SCRATCH=/tmp/promote
python -m exporter run <model> --stage-to $SCRATCH/current            # the pinned build
python -m exporter run <model> --stage-to $SCRATCH/candidate --update # upstream HEAD
```

`--update` resolves upstream HEAD; `--stage-to` writes an inference4j cache layout
(`<dir>/inference4j/<model>/…`) and never uploads.

### 3. Diff the ONNX contract

```bash
python -m exporter contract $SCRATCH/current/inference4j/<model> \
                  --against $SCRATCH/candidate/inference4j/<model>
```

- **"Contract unchanged"** → continue to step 4.
- **Any `BREAKING:` line** (renamed/removed output, rank, dtype, fixed-dim change) →
  stop. The Java wrapper will fail at runtime. Treat it as
  [Path S](#path-s-a-successor-generation): new repo id, not an in-place replacement.

Exits `1` when breaking, so it can gate a script.

### 4. Run the real Java tests against the candidate

```bash
cd ~/projects/java/inference4j
INFERENCE4J_CACHE_DIR=$SCRATCH/candidate ./gradlew :inference4j-core:modelTest --tests '*<Wrapper>*'
```

`HuggingFaceModelSource` reads `INFERENCE4J_CACHE_DIR` and only downloads what is
missing, so a fully staged directory means it exercises your candidate and never
touches the network. Drop `--tests` to run all 24 model tests.

> If the model has external weights (a `model.onnx_data` sidecar), check that the
> wrapper's `requiredFiles` lists **every** file. `resolve()` downloads exactly the
> names it is given, so a missing sidecar yields a graph with no weights.

### 5. Publish

```bash
python -m exporter run <model> --update
```

This uploads and re-pins `exports.lock.json` to the new upstream revision.

### 6. Commit the lockfile

```bash
git add exports.lock.json && git commit -m "Update <model> to upstream <sha>"
```

The lockfile is the record of what is live. A publish without the commit means the next
`check` compares against a stale baseline.

> **Note on blast radius.** `HuggingFaceModelSource` hardcodes `resolve/main/` and caches
> on file presence with no hash check. Publishing reaches every consumer with a cold
> cache immediately, while warm caches keep the old weights indefinitely. There is no
> rollback other than re-uploading the previous bytes, which is why steps 3 and 4 are not
> optional.

---

## B. `upstream-moved`

The upstream repo has new commits, but nothing we consume changed — almost always a
README or metadata edit. No re-export needed.

```bash
python -m exporter check --refresh-lock <model>
git add exports.lock.json && git commit -m "Refresh <model> lock to upstream <sha>"
```

---

## C. `unpublished`

Defined in `exporter/models/` but not in the org. Either it was never run, or it is
deliberately withheld (licensing, WIP). Confirm the licence permits redistribution, then:

```bash
python -m exporter run <model> --dry-run   # inspect staged files and the card
python -m exporter run <model>             # publish
git add exports.lock.json && git commit -m "Publish <model>"
```

---

## D. `published-mismatch`

The org copy differs from what the lockfile says we published — an upload from another
machine, a manual edit through the HF web UI, or an export that never completed.

Do not re-publish blindly. Compare first:

```bash
python -m exporter contract <model>        # what the org currently serves
python -m exporter check --json <model>
```

If the org copy is correct, re-bootstrap the entry. If not, re-run the export.

```bash
python -m exporter lock --bootstrap <model>   # trust the org, rebuild the baseline
python -m exporter run <model>                # trust the code, overwrite the org
```

---

## E. `unlocked`

No baseline. Normal for a model published before the lockfile existed, or one just added.

```bash
python -m exporter lock --bootstrap <model>
```

Mirrors resolve to `verified` (their bytes are provably upstream's). Torch exports
resolve to `inferred` — the upstream commit that was HEAD on our upload date, an
assumption rather than a proof.

---

## F. `unverifiable`

Metadata could not settle the comparison. Two causes:

- **Mixed LFS/plain storage** on a large file. Resolve by hashing real bytes:
  `python -m exporter check --deep <model>`
- **An unversioned source.** `craft-mlt-25k` pulls from Google Drive, which exposes no
  revision at all — its sha256 *is* its version. `--deep` compares against the cached
  `.pth` when present.

---

## G. `error`

Upstream is unreachable, renamed, or newly gated. Confirm by hand:

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('<upstream-repo>'))"
```

A rename means updating `source_repo` in the model definition and re-bootstrapping. A
deletion means the model can no longer be reproduced — what is published is all there
is, and that is worth recording in the model card.

---

## Path S: a successor generation

`python -m exporter successors` reports when an upstream author ships a **new repo**
rather than updating the old one. This never appears in `check`, because the repo you
pinned genuinely has not changed.

**Publish alongside; never replace.** The precedent is yolo: `inference4j/yolov8n` and
`inference4j/yolo26n` both exist, served by separate `YoloV8Detector` and
`Yolo26Detector` wrappers. Users choose. Nobody's pipeline breaks on an upgrade they
didn't ask for.

### 1. Confirm it is a real successor

```bash
python -m exporter successors <model>
```

Candidates are name-matched, not verified. Check that the variant matches (`small` vs
`base`), and read the upstream card — a version bump can change the task entirely.

### 2. Add a new model definition

A new file in `exporter/models/`, with a **new `name` and new `repo_id`**. Leave the
existing definition untouched.

```python
class DepthAnythingV3Small(MirroredModel):
    name = "depth-anything-v3-small"                 # not v2
    repo_id = "inference4j/depth-anything-v3-small"  # new repo
    source_repo = "onnx-community/depth-anything-v3-small"
```

Register it in `exporter/models/__init__.py`.

### 3. Diff the contract against the current generation

```bash
python -m exporter run depth-anything-v3-small --stage-to /tmp/v3
python -m exporter contract depth-anything-v2-small --against /tmp/v3/inference4j/depth-anything-v3-small
```

This decides how much Java work follows:

- **Contract identical** → the existing wrapper can serve both. Publish, and users select
  with `.modelId("inference4j/…-v3-small")`. No new Java class.
- **Contract differs** → a new wrapper class is needed, the `Yolo26Detector` pattern.

Depth Anything v3 is the second case, and a good illustration of why this check comes
first:

```
BREAKING: input 'pixel_values' rank 4 -> 5   (v3 is multi-view: adds num_images)
BREAKING: output 'predicted_depth' rank 3 -> 4
new outputs 'confidence', 'extrinsics', 'intrinsics'
```

Plus it ships external weights (`model.onnx_data`), so `requiredFiles` needs both names.

### 4. Publish the new repo

```bash
python -m exporter run depth-anything-v3-small
git add exports.lock.json exporter/models/ && git commit -m "Add depth-anything-v3-small"
```

### 5. Java side

Add the wrapper (or extend the existing one), give it its own `DEFAULT_MODEL_ID`, add a
`modelTest`, and document both generations. **Leave the old wrapper's `DEFAULT_MODEL_ID`
pointing at the old model** — switching the default is a separate, deliberate decision,
and a breaking change for anyone relying on current behaviour.

---

## Quick reference

```bash
python -m exporter check                       # what needs attention
python -m exporter check --json                # for scripts
python -m exporter check --deep <model>        # settle by hashing bytes
python -m exporter check --refresh-lock        # accept upstream README churn
python -m exporter successors                  # newer generations upstream
python -m exporter contract <a> --against <b>  # will this break the wrapper?
python -m exporter run <m> --stage-to DIR      # build locally, publish nothing
python -m exporter run <m> --update            # publish + re-pin the lock
python -m exporter lock                        # show the baseline
python -m exporter lock --bootstrap <m>        # rebuild a baseline
```
