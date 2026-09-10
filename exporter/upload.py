"""Staging and upload workflow."""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from exporter import lock
from exporter.base import BaseModel, MirroredModel


def _resolve_revision(model: BaseModel, update: bool, api: HfApi) -> str | None:
    """Decide which upstream revision to build from.

    Locked revision by default, so a re-run reproduces the same bytes. `--update` moves
    to upstream HEAD; an unlocked model has no baseline yet and tracks HEAD as before.
    """
    if model.source_type == "url":
        # GitHub raw: the lock holds the commit the branch URL should resolve to.
        if update:
            from exporter.check import _parse_github_raw, github_path_commit

            parsed = _parse_github_raw(model.files[0].src)
            latest = github_path_commit(parsed[0], parsed[2]) if parsed else None
            return latest[0] if latest else None
        entry = lock.get(model.name)
        return entry.source.revision if entry else None

    if model.source_type != "hf":
        return None

    if not model.supports_pinning:
        print(f"  Note: {model.name} cannot pin a revision; building from upstream HEAD")
        return None

    if not update:
        entry = lock.get(model.name)
        if entry and entry.source.revision:
            return entry.source.revision
        return None

    try:
        return api.model_info(model.source_repo).sha
    except Exception as exc:
        print(f"  Warning: cannot resolve upstream HEAD ({type(exc).__name__}); using default")
        return None


def _record(model: BaseModel, revision: str | None, api: HfApi) -> None:
    """Write the provenance entry for an upload that just succeeded."""
    from exporter.check import file_map

    is_mirror = isinstance(model, MirroredModel)

    if model.source_type == "url":
        entry = lock.get(model.name)
        if entry and revision:
            entry.source.revision = revision
            entry.exported_at = lock.now()
            lock.put(model.name, entry)
            print(f"  Locked to {revision[:8]}")
        return
    if model.source_type != "hf":
        return

    try:
        info = api.model_info(model.source_repo, revision=revision, files_metadata=True)
        upstream = file_map(info)
        published = file_map(api.model_info(model.repo_id, files_metadata=True))
    except Exception as exc:
        print(f"  Warning: could not record provenance ({type(exc).__name__})")
        return

    tracked_names = model.tracked_files(list(upstream))
    lock.put(model.name, lock.LockEntry(
        repo_id=model.repo_id,
        source=lock.Source(type="hf", repo=model.source_repo, revision=info.sha),
        provenance=lock.VERIFIED if is_mirror else lock.INFERRED,
        tracked={n: upstream[n] for n in tracked_names if n in upstream},
        published={n: h for n, h in published.items() if not n.startswith(".")},
        exported_at=lock.now(),
    ))
    print(f"  Locked to {model.source_repo}@{info.sha[:8]}")


def stage_to(model: BaseModel, cache_dir: Path, update: bool = False) -> Path:
    """Stage a model into an inference4j cache layout, without uploading anything.

    HuggingFaceModelSource resolves `<cache>/<repoId>/<file>` and only downloads what is
    missing, so writing a candidate here lets inference4j's modelTest suite exercise it
    against the real Java wrappers before it is published anywhere:

        python -m exporter run <model> --stage-to /tmp/candidate
        INFERENCE4J_CACHE_DIR=/tmp/candidate ./gradlew :inference4j-core:modelTest
    """
    api = HfApi()
    model.revision = _resolve_revision(model, update, api)

    target = cache_dir / model.repo_id
    target.mkdir(parents=True, exist_ok=True)
    print(f"  Staging into {target}")
    model.stage(target)
    (target / "README.md").write_text(model.render_card())

    print(f"\n  Staged {model.name} for local testing:")
    for f in sorted(target.iterdir()):
        print(f"    - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"\n  Test it with:")
    print(f"    INFERENCE4J_CACHE_DIR={cache_dir} ./gradlew :inference4j-core:modelTest")
    return target


def process(model: BaseModel, dry_run: bool = False, update: bool = False) -> None:
    """Stage model files and upload to HuggingFace."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {model.name}")
    print(f"  Target repo: {model.repo_id}")
    print(f"{'=' * 60}")

    api = HfApi()
    model.revision = _resolve_revision(model, update, api)

    staging_dir = Path(tempfile.mkdtemp(prefix=f"inference4j-{model.name}-"))
    try:
        model.stage(staging_dir)

        card_text = model.render_card()
        card_path = staging_dir / "README.md"
        card_path.write_text(card_text)
        print(f"  Model card written to {card_path}")

        if dry_run:
            print(f"\n  [DRY RUN] Would upload to {model.repo_id}")
            print(f"  Staging directory: {staging_dir}")
            print(f"  Files:")
            for f in sorted(staging_dir.iterdir()):
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"    - {f.name} ({size_mb:.1f} MB)")
            print(f"\n  Model card preview:")
            print("  " + "-" * 40)
            for line in card_text.split("\n"):
                print(f"  {line}")
            print("  " + "-" * 40)
            return

        print(f"  Creating repo {model.repo_id} ...")
        api.create_repo(model.repo_id, exist_ok=True)

        print(f"  Uploading files ...")
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=model.repo_id,
            commit_message=f"Upload {model.name} ONNX model",
        )
        print(f"  Uploaded to https://huggingface.co/{model.repo_id}")

        _record(model, model.revision, api)
    finally:
        if not dry_run:
            shutil.rmtree(staging_dir)
            print(f"  Cleaned up staging directory")
