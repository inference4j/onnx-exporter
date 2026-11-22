"""Staging and upload workflow."""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, TypeVar

from huggingface_hub import CommitOperationAdd, HfApi
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from exporter import lock
from exporter.base import BaseModel, MirroredModel


_T = TypeVar("_T")
_HUB_WRITE_ATTEMPTS = 3
_COPY_HELPER_VENV = Path(__file__).resolve().parents[1] / ".venv-hub-uploader"
_COPY_HELPER_REQUIREMENTS = ("huggingface_hub==1.33.0",)


def _retry_hub_write(description: str, operation: Callable[[], _T]) -> _T:
    """Retry transient HTTP failures without hiding non-network upload errors."""
    for attempt in range(1, _HUB_WRITE_ATTEMPTS + 1):
        try:
            return operation()
        except (RequestsConnectionError, RequestsTimeout) as exc:
            if attempt == _HUB_WRITE_ATTEMPTS:
                raise
            delay = 2 ** (attempt - 1)
            print(
                f"  Warning: {description} failed ({type(exc).__name__}); "
                f"retrying in {delay}s ({attempt}/{_HUB_WRITE_ATTEMPTS})"
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def _can_copy_server_side(model: BaseModel) -> bool:
    """Whether publishing can copy every artifact directly from an HF source repo."""
    return (
        isinstance(model, MirroredModel)
        and model.source_type == "hf"
        and type(model).stage is MirroredModel.stage
    )


def _copy_helper_python() -> Path:
    """Return an isolated Hub client new enough for cross-repository copies."""
    python = _COPY_HELPER_VENV / "bin" / "python"
    stamp = _COPY_HELPER_VENV / "requirements.txt"
    wanted = "\n".join(_COPY_HELPER_REQUIREMENTS) + "\n"
    if python.exists() and stamp.exists() and stamp.read_text() == wanted:
        return python

    print(f"  Creating Hub copy helper at {_COPY_HELPER_VENV} ...")
    subprocess.run([sys.executable, "-m", "venv", "--clear", str(_COPY_HELPER_VENV)], check=True)
    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", *_COPY_HELPER_REQUIREMENTS],
        check=True,
    )
    stamp.write_text(wanted)
    return python


def _write_license(model: BaseModel, directory: Path) -> Path | None:
    """Copy the model's license text into a directory as LICENSE, if it has one."""
    source = model.license_path()
    if source is None:
        return None
    target = directory / "LICENSE"
    shutil.copyfile(source, target)
    return target


def _copy_mirror(model: MirroredModel, card_path: Path, license_path: Path | None = None) -> None:
    """Publish an unmodified HF mirror using an isolated, modern Hub client."""
    revision = f"@{model.revision}" if model.revision else ""
    print(f"  Copying files server-side from {model.source_repo}{revision} ...")
    cmd = [
        str(_copy_helper_python()),
        str(Path(__file__).with_name("_hub_copy.py")),
        "--source-repo", model.source_repo,
        "--target-repo", model.repo_id,
        "--files", json.dumps([[file.src, file.dst] for file in model.files]),
        "--card", str(card_path),
        "--message", f"Upload {model.name} ONNX model",
    ]
    if model.revision:
        cmd.extend(["--source-revision", model.revision])
    if license_path is not None:
        cmd.extend(["--license", str(license_path)])
    subprocess.run(cmd, check=True)


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
        # New model definitions may carry the issue-approved revision before their
        # first successful upload creates a lock entry. Preserve that pin instead of
        # silently resolving the source repo's moving default branch.
        return model.revision

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
        if model.files and all(f.sha256 for f in model.files):
            _record_hash_pinned(model, api)
            return
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


def _release_tag(url: str) -> str | None:
    """The tag of a GitHub release asset URL (`.../releases/download/<tag>/<file>`)."""
    parts = url.split("/")
    if "releases" in parts:
        i = parts.index("releases")
        if parts[i + 1:i + 2] == ["download"] and len(parts) > i + 2:
            return parts[i + 2]
    return None


def _content_length(url: str) -> int:
    """Size of a URL artifact from a HEAD request, or 0 when the server does not say."""
    import urllib.request

    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as resp:
            return int(resp.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _record_hash_pinned(model: BaseModel, api: HfApi) -> None:
    """Lock a URL mirror whose files carry their own sha256, such as GitHub release assets.

    A release asset has no commit to pin, so the declared hash is the version: stage()
    refuses other bytes, and `check --deep` re-hashes the URL against this entry.
    """
    from exporter.check import file_map

    try:
        published = file_map(api.model_info(model.repo_id, files_metadata=True))
    except Exception as exc:
        print(f"  Warning: could not record provenance ({type(exc).__name__})")
        return

    unchanged = type(model).stage is MirroredModel.stage
    tracked = {}
    for f in model.files:
        ours = published.get(f.dst)
        if unchanged and ours is not None and ours.sha256 not in (None, f.sha256):
            print(f"  Warning: published {f.dst} does not hash-match {f.src}")
        tracked[f.src] = lock.FileHash(size=_content_length(f.src), sha256=f.sha256)

    lock.put(model.name, lock.LockEntry(
        repo_id=model.repo_id,
        source=lock.Source(type="url", repo=model.source_repo,
                           revision=_release_tag(model.files[0].src)),
        provenance=lock.PINNED_BY_HASH,
        tracked=tracked,
        published={n: h for n, h in published.items() if not n.startswith(".")},
        exported_at=lock.now(),
    ))
    print(f"  Locked {len(tracked)} file(s) by sha256")


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
    _write_license(model, target)

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
    succeeded = False
    try:
        server_side_copy = _can_copy_server_side(model) and not dry_run
        if not server_side_copy:
            model.stage(staging_dir)

        card_text = model.render_card()
        card_path = staging_dir / "README.md"
        card_path.write_text(card_text)
        print(f"  Model card written to {card_path}")
        license_path = _write_license(model, staging_dir)

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

        if server_side_copy:
            _copy_mirror(model, card_path, license_path)
        else:
            print(f"  Uploading files ...")
            _retry_hub_write(
                "upload",
                lambda: api.upload_folder(
                    folder_path=str(staging_dir),
                    repo_id=model.repo_id,
                    commit_message=f"Upload {model.name} ONNX model",
                    delete_patterns=list(model.obsolete_files) or None,
                ),
            )
        print(f"  Uploaded to https://huggingface.co/{model.repo_id}")

        _record(model, model.revision, api)
        succeeded = True
    finally:
        if not dry_run and succeeded:
            shutil.rmtree(staging_dir)
            print(f"  Cleaned up staging directory")
        elif not dry_run:
            print(f"  Upload failed; staging directory preserved at {staging_dir}")


def publish_card(model: BaseModel, dry_run: bool = False) -> None:
    """Publish only the model card, and the LICENSE file if the model has one.

    The model files are left untouched. The lockfile's record of the published files is
    refreshed afterwards, so `check` keeps comparing against what is live.
    """
    print(f"\nPublishing card: {model.name} -> {model.repo_id}")
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix=f"inference4j-card-{model.name}-") as tmp:
        directory = Path(tmp)
        card_path = directory / "README.md"
        card_path.write_text(model.render_card())
        operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path))]
        license_path = _write_license(model, directory)
        if license_path is not None:
            operations.append(CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=str(license_path)))
        if dry_run:
            print(f"  [DRY RUN] Would commit {', '.join(op.path_in_repo for op in operations)}")
            return
        _retry_hub_write(
            "card commit",
            lambda: api.create_commit(
                repo_id=model.repo_id,
                operations=operations,
                commit_message=f"Update {model.name} model card",
            ),
        )
    print(f"  Updated https://huggingface.co/{model.repo_id}")
    _refresh_published(model, api)


def _refresh_published(model: BaseModel, api: HfApi) -> None:
    """Re-record the published file hashes of a model already in the lockfile."""
    from exporter.check import file_map

    entry = lock.get(model.name)
    if entry is None:
        return
    published = file_map(api.model_info(model.repo_id, files_metadata=True))
    entry.published = {n: h for n, h in published.items() if not n.startswith(".")}
    lock.put(model.name, entry)
    print(f"  Lockfile: published files of {model.name} refreshed")
