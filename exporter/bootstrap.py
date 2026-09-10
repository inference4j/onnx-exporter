"""One-time baseline reconstruction for models exported before the lockfile existed.

Nothing recorded which upstream revision each published model came from. For mirrors we
can recover it exactly: the files we published are byte copies of upstream's, so the
upstream commit whose blobs match ours *is* the provenance. For torch exports no such
match exists — a fresh trace never equals upstream bytes — so the baseline is the
upstream commit that was HEAD when we uploaded, marked `inferred`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi

from exporter import lock
from exporter.base import MirroredModel
from exporter.check import Comparison, compare_hashes, file_map
from exporter.lock import INFERRED, PINNED_BY_HASH, UNKNOWN, VERIFIED, LockEntry, Source

# How far back to walk upstream history looking for the commit our bytes came from.
MAX_HISTORY = 40


def bootstrap_model(model, published: set[str], api: HfApi) -> tuple[LockEntry | None, str]:
    """Build a lock entry for one model. Returns (entry, human-readable note)."""
    if model.repo_id not in published:
        return None, "not published — skipped"

    if model.source_type != "hf":
        return _bootstrap_non_hf(model)

    try:
        ours = file_map(api.model_info(model.repo_id, files_metadata=True))
        our_info = api.model_info(model.repo_id)
    except Exception as exc:
        return None, f"cannot read our repo: {type(exc).__name__}"

    if isinstance(model, MirroredModel):
        return _bootstrap_mirror(model, ours, api)
    return _bootstrap_export(model, ours, our_info, api)


def _bootstrap_mirror(model, ours: dict, api: HfApi) -> tuple[LockEntry | None, str]:
    """Find the upstream commit whose bytes match what we published."""
    pairs = [(f.src, f.dst) for f in model.files]

    def matches(upstream: dict) -> bool:
        """True if every mapped file is the same on both sides (unknowns don't count against)."""
        verdicts = [compare_hashes(ours.get(dst), upstream.get(src)) for src, dst in pairs]
        return all(v != Comparison.DIFFERENT for v in verdicts) and Comparison.SAME in verdicts

    try:
        head = api.model_info(model.source_repo, files_metadata=True)
    except Exception as exc:
        return None, f"upstream unreadable: {type(exc).__name__}"

    candidates = [(head.sha, file_map(head))]
    if not matches(candidates[0][1]):
        # Our bytes predate upstream HEAD — walk back to find where they came from.
        try:
            for commit in api.list_repo_commits(model.source_repo)[:MAX_HISTORY]:
                if commit.commit_id == head.sha:
                    continue
                info = api.model_info(model.source_repo, revision=commit.commit_id,
                                      files_metadata=True)
                candidates.append((commit.commit_id, file_map(info)))
                if matches(candidates[-1][1]):
                    break
        except Exception:
            pass

    for sha, upstream in candidates:
        if matches(upstream):
            return LockEntry(
                repo_id=model.repo_id,
                source=Source(type="hf", repo=model.source_repo, revision=sha),
                provenance=VERIFIED,
                tracked={src: upstream[src] for src, _ in pairs if src in upstream},
                published={dst: ours[dst] for _, dst in pairs if dst in ours},
                exported_at=None,
            ), f"verified @ {sha[:8]}"

    # No commit matches. Still record what we shipped so future drift stays detectable.
    head_files = candidates[0][1]
    return LockEntry(
        repo_id=model.repo_id,
        source=Source(type="hf", repo=model.source_repo, revision=head.sha),
        provenance=UNKNOWN,
        tracked={src: head_files[src] for src, _ in pairs if src in head_files},
        published={dst: ours[dst] for _, dst in pairs if dst in ours},
    ), "no matching upstream commit — recorded as unknown"


def _bootstrap_export(model, ours: dict, our_info, api: HfApi) -> tuple[LockEntry | None, str]:
    """Use the upstream commit that was HEAD when we uploaded."""
    uploaded_at = our_info.lastModified
    try:
        commits = api.list_repo_commits(model.source_repo)
    except Exception as exc:
        return None, f"upstream unreadable: {type(exc).__name__}"

    chosen = None
    if uploaded_at is not None:
        for commit in commits:  # newest first
            if commit.created_at <= uploaded_at:
                chosen = commit.commit_id
                break
    if chosen is None:
        chosen = commits[0].commit_id if commits else None
    if chosen is None:
        return None, "no upstream commits"

    try:
        info = api.model_info(model.source_repo, revision=chosen, files_metadata=True)
    except Exception as exc:
        return None, f"cannot read upstream @ {chosen[:8]}: {type(exc).__name__}"

    upstream = file_map(info)
    tracked = model.tracked_files(list(upstream))
    return LockEntry(
        repo_id=model.repo_id,
        source=Source(type="hf", repo=model.source_repo, revision=chosen),
        provenance=INFERRED,
        tracked={name: upstream[name] for name in tracked},
        published={name: h for name, h in ours.items() if not name.startswith(".")},
        exported_at=uploaded_at.isoformat() if uploaded_at else None,
    ), f"inferred @ {chosen[:8]} ({len(tracked)} tracked)"


def _hash_url(url: str, limit: int = 64 * 1024 * 1024) -> tuple[str, int] | None:
    """sha256 and size of a URL's contents, or None if it can't be fetched."""
    import hashlib
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read(limit + 1)
    except Exception:
        return None
    if len(data) > limit:
        return None
    return hashlib.sha256(data).hexdigest(), len(data)


def _published_sha256(model, filename: str) -> str | None:
    """sha256 of a file we published, for comparing against an upstream artifact."""
    from huggingface_hub import hf_hub_download

    from exporter.check import _sha256_file

    try:
        return _sha256_file(hf_hub_download(model.repo_id, filename))
    except Exception:
        return None


def _bootstrap_non_hf(model) -> tuple[LockEntry | None, str]:
    """GitHub raw URLs pin to a commit; Google Drive pins to the artifact hash."""
    from pathlib import Path

    from exporter.check import _parse_github_raw, _sha256_file, github_path_commit
    from exporter.lock import FileHash

    if model.source_type == "url":
        url = model.files[0].src
        parsed = _parse_github_raw(url)
        if parsed is None:
            return None, f"unrecognised URL source: {url[:60]}"
        repo, _ref, path = parsed
        latest = github_path_commit(repo, path)
        if latest is None:
            return None, "GitHub API unavailable"
        sha, date = latest

        # These artifacts are small, so settle provenance for real rather than assuming:
        # if the bytes at this commit are what we published, the pin is proven.
        provenance, tracked, note = INFERRED, {}, f"pinned to github {sha[:8]} ({date})"
        digest = _hash_url(url.replace(f"/{parsed[1]}/", f"/{sha}/", 1))
        if digest:
            tracked = {path: FileHash(size=digest[1], sha256=digest[0])}
            if _published_sha256(model, model.files[0].dst) == digest[0]:
                provenance = VERIFIED
                note = f"verified @ github {sha[:8]} ({date})"
        return LockEntry(
            repo_id=model.repo_id,
            source=Source(type="url", repo=url, revision=sha),
            provenance=provenance,
            tracked=tracked,
        ), note

    # Google Drive: hash the cached weights if we still have them locally.
    url = getattr(model, "weights_url", model.source_repo)
    cache = getattr(model, "weights_cache", None)
    tracked = {}
    note = "unversioned source — hash pinned on next run"
    if cache and Path(cache).exists():
        digest = _sha256_file(str(cache))
        tracked = {Path(cache).name: FileHash(size=Path(cache).stat().st_size, sha256=digest)}
        note = f"hashed local weights {digest[:8]}"
    return LockEntry(
        repo_id=model.repo_id,
        source=Source(type="gdrive", repo=url, revision=None),
        provenance=PINNED_BY_HASH,
        tracked=tracked,
    ), note


def bootstrap_all(models, workers: int = 8) -> tuple[dict[str, LockEntry], list[str]]:
    """Reconstruct baselines for every model. Returns (entries, notes)."""
    from exporter.__main__ import _fetch_published_repos

    api = HfApi()
    published = _fetch_published_repos() or set()
    entries: dict[str, LockEntry] = {}
    notes: list[str] = []

    def one(model):
        try:
            return model, bootstrap_model(model, published, api)
        except Exception as exc:
            return model, (None, f"{type(exc).__name__}: {exc}"[:90])

    with ThreadPoolExecutor(workers) as pool:
        for model, (entry, note) in pool.map(one, models):
            notes.append(f"  {model.name:47s} {note}")
            if entry is not None:
                entries[model.name] = entry
    notes.sort()
    return entries, notes
