"""Drift detection — compare upstream sources against the recorded lockfile baseline."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from huggingface_hub import HfApi, hf_hub_download

from exporter import lock
from exporter.base import MirroredModel
from exporter.lock import FileHash

# Statuses, ordered worst-first for reporting.
DRIFT = "drift"
PUBLISHED_MISMATCH = "published-mismatch"
ERROR = "error"
UNPUBLISHED = "unpublished"
UNLOCKED = "unlocked"
UNVERIFIABLE = "unverifiable"
UPSTREAM_MOVED = "upstream-moved"
OK = "ok"

_SEVERITY = {
    DRIFT: 0, PUBLISHED_MISMATCH: 1, ERROR: 2, UNPUBLISHED: 3,
    UNLOCKED: 4, UNVERIFIABLE: 5, UPSTREAM_MOVED: 6, OK: 7,
}

# Exit codes for scripted use.
EXIT_OK, EXIT_DRIFT, EXIT_ERROR = 0, 1, 2

# Below this, a mixed-hash-type comparison is resolved by downloading both files rather
# than reporting `unverifiable`. Tokenizer/config JSON sits comfortably under it.
SMALL_FILE_BYTES = 2 * 1024 * 1024


@dataclass
class Result:
    name: str
    status: str
    source: str
    detail: str
    changed: list[str] | None = None

    def to_json(self) -> dict:
        out = {"model": self.name, "status": self.status,
               "source": self.source, "detail": self.detail}
        if self.changed:
            out["changed_files"] = self.changed
        return out


class Comparison:
    """Outcome of comparing two file hashes."""
    SAME = "same"
    DIFFERENT = "different"
    INCONCLUSIVE = "inconclusive"


def compare_hashes(a: FileHash | None, b: FileHash | None) -> str:
    """Compare two file identities, comparing like with like.

    HuggingFace reports an LFS sha256 for LFS-tracked files and a git blob sha1 for
    plain ones, and the same file can be LFS in one repo and plain in another —
    upstream's .gitattributes is not ours.

    The two hashes are not interchangeable, and neither are blob_ids across that
    boundary: an LFS file's blob_id is the sha1 of its *pointer file*, not of its
    contents, so a byte-identical pair stored differently yields two unequal blob_ids.
    Comparing across storage kinds therefore reports drift that isn't there (observed
    on deepseek-r1-distill-qwen-1.5b, whose config JSON is LFS upstream and plain in
    our org). Such a pair is INCONCLUSIVE, to be settled by hashing real bytes.
    """
    if a is None or b is None:
        return Comparison.DIFFERENT if a is not b else Comparison.SAME

    a_lfs, b_lfs = a.sha256 is not None, b.sha256 is not None
    if a_lfs and b_lfs:
        return Comparison.SAME if a.sha256 == b.sha256 else Comparison.DIFFERENT
    if not a_lfs and not b_lfs:
        if a.blob_id and b.blob_id:
            return Comparison.SAME if a.blob_id == b.blob_id else Comparison.DIFFERENT
        return Comparison.INCONCLUSIVE

    # Mixed storage. Differing sizes still prove difference; equal sizes prove nothing.
    if a.size and b.size and a.size != b.size:
        return Comparison.DIFFERENT
    return Comparison.INCONCLUSIVE


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_by_download(repo_a: str, file_a: str, rev_a: str | None,
                         repo_b: str, file_b: str, rev_b: str | None) -> str:
    """Settle an INCONCLUSIVE pair by hashing the actual bytes."""
    try:
        a = _sha256_file(hf_hub_download(repo_id=repo_a, filename=file_a, revision=rev_a))
        b = _sha256_file(hf_hub_download(repo_id=repo_b, filename=file_b, revision=rev_b))
        return Comparison.SAME if a == b else Comparison.DIFFERENT
    except Exception:
        return Comparison.INCONCLUSIVE


def file_map(info) -> dict[str, FileHash]:
    """Map filename -> FileHash from a ModelInfo fetched with files_metadata=True."""
    return {s.rfilename: FileHash.from_sibling(s) for s in info.siblings}


def github_path_commit(repo: str, path: str) -> tuple[str, str] | None:
    """Latest commit touching a path in a GitHub repo, without downloading it."""
    url = (f"https://api.github.com/repos/{repo}/commits"
           f"?path={urllib.parse.quote(path)}&per_page=1")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
        if not data:
            return None
        return data[0]["sha"], data[0]["commit"]["committer"]["date"][:10]
    except Exception:
        return None


def _parse_github_raw(url: str) -> tuple[str, str, str] | None:
    """Split a github raw URL into (repo, ref, path)."""
    prefix = "https://github.com/"
    if not url.startswith(prefix):
        return None
    rest = url[len(prefix):].split("/")
    # owner/repo/raw/<ref>/<path...>
    if len(rest) < 5 or rest[2] != "raw":
        return None
    return f"{rest[0]}/{rest[1]}", rest[3], "/".join(rest[4:])


def check_model(model, entry: lock.LockEntry | None, published: set[str],
                api: HfApi, deep: bool = False) -> Result:
    """Resolve the drift status of a single model."""
    source = model.source_repo or "—"
    is_mirror = isinstance(model, MirroredModel)

    if model.repo_id not in published:
        return Result(model.name, UNPUBLISHED, source, "never uploaded to the org")
    if entry is None:
        return Result(model.name, UNLOCKED, source, "no baseline recorded — run `lock --bootstrap`")

    if model.source_type != "hf":
        return _check_non_hf(model, entry, source, deep)

    try:
        info = api.model_info(model.source_repo, files_metadata=True)
    except Exception as exc:
        return Result(model.name, ERROR, source, f"{type(exc).__name__}: {exc}"[:120])

    upstream = file_map(info)
    tracked = model.tracked_files(list(upstream))
    if not tracked:
        return Result(model.name, ERROR, source, "no tracked files matched upstream")

    changed, inconclusive = [], []
    for name in tracked:
        verdict = compare_hashes(entry.tracked.get(name), upstream.get(name))
        if verdict == Comparison.DIFFERENT:
            changed.append(name)
        elif verdict == Comparison.INCONCLUSIVE:
            inconclusive.append(name)

    # Resolve what we can by hashing bytes: always for small files, everything under --deep.
    for name in list(inconclusive):
        up = upstream.get(name)
        if not deep and (up is None or up.size > SMALL_FILE_BYTES):
            continue
        verdict = _resolve_by_download(model.source_repo, name, entry.source.revision,
                                       model.source_repo, name, info.sha)
        if verdict != Comparison.INCONCLUSIVE:
            inconclusive.remove(name)
            if verdict == Comparison.DIFFERENT:
                changed.append(name)

    if changed:
        return Result(model.name, DRIFT, source,
                      f"{len(changed)} tracked file(s) changed: " + ", ".join(changed[:3]),
                      changed=changed)
    if inconclusive:
        return Result(model.name, UNVERIFIABLE, source,
                      f"{len(inconclusive)} file(s) need --deep: " + ", ".join(inconclusive[:3]),
                      changed=inconclusive)
    # Upstream is clean. Now the reverse question: does the org still hold what we think
    # we published? Catches out-of-band uploads and exports that never actually landed.
    mismatch = _check_published(model, entry, api) if is_mirror else None
    if mismatch:
        return Result(model.name, PUBLISHED_MISMATCH, source, mismatch)

    if info.sha != entry.source.revision:
        return Result(model.name, UPSTREAM_MOVED, source,
                      f"repo at {info.sha[:8]}, tracked files unchanged — `--refresh-lock`")
    return Result(model.name, OK, source, f"{len(tracked)} file(s) match @{info.sha[:8]}")


def _check_published(model, entry: lock.LockEntry, api: HfApi) -> str | None:
    """Return a description of how our published repo diverges from the lock, or None."""
    if not entry.published:
        return None
    try:
        ours = file_map(api.model_info(model.repo_id, files_metadata=True))
    except Exception:
        return None  # transient; upstream drift is the question this command exists to answer
    differing = [
        name for name, expected in entry.published.items()
        if compare_hashes(expected, ours.get(name)) == Comparison.DIFFERENT
    ]
    if differing:
        return f"org copy differs from lock: {', '.join(differing[:3])}"
    return None


def _check_non_hf(model, entry: lock.LockEntry, source: str, deep: bool) -> Result:
    """URL and Google Drive sources, which have no HF revision to compare."""
    url = entry.source.repo
    parsed = _parse_github_raw(url)
    if parsed:
        repo, _ref, path = parsed
        latest = github_path_commit(repo, path)
        if latest is None:
            return Result(model.name, UNVERIFIABLE, source, "GitHub API unavailable")
        sha, date = latest
        if sha != entry.source.revision:
            return Result(model.name, DRIFT, source,
                          f"{path} changed upstream on {date} ({sha[:8]})", changed=[path])
        if deep:
            expected = next(iter(entry.tracked.values()), None)
            if expected and expected.sha256:
                from exporter.bootstrap import _hash_url

                digest = _hash_url(url.replace(f"/{_ref}/", f"/{sha}/", 1))
                if digest and digest[0] != expected.sha256:
                    return Result(model.name, DRIFT, source,
                                  f"{path} bytes differ from the pinned hash", changed=[path])
        return Result(model.name, OK, source, f"pinned to {sha[:8]}")

    # Google Drive (craft): the artifact hash is the only version there is.
    if not deep:
        return Result(model.name, UNVERIFIABLE, source, "unversioned source — use --deep")
    return _check_by_download(model, entry, source)


def _check_by_download(model, entry: lock.LockEntry, source: str) -> Result:
    """Last resort: fetch the artifact and compare its hash to the pinned one."""
    from pathlib import Path

    cached = getattr(model, "weights_cache", None)
    if cached and Path(cached).exists():
        digest = _sha256_file(str(cached))
        expected = next(iter(entry.tracked.values()), None)
        if expected is None or expected.sha256 is None:
            return Result(model.name, UNLOCKED, source, "no hash pinned")
        if digest == expected.sha256:
            return Result(model.name, OK, source, f"artifact hash matches {digest[:8]}")
        return Result(model.name, DRIFT, source, f"artifact hash {digest[:8]} != pinned")
    return Result(model.name, UNVERIFIABLE, source, "no cached artifact to hash")


def check_all(models, deep: bool = False, workers: int = 8) -> list[Result]:
    """Check every given model concurrently. Read-only; no downloads unless deep."""
    from exporter.__main__ import _fetch_published_repos

    api = HfApi()
    entries = lock.load()
    published = _fetch_published_repos() or set()

    def one(model):
        try:
            return check_model(model, entries.get(model.name), published, api, deep)
        except Exception as exc:
            return Result(model.name, ERROR, model.source_repo or "—",
                          f"{type(exc).__name__}: {exc}"[:120])

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, models))
    return sorted(results, key=lambda r: (_SEVERITY[r.status], r.name))


def exit_code(results: list[Result]) -> int:
    statuses = {r.status for r in results}
    if statuses & {DRIFT, PUBLISHED_MISMATCH}:
        return EXIT_DRIFT
    if ERROR in statuses:
        return EXIT_ERROR
    return EXIT_OK
