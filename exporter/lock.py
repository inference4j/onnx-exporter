"""Provenance lockfile — records which upstream revision each export came from.

The lockfile is the baseline that makes drift detection meaningful. Without it we can
only ask "does this repo exist?"; with it we can ask "is what we published still what
upstream says it should be?".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parent.parent / "exports.lock.json"

# How much we trust a lock entry's `source.revision`.
VERIFIED = "verified"              # published bytes hash-match this upstream commit
INFERRED = "inferred"              # upstream HEAD at our upload date; not hash-verifiable
PINNED_BY_HASH = "pinned-by-hash"  # no upstream revision exists; the artifact hash is the version
UNKNOWN = "unknown"                # could not place it; published hashes recorded anyway


@dataclass
class FileHash:
    """A file's identity. Exactly one of sha256/blob_id is authoritative.

    HuggingFace reports an LFS sha256 for LFS-tracked files and a git blob sha1 for
    plain ones. The same file can be LFS in one repo and plain in another, so the kind
    of hash matters as much as the value — see check.compare_hashes.
    """
    size: int
    sha256: str | None = None
    blob_id: str | None = None

    @classmethod
    def from_sibling(cls, sibling) -> FileHash:
        """Build from a huggingface_hub RepoSibling."""
        lfs = getattr(sibling, "lfs", None)
        return cls(
            size=sibling.size or 0,
            sha256=getattr(lfs, "sha256", None) if lfs else None,
            blob_id=sibling.blob_id,
        )


@dataclass
class Source:
    type: str                    # "hf" | "url" | "gdrive"
    repo: str                    # HF repo id, or the URL for non-hf sources
    revision: str | None = None  # HF commit sha, GitHub commit sha, or None


@dataclass
class LockEntry:
    repo_id: str
    source: Source
    provenance: str
    tracked: dict[str, FileHash] = field(default_factory=dict)
    published: dict[str, FileHash] = field(default_factory=dict)
    exported_at: str | None = None

    def to_json(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "source": asdict(self.source),
            "provenance": self.provenance,
            "tracked": {k: _strip(asdict(v)) for k, v in sorted(self.tracked.items())},
            "published": {k: _strip(asdict(v)) for k, v in sorted(self.published.items())},
            "exported_at": self.exported_at,
        }

    @classmethod
    def from_json(cls, data: dict) -> LockEntry:
        return cls(
            repo_id=data["repo_id"],
            source=Source(**data["source"]),
            provenance=data["provenance"],
            tracked={k: FileHash(**v) for k, v in data.get("tracked", {}).items()},
            published={k: FileHash(**v) for k, v in data.get("published", {}).items()},
            exported_at=data.get("exported_at"),
        )


def _strip(d: dict) -> dict:
    """Drop null hash fields so the committed lockfile stays readable."""
    return {k: v for k, v in d.items() if v is not None}


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load(path: Path = LOCK_PATH) -> dict[str, LockEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {name: LockEntry.from_json(entry) for name, entry in raw.items()}


def save(entries: dict[str, LockEntry], path: Path = LOCK_PATH) -> None:
    payload = {name: entries[name].to_json() for name in sorted(entries)}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def get(name: str, path: Path = LOCK_PATH) -> LockEntry | None:
    return load(path).get(name)


def put(name: str, entry: LockEntry, path: Path = LOCK_PATH) -> None:
    """Insert or replace a single entry, preserving the rest of the file."""
    entries = load(path)
    entries[name] = entry
    save(entries, path)
