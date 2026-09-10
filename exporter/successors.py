"""Successor detection — find newer generations of the models we mirror.

`check` answers "did the repo we pinned change?". It reports `ok` forever on a model
whose author has moved on to a new repo entirely: depth-anything-v2 stays `ok` while
depth-anything-v3 ships next door. This module answers the other question.

It is deliberately a separate command. Drift detection is exact — hashes match or they
don't. This is a heuristic over repo names, so it produces candidates for a human to
judge, and mixing that noise into `check` would cost `check` its credibility.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from huggingface_hub import HfApi

# Version tokens in repo names, most specific first. `depth-anything-v2-small` and
# `Qwen2.5-1.5B-Instruct` and `gemma-2-2b-it` all encode a generation differently.
_VERSION_PATTERNS = [
    r"(?i)^(?P<stem>.*?)[-_]v(?P<ver>\d+(?:\.\d+)?)(?P<rest>[-_].*)?$",   # ...-v2-small
    r"(?i)^(?P<stem>[a-z]+)(?P<ver>\d+(?:\.\d+)?)(?P<rest>.*)$",          # Qwen2.5..., yolo26n
    r"(?i)^(?P<stem>[a-z]+(?:[-_][a-z]+)*?)[-_](?P<ver>\d+(?:\.\d+)?)(?P<rest>[-_].*)?$",  # gemma-2-2b-it
]

# Tokens that mark a variant rather than a generation — used to judge whether a
# candidate is the *same* size/task as what we ship.
_VARIANT_TOKENS = re.compile(
    r"(?i)\b(tiny|small|base|medium|large|xl|xxl|nano|mini|"
    r"\d+\.?\d*[bm]|l\d+|it|instruct|chat|pose|seg|cls|onnx|hf|gguf)\b")


@dataclass
class Family:
    stem: str
    version: float
    raw_version: str
    rest: str

    @property
    def variant_tokens(self) -> set[str]:
        return {t.lower() for t in _VARIANT_TOKENS.findall(self.rest)}


@dataclass
class Candidate:
    repo_id: str
    version: str
    downloads: int
    likes: int
    created: str
    variant_match: bool

    def score(self) -> tuple:
        """Rank exact-variant matches first, then by adoption."""
        return (self.variant_match, self.downloads, self.likes)


def parse_family(name: str) -> Family | None:
    """Split a repo name into family stem, generation, and variant remainder."""
    for pattern in _VERSION_PATTERNS:
        match = re.match(pattern, name)
        if not match or not match.group("ver"):
            continue
        try:
            version = float(match.group("ver"))
        except ValueError:
            continue
        stem = match.group("stem") or ""
        if not stem:
            continue
        return Family(stem=stem, version=version, raw_version=match.group("ver"),
                      rest=match.group("rest") or "")
    return None


def _same_family(candidate_name: str, family: Family) -> Family | None:
    """Parse a candidate and return its Family if it belongs to the same lineage."""
    parsed = parse_family(candidate_name)
    if parsed is None:
        return None
    if parsed.stem.lower() != family.stem.lower():
        return None
    return parsed


def find_successors(source_repo: str, api: HfApi, limit: int = 200) -> tuple[Family | None, list[Candidate]]:
    """Look for newer generations of `source_repo` under the same author."""
    if "/" not in source_repo:
        return None, []
    author, name = source_repo.split("/", 1)

    family = parse_family(name)
    if family is None:
        return None, []

    try:
        listed = api.list_models(author=author, search=family.stem,
                                 sort="downloads", direction=-1, limit=limit)
    except Exception:
        return family, []

    ours_variant = family.variant_tokens
    candidates = []
    for model in listed:
        cand_name = model.id.split("/", 1)[1]
        if model.id == source_repo:
            continue
        parsed = _same_family(cand_name, family)
        if parsed is None or parsed.version <= family.version:
            continue
        candidates.append(Candidate(
            repo_id=model.id,
            version=parsed.raw_version,
            downloads=model.downloads or 0,
            likes=model.likes or 0,
            created=str(getattr(model, "created_at", "") or "")[:10],
            variant_match=bool(ours_variant) and parsed.variant_tokens == ours_variant,
        ))

    candidates.sort(key=Candidate.score, reverse=True)
    return family, candidates


@dataclass
class Report:
    name: str
    source_repo: str
    family: Family | None
    candidates: list[Candidate]


def scan(models, workers: int = 6, top: int = 3) -> list[Report]:
    """Check every model for newer upstream generations."""
    api = HfApi()

    def one(model):
        if not model.source_repo or model.source_type != "hf":
            return Report(model.name, model.source_repo or "—", None, [])
        family, candidates = find_successors(model.source_repo, api)
        return Report(model.name, model.source_repo, family, candidates[:top])

    with ThreadPoolExecutor(workers) as pool:
        reports = list(pool.map(one, models))
    # Models with candidates first, then the rest alphabetically.
    return sorted(reports, key=lambda r: (not r.candidates, r.name))
