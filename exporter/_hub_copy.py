"""Internal CLI for cross-repository Hugging Face Hub copies.

This runs in a small isolated environment because the ONNX export toolchain requires
Transformers 4.x, whose Hub dependency predates cross-repository copy support.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable, TypeVar

import httpx
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi
from huggingface_hub.errors import HfHubHTTPError


_T = TypeVar("_T")
_ATTEMPTS = 3
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return (
        isinstance(exc, HfHubHTTPError)
        and exc.response is not None
        and exc.response.status_code in _TRANSIENT_STATUS_CODES
    )


def _retry(operation: Callable[[], _T]) -> _T:
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:
            if not _is_transient(exc) or attempt == _ATTEMPTS:
                raise
            delay = 2 ** (attempt - 1)
            print(
                f"  Warning: server-side copy failed ({type(exc).__name__}); "
                f"retrying in {delay}s ({attempt}/{_ATTEMPTS})"
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def copy_mirror(
    *,
    source_repo: str,
    source_revision: str | None,
    target_repo: str,
    files: list[list[str]],
    card: Path,
    message: str,
) -> None:
    api = HfApi()

    def create_commit():
        operations = [
            CommitOperationCopy(
                src_path_in_repo=source,
                path_in_repo=target,
                src_revision=source_revision,
                src_repo_id=source_repo,
                src_repo_type="model",
            )
            for source, target in files
        ]
        operations.append(
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card)
        )
        return api.create_commit(
            repo_id=target_repo,
            operations=operations,
            commit_message=message,
        )

    _retry(create_commit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--files", required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    copy_mirror(
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        target_repo=args.target_repo,
        files=json.loads(args.files),
        card=args.card,
        message=args.message,
    )


if __name__ == "__main__":
    main()
