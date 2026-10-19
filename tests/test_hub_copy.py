from pathlib import Path

import httpx
import pytest

from exporter import _hub_copy


class FakeOperation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeApi:
    def __init__(self):
        self.commit = None

    def create_commit(self, **kwargs):
        self.commit = kwargs


def test_copy_mirror_builds_cross_repo_commit(monkeypatch, tmp_path: Path):
    api = FakeApi()
    card = tmp_path / "README.md"
    card.write_text("# model\n")
    monkeypatch.setattr(_hub_copy, "HfApi", lambda: api)
    monkeypatch.setattr(_hub_copy, "CommitOperationCopy", FakeOperation)
    monkeypatch.setattr(_hub_copy, "CommitOperationAdd", FakeOperation)

    _hub_copy.copy_mirror(
        source_repo="upstream/model",
        source_revision="abc123",
        target_repo="inference4j/model",
        files=[["onnx/model.onnx", "model.onnx"]],
        card=card,
        message="Upload model",
    )

    assert api.commit["repo_id"] == "inference4j/model"
    copy, readme = api.commit["operations"]
    assert copy.src_repo_id == "upstream/model"
    assert copy.src_repo_type == "model"
    assert copy.src_revision == "abc123"
    assert copy.src_path_in_repo == "onnx/model.onnx"
    assert copy.path_in_repo == "model.onnx"
    assert readme.path_in_repo == "README.md"
    assert readme.path_or_fileobj == card


def test_retry_retries_transport_errors(monkeypatch):
    attempts = 0
    monkeypatch.setattr(_hub_copy.time, "sleep", lambda seconds: None)

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("reset")
        return "copied"

    assert _hub_copy._retry(operation) == "copied"
    assert attempts == 3


def test_retry_does_not_hide_non_network_errors():
    with pytest.raises(ValueError, match="invalid copy"):
        _hub_copy._retry(lambda: (_ for _ in ()).throw(ValueError("invalid copy")))
