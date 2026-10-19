import json
from pathlib import Path

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError

from exporter import upload
from exporter.base import BaseModel, FileMapping, MirroredModel


class DirectMirror(MirroredModel):
    name = "direct-mirror"
    repo_id = "inference4j/direct-mirror"
    source_repo = "upstream/direct-mirror"
    revision = "abc123"
    files = [FileMapping(src="onnx/model.onnx", dst="model.onnx")]

    def render_card(self) -> str:
        return "# Direct mirror\n"


class GeneratedModel(BaseModel):
    name = "generated"
    repo_id = "inference4j/generated"
    revision = None

    def stage(self, staging_dir: Path) -> None:
        (staging_dir / "model.onnx").write_bytes(b"model")

    def tracked_files(self, siblings: list[str]) -> list[str]:
        return []

    def render_card(self) -> str:
        return "# Generated\n"


class FakeApi:
    def __init__(self):
        self.upload_attempts = 0
        self.upload_error = None

    def create_repo(self, repo_id, exist_ok):
        return None

    def upload_folder(self, **kwargs):
        self.upload_attempts += 1
        if self.upload_error is not None:
            raise self.upload_error


def _fixed_staging(monkeypatch, tmp_path: Path) -> Path:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(upload.tempfile, "mkdtemp", lambda **kwargs: str(staging))
    return staging


def test_direct_hf_mirror_uses_atomic_server_side_copy(monkeypatch, tmp_path):
    api = FakeApi()
    staging = _fixed_staging(monkeypatch, tmp_path)
    copies = []
    monkeypatch.setattr(upload, "HfApi", lambda: api)
    monkeypatch.setattr(upload, "_copy_mirror", lambda model, card: copies.append((model, card)))
    monkeypatch.setattr(upload, "_record", lambda *args: None)

    upload.process(DirectMirror())

    assert not staging.exists()
    assert api.upload_attempts == 0
    assert len(copies) == 1
    copied_model, card_path = copies[0]
    assert copied_model.source_repo == "upstream/direct-mirror"
    assert copied_model.revision == "abc123"
    assert card_path == staging / "README.md"


def test_copy_mirror_invokes_isolated_helper(monkeypatch, tmp_path):
    card = tmp_path / "README.md"
    card.write_text("# Direct mirror\n")
    commands = []
    monkeypatch.setattr(upload, "_copy_helper_python", lambda: Path("/helper/python"))
    monkeypatch.setattr(
        upload.subprocess,
        "run",
        lambda command, check: commands.append((command, check)),
    )

    upload._copy_mirror(DirectMirror(), card)

    assert len(commands) == 1
    command, check = commands[0]
    assert check is True
    assert command[0] == "/helper/python"
    assert command[command.index("--source-repo") + 1] == "upstream/direct-mirror"
    assert command[command.index("--target-repo") + 1] == "inference4j/direct-mirror"
    assert command[command.index("--source-revision") + 1] == "abc123"
    assert json.loads(command[command.index("--files") + 1]) == [
        ["onnx/model.onnx", "model.onnx"]
    ]


def test_transient_upload_failure_retries_and_preserves_staging(monkeypatch, tmp_path):
    api = FakeApi()
    api.upload_error = RequestsConnectionError("reset")
    staging = _fixed_staging(monkeypatch, tmp_path)
    monkeypatch.setattr(upload, "HfApi", lambda: api)
    monkeypatch.setattr(upload.time, "sleep", lambda seconds: None)

    with pytest.raises(RequestsConnectionError, match="reset"):
        upload.process(GeneratedModel())

    assert api.upload_attempts == 3
    assert (staging / "model.onnx").read_bytes() == b"model"
    assert (staging / "README.md").read_text() == "# Generated\n"


def test_successful_generated_upload_cleans_staging(monkeypatch, tmp_path):
    api = FakeApi()
    staging = _fixed_staging(monkeypatch, tmp_path)
    monkeypatch.setattr(upload, "HfApi", lambda: api)
    monkeypatch.setattr(upload, "_record", lambda *args: None)

    upload.process(GeneratedModel())

    assert api.upload_attempts == 1
    assert not staging.exists()
