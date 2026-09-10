"""Base classes for model definitions."""

import shutil
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download



@dataclass
class FileMapping:
    """Maps a source file to a destination filename."""
    src: str
    dst: str


@dataclass
class ModelCard:
    """Metadata for generating a HuggingFace model card."""
    title: str
    description: str
    license: str
    pipeline_tag: str
    tags: list[str]
    original_source_url: str
    original_author: str
    java_usage: str
    model_details: dict[str, str]
    license_text: str
    datasets: list[str] = field(default_factory=list)
    extra_sections: str = ""


class BaseModel(ABC):
    """Abstract base for all model definitions."""
    name: str
    repo_id: str
    source_repo: str = ""
    source_type: str = "hf"  # "hf" | "url" | "gdrive"
    card: ModelCard

    # Upstream revision to build from. Set by upload.process() from the lockfile before
    # stage() runs; None means "whatever upstream HEAD is", the pre-lockfile behaviour.
    revision: str | None = None

    # False when stage() cannot honour a pinned revision — e.g. it shells out to a
    # converter that only accepts a bare model id. Such a model still gets drift
    # detection, but its exports are not reproducible; upload.process() says so.
    supports_pinning: bool = True

    @abstractmethod
    def stage(self, staging_dir: Path) -> None:
        """Download/export model files into the staging directory."""

    def tracked_files(self, siblings: list[str]) -> list[str]:
        """Which upstream files this export actually consumes.

        Given every filename in the upstream repo, return the subset whose contents
        determine our output. Drift in these files means a re-export is needed; drift
        anywhere else (READMEs, other people's ONNX exports) is noise.
        """
        raise NotImplementedError

    def render_card(self) -> str:
        """Render the model card as markdown."""
        from exporter.card import render
        return render(self.card)


def _pin_url(url: str, revision: str | None) -> str:
    """Swap a GitHub raw URL's branch for a pinned commit sha.

    The model definition declares the branch URL, which is what a human reads; the
    lockfile supplies the commit that makes the download reproducible. Substituting
    here means the definition never has to carry a sha that goes stale on every update.
    """
    if not revision:
        return url
    parts = url.split("/")
    try:
        raw = parts.index("raw")
    except ValueError:
        return url
    parts[raw + 1] = revision
    return "/".join(parts)


class MirroredModel(BaseModel):
    """A model mirrored from an upstream source (HuggingFace repo or URL)."""
    source_repo: str
    files: list[FileMapping] = ()

    def tracked_files(self, siblings: list[str]) -> list[str]:
        """The mapped source files — already declared precisely, no heuristic needed."""
        return [f.src for f in self.files]

    def stage(self, staging_dir: Path) -> None:
        if self.source_type == "hf":
            if self.revision:
                print(f"  Pinned to {self.source_repo}@{self.revision[:8]}")
            for f in self.files:
                print(f"  Downloading {self.source_repo}/{f.src} ...")
                downloaded = hf_hub_download(
                    repo_id=self.source_repo,
                    filename=f.src,
                    revision=self.revision,
                )
                dst_path = staging_dir / f.dst
                shutil.copy2(downloaded, dst_path)
                size_mb = dst_path.stat().st_size / 1024 / 1024
                print(f"    -> {dst_path} ({size_mb:.1f} MB)")
        elif self.source_type == "url":
            for f in self.files:
                dst_path = staging_dir / f.dst
                url = _pin_url(f.src, self.revision)
                print(f"  Downloading {url} ...")
                urllib.request.urlretrieve(url, dst_path)
                size_mb = dst_path.stat().st_size / 1024 / 1024
                print(f"    -> {dst_path} ({size_mb:.1f} MB)")


class ExportedModel(BaseModel):
    """A model with custom PyTorch-to-ONNX export logic."""
    card: ModelCard = None  # Exported models use render_card() directly

    # Files a torch export reads: weights, config, tokenizer. Our ONNX output is a fresh
    # trace, so it never hash-matches upstream — these inputs are the only drift signal.
    tracked_patterns: tuple[str, ...] = (
        "config.json",
        "generation_config.json",
        "*.safetensors",
        "*.safetensors.index.json",
        "pytorch_model*.bin",
        "tokenizer*.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.*",
        "merges.txt",
        "spiece.model",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "source.spm",   # Marian (opus-mt)
        "target.spm",
    )
    # Other people's exports and framework variants we never read.
    ignored_prefixes: tuple[str, ...] = ("onnx/", "openvino/", "coreml/", ".")

    def tracked_files(self, siblings: list[str]) -> list[str]:
        import fnmatch

        matched = []
        for name in siblings:
            if name.startswith(self.ignored_prefixes):
                continue
            if any(fnmatch.fnmatch(name, pat) for pat in self.tracked_patterns):
                matched.append(name)
        return sorted(matched)

    @abstractmethod
    def stage(self, staging_dir: Path) -> None:
        """Export the model to ONNX and place files in the staging directory."""

    @abstractmethod
    def render_card(self) -> str:
        """Return the full model card markdown."""
