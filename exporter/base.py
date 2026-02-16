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
    card: ModelCard

    @abstractmethod
    def stage(self, staging_dir: Path) -> None:
        """Download/export model files into the staging directory."""

    def render_card(self) -> str:
        """Render the model card as markdown."""
        from exporter.card import render
        return render(self.card)


class MirroredModel(BaseModel):
    """A model mirrored from an upstream source (HuggingFace repo or URL)."""
    source_repo: str
    source_type: str = "hf"  # "hf" or "url"
    files: list[FileMapping] = ()

    def stage(self, staging_dir: Path) -> None:
        if self.source_type == "hf":
            for f in self.files:
                print(f"  Downloading {self.source_repo}/{f.src} ...")
                downloaded = hf_hub_download(
                    repo_id=self.source_repo,
                    filename=f.src,
                )
                dst_path = staging_dir / f.dst
                shutil.copy2(downloaded, dst_path)
                size_mb = dst_path.stat().st_size / 1024 / 1024
                print(f"    -> {dst_path} ({size_mb:.1f} MB)")
        elif self.source_type == "url":
            for f in self.files:
                dst_path = staging_dir / f.dst
                print(f"  Downloading {f.src} ...")
                urllib.request.urlretrieve(f.src, dst_path)
                size_mb = dst_path.stat().st_size / 1024 / 1024
                print(f"    -> {dst_path} ({size_mb:.1f} MB)")


class ExportedModel(BaseModel):
    """A model with custom PyTorch-to-ONNX export logic."""
    card: ModelCard = None  # Exported models use render_card() directly

    @abstractmethod
    def stage(self, staging_dir: Path) -> None:
        """Export the model to ONNX and place files in the staging directory."""

    @abstractmethod
    def render_card(self) -> str:
        """Return the full model card markdown."""
