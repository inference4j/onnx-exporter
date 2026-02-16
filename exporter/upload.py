"""Staging and upload workflow."""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from exporter.base import BaseModel


def process(model: BaseModel, dry_run: bool = False) -> None:
    """Stage model files and upload to HuggingFace."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {model.name}")
    print(f"  Target repo: {model.repo_id}")
    print(f"{'=' * 60}")

    staging_dir = Path(tempfile.mkdtemp(prefix=f"inference4j-{model.name}-"))
    try:
        model.stage(staging_dir)

        card_text = model.render_card()
        card_path = staging_dir / "README.md"
        card_path.write_text(card_text)
        print(f"  Model card written to {card_path}")

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

        api = HfApi()
        print(f"  Creating repo {model.repo_id} ...")
        api.create_repo(model.repo_id, exist_ok=True)

        print(f"  Uploading files ...")
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=model.repo_id,
            commit_message=f"Upload {model.name} ONNX model",
        )
        print(f"  Uploaded to https://huggingface.co/{model.repo_id}")
    finally:
        shutil.rmtree(staging_dir)
        print(f"  Cleaned up staging directory")
