"""Model card renderer for mirrored models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exporter.base import ModelCard


def render(card: ModelCard) -> str:
    """Render a ModelCard as HuggingFace README.md markdown."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in card.tags + ["onnx", "inference4j"])

    details_rows = ""
    for key, value in card.model_details.items():
        details_rows += f"| {key} | {value} |\n"

    datasets_yaml = ""
    if card.datasets:
        datasets_yaml = "datasets:\n" + "\n".join(f"  - {d}" for d in card.datasets) + "\n"

    extra = ""
    if card.extra_sections:
        extra = "\n" + card.extra_sections + "\n"

    return f"""\
---
library_name: onnx
tags:
{tags_yaml}
license: {card.license}
pipeline_tag: {card.pipeline_tag}
{datasets_yaml}---

# {card.title}

{card.description}

Mirrored for use with [inference4j](https://github.com/inference4j/inference4j), an inference-only AI library for Java.

## Original Source

- **Repository:** [{card.original_author}]({card.original_source_url})
- **License:** {card.license}

## Usage with inference4j

```java
{card.java_usage}
```

## Model Details

| Property | Value |
|----------|-------|
{details_rows}
## License

{card.license_text}
{extra}"""
