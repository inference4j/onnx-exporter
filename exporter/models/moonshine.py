import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register

_SOURCE_REPO = "moonshine-ai/moonshine-voice-assets"

_GRAPHS = [
    "frontend.model.ort",
    "frontend.weights.ort",
    "encoder.ort",
    "adapter.ort",
    "cross_kv.ort",
    "decoder_kv.ort",
    "streaming_config.json",
]


class MoonshineStreamingModel(MirroredModel):
    """Moonshine v2 streaming ASR model in quantized .ort format."""

    source_repo = _SOURCE_REPO
    source_type = "hf"

    def __init__(self, size: str) -> None:
        self.size = size
        self.name = f"moonshine-streaming-{size}-en"
        self.repo_id = f"inference4j/moonshine-streaming-{size}-en"
        self.tokenizer_repo = f"moonshine-ai/moonshine-streaming-{size}"

        src_dir = f"model/{size}-streaming-en/quantized_26_08_21"
        self.files = [
            FileMapping(src=f"{src_dir}/{filename}", dst=filename)
            for filename in _GRAPHS
        ]

        self.card = ModelCard(
            title=f"Moonshine Streaming ({size.capitalize()}) — ONNX Runtime (.ort)",
            description=(
                f"Moonshine v2 streaming {size} speech recognition model for English, "
                "quantized to ONNX Runtime (.ort) format for efficient on-device transcription."
            ),
            license="mit",
            pipeline_tag="automatic-speech-recognition",
            tags=["moonshine", "speech-to-text", "automatic-speech-recognition", "audio", "streaming"],
            original_source_url="https://huggingface.co/moonshine-ai/moonshine-voice-assets",
            original_author="Moonshine AI (moonshine-ai/moonshine)",
            java_usage=f"""\
try (var recognizer = MoonshineRecognizer.builder()
        .modelId("{self.repo_id}")
        .build()) {{
    RecognitionResult result = recognizer.recognize(audioStream);
    System.out.println(result.text());
}}""",
            model_details={
                "Architecture": f"Moonshine v2 Streaming ({size.capitalize()})",
                "Release": "quantized_26_08_21",
                "Task": "Streaming automatic speech recognition",
                "Precision": "Quantized (.ort)",
                "Language": "English (en)",
                "Limits": "At most 30 s per call (hard graph limit: 4096 frames / ~82 s)",
            },
            license_text=(
                "This model is licensed under the [MIT License](https://opensource.org/licenses/MIT). "
                "Original models by [Moonshine AI](https://github.com/moonshine-ai/moonshine)."
            ),
            extra_sections="""\
## Limits

- **Language:** English only.
- **Audio duration:** At most 30 s per call; use `SegmentedSpeechRecognizer` for longer audio.
- **Graph frame limit:** Hard limit inside the graphs is 4096 frames (about 82 s).
""",
        )

    def stage(self, staging_dir: Path) -> None:
        super().stage(staging_dir)
        print(f"  Downloading {self.tokenizer_repo}/tokenizer.json ...")
        downloaded = hf_hub_download(
            repo_id=self.tokenizer_repo,
            filename="tokenizer.json",
        )
        dst_path = staging_dir / "tokenizer.json"
        shutil.copy2(downloaded, dst_path)
        size_mb = dst_path.stat().st_size / 1024 / 1024
        print(f"    -> {dst_path} ({size_mb:.1f} MB)")


register(MoonshineStreamingModel("tiny"))
register(MoonshineStreamingModel("small"))
register(MoonshineStreamingModel("medium"))
