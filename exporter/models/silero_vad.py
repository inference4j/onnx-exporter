from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class SileroVad(MirroredModel):
    name = "silero-vad"
    repo_id = "inference4j/silero-vad"
    source_repo = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    source_type = "url"
    files = [
        FileMapping(
            src="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
            dst="model.onnx",
        ),
    ]
    card = ModelCard(
        title="Silero VAD — ONNX",
        description="ONNX export of [Silero VAD](https://github.com/snakers4/silero-vad), a lightweight and fast voice activity detection model. Detects speech segments in audio with high accuracy and low latency.",
        license="mit",
        pipeline_tag="voice-activity-detection",
        tags=["silero", "voice-activity-detection", "vad", "audio"],
        original_source_url="https://github.com/snakers4/silero-vad",
        original_author="Silero Team (snakers4)",
        java_usage="""\
try (SileroVAD vad = SileroVAD.fromPretrained("models/silero-vad")) {
    List<VoiceSegment> segments = vad.detect(Path.of("meeting.wav"));
    for (VoiceSegment segment : segments) {
        System.out.printf("Speech: %.2fs - %.2fs%n", segment.start(), segment.end());
    }
}""",
        model_details={
            "Architecture": "Silero VAD (lightweight CNN + LSTM)",
            "Task": "Voice activity detection",
            "Input": "16kHz mono audio (float32 waveform, 512-sample chunks)",
            "Output": "Speech probability per chunk",
            "Model size": "~2 MB",
            "Original source": "[snakers4/silero-vad](https://github.com/snakers4/silero-vad)",
        },
        license_text="This model is licensed under the [MIT License](https://opensource.org/licenses/MIT). Original model by [Silero Team](https://github.com/snakers4/silero-vad).",
    )


register(SileroVad())
