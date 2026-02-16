from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class Wav2Vec2Base960h(MirroredModel):
    name = "wav2vec2-base-960h"
    repo_id = "inference4j/wav2vec2-base-960h"
    source_repo = "Xenova/wav2vec2-base-960h"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.json", dst="vocab.json"),
    ]
    card = ModelCard(
        title="Wav2Vec2 Base 960h — ONNX",
        description="ONNX export of [wav2vec2-base-960h](https://huggingface.co/Xenova/wav2vec2-base-960h), a Wav2Vec2 model fine-tuned on 960 hours of LibriSpeech for automatic speech recognition using CTC decoding.",
        license="mit",
        pipeline_tag="automatic-speech-recognition",
        tags=["wav2vec2", "speech-to-text", "automatic-speech-recognition", "ctc", "audio"],
        original_source_url="https://huggingface.co/Xenova/wav2vec2-base-960h",
        original_author="Xenova (originally facebook/wav2vec2-base-960h)",
        java_usage="""\
try (Wav2Vec2 model = Wav2Vec2.fromPretrained("models/wav2vec2-base-960h")) {
    Transcription result = model.transcribe(Path.of("audio.wav"));
    System.out.println(result.text());
}""",
        model_details={
            "Architecture": "Wav2Vec2 Base (12 transformer layers)",
            "Task": "Automatic speech recognition (CTC decoding)",
            "Training data": "LibriSpeech 960h",
            "Input": "16kHz mono audio (float32 waveform)",
            "Output": "CTC logits → greedy-decoded text",
            "Original framework": "PyTorch (HuggingFace Transformers)",
            "ONNX export": "By Xenova (Transformers.js)",
        },
        license_text="This model is licensed under the [MIT License](https://opensource.org/licenses/MIT). Original model by [Facebook AI](https://huggingface.co/facebook/wav2vec2-base-960h), ONNX export by [Xenova](https://huggingface.co/Xenova).",
    )


register(Wav2Vec2Base960h())
