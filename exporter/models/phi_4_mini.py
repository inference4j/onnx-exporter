from pathlib import Path

from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.genai_builder import ensure_chat_template
from exporter.registry import register

_PREFIX = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"


class Phi4MiniInstruct(MirroredModel):
    name = "phi-4-mini-instruct"
    repo_id = "inference4j/phi-4-mini-instruct"
    source_repo = "microsoft/Phi-4-mini-instruct-onnx"
    source_type = "hf"
    files = [
        FileMapping(
            src=f"{_PREFIX}/genai_config.json",
            dst="genai_config.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/config.json",
            dst="config.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/model.onnx",
            dst="model.onnx",
        ),
        FileMapping(
            src=f"{_PREFIX}/model.onnx.data",
            dst="model.onnx.data",
        ),
        FileMapping(
            src=f"{_PREFIX}/tokenizer.json",
            dst="tokenizer.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/tokenizer_config.json",
            dst="tokenizer_config.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/special_tokens_map.json",
            dst="special_tokens_map.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/added_tokens.json",
            dst="added_tokens.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/vocab.json",
            dst="vocab.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/merges.txt",
            dst="merges.txt",
        ),
    ]

    def stage(self, staging_dir: Path) -> None:
        super().stage(staging_dir)
        ensure_chat_template(None, staging_dir)

    card = ModelCard(
        title="Phi-4-mini-instruct — ONNX (INT4)",
        description=(
            "INT4-quantized ONNX export of "
            "[Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct), "
            "a 3.8B-parameter lightweight language model from Microsoft. "
            "Optimized for CPU inference with INT4 RTN block-32 quantization in onnxruntime-genai format."
        ),
        license="mit",
        pipeline_tag="text-generation",
        tags=["text-generation", "phi-4", "onnx", "int4", "cpu", "onnxruntime-genai", "inference4j"],
        original_source_url="https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx",
        original_author="Microsoft",
        java_usage="""\
try (TextGenerator gen = TextGenerator.builder()
        .model(ModelSources.phi4Mini())
        .build()) {
    GenerationResult result = gen.generate("What is Java in one sentence?");
    System.out.println(result.text());
}""",
        model_details={
            "Architecture": "Phi-4-mini (3.8B parameters, 32 layers, 3072 hidden)",
            "Task": "Text generation / chat",
            "Context length": "128k tokens",
            "Quantization": "INT4 RTN block-32 acc-level-4",
            "Original framework": "PyTorch (transformers)",
        },
        license_text=(
            "This model is licensed under the "
            "[MIT License](https://opensource.org/licenses/MIT). "
            "Original model and ONNX conversion by [Microsoft](https://huggingface.co/microsoft)."
        ),
    )


register(Phi4MiniInstruct())
