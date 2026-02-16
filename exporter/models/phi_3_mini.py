from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register

_PREFIX = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"


class Phi3Mini(MirroredModel):
    name = "phi-3-mini"
    repo_id = "inference4j/phi-3-mini-4k-instruct"
    source_repo = "microsoft/Phi-3-mini-4k-instruct-onnx"
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
            src=f"{_PREFIX}/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
            dst="phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
        ),
        FileMapping(
            src=f"{_PREFIX}/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
            dst="phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
        ),
        FileMapping(
            src=f"{_PREFIX}/tokenizer.json",
            dst="tokenizer.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/tokenizer.model",
            dst="tokenizer.model",
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
    ]
    card = ModelCard(
        title="Phi-3-mini-4k-instruct — ONNX (INT4)",
        description=(
            "INT4-quantized ONNX export of "
            "[Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), "
            "a 3.8B-parameter lightweight language model from Microsoft. "
            "Optimized for CPU inference with int4 RTN block-32 quantization."
        ),
        license="mit",
        pipeline_tag="text-generation",
        tags=["text-generation", "phi-3", "onnx", "int4", "cpu"],
        original_source_url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx",
        original_author="Microsoft",
        java_usage="""\
try (TextGenerator gen = TextGenerator.builder().build()) {
    GenerationResult result = gen.generate("What is Java in one sentence?");
    System.out.println(result.text());
}""",
        model_details={
            "Architecture": "Phi-3 (3.8B parameters, 32 layers, 3072 hidden)",
            "Task": "Text generation / chat",
            "Context length": "4096 tokens",
            "Quantization": "INT4 RTN block-32 acc-level-4",
            "Original framework": "PyTorch (transformers)",
        },
        license_text=(
            "This model is licensed under the "
            "[MIT License](https://opensource.org/licenses/MIT). "
            "Original model by [Microsoft](https://huggingface.co/microsoft)."
        ),
    )


register(Phi3Mini())
