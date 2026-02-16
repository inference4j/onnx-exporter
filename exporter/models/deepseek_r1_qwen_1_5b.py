from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register

_PREFIX = "deepseek-r1-distill-qwen-1.5B/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"


class DeepSeekR1Qwen1_5B(MirroredModel):
    name = "deepseek-r1-distill-qwen-1.5b"
    repo_id = "inference4j/deepseek-r1-distill-qwen-1.5b"
    source_repo = "onnxruntime/DeepSeek-R1-Distill-ONNX"
    source_type = "hf"
    files = [
        FileMapping(
            src=f"{_PREFIX}/genai_config.json",
            dst="genai_config.json",
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
    ]
    card = ModelCard(
        title="DeepSeek-R1-Distill-Qwen-1.5B — ONNX (INT4)",
        description=(
            "INT4-quantized ONNX export of "
            "[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), "
            "a 1.5B-parameter reasoning model distilled from DeepSeek-R1. "
            "Optimized for CPU inference with int4 RTN block-32 quantization."
        ),
        license="mit",
        pipeline_tag="text-generation",
        tags=["text-generation", "deepseek", "reasoning", "onnx", "int4", "cpu"],
        original_source_url="https://huggingface.co/onnxruntime/DeepSeek-R1-Distill-ONNX",
        original_author="DeepSeek / Microsoft",
        java_usage="""\
try (TextGenerator gen = TextGenerator.builder()
        .modelSource(ModelSources.deepSeekR1_1_5B())
        .build()) {
    GenerationResult result = gen.generate("What is 2 + 2? Think step by step.");
    System.out.println(result.text());
}""",
        model_details={
            "Architecture": "Qwen2 (1.5B parameters, 28 layers, 1536 hidden)",
            "Task": "Text generation / reasoning",
            "Context length": "131072 tokens",
            "Quantization": "INT4 RTN block-32 acc-level-4",
            "Original framework": "PyTorch (transformers)",
        },
        license_text=(
            "This model is licensed under the "
            "[MIT License](https://opensource.org/licenses/MIT). "
            "Original model by [DeepSeek](https://huggingface.co/deepseek-ai), "
            "ONNX conversion by [Microsoft](https://huggingface.co/onnxruntime)."
        ),
    )


register(DeepSeekR1Qwen1_5B())
