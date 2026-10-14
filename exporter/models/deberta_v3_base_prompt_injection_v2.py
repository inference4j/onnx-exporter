from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class DebertaV3BasePromptInjectionV2(MirroredModel):
    name = "deberta-v3-base-prompt-injection-v2"
    repo_id = "inference4j/deberta-v3-base-prompt-injection-v2"
    source_repo = "protectai/deberta-v3-base-prompt-injection-v2"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="onnx/tokenizer.json", dst="tokenizer.json"),
        FileMapping(src="onnx/config.json", dst="config.json"),
        FileMapping(src="onnx/tokenizer_config.json", dst="tokenizer_config.json"),
        FileMapping(src="onnx/special_tokens_map.json", dst="special_tokens_map.json"),
    ]
    card = ModelCard(
        title="deberta-v3-base-prompt-injection-v2 — ONNX",
        description="ONNX export of [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2), a prompt injection detection model fine-tuned on DeBERTa-v3-base to detect prompt injections and jailbreak attempts.",
        license="apache-2.0",
        pipeline_tag="text-classification",
        tags=["prompt-injection", "deberta-v3", "security", "guardrails"],
        original_source_url="https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2",
        original_author="Protect AI",
        java_usage="""\
try (PromptInjectionDetector guard = DebertaPromptInjectionDetector.builder().build()) {
    InjectionCheck check = guard.check(userInput);
}""",
        model_details={
            "Architecture": "DeBERTa-v3-base (model_type: deberta-v2)",
            "Task": "Text classification (prompt injection detection)",
            "Classes": "0: SAFE, 1: INJECTION",
            "Max sequence length": "512",
            "Vocabulary size": "128,100",
            "Original framework": "PyTorch (Transformers)",
        },
        license_text="This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Original model by [Protect AI](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2).",
    )


register(DebertaV3BasePromptInjectionV2())
