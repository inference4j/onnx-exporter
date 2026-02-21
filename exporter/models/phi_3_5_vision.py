from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register

_PREFIX = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"


class Phi35Vision(MirroredModel):
    name = "phi-3.5-vision"
    repo_id = "inference4j/phi-3.5-vision-instruct"
    source_repo = "microsoft/Phi-3.5-vision-instruct-onnx"
    source_type = "hf"
    files = [
        FileMapping(
            src=f"{_PREFIX}/genai_config.json",
            dst="genai_config.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/processor_config.json",
            dst="processor_config.json",
        ),
        FileMapping(
            src=f"{_PREFIX}/phi-3.5-v-instruct-text.onnx",
            dst="phi-3.5-v-instruct-text.onnx",
        ),
        FileMapping(
            src=f"{_PREFIX}/phi-3.5-v-instruct-text.onnx.data",
            dst="phi-3.5-v-instruct-text.onnx.data",
        ),
        FileMapping(
            src=f"{_PREFIX}/phi-3.5-v-instruct-embedding.onnx",
            dst="phi-3.5-v-instruct-embedding.onnx",
        ),
        FileMapping(
            src=f"{_PREFIX}/phi-3.5-v-instruct-embedding.onnx.data",
            dst="phi-3.5-v-instruct-embedding.onnx.data",
        ),
        FileMapping(
            src=f"{_PREFIX}/phi-3.5-v-instruct-vision.onnx",
            dst="phi-3.5-v-instruct-vision.onnx",
        ),
        FileMapping(
            src=f"{_PREFIX}/phi-3.5-v-instruct-vision.onnx.data",
            dst="phi-3.5-v-instruct-vision.onnx.data",
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
        title="Phi-3.5-vision-instruct — ONNX (INT4)",
        description=(
            "INT4-quantized ONNX export of "
            "[Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct), "
            "a 4.2B-parameter multimodal vision-language model from Microsoft. "
            "Accepts images and text prompts, generates text output. "
            "Optimized for CPU inference with int4 RTN block-32 quantization."
        ),
        license="mit",
        pipeline_tag="image-text-to-text",
        tags=["image-text-to-text", "phi-3", "vision", "multimodal", "onnx", "int4", "cpu"],
        original_source_url="https://huggingface.co/microsoft/Phi-3.5-vision-instruct-onnx",
        original_author="Microsoft",
        java_usage="""\
try (VisionLanguageModel vision = VisionLanguageModel.builder()
        .model(ModelSources.phi3Vision())
        .build()) {
    GenerationResult result = vision.describe(Path.of("photo.jpg"));
    System.out.println(result.text());
}""",
        model_details={
            "Architecture": "Phi-3.5 Vision (4.2B parameters — CLIP ViT encoder + MLP projector + Phi-3 decoder)",
            "Task": "Image description, visual Q&A, multimodal chat",
            "Context length": "128K tokens",
            "Quantization": "INT4 RTN block-32 acc-level-4",
            "ONNX files": "3 models (vision encoder, embedding projector, text decoder)",
            "Original framework": "PyTorch (transformers)",
        },
        license_text=(
            "This model is licensed under the "
            "[MIT License](https://opensource.org/licenses/MIT). "
            "Original model by [Microsoft](https://huggingface.co/microsoft)."
        ),
    )


register(Phi35Vision())
