from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class GteBase(MirroredModel):
    name = "gte-base"
    repo_id = "inference4j/gte-base"
    source_repo = "Xenova/gte-base"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.txt", dst="vocab.txt"),
    ]
    card = ModelCard(
        title="GTE Base — ONNX",
        description="ONNX export of [thenlper/gte-base](https://huggingface.co/thenlper/gte-base), an English embedding model from Alibaba. Maps sentences to 768-dimensional dense vectors using CLS pooling with L2 normalization.",
        license="mit",
        pipeline_tag="sentence-similarity",
        tags=["gte", "sentence-transformers", "sentence-similarity", "feature-extraction"],
        original_source_url="https://huggingface.co/thenlper/gte-base",
        original_author="thenlper/Alibaba (ONNX by Xenova)",
        java_usage="""\
try (SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
        .modelId("inference4j/gte-base")
        .poolingStrategy(PoolingStrategy.CLS)
        .normalize()
        .build()) {
    float[] embedding = model.encode("Hello, world!");
    System.out.println("Dimension: " + embedding.length); // 768
}""",
        model_details={
            "Architecture": "BERT Base (12 layers, 768 hidden)",
            "Task": "Sentence embeddings / semantic similarity",
            "Output dimension": "768",
            "Pooling": "CLS",
            "Normalization": "L2",
            "Max sequence length": "512",
            "Original framework": "PyTorch (HuggingFace Transformers)",
        },
        license_text="This model is licensed under the [MIT License](https://opensource.org/licenses/MIT). Original model by [thenlper/Alibaba](https://huggingface.co/thenlper/gte-base), ONNX export by [Xenova](https://huggingface.co/Xenova).",
    )


register(GteBase())
