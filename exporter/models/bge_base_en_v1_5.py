from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class BgeBaseEnV15(MirroredModel):
    name = "bge-base-en-v1.5"
    repo_id = "inference4j/bge-base-en-v1.5"
    source_repo = "Xenova/bge-base-en-v1.5"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.txt", dst="vocab.txt"),
    ]
    card = ModelCard(
        title="BGE Base EN v1.5 — ONNX",
        description="ONNX export of [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), a high-quality English embedding model. Maps sentences to 768-dimensional dense vectors using CLS pooling with L2 normalization.",
        license="mit",
        pipeline_tag="sentence-similarity",
        tags=["bge", "sentence-transformers", "sentence-similarity", "feature-extraction"],
        original_source_url="https://huggingface.co/BAAI/bge-base-en-v1.5",
        original_author="BAAI (ONNX by Xenova)",
        java_usage="""\
try (SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
        .modelId("inference4j/bge-base-en-v1.5")
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
            "MTEB average": "63.55",
            "Max sequence length": "512",
            "Original framework": "PyTorch (HuggingFace Transformers)",
        },
        license_text="This model is licensed under the [MIT License](https://opensource.org/licenses/MIT). Original model by [BAAI](https://huggingface.co/BAAI/bge-base-en-v1.5), ONNX export by [Xenova](https://huggingface.co/Xenova).",
    )


register(BgeBaseEnV15())
