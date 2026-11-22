from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class NomicEmbedTextV15(MirroredModel):
    name = "nomic-embed-text-v1.5"
    repo_id = "inference4j/nomic-embed-text-v1.5"
    source_repo = "nomic-ai/nomic-embed-text-v1.5"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.txt", dst="vocab.txt"),
    ]
    card = ModelCard(
        title="Nomic Embed Text v1.5 — ONNX",
        description="ONNX export of [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), a long-context English embedding model. Maps text to 768-dimensional dense vectors using mean pooling, with Matryoshka training so the vectors can be truncated to 512, 256, 128 or 64 dimensions. Inputs need a task prefix such as `search_query: ` or `search_document: `.",
        license="apache-2.0",
        pipeline_tag="sentence-similarity",
        tags=["nomic", "nomic-bert", "sentence-transformers", "sentence-similarity", "feature-extraction", "matryoshka"],
        original_source_url="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
        original_author="Nomic AI",
        java_usage="""\
try (SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
        .modelId("inference4j/nomic-embed-text-v1.5")
        .textPrefix("search_query: ")
        .normalize()
        .build()) {
    float[] embedding = model.encode("What is the capital of France?");
    System.out.println("Dimension: " + embedding.length); // 768
}""",
        model_details={
            "Architecture": "NomicBERT (12 layers, 768 hidden, rotary positions, SwiGLU)",
            "Task": "Sentence embeddings / semantic search",
            "Output dimension": "768 (Matryoshka: 512, 256, 128, 64)",
            "Pooling": "Mean",
            "Normalization": "L2",
            "Task prefixes": "`search_query: `, `search_document: `, `clustering: `, `classification: `",
            "Max sequence length": "8192",
            "Original framework": "PyTorch (HuggingFace Transformers)",
        },
        license_text="This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Original model and ONNX export by [Nomic AI](https://huggingface.co/nomic-ai).",
    )


register(NomicEmbedTextV15())
