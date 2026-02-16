from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class MsMarcoMiniLM(MirroredModel):
    name = "ms-marco-MiniLM-L-6-v2"
    repo_id = "inference4j/ms-marco-MiniLM-L-6-v2"
    source_repo = "Xenova/ms-marco-MiniLM-L-6-v2"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.txt", dst="vocab.txt"),
    ]
    card = ModelCard(
        title="ms-marco-MiniLM-L-6-v2 (Cross-Encoder) — ONNX",
        description="ONNX export of [ms-marco-MiniLM-L-6-v2](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2), a cross-encoder model trained on MS MARCO passage ranking data. Scores query-document pairs for search result reranking.",
        license="apache-2.0",
        pipeline_tag="text-classification",
        tags=["cross-encoder", "text-classification", "reranking", "ms-marco", "search"],
        original_source_url="https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2",
        original_author="Xenova (originally cross-encoder by sentence-transformers)",
        java_usage="""\
try (MiniLMReranker reranker = MiniLMReranker.fromPretrained("models/ms-marco-MiniLM-L-6-v2")) {
    float score = reranker.score("What is Java?", "Java is a programming language.");
    System.out.printf("Relevance score: %.4f%n", score);
}""",
        model_details={
            "Architecture": "MiniLM-L6 cross-encoder (6 layers, 384 hidden)",
            "Task": "Query-document relevance scoring / search reranking",
            "Training data": "MS MARCO passage ranking",
            "Max sequence length": "512",
            "Output": "Single relevance score per query-document pair",
            "Original framework": "PyTorch (sentence-transformers cross-encoder)",
            "ONNX export": "By Xenova (Transformers.js)",
        },
        license_text="This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Original model by [sentence-transformers](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2), ONNX export by [Xenova](https://huggingface.co/Xenova).",
    )


register(MsMarcoMiniLM())
