from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class AllMiniLML12V2(MirroredModel):
    name = "all-MiniLM-L12-v2"
    repo_id = "inference4j/all-MiniLM-L12-v2"
    source_repo = "sentence-transformers/all-MiniLM-L12-v2"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.txt", dst="vocab.txt"),
    ]
    card = ModelCard(
        title="all-MiniLM-L12-v2 — ONNX",
        description="ONNX export of [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2), a sentence-transformers model that maps sentences to a 384-dimensional dense vector space. Deeper variant (12 layers) of MiniLM with slightly better quality at the cost of speed.",
        license="apache-2.0",
        pipeline_tag="sentence-similarity",
        tags=["sentence-transformers", "sentence-similarity", "feature-extraction"],
        original_source_url="https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2",
        original_author="sentence-transformers",
        java_usage="""\
try (SentenceTransformer model = SentenceTransformer.fromPretrained("models/all-MiniLM-L12-v2")) {
    float[] embedding = model.encode("Hello, world!");
    System.out.println("Dimension: " + embedding.length); // 384
}""",
        model_details={
            "Architecture": "MiniLM-L12 (12 layers, 384 hidden)",
            "Task": "Sentence embeddings / semantic similarity",
            "Output dimension": "384",
            "Max sequence length": "256",
            "Training data": "1B+ sentence pairs",
            "Original framework": "PyTorch (sentence-transformers)",
        },
        license_text="This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Original model by [sentence-transformers](https://huggingface.co/sentence-transformers).",
    )


register(AllMiniLML12V2())
