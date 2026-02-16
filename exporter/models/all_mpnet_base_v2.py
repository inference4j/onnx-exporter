from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class AllMpnetBaseV2(MirroredModel):
    name = "all-mpnet-base-v2"
    repo_id = "inference4j/all-mpnet-base-v2"
    source_repo = "sentence-transformers/all-mpnet-base-v2"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.txt", dst="vocab.txt"),
    ]
    card = ModelCard(
        title="all-mpnet-base-v2 — ONNX",
        description="ONNX export of [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), the highest-quality sentence-transformers model based on MPNet. Maps sentences to a 768-dimensional dense vector space.",
        license="apache-2.0",
        pipeline_tag="sentence-similarity",
        tags=["sentence-transformers", "sentence-similarity", "feature-extraction", "mpnet"],
        original_source_url="https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
        original_author="sentence-transformers",
        java_usage="""\
try (SentenceTransformer model = SentenceTransformer.fromPretrained("models/all-mpnet-base-v2")) {
    float[] embedding = model.encode("Hello, world!");
    System.out.println("Dimension: " + embedding.length); // 768
}""",
        model_details={
            "Architecture": "MPNet-base (12 layers, 768 hidden)",
            "Task": "Sentence embeddings / semantic similarity",
            "Output dimension": "768",
            "Max sequence length": "384",
            "Training data": "1B+ sentence pairs",
            "Original framework": "PyTorch (sentence-transformers)",
        },
        license_text="This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Original model by [sentence-transformers](https://huggingface.co/sentence-transformers).",
    )


register(AllMpnetBaseV2())
