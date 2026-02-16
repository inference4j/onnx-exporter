from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class DistilbertSst2(MirroredModel):
    name = "distilbert-base-uncased-finetuned-sst-2-english"
    repo_id = "inference4j/distilbert-base-uncased-finetuned-sst-2-english"
    source_repo = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.txt", dst="vocab.txt"),
        FileMapping(src="config.json", dst="config.json"),
    ]
    card = ModelCard(
        title="DistilBERT SST-2 — ONNX",
        description="ONNX export of [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english), a DistilBERT model fine-tuned on the Stanford Sentiment Treebank (SST-2) for binary sentiment classification (POSITIVE/NEGATIVE).",
        license="apache-2.0",
        pipeline_tag="text-classification",
        tags=["distilbert", "text-classification", "sentiment-analysis", "sst-2"],
        original_source_url="https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english",
        original_author="Xenova (originally distilbert-base by HuggingFace)",
        java_usage="""\
try (DistilBertClassifier model = DistilBertClassifier.fromPretrained("models/distilbert-sst2")) {
    List<TextClassification> results = model.classify("This movie was fantastic!");
    System.out.println(results.get(0).label()); // "POSITIVE"
    System.out.printf("Score: %.4f%n", results.get(0).score());
}""",
        model_details={
            "Architecture": "DistilBERT (6 layers, 768 hidden)",
            "Task": "Binary sentiment classification (POSITIVE / NEGATIVE)",
            "Training data": "SST-2 (Stanford Sentiment Treebank)",
            "Max sequence length": "512",
            "Original framework": "PyTorch (HuggingFace Transformers)",
            "ONNX export": "By Xenova (Transformers.js)",
        },
        license_text="This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Original model by [HuggingFace](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), ONNX export by [Xenova](https://huggingface.co/Xenova).",
    )


register(DistilbertSst2())
