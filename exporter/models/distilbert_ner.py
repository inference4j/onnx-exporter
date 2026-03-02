from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class DistilbertNer(MirroredModel):
    name = "distilbert-NER"
    repo_id = "inference4j/distilbert-NER"
    source_repo = "onnx-community/distilbert-NER-ONNX"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.txt", dst="vocab.txt"),
        FileMapping(src="config.json", dst="config.json"),
    ]
    card = ModelCard(
        title="DistilBERT NER — ONNX",
        description="ONNX export of [dslim/distilbert-NER](https://huggingface.co/dslim/distilbert-NER), a DistilBERT model fine-tuned on CoNLL-2003 for Named Entity Recognition. Identifies persons, organizations, locations, and miscellaneous entities in text using IOB2 tagging.",
        license="apache-2.0",
        pipeline_tag="token-classification",
        tags=["distilbert", "ner", "named-entity-recognition", "token-classification", "conll2003"],
        original_source_url="https://huggingface.co/dslim/distilbert-NER",
        original_author="dslim (ONNX by onnx-community)",
        java_usage="""\
try (BertNerRecognizer ner = BertNerRecognizer.builder()
        .modelId("inference4j/distilbert-NER")
        .build()) {
    List<NamedEntity> entities = ner.recognize("John works at Google in London.");
    for (NamedEntity e : entities) {
        System.out.printf("%s (%s)%n", e.text(), e.label());
    }
    // John (PER)
    // Google (ORG)
    // London (LOC)
}""",
        model_details={
            "Architecture": "DistilBERT (6 layers, 768 hidden, 66M params)",
            "Task": "Named Entity Recognition (IOB2 tagging)",
            "Labels": "O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC",
            "Training data": "CoNLL-2003",
            "F1 score": "92.17",
            "Max sequence length": "512",
            "Tokenizer": "WordPiece (cased)",
            "Original framework": "PyTorch (HuggingFace Transformers)",
        },
        license_text="This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Original model by [dslim](https://huggingface.co/dslim/distilbert-NER), ONNX export by [onnx-community](https://huggingface.co/onnx-community).",
    )


register(DistilbertNer())
