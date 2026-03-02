from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class BertBaseNer(MirroredModel):
    name = "bert-base-NER"
    repo_id = "inference4j/bert-base-NER"
    source_repo = "Xenova/bert-base-NER"
    source_type = "hf"
    files = [
        FileMapping(src="onnx/model.onnx", dst="model.onnx"),
        FileMapping(src="vocab.txt", dst="vocab.txt"),
        FileMapping(src="config.json", dst="config.json"),
    ]
    card = ModelCard(
        title="BERT Base NER — ONNX",
        description="ONNX export of [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER), a BERT model fine-tuned on CoNLL-2003 for Named Entity Recognition. Identifies persons, organizations, locations, and miscellaneous entities in text using IOB2 tagging.",
        license="mit",
        pipeline_tag="token-classification",
        tags=["bert", "ner", "named-entity-recognition", "token-classification", "conll2003"],
        original_source_url="https://huggingface.co/dslim/bert-base-NER",
        original_author="dslim (ONNX by Xenova)",
        java_usage="""\
try (BertNerRecognizer ner = BertNerRecognizer.builder()
        .modelId("inference4j/bert-base-NER")
        .build()) {
    List<NamedEntity> entities = ner.recognize("John works at Google in London.");
    for (NamedEntity e : entities) {
        System.out.printf("%s (%s)%n", e.text(), e.label());
    }
}""",
        model_details={
            "Architecture": "BERT Base (12 layers, 768 hidden, 110M params)",
            "Task": "Named Entity Recognition (IOB2 tagging)",
            "Labels": "O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC",
            "Training data": "CoNLL-2003",
            "F1 score": "91.3",
            "Max sequence length": "512",
            "Tokenizer": "WordPiece (cased)",
            "Original framework": "PyTorch (HuggingFace Transformers)",
        },
        license_text="This model is licensed under the [MIT License](https://opensource.org/licenses/MIT). Original model by [dslim](https://huggingface.co/dslim/bert-base-NER), ONNX export by [Xenova](https://huggingface.co/Xenova).",
    )


register(BertBaseNer())
