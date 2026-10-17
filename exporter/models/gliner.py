from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.licenses import APACHE_LICENSE_2_0
from exporter.registry import register

_JAVA_USAGE = """\
try (var ner = GlinerEntityRecognizer.builder().build()) {
    List<NamedEntity> entities = ner.recognize(
            text, List.of("person", "organization", "invoice id"));
}"""

_COMMON_FILES = [
    FileMapping(src="onnx/model.onnx", dst="model.onnx"),
    FileMapping(src="gliner_config.json", dst="gliner_config.json"),
    FileMapping(src="tokenizer.json", dst="tokenizer.json"),
    FileMapping(src="tokenizer_config.json", dst="tokenizer_config.json"),
    FileMapping(src="special_tokens_map.json", dst="special_tokens_map.json"),
    FileMapping(src="added_tokens.json", dst="added_tokens.json"),
]


class GlinerSmallV21(MirroredModel):
    name = "gliner-small-v2.1"
    repo_id = "inference4j/gliner-small-v2.1"
    source_repo = "onnx-community/gliner_small-v2.1"
    source_type = "hf"
    revision = "8142fb00740ccea973e64b1272949ff48653df5e"
    files = _COMMON_FILES
    card = ModelCard(
        title="GLiNER small v2.1 — ONNX",
        description=(
            "FP32 ONNX mirror of "
            "[onnx-community/gliner_small-v2.1]"
            "(https://huggingface.co/onnx-community/gliner_small-v2.1), parity-checked "
            "against the Apache-2.0 PyTorch weights from "
            "[urchade/gliner_small-v2.1](https://huggingface.co/urchade/gliner_small-v2.1). "
            "GLiNER performs zero-shot named-entity recognition using labels supplied at inference time."
        ),
        license="apache-2.0",
        pipeline_tag="token-classification",
        tags=["gliner", "ner", "zero-shot"],
        original_source_url="https://huggingface.co/urchade/gliner_small-v2.1",
        original_author="Urchade Zaratiana et al.",
        java_usage=_JAVA_USAGE,
        model_details={
            "Architecture": "GLiNER small v2.1 (DeBERTa-v3-small backbone)",
            "Task": "Zero-shot named-entity recognition",
            "ONNX source revision": "`8142fb00740ccea973e64b1272949ff48653df5e`",
            "PyTorch parity revision": "`4e091416cf7c3481db542c2a3d26156916f3a47f`",
            "Span mode": "`markerV0`",
            "Maximum span width": "12 words",
            "Maximum text length": "384 words",
            "Word splitting": "Whitespace",
            "Subtoken pooling": "First subtoken",
            "Prompt tokens": "`<<ENT>>`, `<<SEP>>`",
            "Inputs": "Five int64 tensors plus boolean `span_mask`; see the contract below",
            "Primary output": "Raw float32 `logits` `[batch, words, max_width, labels]`; apply sigmoid when decoding",
            "Auxiliary outputs": "`prompts_embedding`, `2973`, `onnx::Shape_3287`, `onnx::Cast_3115`",
        },
        license_text=APACHE_LICENSE_2_0,
        extra_sections="""\
## ONNX input contract

| Input | Dtype | Shape |
|-------|-------|-------|
| `input_ids` | int64 | `[batch, sequence]` |
| `attention_mask` | int64 | `[batch, sequence]` |
| `words_mask` | int64 | `[batch, sequence]` |
| `text_lengths` | int64 | `[batch, 1]` |
| `span_idx` | int64 | `[batch, words * 12, 2]` |
| `span_mask` | bool | `[batch, words * 12]` |

## Credits

GLiNER was created by Urchade Zaratiana and contributors. See the original paper,
*GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer*,
and the upstream repository for training and implementation details.
""",
    )


class GlinerPiiBaseV10(MirroredModel):
    name = "gliner-pii-base-v1.0"
    repo_id = "inference4j/gliner-pii-base-v1.0"
    source_repo = "knowledgator/gliner-pii-base-v1.0"
    source_type = "hf"
    revision = "61726e0ad791dcab3e29339bbec3ad42ded65641"
    files = _COMMON_FILES
    card = ModelCard(
        title="GLiNER PII base v1.0 — ONNX",
        description=(
            "FP32 ONNX mirror of "
            "[knowledgator/gliner-pii-base-v1.0]"
            "(https://huggingface.co/knowledgator/gliner-pii-base-v1.0), a GLiNER model "
            "specialized for zero-shot detection of personally identifiable information."
        ),
        license="apache-2.0",
        pipeline_tag="token-classification",
        tags=["gliner", "ner", "zero-shot", "pii"],
        original_source_url="https://huggingface.co/knowledgator/gliner-pii-base-v1.0",
        original_author="Knowledgator and Urchade Zaratiana et al.",
        java_usage=_JAVA_USAGE,
        model_details={
            "Architecture": "GLiNER PII base v1.0",
            "Task": "Zero-shot PII / named-entity recognition",
            "Source revision": "`61726e0ad791dcab3e29339bbec3ad42ded65641`",
            "Span mode": "`markerV0`",
            "Maximum span width": "12 words",
            "Maximum text length": "2048 words",
            "Word splitting": "Whitespace",
            "Subtoken pooling": "First subtoken",
            "Prompt tokens": "`<<ENT>>`, `<<SEP>>`",
            "Inputs": "Five int64 tensors plus boolean `span_mask`; see the contract below",
            "Output": "Raw float32 `logits` `[batch, words, max_width, labels]`; apply sigmoid when decoding",
        },
        license_text=APACHE_LICENSE_2_0,
        extra_sections="""\
## ONNX input contract

| Input | Dtype | Shape |
|-------|-------|-------|
| `input_ids` | int64 | `[batch, sequence]` |
| `attention_mask` | int64 | `[batch, sequence]` |
| `words_mask` | int64 | `[batch, sequence]` |
| `text_lengths` | int64 | `[batch, 1]` |
| `span_idx` | int64 | `[batch, words * 12, 2]` |
| `span_mask` | bool | `[batch, words * 12]` |

## Credits

This PII-specialized checkpoint was published by Knowledgator and is based on GLiNER,
created by Urchade Zaratiana and contributors.

## PII detection disclaimer

PII detection is probabilistic. This model can miss sensitive information or produce false
positives and is not a guarantee that text has been fully de-identified. Use defense in depth,
appropriate review, and controls suited to the risk of the data being processed.
""",
    )


register(GlinerSmallV21())
register(GlinerPiiBaseV10())
