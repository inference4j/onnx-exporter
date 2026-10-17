from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register

_JAVA_USAGE = """\
try (var ocr = OcrPipeline.builder().build()) {
    List<TextRegion> lines = ocr.read(image);
}"""

_LICENSE_TEXT = (
    "This model is licensed under the [Apache License 2.0]"
    "(https://www.apache.org/licenses/LICENSE-2.0). Original model by "
    "[PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR)."
)

_FILES = [
    FileMapping(src="inference.onnx", dst="model.onnx"),
    FileMapping(src="inference.yml", dst="inference.yml"),
]


class PpOcrV6SmallDet(MirroredModel):
    name = "pp-ocrv6-small-det"
    repo_id = "inference4j/pp-ocrv6-small-det"
    source_repo = "PaddlePaddle/PP-OCRv6_small_det_onnx"
    source_type = "hf"
    files = list(_FILES)
    card = ModelCard(
        title="PP-OCRv6 Small Det — ONNX",
        description=(
            "ONNX export of [PaddlePaddle/PP-OCRv6_small_det_onnx]"
            "(https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx), "
            "the small text detection model from PaddleOCR PP-OCRv6. "
            "Designed for robust, real-time multilingual text detection."
        ),
        license="apache-2.0",
        pipeline_tag="object-detection",
        tags=["ocr", "paddleocr", "pp-ocrv6", "text-detection"],
        original_source_url="https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx",
        original_author="PaddlePaddle",
        java_usage=_JAVA_USAGE,
        model_details={
            "Architecture": "PP-OCRv6 Small Detection (DBNet)",
            "Task": "Text detection",
            "Input format": "[batch, 3, H, W] (BGR, dynamic shape)",
            "Original framework": "PaddlePaddle (PP-OCRv6)",
            "Technical report": "[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR)",
        },
        license_text=_LICENSE_TEXT,
    )


class PpOcrV6SmallRec(MirroredModel):
    name = "pp-ocrv6-small-rec"
    repo_id = "inference4j/pp-ocrv6-small-rec"
    source_repo = "PaddlePaddle/PP-OCRv6_small_rec_onnx"
    source_type = "hf"
    files = list(_FILES)
    card = ModelCard(
        title="PP-OCRv6 Small Rec — ONNX",
        description=(
            "ONNX export of [PaddlePaddle/PP-OCRv6_small_rec_onnx]"
            "(https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx), "
            "the small text recognition model from PaddleOCR PP-OCRv6. "
            "Features a CTC head covering 18,708 characters for general text recognition."
        ),
        license="apache-2.0",
        pipeline_tag="image-to-text",
        tags=["ocr", "paddleocr", "pp-ocrv6", "text-recognition"],
        original_source_url="https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx",
        original_author="PaddlePaddle",
        java_usage=_JAVA_USAGE,
        model_details={
            "Architecture": "PP-OCRv6 Small Recognition (SVTR-CTC)",
            "Task": "Text recognition",
            "Vocabulary size": "18,708 characters (+ blank and space)",
            "Input format": "[batch, 3, 48, W] (BGR, height 48, dynamic width)",
            "Output format": "[batch, time, 18710] (CTC softmax probabilities)",
            "Original framework": "PaddlePaddle (PP-OCRv6)",
            "Technical report": "[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR)",
        },
        license_text=_LICENSE_TEXT,
    )


class PpOcrV6TinyDet(MirroredModel):
    name = "pp-ocrv6-tiny-det"
    repo_id = "inference4j/pp-ocrv6-tiny-det"
    source_repo = "PaddlePaddle/PP-OCRv6_tiny_det_onnx"
    source_type = "hf"
    files = list(_FILES)
    card = ModelCard(
        title="PP-OCRv6 Tiny Det — ONNX",
        description=(
            "ONNX export of [PaddlePaddle/PP-OCRv6_tiny_det_onnx]"
            "(https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx), "
            "the lightweight tiny text detection model from PaddleOCR PP-OCRv6 (~1.8 MB). "
            "Designed for resource-constrained environments."
        ),
        license="apache-2.0",
        pipeline_tag="object-detection",
        tags=["ocr", "paddleocr", "pp-ocrv6", "text-detection"],
        original_source_url="https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx",
        original_author="PaddlePaddle",
        java_usage=_JAVA_USAGE,
        model_details={
            "Architecture": "PP-OCRv6 Tiny Detection (DBNet)",
            "Task": "Text detection",
            "Model size": "~1.8 MB",
            "Input format": "[batch, 3, H, W] (BGR, dynamic shape)",
            "Original framework": "PaddlePaddle (PP-OCRv6)",
            "Technical report": "[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR)",
        },
        license_text=_LICENSE_TEXT,
    )


class PpOcrV6TinyRec(MirroredModel):
    name = "pp-ocrv6-tiny-rec"
    repo_id = "inference4j/pp-ocrv6-tiny-rec"
    source_repo = "PaddlePaddle/PP-OCRv6_tiny_rec_onnx"
    source_type = "hf"
    files = list(_FILES)
    card = ModelCard(
        title="PP-OCRv6 Tiny Rec — ONNX",
        description=(
            "ONNX export of [PaddlePaddle/PP-OCRv6_tiny_rec_onnx]"
            "(https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx), "
            "the lightweight tiny text recognition model from PaddleOCR PP-OCRv6 (~4.5 MB). "
            "Features a compact 6,904-character CTC dictionary."
        ),
        license="apache-2.0",
        pipeline_tag="image-to-text",
        tags=["ocr", "paddleocr", "pp-ocrv6", "text-recognition"],
        original_source_url="https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        original_author="PaddlePaddle",
        java_usage=_JAVA_USAGE,
        model_details={
            "Architecture": "PP-OCRv6 Tiny Recognition (SVTR-CTC)",
            "Task": "Text recognition",
            "Model size": "~4.5 MB",
            "Vocabulary size": "6,904 characters (+ blank and space)",
            "Input format": "[batch, 3, 48, W] (BGR, height 48, dynamic width)",
            "Output format": "[batch, time, 6906] (CTC softmax probabilities)",
            "Original framework": "PaddlePaddle (PP-OCRv6)",
            "Technical report": "[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR)",
        },
        license_text=_LICENSE_TEXT,
    )


register(PpOcrV6SmallDet())
register(PpOcrV6SmallRec())
register(PpOcrV6TinyDet())
register(PpOcrV6TinyRec())
