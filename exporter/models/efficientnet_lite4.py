from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class EfficientnetLite4(MirroredModel):
    name = "efficientnet-lite4"
    repo_id = "inference4j/efficientnet-lite4"
    source_repo = "onnx/EfficientNet-Lite4"
    source_type = "hf"
    files = [
        FileMapping(src="efficientnet-lite4-11.onnx", dst="model.onnx"),
    ]
    card = ModelCard(
        title="EfficientNet-Lite4 — ONNX",
        description="ONNX export of [EfficientNet-Lite4](https://huggingface.co/onnx/EfficientNet-Lite4), a lightweight and efficient image classification model optimized for mobile/edge deployment. Trained on ImageNet with 1000-class output.",
        license="apache-2.0",
        pipeline_tag="image-classification",
        tags=["efficientnet", "image-classification", "imagenet", "computer-vision"],
        original_source_url="https://huggingface.co/onnx/EfficientNet-Lite4",
        original_author="ONNX",
        java_usage="""\
try (EfficientNet model = EfficientNet.fromPretrained("models/efficientnet-lite4")) {
    List<Classification> results = model.classify(Path.of("cat.jpg"));
    results.forEach(c -> System.out.printf("%s: %.2f%%%n", c.label(), c.score() * 100));
}""",
        model_details={
            "Architecture": "EfficientNet-Lite4 (compound-scaled CNN)",
            "Task": "Image classification (ImageNet 1000 classes)",
            "Input": "`[batch, 224, 224, 3]` — RGB, pixel values 0-255",
            "Output": "`[batch, 1000]` — class probabilities",
            "ONNX opset": "11",
            "Original framework": "TensorFlow Lite → ONNX",
        },
        license_text="This model is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Original model from [ONNX](https://huggingface.co/onnx).",
    )


register(EfficientnetLite4())
