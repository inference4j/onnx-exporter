from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register


class Resnet50V17(MirroredModel):
    name = "resnet50-v1-7"
    repo_id = "inference4j/resnet50-v1-7"
    source_repo = "onnxmodelzoo/resnet50-v1-7"
    source_type = "hf"
    files = [
        FileMapping(src="resnet50-v1-7.onnx", dst="model.onnx"),
    ]
    card = ModelCard(
        title="ResNet-50 v1.7 — ONNX",
        description="ONNX export of [ResNet-50](https://huggingface.co/onnxmodelzoo/resnet50-v1-7), a deep residual network for image classification trained on ImageNet. 50-layer variant with 1000-class output.",
        license="mit",
        pipeline_tag="image-classification",
        tags=["resnet", "image-classification", "imagenet", "computer-vision"],
        original_source_url="https://huggingface.co/onnxmodelzoo/resnet50-v1-7",
        original_author="ONNX Model Zoo",
        java_usage="""\
try (ResNet model = ResNet.fromPretrained("models/resnet50")) {
    List<Classification> results = model.classify(Path.of("cat.jpg"));
    results.forEach(c -> System.out.printf("%s: %.2f%%%n", c.label(), c.score() * 100));
}""",
        model_details={
            "Architecture": "ResNet-50 (50 layers, residual connections)",
            "Task": "Image classification (ImageNet 1000 classes)",
            "Input": "`[batch, 3, 224, 224]` — RGB, ImageNet-normalized",
            "Output": "`[batch, 1000]` — class probabilities",
            "Original framework": "ONNX Model Zoo",
        },
        license_text="This model is licensed under the [MIT License](https://opensource.org/licenses/MIT). Original model from the [ONNX Model Zoo](https://github.com/onnx/models).",
    )


register(Resnet50V17())
