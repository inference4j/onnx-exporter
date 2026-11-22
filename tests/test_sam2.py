from exporter import registry, upload
from exporter.base import MirroredModel

import exporter.models  # noqa: F401


def test_sam21_hiera_tiny_registration_and_files():
    model = registry.get("sam2.1-hiera-tiny")

    assert isinstance(model, MirroredModel)
    assert model.repo_id == "inference4j/sam2.1-hiera-tiny"
    assert model.source_repo == "onnx-community/sam2.1-hiera-tiny-ONNX"
    assert model.revision == "814a066640debee5a91e70aa401fb8e17e030503"
    assert [(mapping.src, mapping.dst) for mapping in model.files] == [
        ("onnx/vision_encoder.onnx", "vision_encoder.onnx"),
        ("onnx/vision_encoder.onnx_data", "vision_encoder.onnx_data"),
        (
            "onnx/prompt_encoder_mask_decoder.onnx",
            "prompt_encoder_mask_decoder.onnx",
        ),
        (
            "onnx/prompt_encoder_mask_decoder.onnx_data",
            "prompt_encoder_mask_decoder.onnx_data",
        ),
        ("config.json", "config.json"),
        ("preprocessor_config.json", "preprocessor_config.json"),
        ("processor_config.json", "processor_config.json"),
    ]


def test_sam21_hiera_tiny_uses_server_side_copy():
    model = registry.get("sam2.1-hiera-tiny")
    assert upload._can_copy_server_side(model)


def test_sam21_hiera_tiny_card_documents_external_data_and_license():
    card = registry.get("sam2.1-hiera-tiny").render_card()

    assert "license: apache-2.0" in card
    assert "pipeline_tag: mask-generation" in card
    assert "Meta" in card
    assert "onnx-community" in card
    assert "vision_encoder.onnx_data" in card
    assert "prompt_encoder_mask_decoder.onnx_data" in card
    assert "deliberately not" in card
    assert "mask_input" in card


def test_sam21_declared_revision_is_used_before_first_lock_entry(monkeypatch):
    monkeypatch.setattr(upload.lock, "get", lambda _name: None)
    model = registry.get("sam2.1-hiera-tiny")

    assert upload._resolve_revision(model, update=False, api=None) == model.revision
