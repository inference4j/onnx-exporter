import pytest

from exporter import registry, rtmo_builder, upload
from exporter.base import ExportedModel

import exporter.models  # noqa: F401

SIZES = ("s", "m", "l")


@pytest.mark.parametrize("size", SIZES)
def test_rtmo_registration_pins_the_coco_only_checkpoint(size):
    model = registry.get(f"rtmo-{size}-coco")

    assert isinstance(model, ExportedModel)
    assert model.repo_id == f"inference4j/rtmo-{size}-coco"
    assert model.source_type == "url"
    [mapping] = model.files
    assert mapping.dst == "model.onnx"
    assert mapping.src == model.source_repo
    assert mapping.src.startswith("https://download.openmmlab.com/mmpose/v1/projects/rtmo/")
    assert f"rtmo-{size}_" in mapping.src
    assert "_coco-640x640-" in mapping.src
    assert "body7" not in mapping.src
    # The zoo's file names carry the first 8 hex digits of the sha256.
    assert mapping.sha256 is not None and len(mapping.sha256) == 64
    assert mapping.src.rsplit("-", 1)[-1].split("_")[0] == mapping.sha256[:8]
    assert model.mmpose_config.startswith("body_2d_keypoint/rtmo/coco/")


@pytest.mark.parametrize("size", SIZES)
def test_rtmo_is_locked_by_hash_and_never_copied_server_side(size):
    model = registry.get(f"rtmo-{size}-coco")

    assert not upload._can_copy_server_side(model)
    assert upload._resolve_revision(model, update=False, api=None) is None


@pytest.mark.parametrize("size", SIZES)
def test_rtmo_card_documents_license_data_and_contract(size):
    card = registry.get(f"rtmo-{size}-coco").render_card()

    assert "license: apache-2.0" in card
    assert "pipeline_tag: keypoint-detection" in card
    assert "Exported for use with" in card
    assert "COCO 2017 train person keypoints only" in card
    assert "body7" in card  # explains why the stock ONNX is not used
    assert "**BGR**" in card
    assert "pad right and bottom with 114" in card
    assert "`dets`: `[batch, N, 5]`" in card
    assert "`keypoints`: `[batch, N, 17, 3]`" in card
    assert "padding row with score 0" in card
    assert "MMDeploy 1.3.1" in card
    assert "nose, left_eye, right_eye" in card


def test_rtmo_requirements_are_fully_pinned():
    pins = rtmo_builder._pins()
    names = {p.split("==")[0].lower() for p in pins}

    assert all("==" in p for p in pins)
    assert {"torch", "mmpose", "mmengine", "mmdet", "onnx", "numpy"} <= names
    assert "torch==2.1.2" in pins
    assert "mmpose==1.3.2" in pins
    assert "mmcv" not in names  # built from source separately
    assert rtmo_builder.MMCV_SDIST.endswith("mmcv-2.1.0.tar.gz")
    assert len(rtmo_builder.MMDEPLOY_SHA256) == 64


def test_contract_check_rejects_a_graph_inference4j_cannot_read(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from exporter.models.rtmo import _check_contract

    def graph(outputs):
        nodes = [helper.make_node("Identity", ["input"], [name]) for name, _ in outputs]
        return helper.make_model(helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 640, 640])],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
        ), opset_imports=[helper.make_opsetid("", 11)])

    good = tmp_path / "good.onnx"
    onnx.save(graph([("dets", ["batch", "n", 5]), ("keypoints", ["batch", "n", 17, 3])]), good)
    _check_contract(good)

    bad = tmp_path / "bad.onnx"
    onnx.save(graph([("dets", ["batch", "n", 5]), ("keypoints", ["batch", "n", 26, 3])]), bad)
    with pytest.raises(AssertionError):
        _check_contract(bad)
