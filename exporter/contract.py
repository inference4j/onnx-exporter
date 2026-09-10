"""ONNX input/output contract — what the Java wrapper actually depends on.

Weights can change freely; the Java side never notices. What breaks inference4j is a
change to the graph's *signature* — a renamed output, a lost dimension, a different
dtype. `DepthAnythingEstimator` reads the output named `predicted_depth` and expects
rank 3; a successor export emitting `depth` or `[1,1,H,W]` breaks it at runtime.

Comparing signatures is far cheaper than running the Java suite, so it belongs first in
the promotion path.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: list[str]

    def render(self) -> str:
        return f"{self.name}: {self.dtype}[{', '.join(self.shape)}]"


@dataclass
class Contract:
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]

    def render(self) -> str:
        lines = ["  inputs:"]
        lines += [f"    {s.render()}" for s in self.inputs]
        lines.append("  outputs:")
        lines += [f"    {s.render()}" for s in self.outputs]
        return "\n".join(lines)


def _spec(value) -> TensorSpec:
    shape = ["?" if d is None else str(d) for d in (value.shape or [])]
    return TensorSpec(name=value.name, dtype=str(value.type), shape=shape)


class ExternalDataMissing(Exception):
    """The graph references weights held in a sidecar file that isn't present."""


def external_data_files(onnx_path: Path) -> list[str]:
    """Sidecar weight files the graph references, if any.

    Large exports keep weights outside the .onnx. That matters to inference4j beyond
    loading: HuggingFaceModelSource downloads exactly the filenames a wrapper asks for,
    so a model split into `model.onnx` + `model.onnx_data` needs both listed or it
    resolves to a graph with no weights.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    names = set()
    for tensor in model.graph.initializer:
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    names.add(entry.value)
    return sorted(names)


def read(onnx_path: Path) -> Contract:
    """Read the signature of an ONNX file without running it."""
    import onnxruntime as ort

    missing = [
        name for name in external_data_files(onnx_path)
        if not (onnx_path.parent / name).exists()
    ]
    if missing:
        raise ExternalDataMissing(
            f"{onnx_path.name} keeps its weights in {', '.join(missing)}, "
            f"which is not alongside it in {onnx_path.parent}")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    with _materialized(onnx_path) as path:
        session = ort.InferenceSession(str(path), opts,
                                       providers=["CPUExecutionProvider"])
        return Contract(
            inputs=[_spec(v) for v in session.get_inputs()],
            outputs=[_spec(v) for v in session.get_outputs()],
        )


@contextmanager
def _materialized(onnx_path: Path):
    """Yield a path onnxruntime will accept for a model with external weights.

    The HuggingFace cache stores files as symlinks into a shared blob store, and
    onnxruntime refuses external data whose real path escapes the model directory.
    Copying graph and sidecars into one temp directory sidesteps that; models without
    external data are read in place.
    """
    sidecars = external_data_files(onnx_path)
    if not sidecars or not onnx_path.is_symlink():
        yield onnx_path
        return

    tmp = Path(tempfile.mkdtemp(prefix="onnx-contract-"))
    try:
        target = tmp / onnx_path.name
        shutil.copy2(onnx_path.resolve(), target)
        for name in sidecars:
            shutil.copy2((onnx_path.parent / name).resolve(), tmp / name)
        yield target
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def diff(current: Contract, candidate: Contract) -> list[str]:
    """Describe every signature change that could break a consumer."""
    problems = []

    for label, a, b in (("input", current.inputs, candidate.inputs),
                        ("output", current.outputs, candidate.outputs)):
        a_by_name = {s.name: s for s in a}
        b_by_name = {s.name: s for s in b}

        for name in a_by_name.keys() - b_by_name.keys():
            problems.append(f"BREAKING: {label} '{name}' no longer exists")
        for name in b_by_name.keys() - a_by_name.keys():
            problems.append(f"new {label} '{name}' appeared")

        for name in a_by_name.keys() & b_by_name.keys():
            old, new = a_by_name[name], b_by_name[name]
            if old.dtype != new.dtype:
                problems.append(
                    f"BREAKING: {label} '{name}' dtype {old.dtype} -> {new.dtype}")
            if len(old.shape) != len(new.shape):
                problems.append(
                    f"BREAKING: {label} '{name}' rank {len(old.shape)} -> {len(new.shape)} "
                    f"([{', '.join(old.shape)}] -> [{', '.join(new.shape)}])")
            elif old.shape != new.shape:
                fixed_changed = [
                    (i, o, n) for i, (o, n) in enumerate(zip(old.shape, new.shape))
                    if o != n and o.isdigit() and n.isdigit()
                ]
                severity = "BREAKING: " if fixed_changed else ""
                problems.append(
                    f"{severity}{label} '{name}' shape [{', '.join(old.shape)}] "
                    f"-> [{', '.join(new.shape)}]")

    return problems


def find_onnx(directory: Path) -> list[Path]:
    """The ONNX files in a staged directory, largest first."""
    files = sorted(directory.glob("*.onnx"), key=lambda p: p.stat().st_size, reverse=True)
    return files
