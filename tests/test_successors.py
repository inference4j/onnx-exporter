"""Tests for successor detection and contract diffing."""

from exporter.contract import Contract, TensorSpec, diff
from exporter.successors import parse_family


class TestParseFamily:
    def test_v_prefixed_generation(self):
        f = parse_family("depth-anything-v2-small")
        assert (f.stem, f.raw_version, f.rest) == ("depth-anything", "2", "-small")

    def test_attached_decimal_generation(self):
        f = parse_family("Qwen2.5-1.5B-Instruct")
        assert (f.stem, f.version) == ("Qwen", 2.5)

    def test_attached_integer_generation(self):
        f = parse_family("SmolLM2-1.7B-Instruct")
        assert (f.stem, f.version) == ("SmolLM", 2.0)

    def test_hyphen_separated_generation(self):
        f = parse_family("gemma-2-2b-it")
        assert (f.stem, f.version) == ("gemma", 2.0)

    def test_generation_with_no_separator_before_variant(self):
        f = parse_family("yolo26n-ONNX")
        assert (f.stem, f.version) == ("yolo", 26.0)

    def test_unversioned_names_yield_nothing(self):
        for name in ("flan-t5-base", "whisper-small", "opus-mt-en-fr"):
            assert parse_family(name) is None, name

    def test_variant_tokens_drive_same_variant_matching(self):
        small = parse_family("depth-anything-v2-small")
        base = parse_family("depth-anything-v2-base")
        assert small.variant_tokens == {"small"}
        assert base.variant_tokens == {"base"}
        assert small.variant_tokens != base.variant_tokens


class TestContractDiff:
    def _spec(self, name, shape, dtype="tensor(float)"):
        return TensorSpec(name=name, dtype=dtype, shape=shape)

    def test_identical_contracts_have_no_problems(self):
        c = Contract(inputs=[self._spec("pixel_values", ["b", "3", "h", "w"])],
                     outputs=[self._spec("predicted_depth", ["b", "h", "w"])])
        assert diff(c, c) == []

    def test_rank_change_is_breaking(self):
        """The real depth-anything v2 -> v3 case: v3 is multi-view, so rank grows."""
        v2 = Contract(inputs=[self._spec("pixel_values", ["b", "3", "h", "w"])],
                      outputs=[self._spec("predicted_depth", ["b", "h", "w"])])
        v3 = Contract(inputs=[self._spec("pixel_values", ["b", "n", "3", "h", "w"])],
                      outputs=[self._spec("predicted_depth", ["b", "n", "h", "w"])])
        problems = diff(v2, v3)
        breaking = [p for p in problems if p.startswith("BREAKING")]
        assert len(breaking) == 2
        assert any("rank 4 -> 5" in p for p in breaking)
        assert any("rank 3 -> 4" in p for p in breaking)

    def test_removed_output_is_breaking(self):
        old = Contract(inputs=[], outputs=[self._spec("predicted_depth", ["b", "h", "w"])])
        new = Contract(inputs=[], outputs=[self._spec("depth", ["b", "h", "w"])])
        problems = diff(old, new)
        assert any("BREAKING" in p and "predicted_depth" in p for p in problems)
        assert any("new output 'depth'" in p for p in problems)

    def test_added_output_alone_is_not_breaking(self):
        old = Contract(inputs=[], outputs=[self._spec("predicted_depth", ["b", "h", "w"])])
        new = Contract(inputs=[], outputs=[self._spec("predicted_depth", ["b", "h", "w"]),
                                           self._spec("confidence", ["b", "h", "w"])])
        problems = diff(old, new)
        assert problems and not [p for p in problems if p.startswith("BREAKING")]

    def test_dtype_change_is_breaking(self):
        old = Contract(inputs=[self._spec("x", ["b"], "tensor(float)")], outputs=[])
        new = Contract(inputs=[self._spec("x", ["b"], "tensor(float16)")], outputs=[])
        assert any(p.startswith("BREAKING") and "dtype" in p for p in diff(old, new))

    def test_symbolic_dim_rename_is_not_breaking(self):
        """`height` -> `h` is a naming change in the export, not a contract change."""
        old = Contract(inputs=[self._spec("x", ["batch_size", "height"])], outputs=[])
        new = Contract(inputs=[self._spec("x", ["b", "h"])], outputs=[])
        problems = diff(old, new)
        assert not [p for p in problems if p.startswith("BREAKING")]

    def test_fixed_dim_change_is_breaking(self):
        old = Contract(inputs=[self._spec("x", ["1", "384"])], outputs=[])
        new = Contract(inputs=[self._spec("x", ["1", "768"])], outputs=[])
        assert any(p.startswith("BREAKING") for p in diff(old, new))
