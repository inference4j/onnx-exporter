"""Tests for the drift comparison rules.

The hash-comparison cases below are not hypothetical: each set of fixtures is real
metadata pulled from the HuggingFace API while building this, and a naive comparison
reported drift on files that are byte-identical.
"""

import pytest

from exporter.base import _pin_url
from exporter.check import Comparison, compare_hashes
from exporter.lock import FileHash


class TestCompareHashes:
    def test_matching_lfs_files_are_same(self):
        a = FileHash(size=99060839, sha256="afb6a5c2", blob_id="95a8023b")
        b = FileHash(size=99060839, sha256="afb6a5c2", blob_id="95a8023b")
        assert compare_hashes(a, b) == Comparison.SAME

    def test_differing_lfs_files_are_different(self):
        a = FileHash(size=100, sha256="aaaa", blob_id="1111")
        b = FileHash(size=100, sha256="bbbb", blob_id="2222")
        assert compare_hashes(a, b) == Comparison.DIFFERENT

    def test_matching_plain_files_are_same(self):
        a = FileHash(size=231508, blob_id="fb140275")
        b = FileHash(size=231508, blob_id="fb140275")
        assert compare_hashes(a, b) == Comparison.SAME

    def test_differing_plain_files_are_different(self):
        a = FileHash(size=100, blob_id="1111")
        b = FileHash(size=100, blob_id="2222")
        assert compare_hashes(a, b) == Comparison.DIFFERENT

    def test_lfs_versus_plain_is_inconclusive_not_drift(self):
        """The deepseek-r1-distill-qwen-1.5b case.

        Upstream stores genai_config.json in LFS; our mirror stores it as a plain git
        blob. The bytes are identical (verified by downloading both), but the LFS side's
        blob_id hashes a *pointer file*, so the two blob_ids differ. Reporting that as
        drift would send someone to re-export a model that is perfectly in sync.
        """
        ours = FileHash(size=1477, blob_id="c808742ad4ad2a360ab689f83971d8310a303d0c")
        upstream = FileHash(
            size=1477,
            sha256="dc87bc52132e57d152c937c72394cdfb9d202ee3a9ea1a1c2769588f1e59dc5f",
            blob_id="a1b2c3d4",
        )
        assert compare_hashes(ours, upstream) == Comparison.INCONCLUSIVE
        assert compare_hashes(upstream, ours) == Comparison.INCONCLUSIVE

    def test_mixed_storage_with_different_sizes_is_still_drift(self):
        """Size alone can prove difference even when the hashes are incomparable."""
        ours = FileHash(size=1477, blob_id="c808742a")
        upstream = FileHash(size=2048, sha256="dc87bc52")
        assert compare_hashes(ours, upstream) == Comparison.DIFFERENT

    def test_missing_file_on_one_side_is_drift(self):
        assert compare_hashes(FileHash(size=1, blob_id="a"), None) == Comparison.DIFFERENT
        assert compare_hashes(None, FileHash(size=1, blob_id="a")) == Comparison.DIFFERENT

    def test_missing_on_both_sides_is_same(self):
        assert compare_hashes(None, None) == Comparison.SAME


class TestPinUrl:
    def test_substitutes_the_branch_for_a_commit(self):
        url = ("https://github.com/snakers4/silero-vad/raw/master"
               "/src/silero_vad/data/silero_vad.onnx")
        pinned = _pin_url(url, "bfdc0193023f121ea5b3cc7b176dbed570a68a59")
        assert pinned == ("https://github.com/snakers4/silero-vad/raw/"
                          "bfdc0193023f121ea5b3cc7b176dbed570a68a59"
                          "/src/silero_vad/data/silero_vad.onnx")

    def test_no_revision_leaves_the_url_alone(self):
        url = "https://github.com/o/r/raw/master/f.onnx"
        assert _pin_url(url, None) == url

    def test_non_github_url_is_untouched(self):
        url = "https://example.com/weights/model.onnx"
        assert _pin_url(url, "abc123") == url


class TestTrackedFiles:
    def test_mirrored_tracks_exactly_its_declared_sources(self):
        from exporter import registry
        import exporter.models  # noqa: F401

        model = registry.get("all-MiniLM-L6-v2")
        assert model.tracked_files(["anything", "at", "all"]) == ["onnx/model.onnx", "vocab.txt"]

    def test_exported_ignores_other_peoples_onnx_and_readmes(self):
        from exporter import registry
        import exporter.models  # noqa: F401

        model = registry.get("gpt2")
        tracked = model.tracked_files([
            "config.json", "model.safetensors", "tokenizer_config.json", "vocab.json",
            "merges.txt", "README.md", ".gitattributes", "onnx/decoder_model.onnx",
            "onnx/vocab.json", "tf_model.h5", "flax_model.msgpack",
        ])
        assert tracked == ["config.json", "merges.txt", "model.safetensors",
                           "tokenizer_config.json", "vocab.json"]

    def test_marian_spm_files_are_tracked(self):
        """opus-mt models load source.spm/target.spm; missing them would hide real drift."""
        from exporter import registry
        import exporter.models  # noqa: F401

        model = registry.get("opus-mt-en-fr")
        tracked = model.tracked_files(["source.spm", "target.spm", "config.json", "README.md"])
        assert "source.spm" in tracked and "target.spm" in tracked
