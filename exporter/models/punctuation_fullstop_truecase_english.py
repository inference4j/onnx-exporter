from pathlib import Path

from exporter.base import FileMapping, MirroredModel, ModelCard
from exporter.registry import register

SPM_FILE = "spe_32k_lc_en.model"

# Expected IDs come from the official sentencepiece library. tokenizer.json must
# reproduce every one of them, or the export fails.
PARITY_CASES = {
    "hello world": [8015, 140],
    "marie curie moved from poland to paris": [6608, 7330, 1345, 873, 34, 3835, 7, 1767],
    "numbers 1891 and 8 years": [989, 455, 7271, 10, 540, 104],
    "zxqv unseenword": [2449, 885, 3300, 1359, 15487, 21943],
    "naïve café": [2792, 0, 106, 14459],  # ï is <unk>: no byte fallback
    "<s> literal": [101, 0, 8, 0, 19196],  # control pieces never match text
    "<<s>> </s><pad> <unk>": [101, 0, 8, 0, 101, 0, 8, 0, 14408, 0, 101, 0, 18487, 0],
}

# sentencepiece's remove_extra_whitespaces collapses and trims runs of spaces before
# tokenizing; a Metaspace pre-tokenizer keeps each space as its own "▁". Callers must
# collapse whitespace themselves. Checked and reported, but not fatal.
KNOWN_DIVERGENCES = {
    "  leading and   multiple   spaces  ": [970, 10, 1147, 3685],
}


def build_tokenizer_json(spm_path: Path, dst: Path) -> None:
    """Convert the SentencePiece Unigram model to a tokenizers Unigram tokenizer.json."""
    from sentencepiece import sentencepiece_model_pb2
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import Unigram

    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(spm_path.read_bytes())
    if proto.trainer_spec.model_type != sentencepiece_model_pb2.TrainerSpec.UNIGRAM:
        raise ValueError(f"{spm_path.name} is not a Unigram model")
    if proto.normalizer_spec.name != "identity":
        raise ValueError(f"unexpected normalizer {proto.normalizer_spec.name!r}")

    normal = sentencepiece_model_pb2.ModelProto.SentencePiece.NORMAL
    # sentencepiece never matches control pieces (<s>, </s>, <pad>) against text, but a
    # tokenizers Unigram lattice considers the whole vocab and would emit <s> for a
    # literal "<s>". No normal piece contains "<", so it is always <unk> upstream;
    # rewriting it to another out-of-vocab character keeps the IDs identical while
    # making the control pieces unreachable.
    if any("<" in p.piece for p in proto.pieces if p.type == normal):
        raise ValueError("a normal piece contains '<'; the control-piece guard is unsafe")

    tok = Tokenizer(Unigram([(p.piece, p.score) for p in proto.pieces],
                            unk_id=proto.trainer_spec.unk_id, byte_fallback=False))
    tok.normalizer = normalizers.Replace("<", "\ufffd")
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tok.save(str(dst))


def check_parity(spm_path: Path, tokenizer_json: Path) -> None:
    """Fail unless tokenizer.json encodes exactly like sentencepiece."""
    import sentencepiece as spm
    from tokenizers import Tokenizer

    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    tok = Tokenizer.from_file(str(tokenizer_json))

    failures = []
    for text, expected in {**PARITY_CASES, **KNOWN_DIVERGENCES}.items():
        reference = sp.encode(text)
        if reference != expected:
            failures.append(f"{text!r}: sentencepiece gave {reference}, expected {expected}")
            continue
        got = tok.encode(text).ids
        if got == reference:
            print(f"    ok        {text!r} -> {got}")
        elif text in KNOWN_DIVERGENCES:
            print(f"    DIVERGES  {text!r}: sentencepiece {reference}, tokenizers {got}")
        else:
            failures.append(f"{text!r}: sentencepiece {reference}, tokenizers {got}")
    if failures:
        raise RuntimeError("tokenizer.json parity failed:\n  " + "\n  ".join(failures))


class PunctuationFullstopTruecaseEnglish(MirroredModel):
    name = "punctuation-fullstop-truecase-english"
    repo_id = "inference4j/punctuation-fullstop-truecase-english"
    source_repo = "1-800-BAD-CODE/punctuation_fullstop_truecase_english"
    source_type = "hf"
    files = [
        FileMapping(src="punct_cap_seg_en.onnx", dst="model.onnx"),
        FileMapping(src=SPM_FILE, dst=SPM_FILE),
        FileMapping(src="config.yaml", dst="config.yaml"),
    ]

    def stage(self, staging_dir: Path) -> None:
        super().stage(staging_dir)
        tok_dst = staging_dir / "tokenizer.json"
        print(f"  Building tokenizer.json from {SPM_FILE} (Unigram)...")
        build_tokenizer_json(staging_dir / SPM_FILE, tok_dst)
        print("  Checking tokenizer.json parity against sentencepiece...")
        check_parity(staging_dir / SPM_FILE, tok_dst)
    card = ModelCard(
        title="Punctuation, True-casing & Sentence Boundary Detection (English) — ONNX",
        description="Mirror of [1-800-BAD-CODE/punctuation_fullstop_truecase_english](https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english). Takes lower-cased, unpunctuated English text and, in a single pass, restores punctuation, true-cases words (including acronyms like `U.S.` and mixed-case words like `McDonald's`), and detects sentence boundaries.",
        license="apache-2.0",
        pipeline_tag="token-classification",
        tags=["punctuation", "true-casing", "sentence-boundary-detection", "token-classification", "nlp"],
        original_source_url="https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english",
        original_author="1-800-BAD-CODE",
        java_usage="// Java wrapper coming in a future inference4j release.",
        model_details={
            "Architecture": "Transformer encoder (6 layers, d_model 512) + punctuation, sentence-boundary and true-case heads",
            "Task": "Punctuation restoration, true-casing, sentence boundary detection",
            "Tokenizer": "SentencePiece Unigram, 32k vocab, lower-cased (`tokenizer.json`, converted from `spe_32k_lc_en.model`); BOS = 1, EOS = 2, PAD = 3, UNK = 0",
            "Max sequence length": "256 subtokens including BOS/EOS",
            "Input": "`input_ids` — `[batch, seq]` int64, `[BOS] + pieces + [EOS]`",
            "Outputs": "`pre_preds`, `post_preds`, `cap_preds`, `seg_preds` (see below)",
            "Punctuation labels": "`<NULL>`, `<ACRONYM>`, `.`, `,`, `?`",
            "Training data": "WMT News Crawl (~10M lines, 2012 and 2021)",
            "Original framework": "NeMo (fork), exported to ONNX by the author",
        },
        license_text="This model is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). Original model and ONNX export by [1-800-BAD-CODE](https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english).",
        extra_sections="""\
## Files

| File | Description |
| --- | --- |
| `model.onnx` | The model (upstream `punct_cap_seg_en.onnx`) |
| `tokenizer.json` | Hugging Face `tokenizers` Unigram tokenizer, converted from `spe_32k_lc_en.model` |
| `spe_32k_lc_en.model` | Original SentencePiece tokenizer, kept for provenance |
| `config.yaml` | Label sets and max length |

## Outputs

All outputs are argmaxed/thresholded inside the graph; no softmax is needed.

| Name | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `pre_preds` | `[batch, seq]` | int64 | Pre-token punctuation index into `pre_labels`. Always `<NULL>` for English. |
| `post_preds` | `[batch, seq]` | int64 | Post-token punctuation index into `post_labels`: `0` none, `1` acronym (period after every character), `2` `.`, `3` `,`, `4` `?` |
| `cap_preds` | `[batch, seq, 16]` | bool | Per-character upper-case flag for each subtoken; entries past the subtoken's length are ignored |
| `seg_preds` | `[batch, seq]` | bool | Sentence boundary after this subtoken |

## Preprocessing

1. Lower-case the input, strip punctuation, and collapse runs of whitespace to single
   spaces with no leading or trailing space. SentencePiece does this itself;
   `tokenizer.json` does not, and emits extra `▁` tokens for repeated spaces.
2. Encode with `tokenizer.json` (or SentencePiece), then wrap as `[1] + ids + [2]`.
   Do not let `<s>`, `</s>`, `<pad>` or `<unk>` typed in the text match their IDs;
   SentencePiece encodes them as ordinary characters, with `<` and `>` as `<unk>`.
3. Inputs longer than 254 pieces must be split into windows; the
   [punctuators](https://github.com/1-800-BAD-CODE/punctuators) package uses overlapping windows and fuses the results

## Postprocessing

Ignore the BOS/EOS positions. For each subtoken:

1. Upper-case the characters flagged in `cap_preds`. Indices cover the raw piece
   **including** the SentencePiece `▁` word marker, so for `▁marie` index 1 is `m`.
2. Append the punctuation from `post_preds`; for `<ACRONYM>`, put a period after every character.
3. Start a new sentence wherever `seg_preds` is true.

Example: `marie curie moved from poland to paris and later worked with the us radium institute` →
`Marie Curie moved from Poland to Paris and later worked with the U.S. Radium Institute.`""",
    )


register(PunctuationFullstopTruecaseEnglish())
