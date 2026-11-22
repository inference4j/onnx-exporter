"""Shared tokenizer conversion for the MarianMT (opus-mt) exports."""

import json
from pathlib import Path

MARIAN_SPECIAL_TOKENS = ("</s>", "<unk>", "<pad>")


def build_marian_tokenizer(vocab_json: Path, source_spm: Path, out: Path) -> None:
    """Write a Unigram tokenizer.json that reproduces MarianTokenizer's input ids.

    MarianTokenizer segments text with source.spm (a SentencePiece Unigram model with its own
    32k ids) and then maps each piece to the model's id through vocab.json, the shared
    source+target vocabulary the embeddings use. A tokenizer.json taken straight from
    source.spm therefore has the wrong ids, and BPE merges derived from a Unigram model do not
    reproduce its Viterbi segmentation. This builds a Unigram model whose ids are vocab.json's
    and whose scores are source.spm's, so the same segmentation maps to the same ids. The same
    file decodes target ids, since the vocabulary is shared.
    """
    from sentencepiece import sentencepiece_model_pb2
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
    from tokenizers.models import Unigram

    with open(vocab_json, encoding="utf-8") as f:
        vocab = json.load(f)
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(Path(source_spm).read_bytes())

    by_id = sorted(vocab.items(), key=lambda kv: kv[1])
    if [i for _, i in by_id] != list(range(len(by_id))):
        raise ValueError(f"{vocab_json}: ids are not contiguous from 0")

    scores = {p.piece: p.score for p in proto.pieces if p.type == sentencepiece_model_pb2.ModelProto.SentencePiece.NORMAL}
    # SentencePiece scores an unknown character min_score - 10, and MarianTokenizer then looks
    # its surface up in vocab.json, so target-only single characters get that score. Longer
    # target-only pieces never come out of source.spm and must not win the Viterbi search.
    unk_score = min(scores.values()) - 10.0
    never = min(scores.values()) - 10000.0

    def score(token: str) -> float:
        if token in MARIAN_SPECIAL_TOKENS:
            return 0.0
        if token in scores:
            return scores[token]
        return unk_score if len(token) == 1 else never

    tokenizer = Tokenizer(Unigram([(t, score(t)) for t, _ in by_id], unk_id=vocab["<unk>"], byte_fallback=False))
    # source.spm normalizes with nmt_nfkc and removes leading, trailing and repeated whitespace
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Precompiled(proto.normalizer_spec.precompiled_charsmap),
        normalizers.Strip(left=True, right=True),
        normalizers.Replace(Regex(" {2,}"), " "),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    tokenizer.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in MARIAN_SPECIAL_TOKENS])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>", pair="$A $B </s>", special_tokens=[("</s>", vocab["</s>"])])
    tokenizer.save(str(out))
