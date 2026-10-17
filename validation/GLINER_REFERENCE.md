# GLiNER ONNX reference

These recordings use `gliner==0.2.29` and the pinned model revisions declared in
`exporter/models/gliner.py`. The complete worked-example tensors and per-text entity
outputs are generated as JSON by `validation/gliner_validate.py`:

```bash
python validation/gliner_validate.py gliner-small-v2.1 \
  --candidate-root /tmp/gliner-candidate --output /tmp/gliner-small.json
python validation/gliner_validate.py gliner-pii-base-v1.0 \
  --candidate-root /tmp/gliner-candidate --output /tmp/gliner-pii.json
```

## Contract

Both graphs accept:

| Input | Dtype | Dynamic shape |
|-------|-------|---------------|
| `input_ids` | int64 | `[batch_size, sequence_length]` |
| `attention_mask` | int64 | `[batch_size, sequence_length]` |
| `words_mask` | int64 | `[batch_size, sequence_length]` |
| `text_lengths` | int64 | `[batch_size, value]` (`value` is 1) |
| `span_idx` | int64 | `[batch_size, num_spans, idx]` (`idx` is 2) |
| `span_mask` | bool | `[batch_size, num_spans]` |

The primary output is raw float32 `logits`; it is not probabilities. At runtime its
shape is `[batch, num_words, max_width, num_labels]`. The small graph also exposes four
auxiliary outputs: `prompts_embedding`, `2973`, `onnx::Shape_3287`, and
`onnx::Cast_3115`. The PII graph exposes only `logits`.

For the worked example, both graphs produce logits shaped `[1, 12, 12, 4]`. Values
outside `[0, 1]` and exact sigmoid comparisons confirm these are raw logits.

## Preprocessing and decoding

The prompt is `<<ENT>> label` repeated in caller-provided label order, followed by
`<<SEP>>` and the whitespace-split text. GLiNER constructs that prompt in
[`BaseProcessor.prepare_inputs`](https://github.com/urchade/GLiNER/blob/v0.2.29/gliner/data_processing/processor.py#L138-L187)
and passes the resulting word list to the tokenizer with `is_split_into_words=True` in
[`tokenize_inputs`](https://github.com/urchade/GLiNER/blob/v0.2.29/gliner/data_processing/processor.py#L264-L289).

`words_mask` uses zero for prompt/special/continuation subtokens and one-based text word
indices on the selected first subtoken. The implementation and pooling behavior are in
[`prepare_word_mask`](https://github.com/urchade/GLiNER/blob/v0.2.29/gliner/data_processing/utils.py#L85-L171).

With `max_width=12`, spans are flattened in start-major, width-minor order as inclusive
`[start, start + width]` pairs. Twelve entries are emitted for every word start; entries
whose end exceeds the text are retained in `span_idx` and disabled by `span_mask`. See
[`prepare_span_idx`](https://github.com/urchade/GLiNER/blob/v0.2.29/gliner/data_processing/utils.py#L205-L241).

Decoding applies sigmoid to logits, keeps candidates with probability strictly greater
than the threshold, and converts width index `k` to the inclusive word span
`[start, start + k]`. See the
[`SpanDecoder`](https://github.com/urchade/GLiNER/blob/v0.2.29/gliner/decoding/decoder.py#L567-L643).
The public defaults are `threshold=0.5`, `flat_ner=true`, and `multi_label=false` in
[`predict_entities`](https://github.com/urchade/GLiNER/blob/v0.2.29/gliner/model.py#L2885-L2922).

Flat NER uses stable descending-score order, greedily accepts non-overlapping inclusive
word spans, then sorts accepted spans by start position. Equal scores therefore retain
candidate enumeration order. With `multi_label=false`, two labels on the exact same span
also conflict; with `multi_label=true`, exact-boundary labels may coexist, but other
overlaps still conflict. See
[`greedy_search`](https://github.com/urchade/GLiNER/blob/v0.2.29/gliner/decoding/decoder.py#L184-L225)
and [`has_overlapping`](https://github.com/urchade/GLiNER/blob/v0.2.29/gliner/decoding/utils.py#L5-L11).

Character offsets are recovered from the whitespace splitter's per-word start and
end-exclusive offsets. The returned entity text is `text[start:end]`.

## Worked example

Text:

```text
Alice Johnson paid invoice INV-2024-0042 to Acme Corp on 3 March.
```

Labels: `person`, `organization`, `invoice id`, `date`.

Words and offsets are:

| Word index | Text | Start | End exclusive |
|------------|------|-------|---------------|
| 0 | Alice | 0 | 5 |
| 1 | Johnson | 6 | 13 |
| 2 | paid | 14 | 18 |
| 3 | invoice | 19 | 26 |
| 4 | INV-2024-0042 | 27 | 40 |
| 5 | to | 41 | 43 |
| 6 | Acme | 44 | 48 |
| 7 | Corp | 49 | 53 |
| 8 | on | 54 | 56 |
| 9 | 3 | 57 | 58 |
| 10 | March | 59 | 64 |
| 11 | . | 64 | 65 |

The exact six input tensors—including all 144 `span_idx` pairs and all 144
`span_mask` values—are emitted under `worked_example.inputs` by the validator.

Top valid spans:

| Model | Start | Width index | Label | Logit | Sigmoid score |
|-------|-------|-------------|-------|-------|---------------|
| small | 0 | 1 | person | 4.0940142 | 0.9836012 |
| small | 6 | 1 | organization | 3.3460510 | 0.9659753 |
| small | 9 | 1 | date | 2.6265652 | 0.9325518 |
| small | 4 | 0 | invoice id | 1.8874927 | 0.8684694 |
| PII | 4 | 0 | invoice id | 4.1557140 | 0.9845673 |
| PII | 6 | 1 | organization | 2.7026644 | 0.9371837 |
| PII | 0 | 1 | person | 2.6686854 | 0.9351534 |
| PII | 9 | 1 | date | 1.8619643 | 0.8655257 |

## Parity results

Three texts and the same four labels were checked for each model. The third text contains
397 whitespace words, which exercises the small model's 384-word truncation path. The
PII model's pinned `max_len` is 2048, so the same text does not truncate there.

| Model | Threshold | Worst max absolute logit difference | Worst entity score difference | Span/text/label lists |
|-------|-----------|---------------------------------------|-------------------------------|-----------------------|
| small | 0.5 | `5.531311e-05` | `3.159046e-06` | identical |
| PII | 0.5 | `5.817413e-05` | `1.847744e-06` | identical |
| PII | 0.3 | `5.817413e-05` | `1.847744e-06` | identical |

For the worked example, both models return `Alice Johnson` as `person`,
`INV-2024-0042` as `invoice id`, `Acme Corp` as `organization`, and `3 March` as
`date`, with end-exclusive character offsets matching the table above.
