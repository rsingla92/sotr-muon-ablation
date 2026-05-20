# Natural-language CoT (Phase 1 mode C)

## What's wired today

`scripts/rewrite_cot_natural.py` reads structured-CoT NPZ shards
(`extract_cot_labels.py --mode structured`), samples a configurable fraction
of positions, and calls an LLM provider (Anthropic Claude or Google
Gemini) to translate each position's structured think-block into 1-2
sentences of prose. Output is one JSONL record per position:

```json
{
  "shard": "shard_000004.npz",
  "row": 173,
  "structured_tokens": [97, 103, 119, 136, 10, 122, 110, 139, 137, 40, 133],
  "structured_text": "WR_EVEN SL_B_TINY GRP_WEAK_2@B8 TAC_ATARI PH_MIDGAME SEP_FACTS TOP_MOVE C7 CONF_HIGH",
  "nl_text": "Black is roughly even with a sliver of a lead, but its group at B8 is on two liberties; playing C7 ataris the surrounding stones and resolves the threat.",
  "provider": "anthropic",
  "model": "claude-haiku-4-5",
  "input_tokens": 42,
  "output_tokens": 51
}
```

The script is **resumable** -- on rerun it reads the existing JSONL, builds
a set of `(shard, row)` keys already done, and skips them. Safe to ctrl-C
and re-launch.

## Provider choice

**Default: Gemini Flash 2.5** -- has a generous free tier (sufficient for
a single full pass over typical Phase 1 data volumes) and the quality is
adequate for this strictly-defined mechanical task.

**Anthropic fallback:** the code defaults to `claude-opus-4-7` (per the
Claude API skill default). For bulk rewriting of 10k-100k+ positions,
`--model claude-haiku-4-5` is the right pick: ~5x cheaper, more than
sufficient for translating a fixed-vocabulary tag stream into prose.
Sonnet 4.6 sits between the two if Haiku turns out underpowered (in
practice it should not on this task).

Approximate cost (Anthropic, ~50 output tokens per call, 15% sample rate
of 100k positions = 15k calls):

| Model               | Approx cost  |
| ------------------- | -----------: |
| claude-haiku-4-5    | ~$4          |
| claude-sonnet-4-6   | ~$12         |
| claude-opus-4-7     | ~$20         |
| gemini-2.5-flash    | $0 (free tier covers it) |

The script prints an estimate before any API calls and prompts for
confirmation above $5 unless `--yes` is passed.

## The tokenization bridge (the part that's NOT built yet)

The rewriter produces *text*. To train on it (Phase-1 mode C), we need
to turn that text into token sequences in a vocabulary the model
embeds. Two approaches:

### Option A: Custom BPE on the rewriter output (recommended)

Train a small Byte-Pair Encoding tokenizer on the corpus of `nl_text`
strings produced by the rewriter. Target vocab size: ~1500-3000 tokens.
Reserve a contiguous range past the current think-token slot for the
new BPE codes, e.g. token IDs 300-2300 (bumping
`RESERVED_THINK_VOCAB` in `gogpt/tokenizer.py` accordingly).

Pros: small embedding-table growth (~2k * d_model -- at d_model=768 that's
~1.5M extra params, < 1% of a 150M model); semantically grounded in the
actual NL distribution; preserves prefix-LM mask logic unchanged.

Cons: requires training the BPE; vocabularies for B and C variants
differ if you change the rewriter prompt.

Concretely:

1. Concatenate all `nl_text` from the rewriter output.
2. Train BPE with `tokenizers` library (Hugging Face) or
   `sentencepiece`, target vocab ~2000.
3. Persist the tokenizer to `data/cot/nl_bpe.json`.
4. Extend `gogpt/tokenizer.py` with `nl_token_to_id` / `id_to_nl_token`
   functions backed by the trained BPE.
5. Add a `--mode natural` branch to `extract_cot_labels.py` that:
   reads the structured shard, looks up `nl_text` from the rewriter
   JSONL by `(shard, row)`, tokenizes it via the BPE, and emits a CoT
   region containing those tokens (wrapped by `[<think>]` /
   `[</think>]`).

### Option B: Piggyback on Llama / GPT-2 tokenizer

Use an off-the-shelf BPE (Llama-3, GPT-2 -- ~50k or ~128k vocab).

Pros: no training; well-tested.

Cons: embedding table grows by 50-128k * d_model = 40-100M extra params
for a 150M model -- that's 25-60% of total. Wasteful and makes the
prefix-LM head significantly larger. Most tokens never used on Go-CoT
text.

**Recommendation:** Option A. The cost of training a small BPE is hours of
CPU; the run-time benefit is permanent.

## Where the integration would land

Concrete checklist for wiring NL-CoT into training (not done yet):

1. **`gogpt/tokenizer.py`** -- bump `RESERVED_THINK_VOCAB` from 200 to
   ~2200 to accommodate the BPE range; add `BPE_VOCAB_START` /
   `BPE_VOCAB_END` constants.
2. **New `gogpt/nl_tokenizer.py`** -- thin wrapper around the trained
   BPE that maps tokens to/from the reserved ID range.
3. **`scripts/train_nl_bpe.py`** -- one-time BPE trainer.
4. **`scripts/extract_cot_labels.py`** -- add `--mode natural` that
   reads the NL JSONL and substitutes the structured think-block with
   NL tokens (loss on all CoT tokens, like mode B).
5. **`gogpt/cot_vocab.py`** -- no changes; the BPE range is disjoint.
6. **Training config** -- a mode-C variant uses NL shards from step 4
   plus a fraction of structured shards (e.g. 20% NL, 80% structured).

Probably 1-2 days of work end-to-end once we have rewriter output.

## When to actually do this

The Phase 1 spec explicitly calls this out as optional:

> Skip this entirely if it adds complexity for marginal benefit -- structured
> CoT is sufficient for the interpretability story.

The headline experiment (does CoT help?) and the headline contribution
(faithfulness measurement) both work with structured CoT alone (mode B).
Mode C is a bonus dimension and a robustness check: does the model
generalize across CoT formats?

Do this if (a) structured CoT clearly helps and you want to test
generalization, OR (b) you want to publish prose CoTs alongside
structured ones in the writeup for legibility. Otherwise skip until
Phase 3 results justify the investment.
