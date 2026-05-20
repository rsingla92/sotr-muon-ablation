# Phase 1 plan and verification gate

## Goal

Add structured reasoning tokens before each move, supervised from KataGo's
analysis output. Show that CoT either improves strength or, at minimum,
produces legible reasoning at no strength cost.

## Scaffold already in place (pre-Phase-1 work)

- **`gogpt/cot_vocab.py`** -- ~55 structured think-block tokens (winrate,
  score, phase, group status, tactics, shapes, confidence, structural
  markers, Phase-2 reflect/revise placeholders), assigned into the
  reserved slots 87..286 of the tokenizer.
- **`gogpt/cot_extractor.py`** -- mechanically translates a (board,
  KataGo analysis dict) into a list of think-block tokens. Average
  length 8-15 tokens.
- **`scripts/extract_cot_labels.py`** -- offline pipeline turning
  `generate_selfplay.py`'s SGF + JSONL outputs into sharded NPZ tensors
  ready for training.
- **`gogpt/concepts.py`** -- rule library backing both the extractor's
  weak-group/tactic detection and Phase-3 feature verification.

## Phase-1-specific work remaining

### 1. Head expansion (architectural change)

The Phase-0 model has a 82-way move-logit head. Phase 1 needs the model
to predict think-block tokens too, so the head must be enlarged to the
full ``VOCAB_SIZE`` (currently 287).

Implementation sketch in `gogpt/model.py`:

```python
# Replace
self.move_head = nn.Linear(cfg.d_model, cfg.num_move_outputs, bias=False)
# With
self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
```

Loss in `train.py` already uses cross-entropy with -100 ignore -- only
the head shape changes. Inference becomes: at any position, mask the
logit slice to the legal next-token type (think-block tokens inside
`[<think>]...[</think>]`, move tokens after `[</think>]`).

For 30M / 150M models this adds ~150k extra parameters; negligible.

### 2. CoT-aware inference

`gogpt/inference.py` currently exposes `generate_move` and
`best_of_n_sample`. Phase 1 needs:
- `generate_with_cot(model, board, ...)`: greedy-or-sampled generation
  of `[<think>]` ... `[</think>]` followed by one move token. Masks
  next-token logits by region.
- KV-cache reuse across the N samples of `best_of_n_sample` once CoT
  trajectories get long enough to matter. (For 9x9 + ~15-token CoT,
  batched-from-scratch is fine; revisit if Phase-2's reflect-and-revise
  pushes trajectories to 60+ tokens.)

### 3. Four-way ablation training (the central experiment)

All four at 150M params, matched compute:
- A: No-CoT (think blocks present but empty)
- B: Structured-CoT (template tokens from `cot_extractor`)
- C: Natural-language-CoT (LLM-rewritten subset mixed with B)
- D: Free-CoT (think blocks present, no supervision inside them)

Models B and D differ only in whether the loss is computed over the
think-block tokens. Implementation: an additional config flag
``mask_think_block_in_loss: bool``.

### 4. Optional NL-CoT rewriter (`scripts/rewrite_cot_natural.py`)

10-20% of training data: take a structured think-block and rewrite as
prose using a local model (Llama-3-8B-Instruct via vLLM). Train jointly
with structured CoT; model learns both. SKIP if it adds complexity for
marginal benefit -- the structured CoT is sufficient for the
interpretability story.

### 5. CoT inspection tool

A script that loads a model, plays a game vs KataGo at low visits, and
prints the CoT alongside each move (using `gogpt.cot_vocab.token_name`
for human-readable names and `gogpt.render.board_to_ascii` for the
position).

## Verification gate

- [ ] **CoT extraction validation.** For 1000 randomly sampled positions,
  the extractor's output parses cleanly, all token IDs are valid, and
  re-querying KataGo yields the same top move in >=95% of cases and the
  same score-lead bin in >=90%.
- [ ] **CoT data quality audit.** A Go-knowledgeable human (you, or a
  friend) reviews 50 random structured CoT samples and confirms they're
  plausible descriptions of the position. Saved to
  `docs/cot_audit.md`.
- [ ] **All four models train to convergence** with no instabilities;
  final val loss reported for each.
- [ ] **Strength comparison vs baseline.** Each 150M model plays >=100
  games vs the Phase-0 30M baseline and >=100 games vs KataGo at varied
  visits; Elo reported.
- [ ] **CoT-helps test.** Structured-CoT (B) achieves strictly higher win
  rate vs KataGo than no-CoT (A) at matched compute, OR the gap is
  within noise but qualitative analysis shows B's CoT is legible and
  useful. Document which.
- [ ] **Free-CoT comparison.** D vs A: does giving the model scratchpad
  capacity without supervision help? Report the delta.
- [ ] **Inspection tool.** Loads a model, plays 5 games, prints CoT
  alongside each move; CoT is legible.
- [ ] **Generation legality.** Across 1000 inference calls, model
  produces a legal move >=99.5% of the time. Illegal moves are remapped
  to "pass" and logged.
- [ ] **Tag and document.** `git tag phase-1-complete`. Write-up: "Does
  CoT help a Go transformer?" with the four-way comparison plot.

## Open design questions

- Whether `[<think>]` blocks during training should be variable-length
  (the extractor's natural output) or padded to a fixed length per
  position (simpler for batching but wastes tokens). Default: variable
  length, pack via standard right-padding in the dataloader.
- Whether to include the predicted top move INSIDE the think block as
  `[TOP_MOVE] <vertex>` (current design) or AFTER it. Including-inside
  is the bridge between "thought" and "action" the interpretability
  story wants; AFTER is closer to standard LLM CoT. We pick INSIDE.
- Color-flip semantics: extractor takes `flip_ownership=True` for white
  to-move positions. Verify this matches the always-as-black training
  pipeline in `gogpt/data.py`.
