"""Structured chain-of-thought vocabulary.

The think-block sits between `[<think>]` and `[</think>]` in the trajectory
and is mechanically generated from KataGo analysis output for each
position. Average length: 8-15 tokens.

Token IDs are deterministically assigned into contiguous category blocks
inside the reserved range [PHASE0_VOCAB_SIZE, VOCAB_SIZE). Keeping IDs
clustered by category makes SAE feature slicing in Phase 3 much cleaner
(``ids[GRP_ALIVE:GRP_SEKI+1]`` gives every group-status token).

ID layout (starting at PHASE0_VOCAB_SIZE = 87):

    87 .. 97   : WR_BINS               (11)
    98 ..108   : score-lead SL_*       (11)
   109 ..112   : phase PH_*             (4)
   113 ..117   : group status GRP_*     (5)
   118 ..126   : tactics TAC_*          (9)
   127 ..132   : shapes SH_*            (6)
   133 ..135   : confidence CONF_*      (3)
   136 ..140   : structural             (5)
   141 ..144   : Phase-2 reflect/revise (4)
                                       ----
   Total used: 58 of the 200 reserved.

Think-block emission order is defined in ``cot_extractor.py`` (not here);
the ordering of declarations below is purely for ID layout.

Grammar (informal) of an emitted think-block:

    [<think>]
      <winrate-bin> <score-bin>             ;; state facts
      (<group-status> AT_VERTEX <vertex>)*   ;; up to 3 weak groups
      (<tactic>)*                            ;; up to 3 deduped tactics
      (<shape> AT_VERTEX <vertex>)*           ;; up to 2 shape observations
      <phase>                                 ;; phase comes AFTER grounding
      [SEP_FACTS]
      [TOP_MOVE] <vertex>                     ;; the played move
      <confidence>                            ;; winrate-gap based
    [</think>]
"""

from __future__ import annotations

from .tokenizer import PHASE0_VOCAB_SIZE, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Token-ID assignment
# ---------------------------------------------------------------------------

_NEXT_ID = PHASE0_VOCAB_SIZE


def _assign(name: str) -> int:
    global _NEXT_ID
    tid = _NEXT_ID
    _NEXT_ID += 1
    if tid >= VOCAB_SIZE:
        raise RuntimeError(
            f"think-vocab exhausted at {name!r}; bump RESERVED_THINK_VOCAB "
            f"in tokenizer.py (currently {VOCAB_SIZE - PHASE0_VOCAB_SIZE})"
        )
    return tid


# ===========================================================================
# Block 1: Winrate bins (11 tokens, IDs 87..97)
# WR_00..WR_09 cover [0.0, 0.1), [0.1, 0.2), ... [0.9, 1.0]
# WR_EVEN explicitly marks the [0.45, 0.55] band
# ===========================================================================

WR_BINS: list[int] = [_assign(f"WR_{i:02d}") for i in range(11)]


def winrate_bin_token(winrate: float) -> int:
    """Quantize a winrate (0..1, from side-to-move perspective) to a bin token."""
    if winrate < 0.0:
        winrate = 0.0
    if winrate > 1.0:
        winrate = 1.0
    if 0.45 <= winrate <= 0.55:
        return WR_BINS[10]  # WR_EVEN
    idx = min(int(winrate * 10), 9)
    return WR_BINS[idx]


# ===========================================================================
# Block 2: Score-lead bins (11 tokens, IDs 98..108)
# Tuned for 9x9 komi 7; rescale if porting to 19x19.
# ===========================================================================

SL_B_DOM = _assign("SL_B_DOM")      # >= +20
SL_B_BIG = _assign("SL_B_BIG")      # +10..+20
SL_B_MED = _assign("SL_B_MED")      # +5..+10
SL_B_SMALL = _assign("SL_B_SMALL")  # +2..+5
SL_B_TINY = _assign("SL_B_TINY")    # +0.5..+2
SL_EVEN = _assign("SL_EVEN")        # -0.5..+0.5
SL_W_TINY = _assign("SL_W_TINY")    # -2..-0.5
SL_W_SMALL = _assign("SL_W_SMALL")  # -5..-2
SL_W_MED = _assign("SL_W_MED")      # -10..-5
SL_W_BIG = _assign("SL_W_BIG")      # -20..-10
SL_W_DOM = _assign("SL_W_DOM")      # <= -20


def score_lead_token(score_lead: float) -> int:
    """KataGo score lead from the side-to-move perspective."""
    s = score_lead
    if s >= 20:
        return SL_B_DOM
    if s >= 10:
        return SL_B_BIG
    if s >= 5:
        return SL_B_MED
    if s >= 2:
        return SL_B_SMALL
    if s >= 0.5:
        return SL_B_TINY
    if s > -0.5:
        return SL_EVEN
    if s > -2:
        return SL_W_TINY
    if s > -5:
        return SL_W_SMALL
    if s > -10:
        return SL_W_MED
    if s > -20:
        return SL_W_BIG
    return SL_W_DOM


# ===========================================================================
# Block 3: Phase markers (4 tokens, IDs 109..112)
# ===========================================================================

PH_OPENING = _assign("PH_OPENING")     # moves 0..8
PH_MIDGAME = _assign("PH_MIDGAME")     # 9..30
PH_LATE_MID = _assign("PH_LATE_MID")   # 31..60
PH_ENDGAME = _assign("PH_ENDGAME")     # 61+


def phase_token(move_number: int) -> int:
    if move_number < 9:
        return PH_OPENING
    if move_number < 31:
        return PH_MIDGAME
    if move_number < 61:
        return PH_LATE_MID
    return PH_ENDGAME


# ===========================================================================
# Block 4: Group status (5 tokens, IDs 113..117)
# Always followed by AT_VERTEX <vertex> to bind to a board position.
# ===========================================================================

GRP_ALIVE = _assign("GRP_ALIVE")
GRP_WEAK_1 = _assign("GRP_WEAK_1")  # atari
GRP_WEAK_2 = _assign("GRP_WEAK_2")  # 2 liberties
GRP_DEAD = _assign("GRP_DEAD")
GRP_SEKI = _assign("GRP_SEKI")


def group_status_token(num_liberties: int, dead: bool, seki: bool) -> int:
    if seki:
        return GRP_SEKI
    if dead:
        return GRP_DEAD
    if num_liberties <= 1:
        return GRP_WEAK_1
    if num_liberties <= 2:
        return GRP_WEAK_2
    return GRP_ALIVE


# ===========================================================================
# Block 5: Tactic tokens (9 tokens, IDs 118..126)
# ===========================================================================

TAC_ATARI = _assign("TAC_ATARI")              # puts an opp group in atari
TAC_CAPTURE = _assign("TAC_CAPTURE")          # captures stones
TAC_KO = _assign("TAC_KO")                    # ko-capture
TAC_LADDER_RUN = _assign("TAC_LADDER_RUN")    # an own group is in a captured ladder
TAC_LADDER_BREAK = _assign("TAC_LADDER_BREAK")  # this move breaks a ladder
TAC_EYE_MAKE = _assign("TAC_EYE_MAKE")        # creates a new eye
TAC_INVASION = _assign("TAC_INVASION")        # plays into opp territory
TAC_REDUCTION = _assign("TAC_REDUCTION")      # reduces opp territory
TAC_DEFENSE = _assign("TAC_DEFENSE")          # defends an own weak group


# ===========================================================================
# Block 6: Shape tokens (6 tokens, IDs 127..132)
# Followed by AT_VERTEX <vertex>.
# ===========================================================================

SH_EYE = _assign("SH_EYE")
SH_BAMBOO = _assign("SH_BAMBOO")
SH_TIGER = _assign("SH_TIGER")
SH_HANE = _assign("SH_HANE")
SH_CUT = _assign("SH_CUT")
SH_CONNECT = _assign("SH_CONNECT")


# ===========================================================================
# Block 7: Confidence (3 tokens, IDs 133..135)
# Derived from the winrate gap between the played move and the runner-up,
# which is a more honest "policy confidence" signal than the visit-count
# ratio (KataGo runs to fixed visit budgets, so visit ratios mostly track
# search effort rather than the move's actual quality lead).
# ===========================================================================

CONF_HIGH = _assign("CONF_HIGH")
CONF_MED = _assign("CONF_MED")
CONF_LOW = _assign("CONF_LOW")


def confidence_token(
    top_winrate: float | None,
    runner_up_winrate: float | None,
) -> int:
    """Winrate-gap-based confidence.

    HIGH if the played move beats the runner-up by >= 10 percentage points;
    MED if 3-10 pp; LOW otherwise. If either input is None, return LOW
    (insufficient information).
    """
    if top_winrate is None or runner_up_winrate is None:
        return CONF_LOW
    gap = top_winrate - runner_up_winrate
    if gap >= 0.10:
        return CONF_HIGH
    if gap >= 0.03:
        return CONF_MED
    return CONF_LOW


# ===========================================================================
# Block 8: Structural / control tokens (5 tokens, IDs 136..140)
# ===========================================================================

AT_VERTEX = _assign("AT_VERTEX")      # binds the previous tag to a vertex
TOP_MOVE = _assign("TOP_MOVE")        # prefix for the played move
ALT_MOVE = _assign("ALT_MOVE")        # prefix for an alternative
SEP_FACTS = _assign("SEP_FACTS")      # separates observations from conclusion
NO_FACTS = _assign("NO_FACTS")        # emitted when no facts triggered


# ===========================================================================
# Block 9: Phase-2 reflect/revise (4 tokens, IDs 141..144)
# Pre-allocated so Phase 1 models can be fine-tuned into Phase 2 without
# re-tokenization.
# ===========================================================================

REFLECT_OPEN = _assign("REFLECT_OPEN")
REFLECT_CLOSE = _assign("REFLECT_CLOSE")
REVISE_OPEN = _assign("REVISE_OPEN")
REVISE_CLOSE = _assign("REVISE_CLOSE")


# ---------------------------------------------------------------------------
# Public registry (for debugging, feature interpretation, and the inspector)
# ---------------------------------------------------------------------------

_REGISTRY: dict[int, str] = {}
for _name, _val in dict(globals()).items():
    if _name.isupper() and _name not in {"PHASE0_VOCAB_SIZE", "VOCAB_SIZE"}:
        if isinstance(_val, int) and PHASE0_VOCAB_SIZE <= _val < VOCAB_SIZE:
            _REGISTRY[_val] = _name
for _i, _val in enumerate(WR_BINS):
    _REGISTRY[_val] = f"WR_{'EVEN' if _i == 10 else f'{_i:02d}'}"

THINK_TOKENS_USED = _NEXT_ID - PHASE0_VOCAB_SIZE


def token_name(token_id: int) -> str:
    return _REGISTRY.get(token_id, f"<unknown_token_{token_id}>")


def all_think_token_ids() -> list[int]:
    return sorted(_REGISTRY.keys())


# Category ranges -- useful for SAE slicing in Phase 3.
CATEGORY_RANGES: dict[str, tuple[int, int]] = {
    "winrate": (WR_BINS[0], WR_BINS[-1] + 1),
    "score_lead": (SL_B_DOM, SL_W_DOM + 1),
    "phase": (PH_OPENING, PH_ENDGAME + 1),
    "group_status": (GRP_ALIVE, GRP_SEKI + 1),
    "tactics": (TAC_ATARI, TAC_DEFENSE + 1),
    "shapes": (SH_EYE, SH_CONNECT + 1),
    "confidence": (CONF_HIGH, CONF_LOW + 1),
    "structural": (AT_VERTEX, NO_FACTS + 1),
    "reflect_revise": (REFLECT_OPEN, REVISE_CLOSE + 1),
}


__all__ = [
    "WR_BINS",
    "winrate_bin_token",
    "score_lead_token",
    "phase_token",
    "group_status_token",
    "confidence_token",
    "token_name",
    "all_think_token_ids",
    "THINK_TOKENS_USED",
    "CATEGORY_RANGES",
    # Structural
    "AT_VERTEX",
    "TOP_MOVE",
    "ALT_MOVE",
    "SEP_FACTS",
    "NO_FACTS",
    # Group status
    "GRP_ALIVE",
    "GRP_WEAK_1",
    "GRP_WEAK_2",
    "GRP_DEAD",
    "GRP_SEKI",
    # Tactics
    "TAC_ATARI",
    "TAC_CAPTURE",
    "TAC_KO",
    "TAC_LADDER_RUN",
    "TAC_LADDER_BREAK",
    "TAC_EYE_MAKE",
    "TAC_INVASION",
    "TAC_REDUCTION",
    "TAC_DEFENSE",
    # Shapes
    "SH_EYE",
    "SH_BAMBOO",
    "SH_TIGER",
    "SH_HANE",
    "SH_CUT",
    "SH_CONNECT",
    # Phase
    "PH_OPENING",
    "PH_MIDGAME",
    "PH_LATE_MID",
    "PH_ENDGAME",
    # Confidence
    "CONF_HIGH",
    "CONF_MED",
    "CONF_LOW",
    # Score-lead
    "SL_B_DOM",
    "SL_B_BIG",
    "SL_B_MED",
    "SL_B_SMALL",
    "SL_B_TINY",
    "SL_EVEN",
    "SL_W_TINY",
    "SL_W_SMALL",
    "SL_W_MED",
    "SL_W_BIG",
    "SL_W_DOM",
    # Phase-2 placeholders
    "REFLECT_OPEN",
    "REFLECT_CLOSE",
    "REVISE_OPEN",
    "REVISE_CLOSE",
]
