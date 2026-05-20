"""Natural-language chain-of-thought rewriter.

Translates a position's structured CoT (output of ``gogpt.cot_extractor``)
into a 1-2 sentence prose rendition using either Anthropic Claude or
Google Gemini. Used to build the Phase 1 mode-C training data (mixed
structured + natural-language CoT).

This module produces TEXT, not tokens. Integrating NL-CoT into the
training pipeline requires extending the model's vocabulary -- see
``docs/nl_cot.md`` for the proposed approach.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from . import NUM_POINTS
from . import cot_vocab as V
from .tokenizer import token_to_gtp_vertex

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decode structured CoT tokens to a human-readable string
# ---------------------------------------------------------------------------

def decode_think_tokens(tokens: list[int]) -> str:
    """Render a list of think-block token IDs as a compact human string.

    Combines ``<tag> AT_VERTEX <vertex>`` into ``TAG@<coord>`` and
    ``TOP_MOVE <vertex>`` into ``TOP_MOVE <coord>``.

    >>> decode_think_tokens([V.WR_BINS[10], V.PH_OPENING, V.SEP_FACTS])
    'WR_EVEN PH_OPENING SEP_FACTS'
    """
    parts: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = int(tokens[i])
        # Move-vocab token (board point or PASS)
        if 0 <= t <= NUM_POINTS:
            parts.append(token_to_gtp_vertex(t))
            i += 1
            continue
        if t == V.AT_VERTEX:
            # Standalone AT_VERTEX -- shouldn't happen, but skip cleanly.
            i += 1
            continue
        if t == V.TOP_MOVE and i + 1 < n:
            nxt = int(tokens[i + 1])
            if 0 <= nxt <= NUM_POINTS:
                parts.append(f"TOP_MOVE {token_to_gtp_vertex(nxt)}")
                i += 2
                continue
            parts.append("TOP_MOVE")
            i += 1
            continue
        # Look-ahead for "<tag> AT_VERTEX <vertex>"
        if i + 2 < n and int(tokens[i + 1]) == V.AT_VERTEX:
            nxt = int(tokens[i + 2])
            if 0 <= nxt <= NUM_POINTS:
                parts.append(f"{V.token_name(t)}@{token_to_gtp_vertex(nxt)}")
                i += 3
                continue
        parts.append(V.token_name(t))
        i += 1
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You translate compact structured Go-position reasoning tags into one or two sentences of natural prose.

CONTEXT: A 9x9 Go transformer plays as Black. For each move it emits a structured "think-block" of compact tokens. Your job is to render those tokens as fluent, declarative reasoning a human Go player might write, then state the chosen move.

TAG GLOSSARY

Winrate (from Black's perspective, ranges shown in percent):
  WR_00 = 0-10%   WR_01 = 10-20%   WR_02 = 20-30%   WR_03 = 30-40%
  WR_04 = 40-50%  WR_EVEN = ~50%   WR_05 = 50-60%   WR_06 = 60-70%
  WR_07 = 70-80%  WR_08 = 80-90%   WR_09 = 90-100%

Score lead (in points, Black-relative):
  SL_B_DOM   = Black dominates (>= +20)
  SL_B_BIG   = Black leads big (+10..+20)
  SL_B_MED   = Black leads (+5..+10)
  SL_B_SMALL = Black slightly ahead (+2..+5)
  SL_B_TINY  = Black barely ahead (+0.5..+2)
  SL_EVEN    = essentially even
  SL_W_TINY  = White barely ahead
  SL_W_SMALL = White slightly ahead
  SL_W_MED   = White leads
  SL_W_BIG   = White leads big
  SL_W_DOM   = White dominates

Phase:
  PH_OPENING  = moves 0-8
  PH_MIDGAME  = moves 9-30
  PH_LATE_MID = moves 31-60
  PH_ENDGAME  = move 61+

Group status (suffix @<coord> identifies a stone in the group):
  GRP_ALIVE@X   = healthy group at X
  GRP_WEAK_1@X  = group at X is in atari (one liberty)
  GRP_WEAK_2@X  = group at X has two liberties
  GRP_DEAD@X    = group at X is dead
  GRP_SEKI@X    = group at X is in seki (mutual life)

Tactics (evaluated against the move being played):
  TAC_ATARI        = the move puts an opponent group in atari
  TAC_CAPTURE      = the move captures stones
  TAC_KO           = the move is a ko-capture
  TAC_LADDER_RUN   = an own group is currently in a captured ladder
  TAC_LADDER_BREAK = the move breaks a ladder
  TAC_EYE_MAKE     = the move creates a new eye
  TAC_INVASION     = the move plays into opponent territory
  TAC_REDUCTION    = the move reduces opponent territory
  TAC_DEFENSE      = the move defends an own weak group

Shapes (suffix @<coord> identifies the relevant point):
  SH_EYE@X     = a new eye at X
  SH_BAMBOO@X  = a bamboo joint at X
  SH_TIGER@X   = tiger's mouth at X
  SH_HANE / SH_CUT / SH_CONNECT = local shape pattern

Conclusion:
  NO_FACTS        = nothing specific to mention
  SEP_FACTS       = separator between observations and chosen move
  TOP_MOVE <coord> = the move to play
  CONF_HIGH / CONF_MED / CONF_LOW = confidence in the chosen move

STYLE RULES
- Write 1-2 sentences in present tense.
- State the most important observations first, then the chosen move and confidence.
- Use Go vocabulary: "atari", "ladder", "eye", "ko", "seki", "territory", "weak group", "shape".
- Do NOT mention "the model", "AI", or "structured tokens". Write as if a player is thinking.
- Be terse. Skip filler.
- Use the GTP coordinate exactly as given (e.g. "E5", "B2", "pass"). The board uses columns A-J skipping I.

EXAMPLES

Input: WR_EVEN SL_B_TINY PH_OPENING NO_FACTS SEP_FACTS TOP_MOVE E5 CONF_HIGH
Output: The opening is roughly even with Black a hair ahead. Play the center at E5 with confidence.

Input: WR_06 SL_B_MED GRP_WEAK_2@B2 TAC_ATARI PH_MIDGAME SEP_FACTS TOP_MOVE C2 CONF_HIGH
Output: Black is comfortably ahead in the middlegame, but the Black group at B2 has only two liberties; playing C2 puts the surrounding stones in atari and resolves the threat.

Input: WR_03 SL_W_BIG GRP_DEAD@H8 PH_ENDGAME SEP_FACTS TOP_MOVE pass CONF_MED
Output: White is comfortably ahead in the endgame and the Black group at H8 is dead. Passing is reasonable.

Input: WR_07 SL_B_MED TAC_INVASION SH_EYE@D7 PH_MIDGAME SEP_FACTS TOP_MOVE C3 CONF_MED
Output: Black is winning solidly and has just secured an eye at D7; C3 invades the lower-left to extend the lead.

Now translate the next input."""


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

@dataclass
class RewriteResult:
    nl_text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Provider(ABC):
    name: str = "abstract"
    model: str = ""

    @abstractmethod
    def rewrite(self, structured_text: str) -> RewriteResult: ...


class AnthropicProvider(Provider):
    """Anthropic Claude rewriter.

    Default model is ``claude-opus-4-7`` (per the Claude API skill default).
    For bulk mechanical rewriting of 10k+ short CoTs, ``claude-haiku-4-5``
    is 5x cheaper and quality is more than sufficient -- pass it explicitly
    via ``--model claude-haiku-4-5``. The system prompt is sent with
    ``cache_control: ephemeral`` so subsequent calls within a 5-minute
    window read it at 0.1x cost (effective on models whose minimum
    cacheable prefix the prompt clears -- ~2048 tokens for Sonnet 4.6,
    4096 for Opus 4.7 / Haiku 4.5; check ``cache_read_input_tokens`` in
    the response to confirm).
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-7", max_tokens: int = 256) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed; pip install 'gogpt[nl-cot]' or pip install anthropic"
            ) from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens

    def rewrite(self, structured_text: str) -> RewriteResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Input: {structured_text}\nOutput:",
                }
            ],
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return RewriteResult(
            nl_text=text,
            provider=self.name,
            model=self.model,
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )


class GeminiProvider(Provider):
    """Google Gemini rewriter (default for cost: Flash 2.5 has a free tier).

    Set ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``.
    """

    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", max_tokens: int = 256) -> None:
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "google-genai SDK not installed; pip install 'gogpt[nl-cot]' or "
                "pip install google-genai"
            ) from e
        self._genai = genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini provider")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def rewrite(self, structured_text: str) -> RewriteResult:
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"Input: {structured_text}\nOutput:",
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": self.max_tokens,
                "temperature": 0.3,
            },
        )
        # The Gemini SDK exposes .text as a convenience; fall back to walking
        # candidates if it's missing for some reason.
        text = getattr(response, "text", None)
        if text is None:
            text = ""
            for cand in getattr(response, "candidates", []) or []:
                parts = getattr(cand.content, "parts", []) if getattr(cand, "content", None) else []
                for p in parts:
                    pt = getattr(p, "text", None)
                    if pt:
                        text += pt
        usage = getattr(response, "usage_metadata", None)
        return RewriteResult(
            nl_text=text.strip(),
            provider=self.name,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        )


class MockProvider(Provider):
    """Deterministic mock for tests. Returns ``"PROSE: <input>"``."""

    name = "mock"

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def rewrite(self, structured_text: str) -> RewriteResult:
        return RewriteResult(
            nl_text=f"PROSE: {structured_text}",
            provider=self.name,
            model=self.model,
            input_tokens=len(structured_text.split()),
            output_tokens=len(structured_text.split()) + 2,
        )


def build_provider(name: str, model: str | None = None, **kwargs) -> Provider:
    if name == "anthropic":
        return AnthropicProvider(model=model or "claude-opus-4-7", **kwargs)
    if name == "gemini":
        return GeminiProvider(model=model or "gemini-2.5-flash", **kwargs)
    if name == "mock":
        return MockProvider(model=model or "mock")
    raise ValueError(f"unknown provider {name!r}; choices: anthropic, gemini, mock")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def rewrite_with_retry(
    provider: Provider,
    text: str,
    *,
    max_attempts: int = 4,
    base_delay: float = 2.0,
) -> RewriteResult:
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return provider.rewrite(text)
        except Exception as e:
            last_err = e
            wait = base_delay * (2**attempt)
            log.warning(
                "rewrite failed (attempt %d/%d): %s -- retrying in %.1fs",
                attempt + 1, max_attempts, e, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"rewrite failed after {max_attempts} attempts") from last_err


__all__ = [
    "Provider",
    "AnthropicProvider",
    "GeminiProvider",
    "MockProvider",
    "RewriteResult",
    "SYSTEM_PROMPT",
    "build_provider",
    "decode_think_tokens",
    "rewrite_with_retry",
]
