# optimizers/

Optimizer implementations. Each optimizer is a single file. Public API exported from `__init__.py`.

## Files (planned, written in order)

| File | Purpose |
|---|---|
| `_newton_schulz.py` | Shared NS polynomial routine. Used by SOTR and MuonLike. |
| `_utils.py` | Frobenius norm helper, trust-region clip, parameter-group filter. |
| `lion.py` | Lion optimizer, vendored from `external/lion-pytorch`. |
| `muon_like.py` | Sanity baseline = SOTR with α=1. Same NS as SOTR. |
| `sotr.py` | SOTR (Soft-Orthogonal Trust Region). The flagship. |

## Conventions

- Public class per file: `class Lion(Optimizer)`, `class SOTR(Optimizer)`, etc.
- `step(self, closure=None)` returns `loss | None`, decorated `@torch.no_grad()`.
- Hyperparameters as `__init__` kwargs with sensible defaults documented in docstring.
- Per-parameter group state lives on the optimizer itself; per-step scratch is local.
- All vendored or adapted code carries the header from `CONTRIBUTING.md` §"Comments".

## Public API

```python
# optimizers/__init__.py exposes:
from optimizers import SOTR, Lion, MuonLike
```

We do **not** re-export Muon from `external/Muon`. Use it directly: `from muon import MuonWithAuxAdam` after `pip install -e external/Muon`. This keeps our `optimizers/` clearly "ours."

## What goes in `_utils.py` vs inlined

- Used by ≥2 files → factor into `_utils.py`.
- Used in 1 file → inline. We don't pre-factor for hypothetical reuse.

## Sanity gate

Every optimizer that goes into a primary comparison has a corresponding sanity test in `tests/sanity/`. PROTOCOL §7 lists the required checks.
