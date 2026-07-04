# optimizers/

Optimizer implementations. **Almost everything here is empty by design** — see [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the rationale. The repo's policy is to import from canonical references rather than reimplement.

## Files

| File | Notes |
|---|---|
| `__init__.py` | Public API: `SOTR`, plus `Lion` re-exported from `lion_pytorch` for convenience. Muon is **not** re-exported — use `from muon import Muon` at the import site so the baseline's lineage is explicit. |
| `sotr.py` | **The one novel file.** 225 lines. Muon-byte-compatible at `(α=1, Δ=∞, q=5)`. |

## What we deliberately do *not* have

| Not here | Why | Use instead |
|---|---|---|
| `_newton_schulz.py` | Muon's `zeropower_via_newtonschulz5` is canonical, tuned, validated | `from muon import zeropower_via_newtonschulz5` |
| `lion.py` | lucidrains' Lion is the de-facto reference | `from lion_pytorch import Lion` (pip-installed from `external/lion-pytorch`) |
| `muon.py` / `muon_like.py` | Muon equals `SOTR(α=1, Δ=∞, q=5)` by construction | `from muon import Muon, MuonWithAuxAdam` for baselines; `SOTR(...)` with those args for the equivalence sanity test |
| `_utils.py` | Frobenius norm is `tensor.norm()`; trust-region clip is 4 inline lines | inline in `sotr.py` |
| `adamw.py` | `torch.optim.AdamW` is the standard | `from torch.optim import AdamW` |

## SOTR step ordering

`sotr.py` implements the **Muon-compatible ordering** — see PROTOCOL.md §7 and the §15 amendment for why. Summary:

1. update momentum buffer with grad
2. form Nesterov-mixed value `M`
3. NS the mixed value: `O = zeropower_via_newtonschulz5(M, steps=q)`
4. α-blend: `U = α·O + (1-α)·M / (||M||_F + ε)`
5. Frobenius cap: `if ||U||_F > Δ: U *= Δ / ||U||_F`
6. per-shape RMS scale: `U *= max(1, m/n)**0.5` (matching Muon)
7. apply update: `p -= lr · U`; weight decay applied separately (decoupled, AdamW-style)

At `α=1, Δ=∞, q=5` this is byte-equivalent to `external/Muon`'s `muon_update`. Sanity test #1 in `tests/sanity/test_sotr_limits.py` verifies this.

## Public API

```python
from optimizers import SOTR
# Lion is also re-exported for convenience:
from optimizers import Lion           # equivalent to: from lion_pytorch import Lion
```

We do *not* re-export Muon. Use it directly: `from muon import Muon, MuonWithAuxAdam`. This keeps the `optimizers` package clearly "ours" and makes the lineage of every baseline explicit at the import site.

## Sanity gate

Every optimizer that goes into a primary comparison has a corresponding sanity test in `tests/sanity/`. PROTOCOL §7 lists the eight required checks; `tests/README.md` maps each to its test file. The meta-test `test_sanity_coverage.py` fails if the §7 list and the tests drift apart.
