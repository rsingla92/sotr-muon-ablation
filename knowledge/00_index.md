# Knowledge index — optimizer_experiments

Synthesis of the 5 reference PDFs in the repo root. Generated 2026-05-02.

| File | Topic |
|------|-------|
| [01_muon_landscape.md](01_muon_landscape.md) | Muon-family optimizer literature: Muon, Dion, PolarGrad, AuON, MSign, MuonClip, AdaGC/AGGC, SWAN, SinkGD, CANS, TrasMuon, Lion, Schatten-p Muon, etc. |
| [02_muon_scalability.md](02_muon_scalability.md) | Scalability findings: 20B+/1T-token, distributed Newton-Schulz, fine-tuning vs pretraining vs RL |
| [03_sotr_design.md](03_sotr_design.md) | SOTR (Soft-Orthogonal Trust Region) — full algorithmic spec + ChatGPT's first-pass NanoGPT-integration code |
| [04_proposals_existing.md](04_proposals_existing.md) | Four ChatGPT-developed proposals: Schatten-p Muon, Second-Order Muon, Low-Rank Muon, Robust Muon. Plus the five named placeholders (SOTR/CGPU/SCOO/PSORL/FPPP). |
| [05_open_directions.md](05_open_directions.md) | What is shipped vs what is wide-open. The actual whitespace for novel research. |
| [06_lit_update_2026_05.md](06_lit_update_2026_05.md) | Web search filling the Feb→May 2026 gap. Subsumes/kills several backlog ideas; sharpens recommendations. |
| [07_spectral_interpretation.md](07_spectral_interpretation.md) | Theoretical anchor: SOTR's α-blend is a singular-value rescaling `σ_i ↦ α + (1−α)·σ_i/||M||_F`. Family comparison to PolarGrad / 2602.04669. Reframes the paper's contribution claim. |

**Source PDFs (repo root):**
- `Research Ideas - SOTR Optimizer Design.pdf` (34 pp) — SOTR concept + draft impl
- `Research Ideas - Muon scaling research clarification.pdf` (100 pp) — scalability + 4 proposals + landmark idea seeds + tangential data-valuation proposal
- `Executive Summary (2).pdf` (10 pp) — narrow lit review around SOTR
- `Executive Summary (3).pdf` — duplicate text of (2), different sha256 (metadata diff only)
- `Executive Summary (4).pdf` (16 pp) — broader lit review including Lion/AdaGC/AGGC/SWAN/SinkGD/MuonClip/Pethick/Stiefel methods
