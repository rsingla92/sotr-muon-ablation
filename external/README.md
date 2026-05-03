# External reference repositories

Pinned as git submodules so every reported number is reproducible against a known commit. Initialize with `make submodules` (or `git submodule update --init --recursive`).

## Layout

| Path | Upstream | Pinned commit | Why |
|---|---|---|---|
| `external/Muon` | https://github.com/KellerJordan/Muon | `bd1758a` (HEAD as of 2026-05-02) | Reference Muon optimizer (`MuonWithAuxAdam`). PROTOCOL §6 baseline. |
| `external/modded-nanogpt` | https://github.com/KellerJordan/modded-nanogpt | `6399c65` (HEAD as of 2026-05-02) | The speedrun harness where Muon's published numbers were generated. PROTOCOL §4 dataset/model source. |
| `external/lion-pytorch` | https://github.com/lucidrains/lion-pytorch | `6a74fdc` (HEAD as of 2026-05-02) | Lion (Chen et al. 2023) reference impl. PROTOCOL §6 baseline. |
| `external/dion` | https://github.com/microsoft/dion | `9f7897d` (HEAD as of 2026-05-02) | Microsoft's official Dion implementation (Ahn et al. 2025). Reference for any low-rank distributed comparison and for SOTR-on-Dion ablation. |

## How they're used

We **do not** modify the submodules. They are read-only references. When we need to extract or adapt code:

1. Copy the relevant file into our `optimizers/` or `experiments/` tree.
2. Add a header comment citing the source: upstream URL, commit hash, and what was changed.
3. Match upstream behavior step-by-step in `tests/sanity` (PROTOCOL §7) before claiming the adaptation is faithful.

## Updating a pinned commit

If we deliberately want to update to a newer upstream commit:

```bash
cd external/<repo>
git fetch
git checkout <new-commit>
cd ../..
git add external/<repo>
git commit -m "bump external/<repo> to <new-commit>: <reason>"
```

The bump becomes a protocol amendment if it changes any behavior used in a primary comparison.

## Why submodules and not forks

- We don't intend to push back to upstream
- We do want bit-for-bit reproducibility against a specific commit
- Forks add maintenance overhead (rebasing, syncing) without benefit here

If we ever need to diverge — e.g., to fix a bug we can't easily work around — we'll fork at that point and update this README.
