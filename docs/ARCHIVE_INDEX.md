# Archive Index — Pre-Reconstruction Baseline (2026-08-03)

**Immutable archive pointer**: tag `pre-reconstruction-baseline-v1` on branch `archive/pre-reconstruction-2026-08-03`
**Source branch at archive time**: `new-main` @ `56cf09e5f6f4a5ee1f38c65b22cf28e73d739dd0`
**Quick extraction**: `git checkout archive/pre-reconstruction-2026-08-03 -- <path>`
**Full worktree mounting**: `git worktree add /tmp/legacy-fct archive/pre-reconstruction-2026-08-03`

This document is the lookup map for any future session that needs to reference the pre-reconstruction codebase. It is the *only* place new sessions should learn what was captured. Everything in here should be treated as historical record — do not modify the archive itself.

---

## §1 Key Files Captured at Archive

These are the entry points for module-by-module reasoning during selective migration. Listed by layer (L1→L4 in the project's hierarchy).

**L1 Foundation** — `src/vit_pytorch/core/`
- `config.py` — config primitives & dataclasses
- `constants.py` — `EPS`, `TEMPERATURE_MIN` and other project-wide constants
- `continuous_utils.py` — continuous-position helpers
- `depth_utils.py` — depth scheduling math
- `levels_info.py` — fractal level metadata
- `shape_stabilizer.py` — shape invariants under dynamic depth

**L2 Components** — `src/vit_pytorch/layers/`
- `splitters/hilbert_optimal_splitter.py` — H1SS splitter (the default; H1SS = "Hilbert 1-Shot Splitting/Search")
- `splitters/multi_block_hmft_splitter.py` — multi-block HMFT variant
- `attention/manifold_attention.py` — manifold-aware attention

**Public API entrypoint**: `src/vit_pytorch/__init__.py` (re-exports `FractalCurveViT`)

**L3 Pipeline** (training glue)
- `src/training/config.py` — training configuration
- `src/training/train_fractal_vit.py` — production entrypoint: `python -m src.training.train_fractal_vit`
- `src/training/trainer/epoch_train.py` — 4-hook trainer skeleton (`on_batch_start`, `on_batch_end`, `on_epoch_end`, `on_validation_start`); the spine of training at line 116.
- `src/training/callbacks/auxiliary_routing_loss.py` — STE (Straight-Through Estimator) bridge loss

**L4 Application**: `FractalCurveViT` re-exported from `src/vit_pytorch/__init__.py`

**Test layer** — `tests/`
- `conftest.py` — project-level fixtures (`seeded`, `seeded_rng`, `device`, `default_splitter_config`)
- `unit/L1_foundation/` — L1 tests
- `unit/L2_components/{attention,splitters}/` — L2 tests
- A new `tests/unit/training/conftest.py` is in the archive (read for context; do not migrate)

**Visualization**: `examples/analysis/visualization/hilbert_splitter.py`

---

## §2 Known Design Decisions (Frozen Facts)

These are decisions baked into the codebase at archive time. **They are recorded for context, not endorsement.** When reconstructing, each must be re-evaluated against the new design rather than blindly inherited.

- **Layer hierarchy**: L1 Foundation → L2 Components → L3 Pipeline → L4 Application. Enforced by an import rule (see §3 below).
- **Hierarchical import rule**: L1 imports nothing; L2 imports only L1; L3 imports L1+L2; L4 imports all. Wrong example: `from vit_pytorch.modules.base_splitter import CoreSplitter`. Correct: `from vit_pytorch.core.splitter_protocol import CoreSplitter`.
- **`forward()` contract**: returns a `TrainingStats` object with fields `logits, num_tokens, depth_used, depth_distribution, features, transformer_tokens`. Trainers read these; modules only need to populate them.
- **Logging key format**: `train/{layer}/{metric}` where `{layer}` ∈ {`splitter`, `attn_{i}`, `ffn_{i}`}. Adding a new layer means: implement `xxx_output` property → register in `fractal_vit.py` `auxiliary_outputs` → no trainer changes.
- **Default splitter**: `HilbertOptimalSplitter` (H1SS).
- **`torch.compile` cache pattern**: any function decorated with `lru_cache` that participates in compile must be wrapped via `_dynamo_safe_lru_cache(maxsize=128)` — raw `lru_cache` breaks Dynamo.
- **Hierarchical Top-K + autograd**: top-k routing decisions must `detach()` and use separate loss calls; otherwise routing latency breaks the main computation graph.
- **Public import path**: `from vit_pytorch import FractalCurveViT` (NOT `from fractal_curve_tokenizer import ...`).

---

## §3 Known Incomplete / Risky Areas

> ⚠️ **Placeholder.** The project never explicitly archived a "TODO" or "known bugs" list. This section is intentionally **under-populated** at session-end — the first reconstruction PR must survey the codebase and replace this with concrete findings. Listing guesses here would be self-refuted on first read; better to leave it honest.

When populating, document each item with:
- file:line anchor
- one-sentence problem statement
- suggested resolution (rewrite / migrate / discard) under the new architecture

---

## §4 Local Branches: Worth Knowing vs Noise

The repo has 60+ local branches at archive time, most accumulated from past `refactor/`, `feat/`, and `fix/` work. **For reconstruction, branches are noise — use the archive tag, not local branch state.** The list below is informational only.

**Worth knowing about** (these reveal prior design intent and may surface relevant decisions during reconstruction):
- `refactor/remove-manifold-complexity` — past attempt to simplify attention
- `refactor/vit-core-simplify` — vit-core simplification
- `feat/splitter-gumbel-ste-pipeline` — Gumbel-STE pipeline for splitters
- `feat/learnable-hilbert-decay` — learnable decay over Hilbert levels
- `feat/tripartite-gating` — three-part routing gate
- `wip/save-20251217T211733Z` — manual snapshot from late 2025; review before any delete

**Noise** (do not chase during reconstruction — the archive tag is the source of truth):
- `feat/*` branches with no merged-into-`main` history are all experimental dead-ends
- `fix/*` and `chore/*` branches are point fixes already captured in baseline
- All 60+ branches can be deleted in bulk after reconstruction ships without losing information

---

## §5 Selective Extraction Recipe

The intended way to migrate a module:

```bash
# Inspect old code via worktree (dual-window mode — see root CLAUDE.md Phase 4)
git worktree add /tmp/legacy-fct archive/pre-reconstruction-2026-08-03

# Browse, understand, then CLOSE that window before coding
cd /tmp/legacy-fct && <read the file>

# Single-file extraction (keeps original author via git)
git checkout archive/pre-reconstruction-2026-08-03 -- path/to/file.py

# Check what you extracted
git log archive/pre-reconstruction-2026-08-03 -- path/to/file.py
```

**Hard rules (from the reconstruction protocol)**:

1. **Do NOT migrate test files.** Old tests encode the old architecture. Write new tests against the new design — use old tests as requirements documentation only.
2. **After extraction, rename and place under the new directory tree.** Original paths are FORBIDDEN in the new codebase. This rule exists to defeat "rename-as-you-go" muscle memory.
3. **Annotate non-trivial ports**: `# Ported from legacy: <commit-hash>` near the function/class. The hash is the SHA from the archive branch.
4. **Bug spotted in old code?** Record it in `TODO` for the new codebase. **Do not** amend the archive.
5. **Ghost dependencies**: if a function you migrate turns out to depend on global module state, document the dependency before migration; otherwise refactoring will introduce subtle breakage.

---

## §6 Removal Conditions

This document (`docs/ARCHIVE_INDEX.md`) is to be deprecated but NOT deleted when:
- Reconstruction is feature-complete, AND
- At least one release has shipped on the new architecture, AND
- The archive tag has been replaced by a new baseline tag (or the new architecture no longer needs the old reference).

Deprecation means: prepend "(DEPRECATED — see <new-archive-pointer>)" and move into a `docs/archive/` subdirectory. The git tag itself stays forever (history preservation).
