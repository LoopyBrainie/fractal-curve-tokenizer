# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Sanity Check

```bash
uv run pytest -m "not slow"
```

## Package Import

`src/` path is auto-configured in training scripts. For standalone scripts:
```python
import sys
sys.path.insert(0, 'src')
```

**Always use**: `from vit_pytorch import FractalCurveViT` (NOT `from fractal_curve_tokenizer import ...`)

## Development Rules

- **Mathematical Formalization First**: Derive mathematical forms before any code changes

## Module Hierarchy

| Layer | Purpose |
| ----- | ----- |
| L4 Application | Main model (`FractalCurveViT`) |
| L3 Pipeline | Tokenization & Transformer |
| L2 Components | Splitter, Attention, FFN |
| L1 Foundation | Hilbert curves, config primitives |

## Import Rules

**Hierarchical** (L1→L2→L3→L4):

- L1 imports: None (base)
- L2 imports: L1 only
- L3 imports: L1, L2
- L4 imports: All

**Wrong**: `from vit_pytorch.modules.base_splitter import CoreSplitter`
**Correct**: `from vit_pytorch.core.splitter_protocol import CoreSplitter`

## Model-Trainer Interface

**Principle**: Model defines capabilities, Trainer decides usage.

`forward()` returns `TrainingStats`: `logits`, `num_tokens`, `depth_used`, `depth_distribution`, `features`, `transformer_tokens`

## Logging System Design

Key format: `train/{layer}/{metric}`

| Source | Format |
| ----- | ----- |
| Splitter | `train/splitter/{metric}` |
| Attention | `train/attn_{i}/{metric}` |
| FFN | `train/ffn_{i}/{metric}` |

**Adding a new layer**: Implement `xxx_output` property → add key in `fractal_vit.py` auxiliary_outputs → no trainer changes.

## Critical Patterns

**torch.compile cache** (`lru_cache` breaks Dynamo, must be wrapped):
```python
def _dynamo_safe_lru_cache(maxsize: int = 128):
    cached = lru_cache(maxsize=maxsize)(func)
    return torch._dynamo.disable(cached)
```

## Conventions

- Constants: `from vit_pytorch.core.constants import EPS, TEMPERATURE_MIN`
- Default splitter: `HilbertOptimalSplitter` (H1SS)
- Code comments: `# I24-2: Import Scheme E learnable quota constants`

## Common Pitfalls

- **Hierarchical Top-K breaks computation graph**: use `detach()` + separate loss calls

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): Historical Issue Archive
- [docs/00_introduction.md](docs/00_introduction.md): 架构入口 (详见 docs/00-10 各章节)
- [docs/ARCHIVE_INDEX.md](docs/ARCHIVE_INDEX.md): Pre-reconstruction baseline entry point (active during reconstruction)
- [tests/CLAUDE.md](tests/CLAUDE.md): 测试特定约定 (STE 桥损失、T10 keystone 等)

---

## Reconstruction Workflow (Active — Until Reconstruction Complete)

<!-- TEMPORARY: Active during reconstruction. Remove this section after
reconstruction is fully complete and old archive is deprecated. -->

**Archive pointer**: tag `pre-reconstruction-baseline-v1`, branch `archive/pre-reconstruction-2026-08-03`. See `docs/ARCHIVE_INDEX.md`.

This section is **binding** for any session that opens this repository until explicitly removed. The five phases below are required procedure, not suggestions.

### Phase 1 — Archive (✅ Completed 2026-08-03)

All current working-tree content has been committed and frozen under tag `pre-reconstruction-baseline-v1` on 2026-08-03. Future "reference to old code" actions MUST go through this tag (or `docs/ARCHIVE_INDEX.md`).

**Do not** delete the archive branch, relocate it, or rewrite its tag. Treat the archive as immutable.

### Phase 2 — New Codebase

**Decision (2026-08-03)**: new repository at `D:\myProject\fractal-curve-ViT` (separate repo, not orphan branch).
- First commit: `3a04ec9` on `main`
- Archive bridge wired — see `## Cross-Repo Bridge` below

Day-1 setup as performed in the new repo:
- `pyproject.toml` with stripped runtime deps + ruff/mypy **config** (no dev deps installed) — per "refactor first, manage deps later"
- `pyproject.toml` keeps `requires-python = ">=3.13"` and `[build-system] uv_build`
- `.gitignore` with Python + PyTorch + `.claude` exclusion
- `.python-version` = 3.13
- `README.md` with bootstrap + archive bridge instructions
- No application code yet; no runtime dependencies yet
- No `.gitattributes` (relies on per-user git config for default CRLF)
- Archive remote added in new repo: `git remote add archive https://github.com/LoopyBrainie/fractal-curve-tokenizer.git`
- Tag persisted locally via explicit refspec: `git fetch archive +refs/tags/pre-reconstruction-baseline-v1:refs/tags/pre-reconstruction-baseline-v1`

Day-1 setup checklist (mandatory for any future touch on the new repo):
- Lint + formatter + type checker configured **before** any new code is written
- Empty directory skeleton committed first (architecture over code)
- CI pipeline in place from the first push
- Re-audit `.gitignore`: keep current `.claude` exclusion; add new exclusions only when justified
- Re-review `.gitattributes` for Line Ending Normalization across platforms (Windows CRLF warnings observed during archive commit)

### Phase 3 — Selective Migration

For every existing module, classify into exactly one of:

| Decision | Trigger |
| -------- | ------- |
| **Rewrite** | Interface changes, or coupling to now-removed abstractions |
| **Migrate** | Stable math/algorithm (splitters, Hilbert curves, RoPE, Gumbel-STE pipeline); extract via `git checkout archive/pre-reconstruction-2026-08-03 -- <path>` and re-home under the new directory tree |
| **Discard** | Dead code, replaced by a new abstraction, or only used by tests being rewritten |

**Mandatory rules**:
- Do **not** migrate test files. Old tests encode the old architecture; write new tests against the new design (use old tests as requirements documents).
- After extraction, **rename and place in the new directory tree**. Old paths are forbidden in the new codebase — this defeats "rename-as-you-go" muscle memory.
- Keep an `Old:` comment pointer for non-trivial algorithm ports: `# Ported from legacy: 56cf09e5f6f4a5ee1f38c65b22cf28e73d739dd0` (or similar).

### Phase 4 — Dual-Window Reference Development

For complex ports, mount the archive into a separate directory:

```bash
git worktree add /tmp/legacy-fct archive/pre-reconstruction-2026-08-03
```

IDE dual-open. **Process discipline** (NOT "read while typing"):

1. Read the old implementation fully in the legacy worktree window
2. **Close that window** (or fold the file)
3. Write the new implementation in the main window — *from understanding, not by copy*

Two worktrees share one `.git` database; do NOT run concurrent `git` mutations (rebase, commit, push) from both. Default to treating the legacy worktree as read-only.

If you discover a bug in old code, record it in `TODO` for the new codebase. **Do not** amend the archive.

### Phase 5 — Common Pitfalls

| Pitfall | Consequence | Prevention |
| ------- | ----------- | ---------- |
| "Migrate first, refactor later" | New code becomes old-code-in-new-clothes | Decide new architecture *before* migration; force rename + interface change at extraction |
| Preserving old directory structure | Architecture inertia locks new design into old shape | Draw new directory skeleton on empty repo before writing logic |
| Ignoring data/config migration | New system cannot run, old data formats incompatible | Plan migration scripts + rollback strategy from the start |
| Ghost dependencies | Refactor breaks hidden global-state couplings | Integration verification after each module migration |

### Removal

This entire section (`## Reconstruction Workflow`) is to be removed by an explicit commit **only after** all of these are true:
- (a) reconstruction is feature-complete
- (b) `pre-reconstruction-baseline-v1` is documented as deprecated but still preserved in `docs/ARCHIVE_INDEX.md` (per its own §6 Removal Conditions)
- (c) at least one full release has shipped on the new architecture

---

## Cross-Repo Bridge

The new repo at `D:\myProject\fractal-curve-ViT` has the old repo configured as a **read-only** archive remote.

**Setup (already done — listed for reference):**

```bash
git remote add archive https://github.com/LoopyBrainie/fractal-curve-tokenizer.git
git fetch archive +refs/tags/pre-reconstruction-baseline-v1:refs/tags/pre-reconstruction-baseline-v1
```

The `+refs/tags/...:refs/tags/...` refspec is required — plain `git fetch archive <tagname>` only updates `FETCH_HEAD`, not local `refs/tags/`.

### Inspecting old code via worktree (Phase 4 dual-window mode)

```bash
git worktree add /tmp/legacy-fct pre-reconstruction-baseline-v1
```

The worktree at `/tmp/legacy-fct` is **read-only** by convention. Do not run `git commit` / `git push` from that directory.

**Single-file extraction** (keeps original author via git):

```bash
git checkout pre-reconstruction-baseline-v1 -- path/to/file.py
```

Always rename + re-home extracted files under the new directory tree — original paths are FORBIDDEN in the new codebase.

### Cleaning the bridge

To sever the connection entirely (only after reconstruction is feature-complete, per §Removal above):

```bash
git remote remove archive
git tag -d pre-reconstruction-baseline-v1
git gc --prune=now --aggressive
```

The git tag itself in this old repo MUST stay forever — history preservation. Only the new repo's bridge may be removed.
