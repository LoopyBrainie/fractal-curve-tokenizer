# CLAUDE.md — `src/training/`

> 🔄 **Reconstruction in progress** — for binding workflow rules see root `CLAUDE.md` § "Reconstruction Workflow".

Sub-package guide for the training layer. **Only training-specific contracts live here**;
project-wide rules (L1-L4 imports, `forward()` contract, logging key format, `torch.compile`
wrapper) and test conventions (STE bridge, assertion style, fixture naming) are in
`../../CLAUDE.md` and `../../tests/CLAUDE.md` respectively. Cross-reference, do not duplicate.

---

## 1. Entry points

| Command | Purpose | Model |
|---|---|---|
| `python -m src.training.train_fractal_vit` | **Production** training | `FractalCurveViT` |
| `python -m src.training.main` | Smoke test only | `DummyModel` |

> **Do NOT switch `main.py::create_model` to `FractalCurveViT`** — it is a smoke test by
> design (see §9).

---

## 2. 4-hook training skeleton (PR5c)

`train_one_epoch(model, dataloader, optimizer, scaler, state, ctx)` at
[`trainer/epoch_train.py:116`](trainer/epoch_train.py#L116) is the spine of training.
The 4 hooks the trainer exposes to callbacks:

| Hook | Line | When | Who uses it |
|---|---|---|---|
| `on_batch_start(ctx)` | `:161` | Before forward | `LossComponentsAccumulator`, `NaNDumpCallback` |
| `on_loss_computed(ctx, loss, components)` | `:197` | After `compute_loss` | `FractalTreeRegCallback`, `AuxiliaryRoutingLossCallback` (inject `ctx.aux_losses`) |
| `pre_backward(ctx)` | `:204` | Before `loss.backward()` | — |
| `post_backward(ctx)` | `:207` | After backward, before NaN check | gradient monitors |

`on_batch_end(ctx)` (`:289`) and `on_epoch_end(ctx)` aggregate `ctx.metrics` /
`ctx.loss_components` for the epoch.

**Hard-coded boundaries (CANNOT be callbackized)** — AMP / `GradScaler` sequence (§4),
Mixup application (`_build_targets` at `:66`), OOM catch routed through `on_exception`
(`:178`), gradient accumulation boundary, `ctx.nan_guard.is_healthy(loss, model)` check
(`:213`), `scaler.update()` on the **skip path** (`:240`).

**PR5c skeleton↔callback contract fields** (written by skeleton, read by callbacks):
`ctx.aux_forward` (`:184`, used by `FractalTreeRegCallback` for `parent_logits`),
`ctx.loss_components_batch`, `ctx.epoch_num_batches`.

> **Critical rule**: skip path MUST still call `optimizer.zero_grad()` (else
> skip-poisoned gradients accumulate into the next accumulation window) and
> `scaler.update()` (else AMP scale stalls).

---

## 3. `TrainerContext` write contract (5-allow / 3-forbid)

Single source of truth: [`callbacks/base.py:39`](callbacks/base.py#L39). All callbacks
must obey the **5-allow / 3-forbid** matrix (Q1 + beta-A-r2 D3 locked):

| Allowed writes to `ctx` (5) | Forbidden writes (3) |
|---|---|
| `ctx.metrics` (scalar `dict[str, float]`) | In-place write to `model.*` / `optimizer.*` / `scaler.*` |
| `ctx.loss_components` (Tensor) | `loss.detach()` then write to `ctx` |
| `ctx.aux_losses` (Tensor **with `grad_fn`**) | `aux_loss.detach()` inside `on_loss_computed` |
| `ctx.should_skip_step` (bool) | |
| `ctx.auxiliary_flat_metrics` (scalar) | |

**Why these forbids matter**: detaching `aux_loss` breaks the gradient chain back through
the backbone — the aux loss is the ONLY gradient signal for the routing params (§5, §6).
In-place writes to `model.*` would mutate model state outside the optimizer.

**NaNGuard is a skeleton hard-dep**, accessed via `ctx.nan_guard.is_healthy(loss, model)`
(`callbacks/base.py:54`, `trainer/epoch_train.py:213`). It is **NOT** in `ctx.callbacks`.

---

## 4. Callback system

**ABC**: [`callbacks/base.py:102`](callbacks/base.py#L102) defines 9 lifecycle hooks.

**Priority semantics** (`callbacks/base.py:105-114`):

| Priority | Semantics | Examples |
|---|---|---|
| `< 0` | Reserved for skeleton-boundary | (none currently — hard boundaries inlined) |
| `== 0` | Default | `AuxiliaryRoutingLossCallback`, `FractalTreeRegCallback` |
| `> 0` | Post-hook (logging / monitoring) | `HMFTHProbsCallback` (+10), `GradientMonitorCallback` (+10) |
| `= -10` | Explicit skeleton-binding | `LossComponentsAccumulator`, `NaNDumpCallback` |

Callbacks sorted by `priority` ASC (stable) before each `train_one_epoch` call.

**`on_exception(ctx, exc) -> bool`** (`callbacks/base.py:170`): `True` = skeleton swallows
and continues, `False` (default) = re-raise. OOM is the canonical swallow case
(`trainer/epoch_train.py:178`).

**`build_callbacks(config)`** ([`callbacks/registry.py:25`](callbacks/registry.py#L25))
attaches: `LossComponentsAccumulator` (always), `HMFTHProbsCallback` (default-on),
`AuxiliaryRoutingLossCallback` (gated by `enable_routing_aux_loss`),
`FractalTreeRegCallback` (gated by `enable_r12_aux`), `NaNDumpCallback` (`--debug` only).

**Adding a new callback**: new file in `src/training/callbacks/` → import in `__init__.py`
→ add to `__all__` → re-export in `src/training/__init__.py` (if public) → wire into
`build_callbacks(config)`.

---

## 5. AMP + `GradScaler` + NaNGuard

`GradScaler` initialization ([`train_fractal_vit.py:670`](train_fractal_vit.py#L670)):

```python
scaler = GradScaler(
    init_scale=2048.0,    # conservative (was 65536 — collapsed in 16 steps)
    growth_interval=500,  # longer stable observation
    backoff_factor=0.5,
) if config.amp.enabled else None
```

**AMP protocol** (5 steps, must run in this exact order — `trainer/epoch_train.py:204-242`):

1. `scaler.scale(loss).backward()`  (or `loss.backward()` if no scaler)
2. `scaler.unscale_(optimizer)`  — **BEFORE grad clip**, else unscaled grads leak into clip
3. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)`
4. `scaler.step(optimizer)`  /  `optimizer.step()` (no scaler case)
5. `scaler.update()`  — **EVEN on skip path** (line `:240`)

**NaNGuard contract** (`trainer/epoch_train.py:213`):

```python
should_skip = not ctx.nan_guard.is_healthy(loss, model)
```

`False` → take the skip path: `optimizer.zero_grad()` + `scaler.update()` +
`state.nan_skip_count += 1`. **Do NOT skip `scaler.update()`**.

**Pre-flight check** ([`train_fractal_vit.py:707`](train_fractal_vit.py#L707)): runs one
batch at `model.eval()` then computes `splitter_grad_norm / backbone_grad_norm`. Warns if
**ratio > 1.5** (splitter dominating gradient). Catches init-time pathologies before the
first optimizer step.

**Runtime env knobs** (see `configure_cuda`):

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (MEM-OOM FIX)
- `torch._dynamo.config.cache_size_limit = 16` (P3-Fix)
- `torch.compile(model, mode='default')` only — `'reduce-overhead'` is INCOMPATIBLE with
  gradient checkpointing
- `torch.backends.cuda.matmul.allow_tf32 = True` (Ampere+)

---

## 6. Optimizer setup: 3 param groups + routing freeze

**3 param groups** ([`train_fractal_vit.py:618-639`](train_fractal_vit.py#L618)):

| Group | Prefix match | LR | Notes |
|---|---|---|---|
| `backbone` | (everything not `splitter.*`) | `base_lr` | transformer + mlp_head + other non-splitter |
| `splitter` | `splitter.*` AND NOT geometry prefixes | `base_lr * 0.1` | feature_proj, depth_embedding, roi_norm, conv1d, logit_scale |
| `geometry` | `splitter.geometry_encoder.`, `splitter.fusion.geo_norm`, `splitter.fusion._semantic_ratio` | `base_lr * 0.1` | 3-segment alignment |

> **Dead-field warning**: `TrainingHyperparams.splitter_lr_multiplier: float = 5.0`
> ([`config.py:53`](config.py#L53)) is **NOT used** by `train_fractal_vit.py` — the actual
> multiplier is the hardcoded `0.1` at `:637`. Editing `splitter_lr_multiplier` has no
> effect. (Vestigial from V4 refactor.)

**Post-build safety check**: `verify_optimizer_coverage(model, optimizer)` at
[`train_fractal_vit.py:412, :642`](train_fractal_vit.py#L412). Flags any
`requires_grad=True` parameter missing from the optimizer.

**5 FREEZE_PREFIXES** for routing params ([`train_fractal_vit.py:461-467`](train_fractal_vit.py#L461)):

| Prefix | Owning module | Match |
|---|---|---|
| `splitter.logit_scale` | splitter | exact |
| `splitter._semantic_ratio` | splitter | exact |
| `splitter.conv1d_hilbert.` (trailing dot!) | splitter.conv1d_hilbert | prefix — covers `.weight` + `.bias` |
| `lca_bias_subtractor.bias_table` | lca_bias_subtractor | exact |
| `alpha_modulator.alpha_raw` | alpha_modulator | exact |

**Freeze wiring** ([`_maybe_freeze_routing_params`](train_fractal_vit.py#L470)):

- `enable_routing_aux_loss=True` (default) → **no freeze** (line `:483`). Aux loss is the
  only gradient signal for these params — must stay unfrozen.
- `enable_routing_aux_loss=False` → freeze all matching params via `p.requires_grad_(False)`;
  also filter from all param groups (`:630-632`) to save AdamW state.

> The `freeze_routing_params: bool = True` config flag ([`config.py:88`](config.py#L88))
> is set at `:593` but the actual freeze logic keys off `enable_routing_aux_loss` (not
> `freeze_routing_params`). Adding a new aux loss requires updating `FREEZE_PREFIXES`.

**`set_seed` determinism** ([`train_fractal_vit.py:58`](train_fractal_vit.py#L58)):

```python
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

`cudnn.deterministic=True` AND `benchmark=False` is the deliberate combo — flipping
`benchmark=True` for speed **breaks reproducibility**.

**τ annealing** ([`train_fractal_vit.py:500`](train_fractal_vit.py#L500)):
linear `tau_start=1.0 → tau_end=0.1` over `tau_epochs=20`. **Phase 4 is the only retained
"scheduler"** (no LR scheduler per R7 hard constraint — see §11).

---

## 7. Auxiliary losses catalog

All 8 aux losses are injected via `ctx.aux_losses` by callbacks during
`on_loss_computed`. The skeleton sums them into `loss` before backward.

| Loss | Math | Target | Default weight | Kill-switch |
|---|---|---|---|---|
| `r12_tree` | L2 tree reg | — | `r12_lambda_tree=0.10` | `enable_r12_aux` |
| `r12_skew` | skew reg | — | `r12_lambda_skew=0.10` | `enable_r12_aux` |
| `routing_entropy` | `(H_target − H_actual)²` | 0.7 | 0.05 | `enable_routing_aux_loss` |
| `routing_budget` | `(K_target − mean(active_mask))²` | 0.25 | 0.08 | `enable_routing_aux_loss` |
| `routing_locality` | `mean(Δhilbert_probs²)` | 0 | 0.04 | `enable_routing_aux_loss` |
| `routing_bias_reg` | `mean(bias_table²)` (L2) | 0 | 0.02 | `enable_routing_aux_loss` |
| `i165_3a_rot` | `‖WᵀW − I‖²_F / d²` on `rot_proj.weight` | 0 | shares scale | `enable_i165_3a_aux_loss` |
| `i165_3a_roi` | `mean((γ²−1)²)` on `roi_norm.weight` | 1 | shares scale | `enable_i165_3a_aux_loss` |
| `i165_3a_geo` | `mean((seg_mean_sq−1)²) + 0.1·var(seg_mean_sq)` on `geo_norm.weight` | 1 | shares scale | `enable_i165_3a_aux_loss` |

> **I165-3a grouped scaling**: single `i165_3a_global_scale=0.05` controls all 3 (internal
> 1:1:1 配比). **Do NOT split into per-loss scales** — this is a deliberate design choice.

**Sources** (exact math): 4 routing losses at
[`callbacks/auxiliary_routing_loss.py:147,159,185,195`](callbacks/auxiliary_routing_loss.py#L147);
3 I165-3a losses at `:224,235,253`; `r12` at
[`callbacks/fractal_tree_reg.py:122`](callbacks/fractal_tree_reg.py#L122).

**Aux-loss ON/OFF ↔ FREEZE wiring**: when `enable_routing_aux_loss=False`, the 5 routing
prefixes (§6) are frozen. **The aux loss is the only gradient signal for those params**.
Do not change one without the other.

---

## 8. CLI flag triple

| Form | Use case | Example |
|---|---|---|
| `--enable-X` | opt-in, default OFF | `--enable-r12-aux` |
| `--disable-X` | opt-out, default ON | `--disable-routing-aux-loss` |
| `--no-enable-X` | kill-switch for default-ON | `--no-enable-routing-aux-loss` |

> **R7-A rule**: features that default ON **MUST** expose `--no-enable-X` form (not just
> `--disable-*`) for symmetry with the OFF-defaulted features.

For the v1.3 flag inventory, see [`config.py:73-103`](config.py#L73). `build_parser()` is
extracted at [`train_fractal_vit.py:1340`](train_fractal_vit.py#L1340) (deliberately
testable; see `tests/unit/training/test_v13_cli_flags.py`).

---

## 9. Mixup / target pipeline

`MixupCutmixLoss.__call__(images, labels, apply_aug=True) → (mixed_images, mixed_labels)`
returns **one-hot** targets (Mixup uses `Beta(α,α)`; CutMix does standard box-paste).

When `mixup is None`, [`_build_targets`](trainer/epoch_train.py#L66) one-hot encodes via
`torch.nn.functional.one_hot(labels, n_classes).float()`.

> **`model.num_classes` is REQUIRED** when `mixup is None` — `ValueError` raised otherwise
> ([`trainer/epoch_train.py:78`](trainer/epoch_train.py#L78)).

`compute_loss(forward_output.logits, targets)` ([`trainer/loss.py:196`](trainer/loss.py#L196))
is the unified entry. **The current PR5c skeleton does NOT pass `aux_losses=`** — each
callback multiplies by its own weight before writing to `ctx.aux_losses`.

---

## 10. Dead code & stubs

| Stub | Location | Why |
|---|---|---|
| `MetricsComputer.compute()` returns `{}` | `training_logs/metrics.py` | PR0 deleted dead metrics |
| `train_one_epoch_simple` / `evaluate_simple` | `trainer/__init__.py` (F-X1) | Returns `None`; do not call |
| `LossMonitor` / `LossTracker` / `CombinedLossTracker` | `__init__.py` (PR6) | Returns `None`; do not call |
| `main.py::create_model → DummyModel` | `main.py` | Smoke test by design; do NOT "fix" |
| `ModelArchitectureConfig` / `OptimizerConfig` / `LossConfig` | `config.py` (I147) | Test placeholders only |
| `src/training/diagnostic/` (deleted) | n/a | I165-3 cleanup; do not recreate |
| `splitter_lr_multiplier` | `config.py:53` | **Dead field** — defined but unused (see §6) |

**Top PR/issue shorthand** — preserve when editing: `=== PR5c ===` (4-hook skeleton),
`=== F-X3 ===` (aux-loss `grad_fn` contract), `I165-3a` (rot/roi/geo losses), `R12`
(tree reg callback), `Q1` (5-allow/3-forbid lock). Full list in `IMPROVEMENT_PLAN.md`.

---

## 11. Evaluation, checkpointing, and logging

**Evaluation** ([`trainer/epoch_eval.py`](trainer/epoch_eval.py)):

- `evaluate(model, dataloader, device, ...)` returns `EvaluationMetrics` (distinct from
  `EpochMetrics`)
- ECE (Expected Calibration Error) — eval-only, via `torch.searchsorted` + `scatter_add` over `num_bins=15`
- Per-class accuracy via `scatter_add` on GPU-resident tensors (no per-batch `.item()` syncs)
- Optional confusion matrix save to `.pt`

**Experiment directory layout** (`create_experiment_dir`,
[`train_fractal_vit.py:68-97`](train_fractal_vit.py#L68)):

```
<experiments_dir>/fractal_vit_YYYYMMDD_HHMMSS/
├── logs/
│   ├── config.json
│   └── epoch_NNNN_stats.json
├── checkpoints/
│   ├── checkpoint_epoch_N.pth
│   ├── best.pth          (only if is_best)
│   └── last.pth          (always)
└── training_history.json
```

**Checkpoint contents** ([`checkpoint/saver.py:67`](checkpoint/saver.py#L67)):
`model_state_dict`, `optimizer_state_dict`, `metrics`, plus optional
`scheduler_state_dict`, `scaler_state_dict`, `training_state` (full round-trip),
`model_config` (auto-detected via `model.get_config()`). All three round-trip via
`load_checkpoint(path, device, load_optimizer=True, load_scheduler=True, load_scaler=True)`.

The model-side `auxiliary_outputs` (keys: `splitter`, `attn_{i}`, `ffn_{i}`, `embed`,
`decay_conv`, `levels`, `routing_tensors`) flows via callback-set `ctx.metrics` keys
(e.g. `hmft/h_prob_bin_{i}`, `train/loss_component/{name}`). See root `CLAUDE.md` for the
`train/{layer}/{metric}` key-prefix convention.

---

## 12. Cross-references

- **Root [`../../CLAUDE.md`](../../CLAUDE.md)** — L1-L4 hierarchy, `forward()` returns
  `TrainingStats` contract, logging key format, `torch.compile` + `lru_cache` wrapper,
  default splitter (`HilbertOptimalSplitter` / H1SS), constants (`EPS`, `TEMPERATURE_MIN`).
- **[`../../tests/CLAUDE.md`](../../tests/CLAUDE.md)** — STE bridge gradient testing
  (`ste_gradient_loss`, `manual_seed(42)`, `hard=True`, T10 keystone), assertion style,
  fixture naming, conftest inheritance, new-test checklist.
- **[`../../IMPROVEMENT_PLAN.md:488-497`](../../IMPROVEMENT_PLAN.md)** — R7 hard
  constraints (locally restated): **不引入 LR schedule** (τ annealing is the only retained
  "scheduler"), **不污染 `TrainingStats` 接口** (trainer cannot add fields to `forward()`
  output).
- **[`../../IMPROVEMENT_PLAN.md:43-44`](../../IMPROVEMENT_PLAN.md)** — trainer-side
  modules must follow: **static pre-registration of `nn.Module`** (no dynamic registration
  in `forward()`), **constructor injection of `splitter._config`** (no post-`__init__` config
  mutation).
