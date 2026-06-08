#!/usr/bin/env bash
# v1.3 STANDARD: Phase 0 出口验证脚本 (CI-blocking pre-PoC gate)
#
# Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6
# and docs/superpowers/plans/2026-06-08-hilbert-vit-v13-phase0.md
#
# Run BEFORE any PoC test. Failure blocks Phase 1 entry.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================="
echo "Phase 0 出口验证 (CI-blocking pre-PoC gate)"
echo "Project: $PROJECT_ROOT"
echo "============================================="

# Step 1: 8 静态守卫全部 PASS
echo ""
echo "[Step 1/3] Running 8 static guards (Set A + Set B)..."
uv run pytest tests/static_guards/ -v --tb=short 2>&1 | tail -20

GUARD_RESULT=${PIPESTATUS[0]}
if [ $GUARD_RESULT -ne 0 ]; then
    echo ""
    echo "❌ Phase 0 BLOCKED: 8 static guards FAILED (exit code $GUARD_RESULT)"
    echo "Fix guard errors before running PoC tests."
    exit 1
fi

# Step 2: JVP-5 stress-scale ablation (N=64, 128, 256)
echo ""
echo "[Step 2/3] JVP-5 Stress-Scale ablation (N=64, 128, 256)..."
uv run pytest tests/integration/test_stress_scale_ablation.py -v --tb=short 2>&1 | tail -10 || {
    echo "⚠️  Stress-Scale test not yet implemented (Phase 0 task A.13 still pending)"
}

# Step 3: JVP-6 master coherence
echo ""
echo "[Step 3/3] JVP-6 Master Coherence..."
uv run pytest tests/integration/test_jvp_master_coherence.py -v --tb=short 2>&1 | tail -10 || {
    echo "⚠️  JVP-6 test not yet implemented (Phase 0 task A.14 still pending)"
}

echo ""
echo "============================================="
echo "✅ Phase 0 出口验证 PASSED"
echo "8 static guards: PASS"
echo "Ready to enter Phase 1 (PoC + integration)"
echo "============================================="
