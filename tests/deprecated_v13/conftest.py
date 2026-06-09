"""Auto-skip all tests in the v1.3 opt-in feature archive.

The tests under tests/deprecated_v13/ correspond to v1.3 opt-in features
T2 (Shadow Monitor), T4 (EAHBP 3-gate), and T5 (Paced Window staging),
which were culled in PR1 of the trainer refactor plan
(``fluffy-watching-turing.md`` §3 PR1, Q2 decision).

These tests are kept in-tree as historical reference (git history of the
v1.3 design space) but are not part of the regression matrix. A
``--include-deprecated-v13`` opt-in flag exists for one-off inspection
runs only.

See also:
  - plan: ``C:\\Users\\LamKo\\.claude\\plans\\fluffy-watching-turing.md`` §3 PR1
  - Q2 decision rationale: Q2 — Precision pruning (T2/T4/T5 broken-window
    effects, torch.compile graph breakage, DDP reducer perturbation,
    full-model clone memory fragmentation)
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip all tests in this directory unless --include-deprecated-v13."""
    if config.getoption("--include-deprecated-v13", default=False):
        return
    skip_deprecated = pytest.mark.skip(
        reason="v1.3 opt-in T2/T4/T5 features removed in PR1 (trainer refactor). "
        "Use --include-deprecated-v13 to run these archived tests."
    )
    for item in items:
        item.add_marker(skip_deprecated)


def pytest_addoption(parser):
    """Add --include-deprecated-v13 flag for one-off inspection runs."""
    parser.addoption(
        "--include-deprecated-v13",
        action="store_true",
        default=False,
        help="Run tests in tests/deprecated_v13/ (v1.3 opt-in archive). "
        "Off by default; these tests are skipped at collection.",
    )
