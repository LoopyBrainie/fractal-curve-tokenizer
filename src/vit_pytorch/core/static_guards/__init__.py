"""v1.3 STANDARD: 8 Static Code Guards (双套系统)

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6:

    Set A (经典控制流/优化, v1.2 retained, frozen):
        SMA:  State Machine Alignment
        DGC:  Dual-end Gradient Conservation
        FFMI: Fractal Feature Map Isolation
        PCAR: P-Controller Anti-saturation

    Set B (几何/结构, v1.3 NEW, collision-free names):
        TMA:  Telemetry Monitor Alignment
        ADGC: Adaptive Depthwise Gradient Clipping
        FMIG: Feature Map Integrity Guard
        PCA:  Path-Coordinate Alignment

CI-blocking priority: HIGHEST. 失败阻塞构建,不阻塞训练。
Run via: uv run pytest tests/static_guards/ -v
Or:    bash scripts/verify_phase0.sh
"""
from .sma import check_state_machine_alignment, SMAError
from .dgc import check_dual_end_gradient_conservation, DGCError
from .ffmi import check_fractal_feature_isolation, FFMIError
from .pcar import check_p_controller_anti_saturation, PCARError
from .tma import check_telemetry_alignment, TMAError
from .adgc import (
    compute_delta_c,
    check_adgc_monotonicity,
    ADGCError,
)
from .fmig import (
    check_feature_map_integrity,
    inject_orthogonal_noise,
    FMIGError,
    FMIG_CHANNEL_VARIANCE_THRESHOLD,
)
from .pca import check_path_coordinate_alignment, PCAError
from .runner import (
    run_all_guards,
    assert_all_guards_pass,
    GUARD_NAMES,
)

__all__ = [
    # Set A
    "check_state_machine_alignment", "SMAError",
    "check_dual_end_gradient_conservation", "DGCError",
    "check_fractal_feature_isolation", "FFMIError",
    "check_p_controller_anti_saturation", "PCARError",
    # Set B
    "check_telemetry_alignment", "TMAError",
    "compute_delta_c", "check_adgc_monotonicity", "ADGCError",
    "check_feature_map_integrity", "inject_orthogonal_noise", "FMIGError",
    "FMIG_CHANNEL_VARIANCE_THRESHOLD",
    "check_path_coordinate_alignment", "PCAError",
    # Runner
    "run_all_guards", "assert_all_guards_pass", "GUARD_NAMES",
]
