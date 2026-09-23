"""Locked CUDA-Graph entry points for the two paper deployment profiles.

The profiles intentionally share one frozen numerical core.  They differ only
in the finite active-set budget and in whether non-decision diagnostics are
evaluated on the foreground path.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from torch_sbl import (
    PAPER_RELEASE_ID,
    CUDAGraphSBLPlan,
    RealtimeSBLPlan,
    SBLConfig,
)

DUAL_RELEASE_ID = "FBME-SBL-FASTEST-STABLE-v1.0"
CORE_RELEASE_ID = "FBME-SBL-TAES-v1.1"


def _require_frozen_core() -> None:
    if PAPER_RELEASE_ID != CORE_RELEASE_ID:
        raise RuntimeError(
            f"{DUAL_RELEASE_ID} requires core {CORE_RELEASE_ID}, "
            f"but imported {PAPER_RELEASE_ID}. Extract/install the bundled "
            "core archive before benchmarking or deployment."
        )


class FastestSBLPlan(CUDAGraphSBLPlan):
    """Minimum-latency locked profile.

    This profile keeps the full-grid Triton UAMP proposal and exact
    evidence-gated reduced posterior update.  It uses two active-set
    evaluations, no residual candidate exchange, and omits fixed-point and
    truncation diagnostics that do not change the returned reconstruction.
    """

    profile = "fbme-sbl-fastest-locked"
    certificate_scope = (
        "exact-evidence-gated-fixed-support;"
        "foreground-fixed-point-diagnostics-disabled"
    )

    def __init__(
        self,
        example_data: Tensor,
        grid_shape: Sequence[int],
        *,
        polish_size: int = 32,
        warmup: int = 3,
    ) -> None:
        _require_frozen_core()
        if example_data.dtype != torch.complex64:
            raise ValueError(
                "FastestSBLPlan is locked to CUDA complex64 arithmetic."
            )
        config = SBLConfig.paper_hybrid_uamp(
            device=example_data.device,
            uamp_polish_size=polish_size,
            uamp_polish_iterations=2,
            uamp_polish_residual_exchange_size=0,
            uamp_polish_fixed_budget_diagnostics=False,
            uamp_polish_compute_truncation_diagnostic=False,
        )
        if config.uamp_backend != "triton":
            raise RuntimeError("The fastest profile requires Triton UAMP.")
        super().__init__(
            example_data,
            grid_shape,
            config=config,
            warmup=warmup,
        )


class StableSBLPlan(RealtimeSBLPlan):
    """Stable locked profile with runtime mathematical diagnostics.

    This is the frozen v1.1 paper profile: three exact active-set evaluations,
    a 12-candidate residual exchange, exact evidence gating, and final
    finite-budget gamma/noise fixed-point diagnostics.
    """

    profile = "fbme-sbl-stable-locked"
    certificate_scope = "fixed-budget-monotone-type2"

    def __init__(
        self,
        example_data: Tensor,
        grid_shape: Sequence[int],
        *,
        polish_size: int = 32,
        warmup: int = 3,
    ) -> None:
        _require_frozen_core()
        if example_data.dtype != torch.complex64:
            raise ValueError(
                "StableSBLPlan is locked to CUDA complex64 arithmetic."
            )
        super().__init__(
            example_data,
            grid_shape,
            polish_size=polish_size,
            warmup=warmup,
            config_overrides={
                "uamp_polish_iterations": 3,
                "uamp_polish_residual_exchange_size": 12,
                "uamp_polish_fixed_budget_diagnostics": True,
                "uamp_polish_compute_truncation_diagnostic": False,
            },
        )


def make_fastest_plan(
    example_data: Tensor,
    grid_shape: Sequence[int],
    *,
    polish_size: int = 32,
    warmup: int = 3,
) -> FastestSBLPlan:
    """Build the locked fastest plan."""
    return FastestSBLPlan(
        example_data,
        grid_shape,
        polish_size=polish_size,
        warmup=warmup,
    )


def make_stable_plan(
    example_data: Tensor,
    grid_shape: Sequence[int],
    *,
    polish_size: int = 32,
    warmup: int = 3,
) -> StableSBLPlan:
    """Build the locked stable plan."""
    return StableSBLPlan(
        example_data,
        grid_shape,
        polish_size=polish_size,
        warmup=warmup,
    )
