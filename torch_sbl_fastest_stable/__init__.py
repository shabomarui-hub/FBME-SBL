"""Locked fastest/stable profiles for the FBME-SBL v1.1 core."""

from .profiles import (
    CORE_RELEASE_ID,
    DUAL_RELEASE_ID,
    FastestSBLPlan,
    StableSBLPlan,
    make_fastest_plan,
    make_stable_plan,
)

__all__ = [
    "CORE_RELEASE_ID",
    "DUAL_RELEASE_ID",
    "FastestSBLPlan",
    "StableSBLPlan",
    "make_fastest_plan",
    "make_stable_plan",
]
