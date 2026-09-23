"""FBME-SBL numerical core and fixed-shape CUDA Graph interfaces."""

from .version import PAPER_RELEASE_ID, __version__
from .solver import SBLConfig, SBLResult, clear_operator_cache, fast_sbl, sbl_1d, sbl_2d
from .operators import PrefixFourierOperator
from .offgrid import OffGridResult, continuous_fourier_dictionary, refine_offgrid_fourier
from .cuda_graph import (
    CUDAGraphSBLPlan,
    RealtimeSBLPlan,
    make_cuda_graph_sbl_plan,
    make_realtime_sbl_plan,
)
from .triton_kernels import reset_triton_runtime_error

__all__ = [
    "PAPER_RELEASE_ID", "__version__", "SBLConfig", "SBLResult",
    "clear_operator_cache", "fast_sbl", "sbl_1d", "sbl_2d",
    "PrefixFourierOperator", "OffGridResult", "continuous_fourier_dictionary",
    "refine_offgrid_fourier", "CUDAGraphSBLPlan", "RealtimeSBLPlan",
    "make_cuda_graph_sbl_plan", "make_realtime_sbl_plan",
    "reset_triton_runtime_error",
]
