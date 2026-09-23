from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
from torch import Tensor

from .solver import SBLConfig, SBLResult, fast_sbl


class CUDAGraphSBLPlan:
    """Capture a fixed-shape SBL solve for low-latency replay.

    CUDA Graph replay removes Python dispatch and launches the complete
    fixed-budget solver as one device graph.  The input shape, dtype, device,
    grid, configuration, and batch size are fixed at construction time.
    Returned tensors are graph-owned and are overwritten by the next replay.
    Clone fields that must outlive a subsequent call.
    """

    def __init__(
        self,
        example_data: Tensor,
        grid_shape: Sequence[int],
        *,
        config: SBLConfig,
        warmup: int = 3,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph execution requires CUDA.")
        if not isinstance(example_data, Tensor) or not example_data.is_cuda:
            raise ValueError("example_data must be a CUDA tensor.")
        if warmup < 1:
            raise ValueError("warmup must be positive.")
        if config.return_history or config.verbose:
            raise ValueError(
                "CUDA Graph plans require return_history=False and verbose=False."
            )
        if config.tol > 0.0 or config.eager_change_tracking:
            raise ValueError(
                "CUDA Graph plans require a fixed iteration budget: set "
                "tol=0 and eager_change_tracking=False."
            )
        if (
            config.phase_error_correction
            or config.range_migration_correction
            or config.offgrid_refinement
            or config.learn_snapshot_covariance
        ):
            raise ValueError(
                "Capture the core SBL stage only; motion, off-grid, and "
                "snapshot-covariance refinement are not graph-static."
            )

        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.config = deepcopy(config)
        self.device = example_data.device
        self.dtype = example_data.dtype
        self.shape = tuple(example_data.shape)
        self.input_buffer = torch.empty_like(example_data)
        self.input_buffer.copy_(example_data)

        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup):
                fast_sbl(
                    self.input_buffer,
                    self.grid_shape,
                    config=self.config,
                )
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        self._graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self._graph):
                self._result = fast_sbl(
                    self.input_buffer,
                    self.grid_shape,
                    config=self.config,
                )
        except Exception as error:
            raise RuntimeError(
                "CUDA Graph capture failed for this SBL configuration."
            ) from error

    @property
    def result(self) -> SBLResult:
        """Return graph-owned output from the most recent replay."""
        return self._result

    def replay(
        self,
        data: Tensor | None = None,
        *,
        synchronize: bool = False,
    ) -> SBLResult:
        """Replay the captured solve, optionally copying a new CUDA input."""
        if data is not None:
            self._validate_input(data)
            self.input_buffer.copy_(data)
        self._graph.replay()
        if synchronize:
            torch.cuda.synchronize(self.device)
        return self._result

    def _validate_input(self, data: Tensor) -> None:
        if not isinstance(data, Tensor) or not data.is_cuda:
            raise ValueError("Replay data must be a CUDA tensor.")
        if data.device != self.device:
            raise ValueError(
                f"Replay data must be on {self.device}, got {data.device}."
            )
        if data.dtype != self.dtype:
            raise ValueError(
                f"Replay data must have dtype {self.dtype}, got {data.dtype}."
            )
        if tuple(data.shape) != self.shape:
            raise ValueError(
                f"Replay data must have shape {self.shape}, got "
                f"{tuple(data.shape)}."
            )


class RealtimeSBLPlan(CUDAGraphSBLPlan):
    """Low-latency fixed-shape plan for the paper's unified FBME-SBL.

    The foreground result uses complex64 Triton UAMP and three exact reduced
    active-set evaluations inside one CUDA Graph.  Exact evidence gating gives
    a non-increasing finite-budget objective, while final gamma/noise
    fixed-point residuals quantify the remaining optimization error.
    """

    profile = "fbme-sbl-fixed-budget"
    certificate_scope = "fixed-budget-monotone-type2"

    def __init__(
        self,
        example_data: Tensor,
        grid_shape: Sequence[int],
        *,
        polish_size: int = 32,
        warmup: int = 3,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        overrides: dict[str, Any] = {
            "device": example_data.device,
            "uamp_polish_size": polish_size,
        }
        if config_overrides is not None:
            overrides.update(config_overrides)
        config = SBLConfig.realtime_hybrid_uamp(**overrides)
        self._validate_realtime_config(config)
        super().__init__(
            example_data,
            grid_shape,
            config=config,
            warmup=warmup,
        )

    @staticmethod
    def _validate_realtime_config(config: SBLConfig) -> None:
        if not config.uamp_polish_fixed_budget_diagnostics:
            raise ValueError(
                "RealtimeSBLPlan requires fixed-budget diagnostics so the "
                "paper objective and residuals are always reported."
            )
        if any(
            value != 0.0
            for value in (
                config.hyper_shape,
                config.hyper_rate,
                config.noise_shape,
                config.noise_rate,
            )
        ):
            raise ValueError(
                "RealtimeSBLPlan requires zero numerical hyperpriors so its "
                "exact evidence gate matches the stated paper objective."
            )
        forbidden = {
            "uamp_polish_global_certificate": (
                config.uamp_polish_global_certificate
            ),
            "uamp_polish_support_repairs": (
                config.uamp_polish_support_repairs != 0
            ),
            "uamp_polish_support_growth_steps": (
                config.uamp_polish_support_growth_steps != 0
            ),
            "uamp_polish_support_shrink_steps": (
                config.uamp_polish_support_shrink_steps != 0
            ),
            "uamp_polish_coordinate_steps": (
                config.uamp_polish_coordinate_steps != 0
            ),
            "uamp_polish_noise_coordinate_steps": (
                config.uamp_polish_noise_coordinate_steps != 0
            ),
            "uamp_polish_high_precision": (
                config.uamp_polish_high_precision
            ),
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(
                "RealtimeSBLPlan excludes dynamic research certificates; "
                "disable: " + ", ".join(enabled)
            )
        if config.uamp_backend != "triton":
            raise ValueError(
                "RealtimeSBLPlan requires uamp_backend='triton' so a slow "
                "Torch fallback cannot be reported as the real-time path."
            )


def make_cuda_graph_sbl_plan(
    example_data: Any,
    grid_shape: Sequence[int],
    *,
    config: SBLConfig | None = None,
    warmup: int = 3,
) -> CUDAGraphSBLPlan:
    """Convenience constructor for :class:`CUDAGraphSBLPlan`."""
    tensor = torch.as_tensor(example_data)
    active_config = config or SBLConfig.fast_hybrid_uamp(
        device=tensor.device,
    )
    return CUDAGraphSBLPlan(
        tensor,
        grid_shape,
        config=active_config,
        warmup=warmup,
    )


def make_realtime_sbl_plan(
    example_data: Any,
    grid_shape: Sequence[int],
    *,
    polish_size: int = 32,
    warmup: int = 3,
    config_overrides: dict[str, Any] | None = None,
) -> RealtimeSBLPlan:
    """Build the strict fixed-budget real-time CUDA Graph plan."""
    tensor = torch.as_tensor(example_data)
    return RealtimeSBLPlan(
        tensor,
        grid_shape,
        polish_size=polish_size,
        warmup=warmup,
        config_overrides=config_overrides,
    )
