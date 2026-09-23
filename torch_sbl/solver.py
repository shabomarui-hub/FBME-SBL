from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil, log, prod
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from .offgrid import OffGridResult, refine_offgrid_fourier
from .operators import PrefixFourierOperator
from .triton_kernels import (
    complex_residual_energy,
    triton_error,
    triton_is_available,
    uamp_denoise,
    uamp_message_update,
    uamp_noise_update,
    uamp_prior_update,
)

SolverMode = Literal["auto", "dense", "cg", "uamp"]
UAMPBackend = Literal["auto", "torch", "triton"]
NoiseUpdate = Literal["em", "evidence"]
Initialization = Literal["uniform", "backprojection"]
CGMatvec = Literal["tbt", "fourier"]
CGPreconditioner = Literal["circulant", "low_rank", "nystrom", "none"]
ProbeType = Literal["rademacher", "hadamard"]
SpatialBoundary = Literal["zero", "circular"]
SnapshotCovarianceModel = Literal["ar1", "full"]


@dataclass(slots=True)
class SBLConfig:
    """Numerical and iteration settings for fast SBL."""

    max_iter: int = 200
    min_iter: int = 3
    tol: float = 1e-4
    hyper_shape: float = 1e-8
    hyper_rate: float = 1e-8
    noise_shape: float = 1e-8
    noise_rate: float = 1e-8
    initial_noise_fraction: float = 1e-2
    initialization: Initialization = "backprojection"
    initialization_blend: float = 0.1
    fixed_noise_variance: float | None = None
    noise_update: NoiseUpdate = "em"
    damping: float = 0.15
    min_variance: float = 1e-12
    max_precision: float = 1e12
    mode: SolverMode = "auto"
    dense_threshold: int = 2048
    cg_tolerance: float = 1e-5
    cg_initial_tolerance: float = 1e-2
    cg_max_iter: int = 150
    cg_check_interval: int = 8
    cg_fixed_iterations: int = 0
    cg_matvec: CGMatvec = "tbt"
    cg_preconditioner: CGPreconditioner = "low_rank"
    cg_preconditioner_rank: int = 32
    cg_damping: float = 0.9
    probe_type: ProbeType = "rademacher"
    hutchinson_probes: int = 32
    max_hutchinson_probes: int = 128
    initial_hutchinson_probes: int = 16
    auto_probe_scaling: bool = True
    adaptive_cg: bool = True
    adaptive_probes: bool = False
    cg_warm_start: bool = True
    cg_min_iter: int = 20
    convergence_patience: int = 2
    convergence_check_interval: int = 2
    cuda_sync_free_cholesky: bool = True
    cuda_cholesky_jitter_multiplier: float = 8.0
    cache_fourier_operator: bool = True
    dense_joint_solve: bool = True
    eager_change_tracking: bool = False
    uamp_damping: float = 0.1
    uamp_adaptive_damping: bool = True
    uamp_max_damping: float = 0.8
    uamp_min_iter: int = 10
    uamp_auto_iterations: bool = True
    uamp_iteration_offset: float = 10.0
    uamp_iteration_scale: float = 1.1
    uamp_polish_size: int = 0
    uamp_polish_iterations: int = 1
    uamp_polish_prune_tail: bool = False
    uamp_polish_compute_truncation_diagnostic: bool = True
    uamp_polish_reinitialize: bool = False
    uamp_polish_damping: float = 0.0
    uamp_polish_evidence_gate: bool = False
    uamp_polish_evidence_margin_factor: float = 0.0
    uamp_polish_residual_exchange_size: int = 0
    uamp_polish_max_truncation_delta: float | None = None
    uamp_polish_global_certificate: bool = False
    uamp_polish_support_repairs: int = 0
    uamp_polish_support_growth_steps: int = 0
    uamp_polish_support_shrink_steps: int = 0
    uamp_polish_support_shrink_batch_size: int = 1
    uamp_polish_support_prior: bool = False
    uamp_polish_support_prior_strength: float = 1.0
    uamp_polish_high_precision: bool = False
    uamp_polish_coordinate_steps: int = 0
    uamp_polish_coordinate_tolerance: float = 1e-6
    uamp_polish_coordinate_early_stop: bool = True
    uamp_polish_noise_coordinate_steps: int = 0
    uamp_polish_noise_tolerance: float = 1e-4
    uamp_polish_noise_objective_tolerance: float = 1e-10
    uamp_polish_joint_cycles: int = 1
    uamp_polish_familywise_error_rate: float | None = None
    uamp_polish_fixed_budget_diagnostics: bool = False
    uamp_support_false_alarm_budget: float | None = None
    uamp_backend: UAMPBackend = "auto"
    active_set: bool = True
    active_set_start: int = 30
    active_set_check_interval: int = 2
    active_set_min_size: int = 32
    active_set_max_size: int = 256
    active_set_tail_fraction: float = 5e-2
    joint_sparsity: bool = False
    joint_noise_variance: bool = True
    spatial_coupling: float = 0.0
    spatial_boundary: SpatialBoundary = "zero"
    learn_spatial_coupling: bool = False
    spatial_coupling_max: float = 1.0
    spatial_coupling_damping: float = 0.5
    spatial_coupling_steps: int = 4
    spatial_coupling_update_interval: int = 2
    spatial_coupling_penalty: float = 1e-3
    spatial_anisotropy_penalty: float = 1e-2
    learn_snapshot_covariance: bool = False
    snapshot_covariance_model: SnapshotCovarianceModel = "ar1"
    snapshot_covariance_damping: float = 0.5
    snapshot_covariance_shrinkage: float = 0.05
    snapshot_covariance_floor: float = 1e-3
    snapshot_covariance_start: int = 20
    snapshot_covariance_warmup: int = 20
    snapshot_covariance_min_correlation: float = 0.5
    snapshot_covariance_min_snr_db: float = 10.0
    snapshot_covariance_min_undersampling: float = 48.0
    phase_error_correction: bool = False
    phase_error_axis: int = 1
    phase_basis_rank: int = 6
    phase_optimization_steps: int = 80
    phase_learning_rate: float = 0.05
    phase_smoothness: float = 5.0
    phase_evidence_penalty: float = 1.0
    phase_min_evidence_gain_per_parameter: float = 1.0
    phase_slq_probes: int = 8
    phase_slq_steps: int = 20
    range_dependent_phase: bool = False
    phase_range_basis_rank: int = 2
    range_migration_correction: bool = False
    migration_basis_rank: int = 4
    migration_max_shift: float = 4.0
    migration_smoothness: float = 1.0
    motion_phase_amplitude_penalty: float = 0.2
    motion_migration_amplitude_penalty: float = 0.2
    motion_refinement_steps: int = 40
    motion_refinement_learning_rate: float = 0.02
    offgrid_refinement: bool = False
    offgrid_components: int | None = None
    offgrid_max_components: int = 64
    offgrid_relative_threshold: float = 0.1
    offgrid_minimum_separation: float = 1.5
    offgrid_max_iter: int = 15
    offgrid_tolerance: float = 1e-4
    offgrid_max_step: float = 0.35
    offgrid_damping: float = 1e-3
    offgrid_ridge: float = 1e-7
    seed: int = 0
    device: str | torch.device = "auto"
    return_history: bool = True
    verbose: bool = False

    @classmethod
    def fast_uamp(cls, **overrides: Any) -> SBLConfig:
        """Return the low-latency FFT-UAMP configuration.

        The preset uses a deterministic undersampling-aware iteration budget,
        disables per-iteration history transfers, and performs one batched
        exact posterior update on the strongest coefficients.
        """
        settings: dict[str, Any] = {
            "mode": "uamp",
            "tol": 0.0,
            "return_history": False,
            "uamp_auto_iterations": True,
            "uamp_polish_size": 32,
            "active_set": False,
        }
        settings.update(overrides)
        return cls(**settings)

    @classmethod
    def fast_hybrid_uamp(cls, **overrides: Any) -> SBLConfig:
        """Return a few-step UAMP plus exact active-set EM configuration.

        Full-grid message passing uses ``ceil(1 + N/M)`` inexpensive steps.
        The strongest coefficients are then reinitialised from their UAMP
        second moments and refined by two exact reduced posterior updates.
        This preset is intended for sparse scenes whose support fits in the
        configured polish set.
        """
        settings: dict[str, Any] = {
            "mode": "uamp",
            "max_iter": 200,
            "min_iter": 1,
            "tol": 0.0,
            "return_history": False,
            "uamp_auto_iterations": True,
            "uamp_min_iter": 3,
            "uamp_iteration_offset": 1.0,
            "uamp_iteration_scale": 1.0,
            "uamp_polish_size": 32,
            "uamp_polish_iterations": 2,
            "uamp_polish_prune_tail": True,
            "uamp_polish_reinitialize": True,
            "uamp_polish_damping": 0.0,
            "uamp_polish_evidence_gate": True,
            "uamp_polish_residual_exchange_size": 0,
            "active_set": False,
        }
        settings.update(overrides)
        return cls(**settings)

    @classmethod
    def paper_hybrid_uamp(cls, **overrides: Any) -> SBLConfig:
        """Return the unified fixed-budget monotone-evidence SBL method.

        A fixed number of complex64 Triton UAMP steps proposes a support,
        followed by three reduced exact posterior/evidence evaluations.  The
        first transition may exchange residual-selected atoms and the second
        performs a support-fixed variance/noise update; both pass through the
        same exact Type-II evidence gate.  The complete fixed workload is
        CUDA-Graph capturable.

        The output reports monotonicity, accepted objective decrease, and
        scale-free gamma/noise EM fixed-point residuals.  These are finite-
        budget mathematical diagnostics; the method does not relabel them as
        full support-coordinate stationarity.
        """
        settings: dict[str, Any] = {
            "device": "cuda",
            "uamp_backend": "triton",
            "tol": 0.0,
            "return_history": False,
            "eager_change_tracking": False,
            "hyper_shape": 0.0,
            "hyper_rate": 0.0,
            "noise_shape": 0.0,
            "noise_rate": 0.0,
            "uamp_iteration_offset": 0.0,
            "uamp_iteration_scale": 0.8,
            "uamp_damping": 0.05,
            "uamp_adaptive_damping": False,
            "uamp_polish_iterations": 3,
            "uamp_polish_global_certificate": False,
            "uamp_polish_support_repairs": 0,
            "uamp_polish_support_growth_steps": 0,
            "uamp_polish_support_shrink_steps": 0,
            "uamp_polish_support_prior": False,
            "uamp_polish_high_precision": False,
            "uamp_polish_coordinate_steps": 0,
            "uamp_polish_noise_coordinate_steps": 0,
            "uamp_polish_residual_exchange_size": 12,
            "uamp_polish_evidence_margin_factor": 64.0,
            "uamp_polish_compute_truncation_diagnostic": False,
            "uamp_polish_familywise_error_rate": None,
            "uamp_polish_fixed_budget_diagnostics": True,
            "uamp_support_false_alarm_budget": 0.5,
            "phase_error_correction": False,
            "range_migration_correction": False,
            "offgrid_refinement": False,
            "learn_snapshot_covariance": False,
        }
        settings.update(overrides)
        return cls.fast_hybrid_uamp(**settings)

    @classmethod
    def realtime_hybrid_uamp(cls, **overrides: Any) -> SBLConfig:
        """Return the fixed-budget CUDA-Graph real-time configuration.

        This profile deliberately keeps data-dependent support-coordinate and
        noise-certificate loops out of the foreground path.  It executes the
        same complex64 Triton UAMP plus three reduced active-set evaluations as
        :meth:`fast_hybrid_uamp`, but pins every option that affects graph
        capture so callers cannot accidentally benchmark the eager research
        solver under a real-time label.

        The returned reconstruction is a fixed-budget Bayesian estimate.  It
        does *not* claim joint ``(support, gamma, noise)`` stationarity; use
        :meth:`taes_hybrid_uamp` as the independent research/certification
        path when that property is required.
        """
        return cls.paper_hybrid_uamp(**overrides)

    @classmethod
    def certified_hybrid_uamp(cls, **overrides: Any) -> SBLConfig:
        """Return the hybrid preset with exact Type-II coordinate diagnostics.

        UAMP proposes a fixed-capacity active model.  All subsequent posterior
        evaluations, support exchanges, and accept/reject decisions use the
        same exact Type-II negative log evidence with inactive variances fixed
        to zero.  The final full-grid single-coordinate test is a stationarity
        diagnostic, not a global-optimality certificate.
        """
        settings: dict[str, Any] = {
            "uamp_polish_global_certificate": True,
            "uamp_polish_iterations": 4,
            "uamp_polish_support_repairs": 1,
        }
        settings.update(overrides)
        return cls.fast_hybrid_uamp(**settings)

    @classmethod
    def taes_hybrid_uamp(cls, **overrides: Any) -> SBLConfig:
        """Return the evidence-monotone Type-II research configuration.

        This preset separates mathematical coordinate stationarity
        (inactive ratio at most one and no profitable active-coordinate
        update) from the optional familywise statistical screening diagnostic.
        Its complete inference path uses complex128/Torch arithmetic so that
        support decisions and the final certificate are evaluated in the same
        numerical model.  It is intended for Monte-Carlo and measured-ISAR
        experiments; ``fast_hybrid_uamp`` remains the complex64/Triton latency
        configuration.
        """
        settings: dict[str, Any] = {
            "uamp_polish_global_certificate": True,
            "uamp_polish_iterations": 8,
            "uamp_polish_support_repairs": 1,
            "uamp_polish_support_growth_steps": 64,
            "uamp_polish_support_shrink_steps": 64,
            "uamp_polish_support_shrink_batch_size": 8,
            "uamp_polish_support_prior": True,
            "uamp_polish_support_prior_strength": 1.0,
            "uamp_polish_high_precision": True,
            "uamp_polish_coordinate_steps": 64,
            "uamp_polish_coordinate_tolerance": 1e-3,
            "uamp_polish_noise_coordinate_steps": 16,
            "uamp_polish_noise_tolerance": 1e-4,
            "uamp_polish_noise_objective_tolerance": 1e-10,
            "uamp_polish_joint_cycles": 8,
            "uamp_polish_familywise_error_rate": 0.05,
        }
        settings.update(overrides)
        return cls.fast_hybrid_uamp(**settings)

    def validate(self) -> None:
        if self.max_iter < 1 or self.min_iter < 1:
            raise ValueError("Iteration counts must be positive.")
        if not 0.0 <= self.damping < 1.0:
            raise ValueError("damping must satisfy 0 <= damping < 1.")
        if self.mode not in ("auto", "dense", "cg", "uamp"):
            raise ValueError("mode must be 'auto', 'dense', 'cg', or 'uamp'.")
        if self.noise_update not in ("em", "evidence"):
            raise ValueError("noise_update must be 'em' or 'evidence'.")
        if self.initialization not in ("uniform", "backprojection"):
            raise ValueError(
                "initialization must be 'uniform' or 'backprojection'."
            )
        if not 0.0 <= self.initialization_blend <= 1.0:
            raise ValueError("initialization_blend must be between 0 and 1.")
        if self.fixed_noise_variance is not None and self.fixed_noise_variance <= 0:
            raise ValueError("fixed_noise_variance must be positive.")
        if self.hutchinson_probes < 1:
            raise ValueError("hutchinson_probes must be positive.")
        if self.initial_hutchinson_probes < 1:
            raise ValueError("initial_hutchinson_probes must be positive.")
        if self.max_hutchinson_probes < self.hutchinson_probes:
            raise ValueError(
                "max_hutchinson_probes must be >= hutchinson_probes."
            )
        if self.cg_tolerance <= 0.0 or self.cg_initial_tolerance <= 0.0:
            raise ValueError("CG tolerances must be positive.")
        if self.cg_check_interval < 1 or self.convergence_check_interval < 1:
            raise ValueError("convergence check intervals must be positive.")
        if self.cg_fixed_iterations < 0:
            raise ValueError("cg_fixed_iterations cannot be negative.")
        if self.cg_fixed_iterations > self.cg_max_iter:
            raise ValueError("cg_fixed_iterations cannot exceed cg_max_iter.")
        if self.cg_min_iter < 1 or self.convergence_patience < 1:
            raise ValueError("CG minimum iterations and patience must be positive.")
        if self.cuda_cholesky_jitter_multiplier <= 0.0:
            raise ValueError("cuda_cholesky_jitter_multiplier must be positive.")
        if self.cg_matvec not in ("tbt", "fourier"):
            raise ValueError("cg_matvec must be 'tbt' or 'fourier'.")
        if self.cg_preconditioner not in (
            "circulant",
            "low_rank",
            "nystrom",
            "none",
        ):
            raise ValueError(
                "cg_preconditioner must be 'circulant', 'low_rank', "
                "'nystrom', or 'none'."
            )
        if self.cg_preconditioner_rank < 0:
            raise ValueError("cg_preconditioner_rank cannot be negative.")
        if not 0.0 <= self.cg_damping < 1.0:
            raise ValueError("cg_damping must satisfy 0 <= cg_damping < 1.")
        if not 0.0 <= self.uamp_damping < 1.0:
            raise ValueError("uamp_damping must satisfy 0 <= uamp_damping < 1.")
        if not self.uamp_damping <= self.uamp_max_damping < 1.0:
            raise ValueError(
                "uamp_max_damping must be in [uamp_damping, 1)."
            )
        if self.uamp_min_iter < 1:
            raise ValueError("uamp_min_iter must be positive.")
        if self.uamp_iteration_offset < 0.0:
            raise ValueError("uamp_iteration_offset cannot be negative.")
        if self.uamp_iteration_scale <= 0.0:
            raise ValueError("uamp_iteration_scale must be positive.")
        if self.uamp_polish_size < 0:
            raise ValueError("uamp_polish_size cannot be negative.")
        if self.uamp_polish_iterations < 1:
            raise ValueError("uamp_polish_iterations must be positive.")
        if not 0 <= self.uamp_polish_residual_exchange_size <= 32:
            raise ValueError(
                "uamp_polish_residual_exchange_size must be in [0, 32]."
            )
        if (
            self.uamp_polish_residual_exchange_size > 0
            and (
                self.uamp_polish_iterations < 2
                or not self.uamp_polish_prune_tail
                or not self.uamp_polish_evidence_gate
            )
        ):
            raise ValueError(
                "Residual support exchange requires at least two polish "
                "evaluations, zero-tail pruning, and the exact evidence gate."
            )
        if not 0.0 <= self.uamp_polish_damping < 1.0:
            raise ValueError(
                "uamp_polish_damping must satisfy 0 <= damping < 1."
            )
        if self.uamp_polish_evidence_margin_factor < 0.0:
            raise ValueError(
                "uamp_polish_evidence_margin_factor cannot be negative."
            )
        if (
            self.uamp_polish_max_truncation_delta is not None
            and self.uamp_polish_max_truncation_delta <= 0.0
        ):
            raise ValueError(
                "uamp_polish_max_truncation_delta must be positive."
            )
        if not 0 <= self.uamp_polish_support_repairs <= 4:
            raise ValueError(
                "uamp_polish_support_repairs must be between 0 and 4."
            )
        if (
            self.uamp_polish_support_repairs > 0
            and not self.uamp_polish_global_certificate
        ):
            raise ValueError(
                "uamp_polish_support_repairs requires "
                "uamp_polish_global_certificate=True."
            )
        if not 0 <= self.uamp_polish_support_growth_steps <= 64:
            raise ValueError(
                "uamp_polish_support_growth_steps must be between 0 and 64."
            )
        if (
            self.uamp_polish_support_growth_steps > 0
            and not self.uamp_polish_global_certificate
        ):
            raise ValueError(
                "uamp_polish_support_growth_steps requires "
                "uamp_polish_global_certificate=True."
            )
        if not 0 <= self.uamp_polish_support_shrink_steps <= 64:
            raise ValueError(
                "uamp_polish_support_shrink_steps must be between 0 and 64."
            )
        if (
            self.uamp_polish_support_shrink_steps > 0
            and not self.uamp_polish_global_certificate
        ):
            raise ValueError(
                "uamp_polish_support_shrink_steps requires "
                "uamp_polish_global_certificate=True."
            )
        if not 1 <= self.uamp_polish_support_shrink_batch_size <= 32:
            raise ValueError(
                "uamp_polish_support_shrink_batch_size must be in [1, 32]."
            )
        if self.uamp_polish_support_prior_strength < 0.0:
            raise ValueError(
                "uamp_polish_support_prior_strength cannot be negative."
            )
        if (
            self.uamp_polish_support_prior
            and not self.uamp_polish_global_certificate
        ):
            raise ValueError(
                "uamp_polish_support_prior requires "
                "uamp_polish_global_certificate=True."
            )
        if not 0 <= self.uamp_polish_coordinate_steps <= 64:
            raise ValueError(
                "uamp_polish_coordinate_steps must be between 0 and 64."
            )
        if self.uamp_polish_coordinate_tolerance < 0.0:
            raise ValueError(
                "uamp_polish_coordinate_tolerance cannot be negative."
            )
        if (
            self.uamp_polish_coordinate_steps > 0
            and not self.uamp_polish_global_certificate
        ):
            raise ValueError(
                "uamp_polish_coordinate_steps requires "
                "uamp_polish_global_certificate=True."
            )
        if not 0 <= self.uamp_polish_noise_coordinate_steps <= 64:
            raise ValueError(
                "uamp_polish_noise_coordinate_steps must be between 0 and 64."
            )
        if self.uamp_polish_noise_tolerance <= 0.0:
            raise ValueError(
                "uamp_polish_noise_tolerance must be positive."
            )
        if self.uamp_polish_noise_objective_tolerance < 0.0:
            raise ValueError(
                "uamp_polish_noise_objective_tolerance cannot be negative."
            )
        if not 1 <= self.uamp_polish_joint_cycles <= 32:
            raise ValueError(
                "uamp_polish_joint_cycles must be between 1 and 32."
            )
        if (
            self.uamp_polish_noise_coordinate_steps > 0
            and not self.uamp_polish_global_certificate
        ):
            raise ValueError(
                "uamp_polish_noise_coordinate_steps requires "
                "uamp_polish_global_certificate=True."
            )
        if (
            self.uamp_polish_noise_coordinate_steps > 0
            and self.fixed_noise_variance is not None
        ):
            raise ValueError(
                "Noise-coordinate refinement is incompatible with "
                "fixed_noise_variance; either estimate noise or fix it."
            )
        if (
            self.uamp_polish_familywise_error_rate is not None
            and not 0.0 < self.uamp_polish_familywise_error_rate < 1.0
        ):
            raise ValueError(
                "uamp_polish_familywise_error_rate must be in (0, 1)."
            )
        if (
            self.uamp_polish_familywise_error_rate is not None
            and not self.uamp_polish_global_certificate
        ):
            raise ValueError(
                "uamp_polish_familywise_error_rate requires "
                "uamp_polish_global_certificate=True."
            )
        if (
            self.uamp_polish_global_certificate
            and not self.uamp_polish_prune_tail
        ):
            raise ValueError(
                "Exact Type-II coordinate diagnostics require "
                "uamp_polish_prune_tail=True so inactive variances define "
                "the zero-tail active model."
            )
        if (
            self.uamp_polish_fixed_budget_diagnostics
            and (
                not self.uamp_polish_prune_tail
                or not self.uamp_polish_evidence_gate
            )
        ):
            raise ValueError(
                "uamp_polish_fixed_budget_diagnostics requires "
                "uamp_polish_prune_tail=True and "
                "uamp_polish_evidence_gate=True."
            )
        if (
            self.uamp_support_false_alarm_budget is not None
            and self.uamp_support_false_alarm_budget <= 0.0
        ):
            raise ValueError(
                "uamp_support_false_alarm_budget must be positive."
            )
        if self.uamp_backend not in ("auto", "torch", "triton"):
            raise ValueError(
                "uamp_backend must be 'auto', 'torch', or 'triton'."
            )
        if self.uamp_polish_high_precision and self.uamp_backend == "triton":
            raise ValueError(
                "uamp_polish_high_precision=True requires the Torch backend: "
                "Triton UAMP currently supports complex64 only. Use "
                "uamp_backend='torch' (or 'auto') for the research path, or "
                "disable high precision for the latency path."
            )
        if self.probe_type not in ("rademacher", "hadamard"):
            raise ValueError("probe_type must be 'rademacher' or 'hadamard'.")
        if self.active_set_start < 1 or self.active_set_check_interval < 1:
            raise ValueError("Active-set iteration counts must be positive.")
        if self.active_set_min_size < 1:
            raise ValueError("active_set_min_size must be positive.")
        if self.active_set_max_size < self.active_set_min_size:
            raise ValueError(
                "active_set_max_size must be >= active_set_min_size."
            )
        if not 0.0 < self.active_set_tail_fraction < 1.0:
            raise ValueError("active_set_tail_fraction must be in (0, 1).")
        if not 0.0 <= self.spatial_coupling <= 1.0:
            raise ValueError("spatial_coupling must be in [0, 1].")
        if self.spatial_boundary not in ("zero", "circular"):
            raise ValueError("spatial_boundary must be 'zero' or 'circular'.")
        if not 0.0 < self.spatial_coupling_max <= 2.0:
            raise ValueError("spatial_coupling_max must be in (0, 2].")
        if not 0.0 <= self.spatial_coupling_damping < 1.0:
            raise ValueError(
                "spatial_coupling_damping must be in [0, 1)."
            )
        if (
            self.spatial_coupling_steps < 1
            or self.spatial_coupling_update_interval < 1
        ):
            raise ValueError(
                "Spatial coupling steps and update interval must be positive."
            )
        if (
            self.spatial_coupling_penalty < 0.0
            or self.spatial_anisotropy_penalty < 0.0
        ):
            raise ValueError("Spatial coupling penalties cannot be negative.")
        if not 0.0 <= self.snapshot_covariance_damping < 1.0:
            raise ValueError(
                "snapshot_covariance_damping must be in [0, 1)."
            )
        if not 0.0 <= self.snapshot_covariance_shrinkage < 1.0:
            raise ValueError(
                "snapshot_covariance_shrinkage must be in [0, 1)."
            )
        if not 0.0 < self.snapshot_covariance_floor < 1.0:
            raise ValueError("snapshot_covariance_floor must be in (0, 1).")
        if self.snapshot_covariance_model not in ("ar1", "full"):
            raise ValueError(
                "snapshot_covariance_model must be 'ar1' or 'full'."
            )
        if (
            self.snapshot_covariance_start < 0
            or self.snapshot_covariance_warmup < 1
        ):
            raise ValueError(
                "Snapshot-covariance start must be nonnegative and "
                "warmup must be positive."
            )
        if not 0.0 <= self.snapshot_covariance_min_correlation < 1.0:
            raise ValueError(
                "snapshot_covariance_min_correlation must be in [0, 1)."
            )
        if self.snapshot_covariance_min_undersampling < 1.0:
            raise ValueError(
                "snapshot_covariance_min_undersampling must be at least one."
            )
        if self.phase_error_axis not in (0, 1):
            raise ValueError("phase_error_axis must be 0 or 1.")
        if self.phase_basis_rank < 1 or self.phase_optimization_steps < 1:
            raise ValueError(
                "Phase basis rank and optimization steps must be positive."
            )
        if self.phase_learning_rate <= 0.0 or self.phase_smoothness < 0.0:
            raise ValueError("Phase optimization settings are invalid.")
        if self.phase_evidence_penalty < 0.0:
            raise ValueError("phase_evidence_penalty cannot be negative.")
        if self.phase_min_evidence_gain_per_parameter < 0.0:
            raise ValueError(
                "phase_min_evidence_gain_per_parameter cannot be negative."
            )
        if self.phase_slq_probes < 1 or self.phase_slq_steps < 2:
            raise ValueError("SLQ requires positive probes and at least 2 steps.")
        if self.phase_range_basis_rank < 1:
            raise ValueError("phase_range_basis_rank must be positive.")
        if self.migration_basis_rank < 1:
            raise ValueError("migration_basis_rank must be positive.")
        if self.migration_max_shift <= 0.0 or self.migration_smoothness < 0.0:
            raise ValueError("Range-migration settings are invalid.")
        if (
            self.motion_phase_amplitude_penalty < 0.0
            or self.motion_migration_amplitude_penalty < 0.0
        ):
            raise ValueError("Motion-amplitude penalties cannot be negative.")
        if (
            self.motion_refinement_steps < 0
            or self.motion_refinement_learning_rate <= 0.0
        ):
            raise ValueError("Motion-refinement settings are invalid.")
        if self.offgrid_components is not None and self.offgrid_components < 1:
            raise ValueError("offgrid_components must be positive.")
        if self.offgrid_max_components < 1:
            raise ValueError("offgrid_max_components must be positive.")
        if not 0.0 < self.offgrid_relative_threshold < 1.0:
            raise ValueError("offgrid_relative_threshold must be in (0, 1).")
        if self.offgrid_minimum_separation < 0.0:
            raise ValueError("offgrid_minimum_separation cannot be negative.")
        if (
            self.offgrid_max_iter < 1
            or self.offgrid_tolerance <= 0.0
            or self.offgrid_max_step <= 0.0
            or self.offgrid_damping <= 0.0
            or self.offgrid_ridge < 0.0
        ):
            raise ValueError("Off-grid refinement settings are invalid.")


@dataclass(slots=True)
class SBLResult:
    """Reconstruction and final posterior state."""

    x: Tensor
    posterior_variance: Tensor
    precision: Tensor
    noise_variance: Tensor
    effective_dof: Tensor
    iterations: int
    converged: bool
    relative_change: float
    mode: str
    arithmetic_dtype: str = ""
    uamp_backend_used: str | None = None
    active_set_size: int = 0
    offgrid: OffGridResult | None = None
    snapshot_covariance: Tensor | None = None
    snapshot_covariance_active: bool | None = None
    snapshot_snr_db: float | None = None
    learned_spatial_coupling: Tensor | None = None
    phase_error: Tensor | None = None
    phase_candidate: Tensor | None = None
    phase_correction_accepted: bool | None = None
    phase_evidence_gain: float | None = None
    range_migration: Tensor | None = None
    range_migration_candidate: Tensor | None = None
    negative_log_evidence: Tensor | None = None
    evidence_trace: Tensor | None = None
    truncation_delta_bound: Tensor | None = None
    hard_pruning_accepted: Tensor | None = None
    truncation_certified: Tensor | None = None
    inactive_max_evidence_ratio: Tensor | None = None
    inactive_best_index: Tensor | None = None
    inactive_evidence_gain: Tensor | None = None
    inactive_optimal_variance: Tensor | None = None
    support_repairs_accepted: Tensor | None = None
    support_exchanges_accepted: Tensor | None = None
    support_growth_accepted: Tensor | None = None
    support_shrink_accepted: Tensor | None = None
    active_max_evidence_gain: Tensor | None = None
    active_best_index: Tensor | None = None
    active_optimal_variance: Tensor | None = None
    active_max_deletion_gain: Tensor | None = None
    coordinate_steps_accepted: Tensor | None = None
    coordinate_stationary: Tensor | None = None
    inactive_ratio_threshold: Tensor | None = None
    familywise_support_certified: Tensor | None = None
    type2_coordinate_stationary: Tensor | None = None
    stationarity_ratio_threshold: Tensor | None = None
    familywise_ratio_threshold: Tensor | None = None
    familywise_no_discovery: Tensor | None = None
    active_model_selected: Tensor | None = None
    support_log_odds_penalty: Tensor | None = None
    penalized_negative_log_evidence: Tensor | None = None
    penalized_evidence_trace: Tensor | None = None
    inactive_penalized_evidence_gain: Tensor | None = None
    active_penalized_evidence_gain: Tensor | None = None
    type2_map_coordinate_stationary: Tensor | None = None
    noise_coordinate_steps_accepted: Tensor | None = None
    noise_evidence_gradient: Tensor | None = None
    noise_fixed_point_residual: Tensor | None = None
    noise_evidence_gain: Tensor | None = None
    noise_coordinate_stationary: Tensor | None = None
    joint_map_coordinate_stationary: Tensor | None = None
    joint_coordinate_cycles: Tensor | None = None
    fixed_budget_gamma_residual: Tensor | None = None
    fixed_budget_noise_residual: Tensor | None = None
    fixed_budget_objective_decrease: Tensor | None = None
    fixed_budget_monotone: Tensor | None = None
    support_mask: Tensor | None = None
    support_score_threshold: Tensor | None = None
    history: dict[str, list[float]] = field(default_factory=dict)


@dataclass(slots=True)
class _UAMPState:
    """Device-resident state for orthogonal-row Fourier UAMP."""

    mean: Tensor
    variance: Tensor
    message: Tensor
    prediction: Tensor
    data_variance: Tensor | None = None
    next_prior_variance: Tensor | None = None


@dataclass(slots=True)
class _InactiveEvidenceCertificate:
    """Coordinatewise Type-II optimality diagnostic outside an active set."""

    max_ratio: Tensor
    best_index: Tensor
    evidence_gain: Tensor
    optimal_variance: Tensor


@dataclass(slots=True)
class _ActiveEvidenceCertificate:
    """Best exact one-coordinate Type-II update inside an active set."""

    max_gain: Tensor
    best_position: Tensor
    best_index: Tensor
    optimal_variance: Tensor
    max_nondeletion_gain: Tensor
    best_nondeletion_position: Tensor
    best_nondeletion_index: Tensor
    best_nondeletion_variance: Tensor
    deletion_gain: Tensor
    max_deletion_gain: Tensor
    best_deletion_position: Tensor


@dataclass(slots=True)
class _NoiseEvidenceCertificate:
    """Exact first-order diagnostic for the scalar noise-variance block."""

    gradient: Tensor
    fixed_point: Tensor
    projected_residual: Tensor
    candidate_variance: Tensor


@lru_cache(maxsize=4)
def _cached_prefix_fourier_operator(
    grid_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
    device_string: str,
    complex_dtype: torch.dtype,
) -> PrefixFourierOperator:
    """Reuse immutable geometry tensors and lazy dense lag tables by shape."""
    return PrefixFourierOperator(
        grid_shape,
        sample_shape,
        device=torch.device(device_string),
        complex_dtype=complex_dtype,
    )


def clear_operator_cache() -> None:
    """Release cached Fourier geometry, for example between unrelated jobs."""
    _cached_prefix_fourier_operator.cache_clear()


def sbl_1d(
    data: Any,
    grid_size: int,
    *,
    config: SBLConfig | None = None,
    observation_mask: Any | None = None,
) -> SBLResult:
    """Reconstruct one or a batch of 1-D spectra.

    ``data`` has shape ``[..., M]`` and the returned coefficient field has
    shape ``[..., grid_size]``.
    """
    return fast_sbl(
        data,
        (int(grid_size),),
        config=config,
        observation_mask=observation_mask,
    )


def sbl_2d(
    data: Any,
    grid_shape: Sequence[int],
    *,
    config: SBLConfig | None = None,
    observation_mask: Any | None = None,
) -> SBLResult:
    """Reconstruct one or a batch of 2-D range-Doppler images.

    ``data`` has shape ``[..., M1, M2]`` and the returned image has shape
    ``[..., N1, N2]``.
    """
    shape = tuple(int(v) for v in grid_shape)
    if len(shape) != 2:
        raise ValueError("sbl_2d requires grid_shape=(N1, N2).")
    if config is not None and config.phase_error_correction:
        from .autofocus import autofocus_sbl_2d

        return autofocus_sbl_2d(
            data,
            shape,
            config=config,
            observation_mask=observation_mask,
        )
    if config is not None and config.learn_snapshot_covariance:
        if observation_mask is not None:
            raise ValueError(
                "Masked correlated-MMV reconstruction is not yet supported."
            )
        from .correlated_mmv import correlated_mmv_sbl_2d

        return correlated_mmv_sbl_2d(data, shape, config=config)
    return fast_sbl(
        data,
        shape,
        config=config,
        observation_mask=observation_mask,
    )


def fast_sbl(
    data: Any,
    grid_shape: Sequence[int],
    *,
    config: SBLConfig | None = None,
    observation_mask: Any | None = None,
) -> SBLResult:
    """Run Fourier SBL without materialising the super-resolution dictionary.

    The observation model is ``y = crop(fftn(x)) + noise``.  Leading
    dimensions are evaluated together on the selected device.  They are
    independent by default; ``joint_sparsity=True`` turns them into MMV
    snapshots with a shared support variance.
    """
    cfg = config or SBLConfig()
    cfg.validate()
    if cfg.learn_snapshot_covariance:
        raise ValueError(
            "learn_snapshot_covariance is available through sbl_2d only."
        )
    if cfg.phase_error_correction:
        raise ValueError(
            "phase_error_correction is available through sbl_2d only."
        )
    grid_shape = tuple(int(v) for v in grid_shape)
    if len(grid_shape) not in (1, 2):
        raise ValueError("grid_shape must contain one or two dimensions.")
    if (
        cfg.uamp_support_false_alarm_budget is not None
        and cfg.uamp_support_false_alarm_budget >= prod(grid_shape)
    ):
        raise ValueError(
            "uamp_support_false_alarm_budget must be smaller than the "
            "reconstruction grid size."
        )
    spatial_enabled = (
        cfg.spatial_coupling > 0.0 or cfg.learn_spatial_coupling
    )
    if spatial_enabled and len(grid_shape) != 2:
        raise ValueError("spatial_coupling is only defined for a 2-D grid.")

    input_tensor = torch.as_tensor(data)
    if input_tensor.ndim < len(grid_shape):
        raise ValueError("data has fewer dimensions than grid_shape.")
    sample_shape = tuple(int(v) for v in input_tensor.shape[-len(grid_shape) :])
    batch_shape = tuple(int(v) for v in input_tensor.shape[: -len(grid_shape)])
    if any(m > n for m, n in zip(sample_shape, grid_shape)):
        raise ValueError("Observed dimensions cannot exceed reconstruction dimensions.")

    device = _resolve_device(cfg.device, input_tensor)
    # The research certificate must describe the numerical model that was
    # actually optimized.  Promoting only the reduced active-set polish is too
    # late: a complex64 UAMP proposal can already have selected a wrong support.
    # Therefore the high-precision preset promotes data, operator and UAMP from
    # the first operation.  The low-latency preset remains complex64/Triton.
    complex_dtype = (
        torch.complex128
        if cfg.uamp_polish_high_precision
        else _complex_dtype(input_tensor.dtype)
    )
    y_full = input_tensor.to(device=device, dtype=complex_dtype)
    mask = None
    if observation_mask is not None:
        mask = torch.as_tensor(
            observation_mask, device=device, dtype=torch.bool
        )
        if tuple(mask.shape) != sample_shape:
            raise ValueError(
                "observation_mask must match the trailing data dimensions."
            )
        if not bool(mask.any()):
            raise ValueError("observation_mask must retain at least one sample.")
        if cfg.offgrid_refinement:
            raise ValueError(
                "Run off-grid refinement after masked reconstruction."
            )
    batch_size = prod(batch_shape) if batch_shape else 1
    if cfg.joint_sparsity and batch_size < 2:
        raise ValueError(
            "joint_sparsity=True requires at least two leading batch items."
        )
    operator = (
        _cached_prefix_fourier_operator(
            grid_shape,
            sample_shape,
            str(device),
            complex_dtype,
        )
        if cfg.cache_fourier_operator and mask is None
        else PrefixFourierOperator(
            grid_shape,
            sample_shape,
            device=device,
            complex_dtype=complex_dtype,
            observation_mask=mask,
        )
    )
    y = operator.restrict(
        y_full.reshape(batch_size, prod(sample_shape), 1)
    ).squeeze(-1)
    auto_dense_threshold = (
        cfg.dense_threshold
        if device.type == "cuda"
        else min(cfg.dense_threshold, 512)
    )
    mode = (
        "dense"
        if cfg.mode == "auto" and operator.n_samples <= auto_dense_threshold
        else "cg"
        if cfg.mode == "auto"
        else cfg.mode
    )
    uamp_triton_enabled = (
        mode == "uamp"
        and cfg.uamp_backend != "torch"
        and device.type == "cuda"
        and complex_dtype == torch.complex64
        and triton_is_available()
    )
    uamp_backend_used = (
        "triton"
        if mode == "uamp" and uamp_triton_enabled
        else "torch"
        if mode == "uamp"
        else None
    )
    if (
        mode == "uamp"
        and cfg.uamp_backend == "triton"
        and not uamp_triton_enabled
    ):
        reason = triton_error()
        detail = f" Last Triton error: {reason}" if reason is not None else ""
        raise RuntimeError(
            "uamp_backend='triton' requires CUDA, complex64 input, and a "
            f"working Triton runtime.{detail}"
        )

    real_dtype = operator.real_dtype
    eps = torch.finfo(real_dtype).eps
    data_power = y.abs().square().mean(dim=1).real.clamp_min(eps)
    variance_floor = torch.maximum(
        data_power * cfg.min_variance,
        torch.full_like(data_power, torch.finfo(real_dtype).tiny),
    )
    uniform_variance = data_power[:, None] / operator.n_grid
    initial_backprojection: Tensor | None = None
    if cfg.initialization == "backprojection":
        backprojection = operator.adjoint_flat(y)
        if mode == "uamp":
            initial_backprojection = backprojection
        backprojection_energy = backprojection.abs().square().real
        structured_variance = (
            backprojection_energy
            / backprojection_energy.sum(dim=1, keepdim=True).clamp_min(eps)
            * data_power[:, None]
        )
        prior_variance = (
            cfg.initialization_blend * structured_variance
            + (1.0 - cfg.initialization_blend) * uniform_variance
        )
    else:
        prior_variance = uniform_variance.expand(
            -1, operator.n_grid
        ).clone()
    if cfg.joint_sparsity:
        # M-SBL: every snapshot uses the same row variance.  Averaging the
        # scale-aware initial estimates avoids privileging any one snapshot.
        prior_variance = prior_variance.mean(dim=0, keepdim=True).expand(
            batch_size, -1
        ).clone()
        prior_floor = variance_floor.max().expand(batch_size)
    else:
        prior_floor = variance_floor
    prior_variance = prior_variance.clamp_min(prior_floor[:, None])
    directional_coupling = None
    if cfg.learn_spatial_coupling:
        initial_coupling = (
            cfg.spatial_coupling
            if cfg.spatial_coupling > 0.0
            else min(0.1, cfg.spatial_coupling_max)
        )
        directional_coupling = torch.full(
            (batch_size, 2),
            initial_coupling,
            device=device,
            dtype=real_dtype,
        )
        if cfg.joint_sparsity:
            directional_coupling[:] = directional_coupling[:1]
        pattern_precision = _initial_anisotropic_pattern_precision(
            prior_variance,
            grid_shape,
            directional_coupling,
            cfg.spatial_boundary,
        )
    elif cfg.spatial_coupling > 0.0:
        pattern_precision = _initial_pattern_precision(
            prior_variance,
            grid_shape,
            cfg.spatial_coupling,
            cfg.spatial_boundary,
        )
    else:
        pattern_precision = None
    if pattern_precision is not None:
        initial_precision = (
            _effective_anisotropic_pattern_precision(
                pattern_precision,
                grid_shape,
                directional_coupling,
                cfg.spatial_boundary,
            )
            if directional_coupling is not None
            else _effective_pattern_precision(
                pattern_precision,
                grid_shape,
                cfg.spatial_coupling,
                cfg.spatial_boundary,
            )
        )
        prior_variance = initial_precision.clamp(
            min=1.0 / cfg.max_precision,
            max=cfg.max_precision,
        ).reciprocal()
        prior_variance = prior_variance.clamp_min(prior_floor[:, None])
    if cfg.fixed_noise_variance is None:
        noise_variance = (
            cfg.initial_noise_fraction * data_power
        ).clamp_min(variance_floor)
    else:
        noise_variance = torch.full_like(data_power, cfg.fixed_noise_variance)
    shared_mmv = cfg.joint_sparsity and cfg.joint_noise_variance
    if shared_mmv:
        noise_variance = noise_variance.mean().expand(batch_size).clone()
    uamp_state = (
        _UAMPState(
            mean=torch.zeros(
                (batch_size, operator.n_grid),
                device=device,
                dtype=complex_dtype,
            ),
            variance=prior_variance.clone(),
            message=torch.zeros(
                (batch_size, operator.n_samples),
                device=device,
                dtype=complex_dtype,
            ),
            prediction=torch.zeros(
                (batch_size, operator.n_samples),
                device=device,
                dtype=complex_dtype,
            ),
            data_variance=prior_variance.sum(dim=1),
        )
        if mode == "uamp"
        else None
    )
    uamp_damping = cfg.uamp_damping
    if mode == "uamp" and cfg.uamp_adaptive_damping:
        undersampling = operator.n_grid / operator.n_samples
        uamp_damping = max(
            uamp_damping,
            min(cfg.uamp_max_damping, 0.02 * undersampling),
        )

    probes = None
    lifted_probes = None
    maximum_probes = cfg.hutchinson_probes
    if mode == "cg":
        if cfg.auto_probe_scaling:
            undersampling = operator.n_grid / operator.n_samples
            recommended_float = 8.0 * undersampling**0.5
            if undersampling > 16.0:
                # Coherent Fourier atoms amplify diagonal-estimator variance
                # at extreme super-resolution ratios.
                recommended_float = max(
                    recommended_float, 2.0 * undersampling
                )
            recommended = int(recommended_float + 0.999999)
            maximum_probes = min(
                cfg.max_hutchinson_probes,
                max(cfg.hutchinson_probes, recommended),
            )
        probes = _make_probes(
            1 if shared_mmv else batch_size,
            operator.n_samples,
            maximum_probes,
            device=device,
            dtype=complex_dtype,
            seed=cfg.seed,
            probe_type=cfg.probe_type,
            probe_shape=(
                operator.sample_shape if operator.is_complete else None
            ),
        )
        lifted_probes = operator.adjoint(probes)

    history_tensors: dict[str, list[Tensor]] = {
        "relative_change": [],
        "noise_variance": [],
        "effective_dof": [],
        "cg_iterations": [],
        "hutchinson_probes": [],
        "cg_tolerance": [],
        "active_set_size": [],
        "spatial_vertical_coupling": [],
        "spatial_horizontal_coupling": [],
    }
    previous_mean = torch.zeros(
        (batch_size, operator.n_grid), device=device, dtype=complex_dtype
    )
    relative_change = float("inf")
    converged = False
    effective_dof = torch.zeros(batch_size, device=device, dtype=real_dtype)
    cg_solution: Tensor | None = None
    active_indices: Tensor | None = None
    active_columns: Tensor | None = None
    active_gram: Tensor | None = None
    negative_log_evidence: Tensor | None = None
    active_evidence_trace: Tensor | None = None
    truncation_delta_bound: Tensor | None = None
    hard_pruning_accepted: Tensor | None = None
    inactive_max_evidence_ratio: Tensor | None = None
    inactive_best_index: Tensor | None = None
    inactive_evidence_gain: Tensor | None = None
    inactive_optimal_variance: Tensor | None = None
    support_repairs_accepted: Tensor | None = None
    support_growth_accepted: Tensor | None = None
    active_max_evidence_gain: Tensor | None = None
    active_best_index: Tensor | None = None
    active_optimal_variance: Tensor | None = None
    active_max_deletion_gain: Tensor | None = None
    coordinate_steps_accepted: Tensor | None = None
    coordinate_stationary: Tensor | None = None
    inactive_ratio_threshold: Tensor | None = None
    familywise_support_certified: Tensor | None = None
    type2_coordinate_stationary: Tensor | None = None
    stationarity_ratio_threshold: Tensor | None = None
    familywise_ratio_threshold: Tensor | None = None
    familywise_no_discovery: Tensor | None = None
    active_model_selected: Tensor | None = None
    support_shrink_accepted: Tensor | None = None
    support_log_odds_penalty: Tensor | None = None
    penalized_negative_log_evidence: Tensor | None = None
    penalized_evidence_trace: Tensor | None = None
    inactive_penalized_evidence_gain: Tensor | None = None
    active_penalized_evidence_gain: Tensor | None = None
    type2_map_coordinate_stationary: Tensor | None = None
    noise_coordinate_steps_accepted: Tensor | None = None
    noise_evidence_gradient: Tensor | None = None
    noise_fixed_point_residual: Tensor | None = None
    noise_evidence_gain: Tensor | None = None
    noise_coordinate_stationary: Tensor | None = None
    joint_map_coordinate_stationary: Tensor | None = None
    joint_coordinate_cycles: Tensor | None = None
    fixed_budget_gamma_residual: Tensor | None = None
    fixed_budget_noise_residual: Tensor | None = None
    fixed_budget_objective_decrease: Tensor | None = None
    fixed_budget_monotone: Tensor | None = None
    last_change = torch.full((), float("inf"), device=device, dtype=real_dtype)
    convergence_streak = 0
    minimum_iterations = (
        max(cfg.min_iter, cfg.cg_min_iter)
        if mode == "cg"
        else max(cfg.min_iter, cfg.uamp_min_iter)
        if mode == "uamp"
        else cfg.min_iter
    )
    iteration_limit = cfg.max_iter
    if mode == "uamp" and cfg.uamp_auto_iterations:
        # A partial Fourier dictionary has orthogonal rows, so UAMP has no
        # inner linear solve.  Empirically the iterations required for stable
        # support formation grow approximately linearly with N/M.  Capping by
        # max_iter keeps the user's hard latency limit authoritative.
        uamp_budget = ceil(
            cfg.uamp_iteration_offset
            + cfg.uamp_iteration_scale
            * operator.n_grid
            / operator.n_samples
        )
        iteration_limit = min(
            cfg.max_iter,
            max(minimum_iterations, uamp_budget),
        )
    convergence_enabled = (
        cfg.tol > 0.0 and minimum_iterations < iteration_limit
    )

    with torch.no_grad():
        for iteration in range(1, iteration_limit + 1):
            if (
                mode == "cg"
                and cfg.active_set
                and not cfg.joint_sparsity
                and not spatial_enabled
                and active_indices is None
                and batch_size == 1
                and iteration >= cfg.active_set_start
                and (
                    (iteration - cfg.active_set_start)
                    % cfg.active_set_check_interval
                    == 0
                )
            ):
                active_indices = _select_active_indices(
                    prior_variance,
                    minimum_size=cfg.active_set_min_size,
                    maximum_size=cfg.active_set_max_size,
                    tail_fraction=cfg.active_set_tail_fraction,
                )
                if active_indices is not None:
                    active_columns = operator.fourier_columns(active_indices)
                    active_gram = active_columns.mH @ active_columns
                    # The data-space Krylov state has no use after switching
                    # to the exact reduced coefficient-space posterior.
                    cg_solution = None
            if mode == "cg":
                probe_count, cg_tolerance = _cg_schedule(
                    iteration, cfg, maximum_probes
                )
                used_probe_count = (
                    probe_count if active_indices is None else 0
                )
                cg_tolerance = max(
                    cg_tolerance,
                    3.0 * torch.finfo(real_dtype).eps**0.5,
                )
                iteration_probes = (
                    probes[:, :, :probe_count]
                    if probes is not None
                    else None
                )
                iteration_lifted_probes = (
                    lifted_probes[:, :, :probe_count]
                    if lifted_probes is not None
                    else None
                )
                warm_start = (
                    _resize_warm_start(
                        cg_solution,
                        (
                            batch_size + probe_count
                            if shared_mmv
                            else 1 + probe_count
                        ),
                        1 if shared_mmv else batch_size,
                        operator.n_samples,
                        device,
                        complex_dtype,
                    )
                    if cfg.cg_warm_start
                    else None
                )
            else:
                # Dense and UAMP do not use stochastic probes or Krylov
                # schedules.  Keeping this branch out of their inner loop
                # removes Python dispatch and scalar work from low-latency
                # small-grid reconstruction.
                probe_count = 0
                used_probe_count = 0
                cg_tolerance = cfg.cg_tolerance
                iteration_probes = None
                iteration_lifted_probes = None
                warm_start = None
            if mode == "uamp":
                if uamp_state is None:
                    raise RuntimeError("UAMP state was not initialized.")
                (
                    mean,
                    posterior_variance,
                    uamp_effective_dof,
                    uamp_state,
                ) = _uamp_posterior(
                    y,
                    prior_variance,
                    noise_variance,
                    operator,
                    uamp_state,
                    damping=uamp_damping,
                    use_triton=uamp_triton_enabled,
                    strict_triton=cfg.uamp_backend == "triton",
                    initial_backprojection=(
                        initial_backprojection if iteration == 1 else None
                    ),
                    fused_prior_floor=(
                        prior_floor
                        if (
                            uamp_triton_enabled
                            and not cfg.joint_sparsity
                            and not spatial_enabled
                        )
                        else None
                    ),
                    hyper_shape=cfg.hyper_shape,
                    hyper_rate=cfg.hyper_rate,
                    prior_damping=cfg.damping,
                    prior_maximum=cfg.max_precision,
                )
                uamp_prediction = uamp_state.prediction
                cg_iterations = 0
                solved_state = None
            else:
                uamp_effective_dof = None
                (
                    mean,
                    posterior_variance,
                    data_diagonal,
                    cg_iterations,
                    solved_state,
                ) = (
                    _shared_mmv_posterior(
                        y,
                        prior_variance[:1],
                        noise_variance[:1],
                        operator,
                        mode,
                        cfg,
                        iteration_probes,
                        iteration_lifted_probes,
                        cg_tolerance=cg_tolerance,
                        initial_solution=warm_start,
                        active_indices=None,
                        active_columns=None,
                        active_gram=None,
                    )
                    if shared_mmv
                    else _posterior(
                        y,
                        prior_variance,
                        noise_variance,
                        operator,
                        mode,
                        cfg,
                        iteration_probes,
                        iteration_lifted_probes,
                        cg_tolerance=cg_tolerance,
                        initial_solution=warm_start,
                        active_indices=active_indices,
                        active_columns=active_columns,
                        active_gram=active_gram,
                    )
                )
            cg_solution = solved_state if cfg.cg_warm_start else None

            # gamma_i in the MATLAB code: effective degree of freedom.
            if uamp_effective_dof is not None:
                maximum_dof = operator.n_samples * (1.0 - 10.0 * eps)
                effective_dof = uamp_effective_dof.clamp_max(maximum_dof)
                relevance = None
            else:
                relevance = _project_relevance(
                    prior_variance,
                    data_diagonal,
                    operator.n_samples,
                    eps,
                )
                effective_dof = relevance.sum(dim=1)
            keep = (
                max(cfg.damping, cfg.cg_damping)
                if mode == "cg"
                else cfg.damping
            )
            prior_already_damped = False
            if spatial_enabled:
                if pattern_precision is None:
                    raise RuntimeError("Pattern precision was not initialized.")
                if cfg.learn_spatial_coupling:
                    if directional_coupling is None:
                        raise RuntimeError(
                            "Directional coupling was not initialized."
                        )
                    structure_moment = (
                        mean.abs().square() + posterior_variance
                    )
                    if cfg.joint_sparsity:
                        structure_moment = structure_moment.mean(
                            dim=0, keepdim=True
                        )
                        structure_precision = pattern_precision[:1]
                        current_coupling = directional_coupling[:1]
                    else:
                        structure_precision = pattern_precision
                        current_coupling = directional_coupling
                    if (
                        (iteration - 1)
                        % cfg.spatial_coupling_update_interval
                        == 0
                    ):
                        learned_coupling = _learn_anisotropic_coupling(
                            structure_precision,
                            structure_moment,
                            grid_shape,
                            current_coupling,
                            cfg.spatial_boundary,
                            maximum=cfg.spatial_coupling_max,
                            steps=cfg.spatial_coupling_steps,
                            magnitude_penalty=(
                                cfg.spatial_coupling_penalty
                            ),
                            anisotropy_penalty=(
                                cfg.spatial_anisotropy_penalty
                            ),
                        )
                        learned_coupling = (
                            cfg.spatial_coupling_damping
                            * current_coupling
                            + (1.0 - cfg.spatial_coupling_damping)
                            * learned_coupling
                        )
                    else:
                        learned_coupling = current_coupling
                    if cfg.joint_sparsity:
                        directional_coupling = learned_coupling.expand(
                            batch_size, -1
                        ).clone()
                    else:
                        directional_coupling = learned_coupling
                    new_pattern_precision = (
                        _anisotropic_pattern_precision_update(
                            structure_moment,
                            grid_shape,
                            learned_coupling,
                            cfg.spatial_boundary,
                            hyper_shape=cfg.hyper_shape,
                            hyper_rate=cfg.hyper_rate,
                            max_precision=cfg.max_precision,
                        )
                    )
                    if cfg.joint_sparsity:
                        new_pattern_precision = (
                            new_pattern_precision.expand(
                                batch_size, -1
                            )
                        )
                else:
                    new_pattern_precision = _pattern_precision_update(
                        mean,
                        posterior_variance,
                        grid_shape,
                        cfg.spatial_coupling,
                        cfg.spatial_boundary,
                        joint_sparsity=cfg.joint_sparsity,
                        hyper_shape=cfg.hyper_shape,
                        hyper_rate=cfg.hyper_rate,
                        max_precision=cfg.max_precision,
                    )
                pattern_precision = (
                    keep * pattern_precision
                    + (1.0 - keep) * new_pattern_precision
                )
                effective_pattern_precision = (
                    _effective_anisotropic_pattern_precision(
                        pattern_precision,
                        grid_shape,
                        directional_coupling,
                        cfg.spatial_boundary,
                    )
                    if directional_coupling is not None
                    else _effective_pattern_precision(
                        pattern_precision,
                        grid_shape,
                        cfg.spatial_coupling,
                        cfg.spatial_boundary,
                    )
                )
                new_precision = effective_pattern_precision.clamp(
                    min=1.0 / cfg.max_precision,
                    max=cfg.max_precision,
                )
                new_prior_variance = new_precision.reciprocal()
            elif mode == "uamp":
                # Variational UAMP-SBL precision update.  Unlike the MacKay
                # fixed-point rule below, this uses the approximate posterior
                # second moment directly and remains fully elementwise.
                if (
                    uamp_state is not None
                    and uamp_state.next_prior_variance is not None
                ):
                    new_prior_variance = (
                        uamp_state.next_prior_variance
                    )
                    prior_already_damped = True
                elif (
                    uamp_triton_enabled
                    and triton_is_available()
                    and not cfg.joint_sparsity
                ):
                    try:
                        new_prior_variance = uamp_prior_update(
                            mean,
                            posterior_variance,
                            prior_variance,
                            prior_floor,
                            hyper_shape=cfg.hyper_shape,
                            hyper_rate=cfg.hyper_rate,
                            damping=keep,
                            maximum=cfg.max_precision,
                        )
                        prior_already_damped = True
                    except Exception as exc:
                        if cfg.uamp_backend == "triton":
                            raise RuntimeError(
                                "Triton ARD update kernel failed."
                            ) from exc
                if not prior_already_damped:
                    second_moment = (
                        mean.abs().square() + posterior_variance
                    )
                    if cfg.joint_sparsity:
                        second_moment = second_moment.mean(
                            dim=0, keepdim=True
                        )
                    new_prior_variance = (
                        second_moment + cfg.hyper_rate
                    ) / (1.0 + cfg.hyper_shape)
                    minimum_prior = (
                        prior_floor.max()
                        if cfg.joint_sparsity
                        else prior_floor[:, None]
                    )
                    new_prior_variance = torch.maximum(
                        new_prior_variance, minimum_prior
                    ).clamp_max(cfg.max_precision)
                    if cfg.joint_sparsity:
                        new_prior_variance = new_prior_variance.expand(
                            batch_size, -1
                        )
            elif cfg.joint_sparsity:
                # Complex M-SBL EM update:
                # gamma_i <- L^{-1} sum_l E[|x_{i,l}|^2].
                second_moment = (
                    mean.abs().square() + posterior_variance
                ).mean(dim=0, keepdim=True)
                shared_variance = (
                    second_moment + cfg.hyper_rate
                ) / (1.0 + cfg.hyper_shape)
                shared_variance = shared_variance.clamp(
                    min=prior_floor.max(),
                    max=cfg.max_precision,
                )
                new_prior_variance = shared_variance.expand(
                    batch_size, -1
                )
            else:
                new_precision = (
                    relevance + cfg.hyper_shape
                ) / (mean.abs().square() + cfg.hyper_rate)
                new_precision = new_precision.clamp(
                    min=1.0 / cfg.max_precision,
                    max=cfg.max_precision,
                )
                new_prior_variance = new_precision.reciprocal()
            if active_indices is not None:
                active_mask = torch.zeros_like(
                    prior_variance, dtype=torch.bool
                )
                active_mask.scatter_(1, active_indices, True)
                # ARD deletion is permanent.  The omitted coefficients retain
                # their small variance as an isotropic covariance loading in
                # the reduced posterior instead of being spuriously revived.
                new_prior_variance = torch.where(
                    active_mask, new_prior_variance, prior_variance
                )

            residual_energy = None
            noise_already_damped = False
            new_noise_variance = None
            if (
                mode == "uamp"
                and uamp_triton_enabled
                and triton_is_available()
                and cfg.fixed_noise_variance is None
                and not shared_mmv
            ):
                try:
                    new_noise_variance = uamp_noise_update(
                        y,
                        uamp_prediction,
                        noise_variance,
                        effective_dof,
                        variance_floor,
                        noise_shape=cfg.noise_shape,
                        noise_rate=cfg.noise_rate,
                        damping=keep,
                        evidence_update=cfg.noise_update == "evidence",
                        epsilon=eps,
                    )
                    noise_already_damped = (
                        new_noise_variance is not None
                    )
                except Exception as exc:
                    if cfg.uamp_backend == "triton":
                        raise RuntimeError(
                            "Triton fused noise update kernel failed."
                        ) from exc
            if (
                not noise_already_damped
                and
                mode == "uamp"
                and uamp_triton_enabled
                and triton_is_available()
            ):
                try:
                    residual_energy = complex_residual_energy(
                        y, uamp_prediction
                    )
                except Exception as exc:
                    if cfg.uamp_backend == "triton":
                        raise RuntimeError(
                            "Triton residual reduction kernel failed."
                        ) from exc
            if not noise_already_damped and residual_energy is None:
                residual = (
                    y - uamp_prediction
                    if mode == "uamp"
                    else y - operator.forward(mean[:, :, None]).squeeze(-1)
                )
                residual_energy = residual.abs().square().sum(dim=1)
            if noise_already_damped:
                if new_noise_variance is None:
                    raise RuntimeError("Fused noise update lost its result.")
            elif cfg.fixed_noise_variance is not None:
                new_noise_variance = torch.full_like(
                    noise_variance, cfg.fixed_noise_variance
                )
            elif cfg.noise_update == "em":
                # EM form of the noise update.  It has the same fixed point
                # as the evidence formula, but is much safer when dof ~= M.
                new_noise_variance = (
                    residual_energy
                    + noise_variance * effective_dof
                    + cfg.noise_rate
                ) / (operator.n_samples + cfg.noise_shape)
            else:
                noise_denominator = (
                    operator.n_samples
                    - effective_dof
                    + cfg.noise_shape
                ).clamp_min(eps)
                new_noise_variance = (
                    residual_energy + cfg.noise_rate
                ) / noise_denominator
            new_noise_variance = torch.maximum(new_noise_variance, variance_floor)
            if shared_mmv:
                new_noise_variance = (
                    new_noise_variance.mean()
                    .clamp_min(variance_floor.max())
                    .expand(batch_size)
                )

            if spatial_enabled:
                # Damping was applied in the base precision domain so that
                # the coupled prior remains exactly eta=alpha+beta*N(alpha).
                prior_variance = new_prior_variance
            elif prior_already_damped:
                prior_variance = new_prior_variance
            else:
                prior_variance = (
                    keep * prior_variance
                    + (1.0 - keep) * new_prior_variance
                )
            prior_variance = prior_variance.clamp_min(prior_floor[:, None])
            if cfg.fixed_noise_variance is None:
                noise_variance = (
                    new_noise_variance
                    if noise_already_damped
                    else (
                        keep * noise_variance
                        + (1.0 - keep) * new_noise_variance
                    )
                )
            else:
                noise_variance = new_noise_variance

            should_check = (
                convergence_enabled
                and iteration >= minimum_iterations
                and (
                    (iteration - minimum_iterations)
                    % cfg.convergence_check_interval
                    == 0
                    or iteration == iteration_limit
                )
            )
            verbose_check = cfg.verbose and (
                iteration == 1 or iteration % 10 == 0
            )
            need_change = (
                cfg.eager_change_tracking
                or cfg.return_history
                or should_check
                or verbose_check
                or iteration == iteration_limit
            )
            if need_change:
                denominator = mean.norm(dim=1).clamp_min(eps)
                per_batch_change = (
                    mean - previous_mean
                ).norm(dim=1) / denominator
                last_change = per_batch_change.max()
            previous_mean = mean

            if cfg.return_history:
                history_tensors["relative_change"].append(last_change.detach())
                history_tensors["noise_variance"].append(
                    noise_variance.mean().detach()
                )
                history_tensors["effective_dof"].append(
                    effective_dof.mean().detach()
                )
                history_tensors["cg_iterations"].append(
                    torch.as_tensor(
                        cg_iterations, device=device, dtype=real_dtype
                    )
                )
                history_tensors["hutchinson_probes"].append(
                    torch.as_tensor(
                        used_probe_count if mode == "cg" else 0,
                        device=device,
                        dtype=real_dtype,
                    )
                )
                history_tensors["cg_tolerance"].append(
                    torch.as_tensor(
                        cg_tolerance if mode == "cg" else 0.0,
                        device=device,
                        dtype=real_dtype,
                    )
                )
                history_tensors["active_set_size"].append(
                    torch.as_tensor(
                        active_indices.shape[1]
                        if active_indices is not None
                        else 0,
                        device=device,
                        dtype=real_dtype,
                    )
                )
                if directional_coupling is None:
                    vertical_coupling = torch.as_tensor(
                        cfg.spatial_coupling,
                        device=device,
                        dtype=real_dtype,
                    )
                    horizontal_coupling = vertical_coupling
                else:
                    vertical_coupling = directional_coupling[
                        :, 0
                    ].mean()
                    horizontal_coupling = directional_coupling[
                        :, 1
                    ].mean()
                history_tensors["spatial_vertical_coupling"].append(
                    vertical_coupling.detach()
                )
                history_tensors["spatial_horizontal_coupling"].append(
                    horizontal_coupling.detach()
                )
            if should_check:
                relative_change = float(last_change.item())
            if verbose_check:
                if not should_check:
                    relative_change = float(last_change.item())
                print(
                    f"[SBL] iter={iteration:4d} change={relative_change:.3e} "
                    f"noise={noise_variance.mean().item():.3e} "
                    f"dof={effective_dof.mean().item():.2f} "
                    f"cg={cg_iterations:3d} probes={used_probe_count:2d} "
                    f"active={active_indices.shape[1] if active_indices is not None else 0}"
                )
            if should_check:
                convergence_streak = (
                    convergence_streak + 1
                    if relative_change <= cfg.tol
                    else 0
                )
                if convergence_streak >= cfg.convergence_patience:
                    converged = True
                    break

        # Return a posterior consistent with the final hyperparameters.
        if mode == "uamp":
            if uamp_state is None:
                raise RuntimeError("UAMP state was not initialized.")
            (
                mean,
                posterior_variance,
                uamp_effective_dof,
                uamp_state,
            ) = _uamp_posterior(
                y,
                prior_variance,
                noise_variance,
                operator,
                uamp_state,
                damping=uamp_damping,
                use_triton=uamp_triton_enabled,
                strict_triton=cfg.uamp_backend == "triton",
                compute_prediction=False,
            )
            if cfg.uamp_polish_size > 0:
                polish_size = min(
                    cfg.uamp_polish_size, operator.n_grid
                )
                polish_score = mean.abs().square() + posterior_variance
                active_indices = torch.topk(
                    polish_score, polish_size, dim=1
                ).indices
                (
                    mean,
                    posterior_variance,
                    data_diagonal,
                    prior_variance,
                    noise_variance,
                    negative_log_evidence,
                    active_evidence_trace,
                    truncation_delta_bound,
                    hard_pruning_accepted,
                    inactive_max_evidence_ratio,
                    inactive_best_index,
                    inactive_evidence_gain,
                    inactive_optimal_variance,
                    support_repairs_accepted,
                    support_growth_accepted,
                    active_max_evidence_gain,
                    active_best_index,
                    active_optimal_variance,
                    active_max_deletion_gain,
                    coordinate_steps_accepted,
                    coordinate_stationary,
                    inactive_ratio_threshold,
                    familywise_support_certified,
                    active_indices,
                    stationarity_ratio_threshold,
                    familywise_ratio_threshold,
                    familywise_no_discovery,
                    active_model_selected,
                    support_shrink_accepted,
                    support_log_odds_penalty,
                    penalized_negative_log_evidence,
                    penalized_evidence_trace,
                    inactive_penalized_evidence_gain,
                    active_penalized_evidence_gain,
                    type2_map_coordinate_stationary,
                    noise_coordinate_steps_accepted,
                    noise_evidence_gradient,
                    noise_fixed_point_residual,
                    noise_evidence_gain,
                    noise_coordinate_stationary,
                    joint_map_coordinate_stationary,
                    joint_coordinate_cycles,
                ) = _uamp_active_em_polish(
                    y,
                    mean,
                    posterior_variance,
                    prior_variance,
                    noise_variance,
                    operator,
                    active_indices,
                    prior_floor,
                    cfg,
                )
                type2_coordinate_stationary = coordinate_stationary
                if cfg.uamp_polish_fixed_budget_diagnostics:
                    (
                        fixed_budget_gamma_residual,
                        fixed_budget_noise_residual,
                    ) = _fixed_budget_em_residuals(
                        y,
                        mean,
                        posterior_variance,
                        prior_variance,
                        noise_variance,
                        active_indices,
                        prior_floor,
                        operator,
                        cfg,
                    )
                    if active_evidence_trace is None:
                        raise RuntimeError(
                            "Fixed-budget diagnostics require evidence trace."
                        )
                    evidence_finite = torch.isfinite(
                        active_evidence_trace
                    ).all(dim=1)
                    evidence_nonincreasing = (
                        active_evidence_trace[:, 1:]
                        <= active_evidence_trace[:, :-1]
                    ).all(dim=1)
                    fixed_budget_monotone = (
                        evidence_finite & evidence_nonincreasing
                    )
                    fixed_budget_objective_decrease = (
                        active_evidence_trace[:, 0]
                        - active_evidence_trace[:, -1]
                    ).clamp_min(0.0)
                uamp_effective_dof = None
        else:
            final_probes = probes if mode == "cg" else None
            final_warm_start = (
                _resize_warm_start(
                    cg_solution,
                    (
                        batch_size + maximum_probes
                        if shared_mmv
                        else 1 + maximum_probes
                    ),
                    1 if shared_mmv else batch_size,
                    operator.n_samples,
                    device,
                    complex_dtype,
                )
                if mode == "cg" and cfg.cg_warm_start
                else None
            )
            final_arguments = dict(
                cg_tolerance=max(
                    cfg.cg_tolerance,
                    3.0 * torch.finfo(real_dtype).eps**0.5,
                ),
                initial_solution=final_warm_start,
                active_indices=active_indices,
                active_columns=active_columns,
                active_gram=active_gram,
            )
            mean, posterior_variance, data_diagonal, _, _ = (
                _shared_mmv_posterior(
                    y,
                    prior_variance[:1],
                    noise_variance[:1],
                    operator,
                    mode,
                    cfg,
                    final_probes,
                    lifted_probes,
                    **{
                        **final_arguments,
                        "active_indices": None,
                        "active_columns": None,
                        "active_gram": None,
                    },
                )
                if shared_mmv
                else _posterior(
                    y,
                    prior_variance,
                    noise_variance,
                    operator,
                    mode,
                    cfg,
                    final_probes,
                    lifted_probes,
                    **final_arguments,
                )
            )
        effective_dof = (
            uamp_effective_dof.clamp_max(
                operator.n_samples * (1.0 - 10.0 * eps)
            )
            if mode == "uamp" and uamp_effective_dof is not None
            else _project_relevance(
                prior_variance,
                data_diagonal,
                operator.n_samples,
                eps,
            ).sum(1)
        )
        precision = prior_variance.reciprocal()
        # Fixed-budget low-latency runs do not consume relative_change.
        # Avoiding the final scalar extraction removes a CUDA synchronisation
        # and makes the whole solver eligible for CUDA Graph capture.
        relative_change = (
            float(last_change.item())
            if (
                cfg.return_history
                or cfg.eager_change_tracking
                or cfg.tol > 0.0
                or cfg.verbose
            )
            else float("nan")
        )

    support_mask: Tensor | None = None
    support_score_threshold: Tensor | None = None
    if cfg.uamp_support_false_alarm_budget is not None:
        support_score_threshold = torch.full_like(
            noise_variance,
            log(
                prod(grid_shape)
                / cfg.uamp_support_false_alarm_budget
            ),
        )
        support_score = (
            mean.abs().square()
            / posterior_variance.clamp_min(
                torch.finfo(real_dtype).tiny
            )
        )
        support_mask = (
            support_score > support_score_threshold[:, None]
        )

    output_shape = (*batch_shape, *grid_shape)
    history = (
        {
            key: torch.stack(values).cpu().tolist()
            for key, values in history_tensors.items()
        }
        if cfg.return_history
        else {}
    )
    reconstructed = mean.reshape(output_shape)
    offgrid_result = None
    if cfg.offgrid_refinement:
        offgrid_result = refine_offgrid_fourier(
            input_tensor.to(device=device, dtype=complex_dtype),
            reconstructed,
            grid_shape,
            component_count=cfg.offgrid_components,
            max_components=cfg.offgrid_max_components,
            relative_threshold=cfg.offgrid_relative_threshold,
            minimum_separation=cfg.offgrid_minimum_separation,
            max_iter=cfg.offgrid_max_iter,
            tolerance=cfg.offgrid_tolerance,
            max_step=cfg.offgrid_max_step,
            damping=cfg.offgrid_damping,
            ridge=cfg.offgrid_ridge,
        )
    return SBLResult(
        x=reconstructed,
        posterior_variance=posterior_variance.reshape(output_shape),
        precision=precision.reshape(output_shape),
        noise_variance=noise_variance.reshape(batch_shape),
        effective_dof=effective_dof.reshape(batch_shape),
        iterations=iteration,
        converged=converged,
        relative_change=relative_change,
        mode=mode,
        arithmetic_dtype=str(complex_dtype).removeprefix("torch."),
        uamp_backend_used=uamp_backend_used,
        active_set_size=(
            active_indices.shape[1] if active_indices is not None else 0
        ),
        offgrid=offgrid_result,
        learned_spatial_coupling=(
            directional_coupling.reshape(*batch_shape, 2)
            if directional_coupling is not None
            else None
        ),
        negative_log_evidence=(
            negative_log_evidence.reshape(batch_shape)
            if negative_log_evidence is not None
            else None
        ),
        evidence_trace=active_evidence_trace,
        truncation_delta_bound=(
            truncation_delta_bound.reshape(batch_shape)
            if truncation_delta_bound is not None
            else None
        ),
        hard_pruning_accepted=(
            hard_pruning_accepted.reshape(batch_shape)
            if hard_pruning_accepted is not None
            else None
        ),
        truncation_certified=(
            hard_pruning_accepted.reshape(batch_shape)
            if hard_pruning_accepted is not None
            else None
        ),
        inactive_max_evidence_ratio=(
            inactive_max_evidence_ratio.reshape(batch_shape)
            if inactive_max_evidence_ratio is not None
            else None
        ),
        inactive_best_index=(
            inactive_best_index.reshape(batch_shape)
            if inactive_best_index is not None
            else None
        ),
        inactive_evidence_gain=(
            inactive_evidence_gain.reshape(batch_shape)
            if inactive_evidence_gain is not None
            else None
        ),
        inactive_optimal_variance=(
            inactive_optimal_variance.reshape(batch_shape)
            if inactive_optimal_variance is not None
            else None
        ),
        support_repairs_accepted=(
            support_repairs_accepted.reshape(batch_shape)
            if support_repairs_accepted is not None
            else None
        ),
        support_exchanges_accepted=(
            support_repairs_accepted.reshape(batch_shape)
            if support_repairs_accepted is not None
            else None
        ),
        support_growth_accepted=(
            support_growth_accepted.reshape(batch_shape)
            if support_growth_accepted is not None
            else None
        ),
        active_max_evidence_gain=(
            active_max_evidence_gain.reshape(batch_shape)
            if active_max_evidence_gain is not None
            else None
        ),
        active_best_index=(
            active_best_index.reshape(batch_shape)
            if active_best_index is not None
            else None
        ),
        active_optimal_variance=(
            active_optimal_variance.reshape(batch_shape)
            if active_optimal_variance is not None
            else None
        ),
        active_max_deletion_gain=(
            active_max_deletion_gain.reshape(batch_shape)
            if active_max_deletion_gain is not None
            else None
        ),
        coordinate_steps_accepted=(
            coordinate_steps_accepted.reshape(batch_shape)
            if coordinate_steps_accepted is not None
            else None
        ),
        coordinate_stationary=(
            coordinate_stationary.reshape(batch_shape)
            if coordinate_stationary is not None
            else None
        ),
        inactive_ratio_threshold=(
            inactive_ratio_threshold.reshape(batch_shape)
            if inactive_ratio_threshold is not None
            else None
        ),
        familywise_support_certified=(
            familywise_support_certified.reshape(batch_shape)
            if familywise_support_certified is not None
            else None
        ),
        type2_coordinate_stationary=(
            type2_coordinate_stationary.reshape(batch_shape)
            if type2_coordinate_stationary is not None
            else None
        ),
        stationarity_ratio_threshold=(
            stationarity_ratio_threshold.reshape(batch_shape)
            if stationarity_ratio_threshold is not None
            else None
        ),
        familywise_ratio_threshold=(
            familywise_ratio_threshold.reshape(batch_shape)
            if familywise_ratio_threshold is not None
            else None
        ),
        familywise_no_discovery=(
            familywise_no_discovery.reshape(batch_shape)
            if familywise_no_discovery is not None
            else None
        ),
        active_model_selected=(
            active_model_selected.reshape(batch_shape)
            if active_model_selected is not None
            else None
        ),
        support_shrink_accepted=(
            support_shrink_accepted.reshape(batch_shape)
            if support_shrink_accepted is not None
            else None
        ),
        support_log_odds_penalty=(
            support_log_odds_penalty.reshape(batch_shape)
            if support_log_odds_penalty is not None
            else None
        ),
        penalized_negative_log_evidence=(
            penalized_negative_log_evidence.reshape(batch_shape)
            if penalized_negative_log_evidence is not None
            else None
        ),
        penalized_evidence_trace=penalized_evidence_trace,
        inactive_penalized_evidence_gain=(
            inactive_penalized_evidence_gain.reshape(batch_shape)
            if inactive_penalized_evidence_gain is not None
            else None
        ),
        active_penalized_evidence_gain=(
            active_penalized_evidence_gain.reshape(batch_shape)
            if active_penalized_evidence_gain is not None
            else None
        ),
        type2_map_coordinate_stationary=(
            type2_map_coordinate_stationary.reshape(batch_shape)
            if type2_map_coordinate_stationary is not None
            else None
        ),
        noise_coordinate_steps_accepted=(
            noise_coordinate_steps_accepted.reshape(batch_shape)
            if noise_coordinate_steps_accepted is not None
            else None
        ),
        noise_evidence_gradient=(
            noise_evidence_gradient.reshape(batch_shape)
            if noise_evidence_gradient is not None
            else None
        ),
        noise_fixed_point_residual=(
            noise_fixed_point_residual.reshape(batch_shape)
            if noise_fixed_point_residual is not None
            else None
        ),
        noise_evidence_gain=(
            noise_evidence_gain.reshape(batch_shape)
            if noise_evidence_gain is not None
            else None
        ),
        noise_coordinate_stationary=(
            noise_coordinate_stationary.reshape(batch_shape)
            if noise_coordinate_stationary is not None
            else None
        ),
        joint_map_coordinate_stationary=(
            joint_map_coordinate_stationary.reshape(batch_shape)
            if joint_map_coordinate_stationary is not None
            else None
        ),
        joint_coordinate_cycles=(
            joint_coordinate_cycles.reshape(batch_shape)
            if joint_coordinate_cycles is not None
            else None
        ),
        fixed_budget_gamma_residual=(
            fixed_budget_gamma_residual.reshape(batch_shape)
            if fixed_budget_gamma_residual is not None
            else None
        ),
        fixed_budget_noise_residual=(
            fixed_budget_noise_residual.reshape(batch_shape)
            if fixed_budget_noise_residual is not None
            else None
        ),
        fixed_budget_objective_decrease=(
            fixed_budget_objective_decrease.reshape(batch_shape)
            if fixed_budget_objective_decrease is not None
            else None
        ),
        fixed_budget_monotone=(
            fixed_budget_monotone.reshape(batch_shape)
            if fixed_budget_monotone is not None
            else None
        ),
        support_mask=(
            support_mask.reshape(output_shape)
            if support_mask is not None
            else None
        ),
        support_score_threshold=(
            support_score_threshold.reshape(batch_shape)
            if support_score_threshold is not None
            else None
        ),
        history=history,
    )


def _neighbor_sum_2d(
    values: Tensor,
    grid_shape: Sequence[int],
    boundary: SpatialBoundary,
) -> Tensor:
    """Sum the four axial neighbours without materialising a graph matrix."""
    shape = tuple(int(v) for v in grid_shape)
    if len(shape) != 2:
        raise ValueError("The pattern-coupled prior requires a 2-D grid.")
    if values.ndim != 2 or values.shape[1] != prod(shape):
        raise ValueError("values must have shape [B, prod(grid_shape)].")
    field = values.reshape(values.shape[0], *shape)
    if boundary == "circular":
        neighbours = (
            torch.roll(field, 1, dims=1)
            + torch.roll(field, -1, dims=1)
            + torch.roll(field, 1, dims=2)
            + torch.roll(field, -1, dims=2)
        )
    elif boundary == "zero":
        neighbours = torch.zeros_like(field)
        neighbours[:, 1:, :] += field[:, :-1, :]
        neighbours[:, :-1, :] += field[:, 1:, :]
        neighbours[:, :, 1:] += field[:, :, :-1]
        neighbours[:, :, :-1] += field[:, :, 1:]
    else:
        raise ValueError("boundary must be 'zero' or 'circular'.")
    return neighbours.reshape_as(values)


def _directional_neighbor_sums_2d(
    values: Tensor,
    grid_shape: Sequence[int],
    boundary: SpatialBoundary,
) -> tuple[Tensor, Tensor]:
    """Return vertical and horizontal two-neighbour sums separately."""
    shape = tuple(int(v) for v in grid_shape)
    if len(shape) != 2:
        raise ValueError("Directional coupling requires a 2-D grid.")
    if values.ndim != 2 or values.shape[1] != prod(shape):
        raise ValueError("values must have shape [B, prod(grid_shape)].")
    field = values.reshape(values.shape[0], *shape)
    if boundary == "circular":
        vertical = (
            torch.roll(field, 1, dims=1)
            + torch.roll(field, -1, dims=1)
        )
        horizontal = (
            torch.roll(field, 1, dims=2)
            + torch.roll(field, -1, dims=2)
        )
    elif boundary == "zero":
        vertical = torch.zeros_like(field)
        vertical[:, 1:, :] += field[:, :-1, :]
        vertical[:, :-1, :] += field[:, 1:, :]
        horizontal = torch.zeros_like(field)
        horizontal[:, :, 1:] += field[:, :, :-1]
        horizontal[:, :, :-1] += field[:, :, 1:]
    else:
        raise ValueError("boundary must be 'zero' or 'circular'.")
    return (
        vertical.reshape_as(values),
        horizontal.reshape_as(values),
    )


def _effective_anisotropic_pattern_precision(
    base_precision: Tensor,
    grid_shape: Sequence[int],
    coupling: Tensor | None,
    boundary: SpatialBoundary,
) -> Tensor:
    if coupling is None:
        raise ValueError("Anisotropic coupling coefficients are required.")
    vertical, horizontal = _directional_neighbor_sums_2d(
        base_precision, grid_shape, boundary
    )
    return (
        base_precision
        + coupling[:, 0, None] * vertical
        + coupling[:, 1, None] * horizontal
    )


def _initial_anisotropic_pattern_precision(
    prior_variance: Tensor,
    grid_shape: Sequence[int],
    coupling: Tensor,
    boundary: SpatialBoundary,
) -> Tensor:
    ones = torch.ones(
        (prior_variance.shape[0], prior_variance.shape[1]),
        device=prior_variance.device,
        dtype=prior_variance.dtype,
    )
    vertical_degree, horizontal_degree = _directional_neighbor_sums_2d(
        ones, grid_shape, boundary
    )
    scale = (
        1.0
        + coupling[:, 0, None] * vertical_degree
        + coupling[:, 1, None] * horizontal_degree
    )
    return prior_variance.reciprocal() / scale


def _anisotropic_pattern_precision_update(
    second_moment: Tensor,
    grid_shape: Sequence[int],
    coupling: Tensor,
    boundary: SpatialBoundary,
    *,
    hyper_shape: float,
    hyper_rate: float,
    max_precision: float,
) -> Tensor:
    vertical, horizontal = _directional_neighbor_sums_2d(
        second_moment, grid_shape, boundary
    )
    coupled_moment = (
        second_moment
        + coupling[:, 0, None] * vertical
        + coupling[:, 1, None] * horizontal
    )
    return ((1.0 + hyper_shape) / (
        coupled_moment + hyper_rate
    )).clamp(
        min=1.0 / max_precision,
        max=max_precision,
    )


def _learn_anisotropic_coupling(
    base_precision: Tensor,
    second_moment: Tensor,
    grid_shape: Sequence[int],
    coupling: Tensor,
    boundary: SpatialBoundary,
    *,
    maximum: float,
    steps: int,
    magnitude_penalty: float,
    anisotropy_penalty: float,
) -> Tensor:
    """Maximise the concave EM auxiliary function over two couplings.

    With ``eta=alpha+beta_v V(alpha)+beta_h H(alpha)``, the complex
    Gaussian auxiliary function is ``sum(log(eta)-eta*m)``.  Its Hessian
    in ``(beta_v, beta_h)`` is negative semidefinite, so projected Newton
    with a finite backtracking set has a monotone update.
    """
    vertical, horizontal = _directional_neighbor_sums_2d(
        base_precision, grid_shape, boundary
    )
    statistics = torch.stack((vertical, horizontal), dim=2)
    beta = coupling.clamp(0.0, maximum)
    real_dtype = base_precision.dtype
    eps = torch.finfo(real_dtype).eps
    penalty_scale = float(base_precision.shape[1])
    step_scales = torch.tensor(
        [0.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
        device=base_precision.device,
        dtype=real_dtype,
    )
    regularizer = penalty_scale * torch.tensor(
        [
            [
                magnitude_penalty + anisotropy_penalty,
                -anisotropy_penalty,
            ],
            [
                -anisotropy_penalty,
                magnitude_penalty + anisotropy_penalty,
            ],
        ],
        device=base_precision.device,
        dtype=real_dtype,
    )

    for _ in range(steps):
        eta = (
            base_precision
            + (statistics * beta[:, None, :]).sum(dim=2)
        ).clamp_min(torch.finfo(real_dtype).tiny)
        gradient = (
            statistics
            * (eta.reciprocal() - second_moment)[:, :, None]
        ).sum(dim=1)
        difference = beta[:, 0] - beta[:, 1]
        gradient = (
            gradient
            - magnitude_penalty * penalty_scale * beta
            - anisotropy_penalty
            * penalty_scale
            * torch.stack((difference, -difference), dim=1)
        )
        scaled_statistics = statistics / eta[:, :, None]
        information = torch.einsum(
            "bni,bnj->bij",
            scaled_statistics,
            scaled_statistics,
        )
        information = information + regularizer
        ridge = (
            eps**0.5
            * information.diagonal(dim1=-2, dim2=-1)
            .sum(dim=1)
            .clamp_min(1.0)
        )
        diagonal_0 = information[:, 0, 0] + ridge
        diagonal_1 = information[:, 1, 1] + ridge
        off_diagonal = 0.5 * (
            information[:, 0, 1] + information[:, 1, 0]
        )
        determinant = (
            diagonal_0 * diagonal_1 - off_diagonal.square()
        ).clamp_min(torch.finfo(real_dtype).tiny)
        direction = torch.stack(
            (
                (
                    diagonal_1 * gradient[:, 0]
                    - off_diagonal * gradient[:, 1]
                )
                / determinant,
                (
                    diagonal_0 * gradient[:, 1]
                    - off_diagonal * gradient[:, 0]
                )
                / determinant,
            ),
            dim=1,
        )
        candidate_tensor = (
            beta[:, None, :]
            + step_scales[None, :, None] * direction[:, None, :]
        ).clamp(0.0, maximum)
        candidate_eta = (
            base_precision[:, None, :]
            + torch.einsum(
                "bni,bci->bcn", statistics, candidate_tensor
            )
        ).clamp_min(torch.finfo(real_dtype).tiny)
        candidate_difference = (
            candidate_tensor[:, :, 0] - candidate_tensor[:, :, 1]
        )
        scores = (
            (
                candidate_eta.log()
                - candidate_eta * second_moment[:, None, :]
            ).sum(dim=2)
            - 0.5
            * magnitude_penalty
            * penalty_scale
            * candidate_tensor.square().sum(dim=2)
            - 0.5
            * anisotropy_penalty
            * penalty_scale
            * candidate_difference.square()
        )
        best = scores.argmax(dim=1)
        beta = candidate_tensor.gather(
            1, best[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
    return beta


def _effective_pattern_precision(
    base_precision: Tensor,
    grid_shape: Sequence[int],
    coupling: float,
    boundary: SpatialBoundary,
) -> Tensor:
    """Return eta_i = alpha_i + beta sum_{j in N(i)} alpha_j."""
    return base_precision + coupling * _neighbor_sum_2d(
        base_precision, grid_shape, boundary
    )


def _initial_pattern_precision(
    prior_variance: Tensor,
    grid_shape: Sequence[int],
    coupling: float,
    boundary: SpatialBoundary,
) -> Tensor:
    """Construct a scale-matched base precision for the coupled hierarchy."""
    ones = torch.ones(
        (1, prior_variance.shape[1]),
        device=prior_variance.device,
        dtype=prior_variance.dtype,
    )
    degree = _neighbor_sum_2d(ones, grid_shape, boundary)
    return prior_variance.reciprocal() / (1.0 + coupling * degree)


def _pattern_precision_update(
    mean: Tensor,
    posterior_variance: Tensor,
    grid_shape: Sequence[int],
    coupling: float,
    boundary: SpatialBoundary,
    *,
    joint_sparsity: bool,
    hyper_shape: float,
    hyper_rate: float,
    max_precision: float,
) -> Tensor:
    """Complex-valued 2-D pattern-coupled EM/MAP hyperparameter update.

    For a proper complex Gaussian, ``E|x_i|^2`` replaces the half-weighted
    real second moment in the original PCSBL derivation.  If MMV sharing is
    enabled, the sufficient statistic is first averaged over snapshots.
    """
    second_moment = mean.abs().square() + posterior_variance
    if joint_sparsity:
        second_moment = second_moment.mean(dim=0, keepdim=True)
    coupled_moment = second_moment + coupling * _neighbor_sum_2d(
        second_moment, grid_shape, boundary
    )
    base_precision = (1.0 + hyper_shape) / (
        coupled_moment + hyper_rate
    )
    base_precision = base_precision.clamp(
        min=1.0 / max_precision,
        max=max_precision,
    )
    if joint_sparsity:
        base_precision = base_precision.expand(mean.shape[0], -1)
    return base_precision


def _uamp_posterior(
    y: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    state: _UAMPState,
    *,
    damping: float,
    use_triton: bool = False,
    strict_triton: bool = False,
    initial_backprojection: Tensor | None = None,
    compute_prediction: bool = True,
    fused_prior_floor: Tensor | None = None,
    hyper_shape: float = 0.0,
    hyper_rate: float = 0.0,
    prior_damping: float = 0.0,
    prior_maximum: float = 1e12,
) -> tuple[Tensor, Tensor, Tensor, _UAMPState]:
    """One fully vectorized UAMP step for a partial Fourier dictionary.

    Every selected row of the unnormalised DFT has squared norm ``N`` and
    every entry has unit magnitude.  Consequently the UAMP variance
    propagation reduces to batch reductions; no SVD, covariance, probe, or
    inner linear solve is required.
    """
    if state.mean.shape != prior_variance.shape:
        raise ValueError("UAMP mean and prior_variance must have equal shapes.")
    if state.message.shape != y.shape or state.prediction.shape != y.shape:
        raise ValueError("UAMP data-space state must match the observations.")

    tiny = torch.finfo(operator.real_dtype).tiny
    coefficient_variance = state.variance.clamp_min(tiny)
    data_variance = (
        state.data_variance
        if state.data_variance is not None
        else coefficient_variance.sum(dim=1)
    ).clamp_min(tiny)
    triton_step = use_triton and triton_is_available()
    if triton_step:
        try:
            message, pseudo_variance = uamp_message_update(
                y,
                state.prediction,
                state.message,
                data_variance,
                noise_variance,
                damping=damping,
            )
            pseudo_variance = pseudo_variance.clamp_min(tiny)
            lifted_message = (
                initial_backprojection
                * (
                    (1.0 - damping)
                    / (data_variance + noise_variance)
                )[:, None]
                if initial_backprojection is not None
                else operator.adjoint_flat(message)
            )
            (
                mean,
                posterior_variance,
                effective_dof,
                next_data_variance,
                next_prior_variance,
            ) = uamp_denoise(
                prior_variance,
                pseudo_variance,
                lifted_message,
                state.mean,
                coefficient_variance,
                damping=damping,
                tiny=tiny,
                prior_floor=fused_prior_floor,
                hyper_shape=hyper_shape,
                hyper_rate=hyper_rate,
                prior_damping=prior_damping,
                prior_maximum=prior_maximum,
            )
        except Exception as exc:
            if strict_triton:
                raise RuntimeError("Triton UAMP kernel launch failed.") from exc
            triton_step = False
    if not triton_step:
        next_prior_variance = None
        pseudo_data = (
            state.prediction - data_variance[:, None] * state.message
        )
        data_precision = (data_variance + noise_variance).reciprocal()
        candidate_message = data_precision[:, None] * (y - pseudo_data)
        message = (
            damping * state.message
            + (1.0 - damping) * candidate_message
        )

        pseudo_variance = (
            operator.n_samples * data_precision
        ).reciprocal().clamp_min(tiny)
        lifted_message = (
            initial_backprojection
            * (
                (1.0 - damping)
                / (data_variance + noise_variance)
            )[:, None]
            if initial_backprojection is not None
            else operator.adjoint_flat(message)
        )
        pseudo_mean = state.mean + pseudo_variance[:, None] * lifted_message

        denominator = prior_variance + pseudo_variance[:, None]
        gain = prior_variance / denominator.clamp_min(tiny)
        candidate_mean = gain * pseudo_mean
        candidate_variance = (
            prior_variance * pseudo_variance[:, None]
            / denominator.clamp_min(tiny)
        )
        mean = damping * state.mean + (1.0 - damping) * candidate_mean
        posterior_variance = (
            damping * coefficient_variance
            + (1.0 - damping) * candidate_variance
        ).clamp_min(tiny)
        effective_dof = (
            1.0
            - posterior_variance / prior_variance.clamp_min(tiny)
        ).clamp(0.0, 1.0).sum(dim=1)
        next_data_variance = posterior_variance.sum(dim=1)
    # The final consistency E-step is immediately consumed by the exact
    # active posterior (or returned to the caller), so its data prediction
    # is dead work. Keeping the previous buffer makes this opt-out graph
    # shape-stable; callers must not reuse the returned state for another
    # UAMP iteration when ``compute_prediction=False``.
    prediction = (
        operator.forward_flat(mean)
        if compute_prediction
        else state.prediction
    )

    updated_state = _UAMPState(
        mean=mean,
        variance=posterior_variance,
        message=message,
        prediction=prediction,
        data_variance=next_data_variance,
        next_prior_variance=next_prior_variance,
    )
    return mean, posterior_variance, effective_dof, updated_state


def _stable_cholesky(
    matrix: Tensor,
    identity: Tensor,
    operator: PrefixFourierOperator,
    config: SBLConfig,
    *,
    fallback_multipliers: Sequence[float],
) -> Tensor | None:
    """Factor an SPD batch without forcing a CUDA-to-host synchronization.

    Fourier data covariances are positive definite after the explicitly
    modelled noise loading.  On CUDA we therefore use one conservative,
    deterministic loading and leave ``cholesky_ex.info`` on the device.
    CPU execution retains the defensive multi-loading fallback.
    """
    return _stable_cholesky_raw(
        matrix,
        identity,
        operator,
        sync_free_cholesky=config.cuda_sync_free_cholesky,
        jitter_multiplier=config.cuda_cholesky_jitter_multiplier,
        fallback_multipliers=fallback_multipliers,
    )


def _stable_cholesky_raw(
    matrix: Tensor,
    identity: Tensor,
    operator: PrefixFourierOperator,
    *,
    sync_free_cholesky: bool,
    jitter_multiplier: float,
    fallback_multipliers: Sequence[float],
) -> Tensor | None:
    scale = matrix.diagonal(dim1=-2, dim2=-1).real.abs().mean(dim=1)
    scale = scale.clamp_min(torch.finfo(operator.real_dtype).tiny)
    base_jitter = torch.finfo(matrix.real.dtype).eps * scale
    if matrix.is_cuda and sync_free_cholesky:
        candidate, info = torch.linalg.cholesky_ex(
            matrix
            + (jitter_multiplier * base_jitter)[:, None, None] * identity,
            check_errors=False,
        )
        torch._assert_async(
            (info == 0).all(),
            "CUDA Cholesky failed after deterministic diagonal loading.",
        )
        return candidate

    if not bool(torch.isfinite(matrix).all()):
        return None
    for multiplier in fallback_multipliers:
        candidate, info = torch.linalg.cholesky_ex(
            matrix + (multiplier * base_jitter)[:, None, None] * identity,
            check_errors=False,
        )
        if bool((info == 0).all()):
            return candidate
    return None


def _posterior(
    y: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    mode: str,
    config: SBLConfig,
    probes: Tensor | None,
    lifted_probes: Tensor | None,
    *,
    cg_tolerance: float,
    initial_solution: Tensor | None,
    active_indices: Tensor | None,
    active_columns: Tensor | None,
    active_gram: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, int, Tensor | None]:
    if active_indices is not None:
        mean, posterior_variance, data_diagonal = _active_posterior(
            y,
            prior_variance,
            noise_variance,
            operator,
            active_indices,
            columns=active_columns,
            gram=active_gram,
            config=config,
        )
        return mean, posterior_variance, data_diagonal, 0, None
    if mode == "dense":
        q = operator.covariance(prior_variance, noise_variance)
        cholesky = _stable_cholesky(
            q,
            operator.eye,
            operator,
            config,
            fallback_multipliers=(1.0, 10.0, 100.0, 1000.0),
        )
        if cholesky is None:
            raise RuntimeError("Data covariance is not numerically positive definite.")
        if config.dense_joint_solve:
            right_hand_sides = torch.cat(
                (
                    y[:, :, None],
                    operator.eye.expand(y.shape[0], -1, -1),
                ),
                dim=2,
            )
            solved = torch.cholesky_solve(right_hand_sides, cholesky)
            q_inverse_y = solved[:, :, :1]
            q_inverse = solved[:, :, 1:]
        else:
            q_inverse_y = torch.cholesky_solve(y[:, :, None], cholesky)
            q_inverse = torch.cholesky_inverse(cholesky)
        data_diagonal = operator.posterior_data_diagonal(q_inverse)
        cg_iterations = 0
        solved_state = None
        lifted_data = operator.adjoint(q_inverse_y).squeeze(-1)
    else:
        if probes is None or lifted_probes is None:
            raise RuntimeError("CG mode requires Hutchinson probes.")
        right_hand_sides = torch.cat((y[:, :, None], probes), dim=2)
        toeplitz_spectrum = (
            operator.toeplitz_spectrum(prior_variance, noise_variance)
            if config.cg_matvec == "tbt"
            else None
        )
        solutions, cg_iterations = _pcg(
            right_hand_sides,
            prior_variance,
            noise_variance,
            operator,
            initial_solution=initial_solution,
            tolerance=cg_tolerance,
            max_iter=config.cg_max_iter,
            check_interval=config.cg_check_interval,
            preconditioner_mode=config.cg_preconditioner,
            preconditioner_rank=config.cg_preconditioner_rank,
            preconditioner_seed=config.seed,
            fixed_iterations=config.cg_fixed_iterations,
            sync_free_cholesky=config.cuda_sync_free_cholesky,
            jitter_multiplier=config.cuda_cholesky_jitter_multiplier,
            toeplitz_spectrum=toeplitz_spectrum,
        )
        q_inverse_y = solutions[:, :, :1]
        lifted_all_solutions = operator.adjoint(solutions)
        lifted_data = lifted_all_solutions[:, :, 0]
        lifted_solutions = lifted_all_solutions[:, :, 1:]
        data_diagonal = (
            lifted_solutions * lifted_probes.conj()
        ).real.mean(dim=2).clamp_min(0.0)
        solved_state = solutions

    mean = prior_variance * lifted_data
    posterior_variance = (
        prior_variance - prior_variance.square() * data_diagonal
    ).clamp_min(0.0)
    return (
        mean,
        posterior_variance,
        data_diagonal,
        cg_iterations,
        solved_state,
    )


def _shared_mmv_posterior(
    y: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    mode: str,
    config: SBLConfig,
    probes: Tensor | None,
    lifted_probes: Tensor | None,
    *,
    cg_tolerance: float,
    initial_solution: Tensor | None,
    active_indices: Tensor | None,
    active_columns: Tensor | None,
    active_gram: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, int, Tensor | None]:
    """Evaluate shared-gamma MMV as one system with many right-hand sides.

    If all snapshots share both ``gamma`` and the noise variance, their data
    covariance Q is identical.  Factoring Q or estimating
    ``diag(H^H Q^-1 H)`` once is exactly equivalent to repeating the same
    posterior calculation for every snapshot.
    """
    del active_indices, active_columns, active_gram
    if prior_variance.shape[0] != 1 or noise_variance.numel() != 1:
        raise ValueError("Shared MMV posterior requires one gamma and noise.")
    snapshot_count = y.shape[0]
    if mode == "dense":
        covariance = operator.covariance(
            prior_variance, noise_variance
        )
        cholesky = _stable_cholesky(
            covariance,
            operator.eye,
            operator,
            config,
            fallback_multipliers=(1.0, 10.0, 100.0, 1000.0),
        )
        if cholesky is None:
            raise RuntimeError(
                "Shared MMV covariance is not numerically positive definite."
            )
        right_hand_sides = y.mT.unsqueeze(0)
        if config.dense_joint_solve:
            solved = torch.cholesky_solve(
                torch.cat((right_hand_sides, operator.eye.unsqueeze(0)), dim=2),
                cholesky,
            )
            solved_data = solved[:, :, :snapshot_count]
            covariance_inverse = solved[:, :, snapshot_count:]
        else:
            solved_data = torch.cholesky_solve(
                right_hand_sides, cholesky
            )
            covariance_inverse = torch.cholesky_inverse(cholesky)
        data_diagonal_single = operator.posterior_data_diagonal(
            covariance_inverse
        )
        lifted_data = operator.adjoint(solved_data)[0].mT
        cg_iterations = 0
        solved_state = None
    else:
        if probes is None or lifted_probes is None:
            raise RuntimeError("CG mode requires shared Hutchinson probes.")
        right_hand_sides = torch.cat(
            (y.mT.unsqueeze(0), probes), dim=2
        )
        toeplitz_spectrum = (
            operator.toeplitz_spectrum(
                prior_variance, noise_variance
            )
            if config.cg_matvec == "tbt"
            else None
        )
        solutions, cg_iterations = _pcg(
            right_hand_sides,
            prior_variance,
            noise_variance,
            operator,
            initial_solution=initial_solution,
            tolerance=cg_tolerance,
            max_iter=config.cg_max_iter,
            check_interval=config.cg_check_interval,
            preconditioner_mode=config.cg_preconditioner,
            preconditioner_rank=config.cg_preconditioner_rank,
            preconditioner_seed=config.seed,
            fixed_iterations=config.cg_fixed_iterations,
            sync_free_cholesky=config.cuda_sync_free_cholesky,
            jitter_multiplier=config.cuda_cholesky_jitter_multiplier,
            toeplitz_spectrum=toeplitz_spectrum,
        )
        lifted_solutions = operator.adjoint(solutions)
        lifted_data = lifted_solutions[0, :, :snapshot_count].mT
        lifted_random = lifted_solutions[:, :, snapshot_count:]
        data_diagonal_single = (
            lifted_random * lifted_probes.conj()
        ).real.mean(dim=2).clamp_min(0.0)
        solved_state = solutions

    mean = prior_variance * lifted_data
    posterior_variance_single = (
        prior_variance
        - prior_variance.square() * data_diagonal_single
    ).clamp_min(0.0)
    posterior_variance = posterior_variance_single.expand(
        snapshot_count, -1
    )
    data_diagonal = data_diagonal_single.expand(snapshot_count, -1)
    return (
        mean,
        posterior_variance,
        data_diagonal,
        cg_iterations,
        solved_state,
    )


def _project_relevance(
    prior_variance: Tensor,
    data_diagonal: Tensor,
    n_samples: int,
    eps: float,
) -> Tensor:
    relevance = (prior_variance * data_diagonal).clamp(0.0, 1.0)
    return _project_relevance_values(relevance, n_samples, eps)


def _project_relevance_values(
    relevance: Tensor,
    n_samples: int,
    eps: float,
) -> Tensor:
    """Project already formed relevance values onto the valid DOF set."""
    relevance = relevance.clamp(0.0, 1.0)
    # The exact sum is at most M.  Stochastic diagonal estimation in CG mode
    # can violate that identity slightly, so project it back.
    raw_dof = relevance.sum(dim=1)
    max_dof = n_samples * (1.0 - 10.0 * eps)
    scale = torch.minimum(
        torch.ones_like(raw_dof),
        torch.full_like(raw_dof, max_dof) / raw_dof.clamp_min(eps),
    )
    return relevance * scale[:, None]


def _select_active_indices(
    prior_variance: Tensor,
    *,
    minimum_size: int,
    maximum_size: int,
    tail_fraction: float,
) -> Tensor | None:
    """Return a conservative ARD active set once the variance tail is small.

    The switch is made only if at most ``maximum_size`` coefficients explain
    at least ``1-tail_fraction`` of the total prior variance.  This prevents
    early hard pruning while the posterior is still diffuse.
    """
    if prior_variance.shape[0] != 1:
        return None
    if not bool(torch.isfinite(prior_variance).all()):
        return None
    maximum_size = min(maximum_size, prior_variance.shape[1])
    values, indices = torch.topk(
        prior_variance, maximum_size, dim=1, sorted=True
    )
    total = prior_variance.sum(dim=1)
    target = (1.0 - tail_fraction) * total
    if bool((values.sum(dim=1) < target).any()):
        return None
    cumulative = values.cumsum(dim=1)
    required = int(
        torch.searchsorted(cumulative[0], target[0]).clamp_max(
            maximum_size - 1
        ).item()
    ) + 1
    size = min(maximum_size, max(minimum_size, required))
    return indices[:, :size]


def _active_evidence_certificate(
    active_mean: Tensor,
    active_posterior_variance: Tensor,
    active_gamma: Tensor,
    active_indices: Tensor,
) -> _ActiveEvidenceCertificate:
    """Return the best exact Type-II update among active coordinates.

    Let ``C = C_-j + gamma_j a_j a_j^H`` and evaluate
    ``q_j = a_j^H C^-1 y`` and ``s_j = a_j^H C^-1 a_j`` from the posterior:

    ``q_j = mu_j / gamma_j`` and
    ``s_j = (gamma_j - Sigma_jj) / gamma_j^2``.

    Sherman--Morrison then recovers the leave-one-out statistics without a
    new solve.  The globally optimal non-negative coordinate variance is
    zero for ``rho <= 1`` and ``(rho - 1) / s_-j`` otherwise.  ``max_gain``
    is the exact reduction in negative log evidence obtained by replacing
    the current variance with that optimum.  ``max_deletion_gain`` is kept
    separately so a weak atom can be recycled during a support exchange.
    """
    if not (
        active_mean.shape
        == active_posterior_variance.shape
        == active_gamma.shape
        == active_indices.shape
    ):
        raise ValueError("Active posterior tensors must have identical shape.")
    tiny = torch.finfo(active_gamma.dtype).tiny
    safe_gamma = active_gamma.clamp_min(tiny)
    q_current = active_mean / safe_gamma
    s_current = (
        (safe_gamma - active_posterior_variance)
        / safe_gamma.square()
    ).clamp_min(tiny)
    downdate_denominator = (
        active_posterior_variance / safe_gamma
    ).clamp(min=tiny, max=1.0)
    leave_one_out_precision = (
        s_current / downdate_denominator
    ).clamp_min(tiny)
    ratio = (
        q_current.abs().square()
        / (s_current * downdate_denominator).clamp_min(tiny)
    )
    safe_ratio = ratio.clamp_min(tiny)

    optimal_variance = torch.where(
        ratio > 1.0,
        (ratio - 1.0) / leave_one_out_precision,
        torch.zeros_like(ratio),
    )
    current_delta = (
        -downdate_denominator.log()
        - safe_gamma
        * q_current.abs().square()
        / downdate_denominator
    )
    optimal_delta = torch.where(
        ratio > 1.0,
        safe_ratio.log() - ratio + 1.0,
        torch.zeros_like(ratio),
    )
    coordinate_gain = (current_delta - optimal_delta).clamp_min(0.0)
    max_gain, best_position = coordinate_gain.max(dim=1)
    best_index = active_indices.gather(
        1, best_position[:, None]
    ).squeeze(1)
    best_optimal_variance = optimal_variance.gather(
        1, best_position[:, None]
    ).squeeze(1)
    nondeletion_gain = torch.where(
        optimal_variance > 0.0,
        coordinate_gain,
        torch.zeros_like(coordinate_gain),
    )
    max_nondeletion_gain, best_nondeletion_position = (
        nondeletion_gain.max(dim=1)
    )
    best_nondeletion_index = active_indices.gather(
        1, best_nondeletion_position[:, None]
    ).squeeze(1)
    best_nondeletion_variance = optimal_variance.gather(
        1, best_nondeletion_position[:, None]
    ).squeeze(1)
    max_deletion_gain, best_deletion_position = current_delta.max(dim=1)
    return _ActiveEvidenceCertificate(
        max_gain=max_gain,
        best_position=best_position,
        best_index=best_index,
        optimal_variance=best_optimal_variance,
        max_nondeletion_gain=max_nondeletion_gain,
        best_nondeletion_position=best_nondeletion_position,
        best_nondeletion_index=best_nondeletion_index,
        best_nondeletion_variance=best_nondeletion_variance,
        deletion_gain=current_delta,
        max_deletion_gain=max_deletion_gain,
        best_deletion_position=best_deletion_position,
    )


def _noise_evidence_certificate(
    y: Tensor,
    mean: Tensor,
    posterior_variance: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    active_indices: Tensor,
    prior_floor: Tensor,
    operator: PrefixFourierOperator,
    config: SBLConfig,
) -> _NoiseEvidenceCertificate:
    """Return a projected first-order certificate for ``sigma^2``.

    For ``C = sigma^2 I + A Gamma A^H``,

    ``dL/dsigma^2 = tr(C^-1) - y^H C^-2 y``.

    With the exact active posterior this becomes

    ``[sigma^2 (M - dof) - ||y - A mu||^2] / sigma^4``.

    The EM fixed-point proposal

    ``sigma_new^2 = ||y - A mu||^2 / (M - dof)``

    is evaluated by the same exact Cholesky objective before it can be
    accepted.  ``projected_residual`` also implements the KKT sign condition
    at the configured variance bounds.
    """
    active_gamma = prior_variance.gather(1, active_indices).clamp_min(
        torch.finfo(prior_variance.dtype).tiny
    )
    active_posterior = posterior_variance.gather(
        1, active_indices
    )
    active_mean = mean.gather(1, active_indices)
    effective_dof = (
        1.0 - active_posterior / active_gamma
    ).clamp(0.0, 1.0).sum(dim=1)
    denominator = (
        operator.n_samples - effective_dof
    ).clamp_min(torch.finfo(prior_variance.dtype).eps)
    # The polished posterior is exactly zero outside ``active_indices``.
    # Reconstructing its data prediction with the cached-size subdictionary
    # costs O(MK), avoiding a full-grid FFT used only by this diagnostic.
    columns = operator.fourier_columns(active_indices).to(dtype=y.dtype)
    prediction = (
        columns @ active_mean[:, :, None]
    ).squeeze(-1)
    residual_energy = (
        y - prediction
    ).abs().square().sum(dim=1).real
    fixed_point = residual_energy / denominator
    candidate_variance = torch.maximum(
        fixed_point, prior_floor
    ).clamp_max(config.max_precision)
    balance = noise_variance * denominator - residual_energy
    gradient = balance / noise_variance.square().clamp_min(
        torch.finfo(noise_variance.dtype).tiny
    )
    bound_tolerance = 16.0 * torch.finfo(
        noise_variance.dtype
    ).eps**0.5
    at_lower = noise_variance <= prior_floor * (1.0 + bound_tolerance)
    at_upper = noise_variance >= (
        config.max_precision * (1.0 - bound_tolerance)
    )
    kkt_satisfied_at_bound = (
        (at_lower & (gradient >= 0.0))
        | (at_upper & (gradient <= 0.0))
    )
    projected_balance = torch.where(
        kkt_satisfied_at_bound,
        torch.zeros_like(balance),
        balance,
    )
    scale = torch.maximum(
        noise_variance * denominator,
        residual_energy,
    ).clamp_min(torch.finfo(noise_variance.dtype).tiny)
    return _NoiseEvidenceCertificate(
        gradient=gradient,
        fixed_point=fixed_point,
        projected_residual=projected_balance.abs() / scale,
        candidate_variance=candidate_variance,
    )


def _fixed_budget_em_residuals(
    y: Tensor,
    mean: Tensor,
    posterior_variance: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    active_indices: Tensor,
    prior_floor: Tensor,
    operator: PrefixFourierOperator,
    config: SBLConfig,
) -> tuple[Tensor, Tensor]:
    """Return scale-free residuals of the exact active-set EM map.

    The active variance map is

    ``gamma_i^+ = (|mu_i|^2 + Sigma_ii + b) / (1 + a)``.

    For the EM noise update,

    ``beta^+ = (||r||^2 + beta*dof + d) / (M + c)``.

    The returned values are the maximum relative distance from these maps.
    They are cheap a-posteriori finite-budget diagnostics, not claims that
    the unselected full-grid support is coordinate stationary.
    """
    tiny = torch.finfo(prior_variance.dtype).tiny
    active_gamma = prior_variance.gather(
        1, active_indices
    ).clamp_min(tiny)
    active_posterior = posterior_variance.gather(
        1, active_indices
    )
    active_mean = mean.gather(1, active_indices)
    gamma_target = (
        active_mean.abs().square()
        + active_posterior
        + config.hyper_rate
    ) / (1.0 + config.hyper_shape)
    gamma_target = torch.maximum(
        gamma_target,
        prior_floor[:, None].expand_as(gamma_target),
    ).clamp_max(config.max_precision)
    gamma_scale = torch.maximum(
        active_gamma.abs(), gamma_target.abs()
    ).clamp_min(tiny)
    gamma_residual = (
        (active_gamma - gamma_target).abs() / gamma_scale
    ).amax(dim=1)

    if config.fixed_noise_variance is not None:
        return gamma_residual, torch.zeros_like(noise_variance)

    effective_dof = (
        1.0 - active_posterior / active_gamma
    ).clamp(0.0, 1.0).sum(dim=1)
    # ``mean`` is zero outside the selected model.  A reduced GEMV is exact
    # here and avoids an FFT used only by the finite-budget diagnostic.
    columns = operator.fourier_columns(active_indices).to(dtype=y.dtype)
    prediction = (
        columns @ active_mean[:, :, None]
    ).squeeze(-1)
    residual_energy = (
        y - prediction
    ).abs().square().sum(dim=1).real
    if config.noise_update == "em":
        noise_target = (
            residual_energy
            + noise_variance * effective_dof
            + config.noise_rate
        ) / (operator.n_samples + config.noise_shape)
    else:
        denominator = (
            operator.n_samples
            - effective_dof
            + config.noise_shape
        ).clamp_min(torch.finfo(prior_variance.dtype).eps)
        noise_target = (
            residual_energy + config.noise_rate
        ) / denominator
    noise_target = torch.maximum(
        noise_target, prior_floor
    ).clamp_max(config.max_precision)
    noise_scale = torch.maximum(
        noise_variance.abs(), noise_target.abs()
    ).clamp_min(tiny)
    noise_residual = (
        noise_variance - noise_target
    ).abs() / noise_scale
    return gamma_residual, noise_residual


def _ratio_threshold_for_evidence_tolerance(tolerance: float) -> float:
    """Return rho >= 1 satisfying ``rho - 1 - log(rho) = tolerance``.

    The inactive-coordinate Type-II evidence gain is exactly
    ``rho - 1 - log(rho)`` for ``rho > 1``.  Reporting the corresponding ratio
    threshold keeps the stopping rule and its public diagnostic in the same
    evidence units as active-coordinate updates.
    """
    if tolerance <= 0.0:
        return 1.0
    lower = 1.0
    upper = 2.0
    while upper - 1.0 - log(upper) < tolerance:
        upper *= 2.0
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if middle - 1.0 - log(middle) < tolerance:
            lower = middle
        else:
            upper = middle
    return upper


def _inactive_evidence_certificate(
    y: Tensor,
    columns: Tensor,
    active_mean: Tensor,
    effective_noise: Tensor,
    diagonal_scale: Tensor,
    cholesky: Tensor,
    operator: PrefixFourierOperator,
    active_indices: Tensor,
) -> _InactiveEvidenceCertificate:
    """Evaluate every inactive atom's optimal one-coordinate evidence gain.

    For ``C = nu I + A_S Gamma_S A_S^H``, define
    ``q_j = a_j^H C^-1 y`` and ``s_j = a_j^H C^-1 a_j``.  Optimising only a
    new variance ``gamma_j`` lowers the negative log evidence iff
    ``rho_j = |q_j|^2 / s_j > 1``.  The optimal decrease is
    ``rho_j - 1 - log(rho_j)``.  All ``q_j`` are obtained by one adjoint FFT;
    all ``s_j`` reuse the active Cholesky through a multi-right-hand-side
    triangular solve.
    """
    residual = y - (columns @ active_mean[:, :, None]).squeeze(-1)
    inverse_y = residual / effective_noise[:, None]
    q_all = operator.adjoint_flat(inverse_y)

    # A_all^H A_S from K simultaneous adjoint FFTs, then transpose to the
    # K-by-N orientation used by the active precision system.
    cross_gram = operator.adjoint(columns).mH
    scaled_cross = cross_gram / diagonal_scale[:, :, None]
    inverse_factor_cross = torch.linalg.solve_triangular(
        cholesky,
        scaled_cross,
        upper=False,
    )
    posterior_reduction = inverse_factor_cross.abs().square().sum(dim=1)
    atom_precision = (
        operator.n_samples / effective_noise[:, None]
        - posterior_reduction / effective_noise[:, None].square()
    ).clamp_min(torch.finfo(effective_noise.dtype).tiny)
    ratio = q_all.abs().square() / atom_precision
    ratio = ratio.scatter(
        1,
        active_indices,
        torch.zeros_like(ratio.gather(1, active_indices)),
    )
    max_ratio, best_index = ratio.max(dim=1)
    best_precision = atom_precision.gather(
        1, best_index[:, None]
    ).squeeze(1)
    safe_ratio = max_ratio.clamp_min(
        torch.finfo(effective_noise.dtype).tiny
    )
    evidence_gain = torch.where(
        max_ratio > 1.0,
        max_ratio - 1.0 - safe_ratio.log(),
        torch.zeros_like(max_ratio),
    )
    optimal_variance = torch.where(
        max_ratio > 1.0,
        (max_ratio - 1.0) / best_precision,
        torch.zeros_like(max_ratio),
    )
    return _InactiveEvidenceCertificate(
        max_ratio=max_ratio,
        best_index=best_index,
        evidence_gain=evidence_gain,
        optimal_variance=optimal_variance,
    )


def _active_posterior_with_evidence(
    y: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    active_indices: Tensor,
    *,
    columns: Tensor | None = None,
    gram: Tensor | None = None,
    column_data: Tensor | None = None,
    data_energy: Tensor | None = None,
    identity: Tensor | None = None,
    tail_loading: Tensor | None = None,
    global_certificate: bool = False,
    config: SBLConfig | None = None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    _InactiveEvidenceCertificate | None,
]:
    """Return a reduced posterior and its internally consistent objective.

    ``tail_loading=0`` gives the exact Type-II objective for the selected
    zero-tail model.  A positive or implicit tail loading is an isotropic
    approximation to omitted atoms and must not be described as the full-grid
    Type-II evidence.  The negative log evidence is evaluated from the same
    equilibrated Cholesky factor, so no data-space covariance or additional
    factorisation is required.
    """
    if prior_variance.shape[0] != active_indices.shape[0]:
        raise ValueError("Active indices must match the posterior batch size.")
    active_gamma = prior_variance.gather(1, active_indices)
    if columns is None:
        columns = operator.fourier_columns(active_indices).to(
            dtype=y.dtype
        )
    elif columns.dtype != y.dtype:
        columns = columns.to(dtype=y.dtype)
    omitted_variance = (
        (
            prior_variance.sum(dim=1) - active_gamma.sum(dim=1)
        ).clamp_min(0.0)
        if tail_loading is None
        else tail_loading
    )
    effective_noise = noise_variance + omitted_variance

    if gram is None:
        gram = columns.mH @ columns
    precision = torch.diag_embed(active_gamma.reciprocal()).to(y.dtype)
    system = precision + gram / effective_noise[:, None, None]
    system = 0.5 * (system + system.mH)
    diagonal_scale = system.diagonal(
        dim1=-2, dim2=-1
    ).real.clamp_min(torch.finfo(y.real.dtype).tiny).sqrt()
    equilibrated = (
        system
        / diagonal_scale[:, :, None]
        / diagonal_scale[:, None, :]
    )
    equilibrated = 0.5 * (equilibrated + equilibrated.mH)
    if identity is None:
        identity = torch.eye(
            active_indices.shape[1],
            device=operator.device,
            dtype=y.dtype,
        )
    active_config = config or SBLConfig(device=operator.device)
    cholesky = _stable_cholesky(
        equilibrated,
        identity,
        operator,
        active_config,
        fallback_multipliers=(1.0, 10.0, 1e2, 1e3, 1e4, 1e5),
    )
    if cholesky is None:
        raise RuntimeError("Reduced active-set posterior is not positive definite.")

    if column_data is None:
        column_data = columns.mH @ y[:, :, None]
    rhs = column_data / effective_noise[:, None, None]
    scaled_rhs = rhs / diagonal_scale[:, :, None]
    if cholesky.is_cuda:
        # MAGMA's batched cholesky_solve/cholesky_inverse allocate temporary
        # device memory and cannot be captured by CUDA Graphs on Windows.
        # Two triangular solves are algebraically identical and use
        # graph-safe cuBLAS kernels.  L^{-H}L^{-1} gives the inverse needed
        # for the posterior diagonal.
        forward_solution = torch.linalg.solve_triangular(
            cholesky,
            scaled_rhs,
            upper=False,
        )
        equilibrated_mean = torch.linalg.solve_triangular(
            cholesky.mH,
            forward_solution,
            upper=True,
        )
    else:
        equilibrated_mean = torch.cholesky_solve(
            scaled_rhs, cholesky
        )
    batched_identity = identity.expand(
        active_indices.shape[0], -1, -1
    )
    inverse_factor = torch.linalg.solve_triangular(
        cholesky,
        batched_identity,
        upper=False,
    )
    # diag(L^-H L^-1) is the columnwise squared norm of L^-1. The previous
    # implementation formed the complete inverse with a K^3 GEMM although
    # every downstream equation consumes only this diagonal.
    equilibrated_variance = (
        inverse_factor.abs().square().sum(dim=-2)
    )
    active_mean = (
        equilibrated_mean / diagonal_scale[:, :, None]
    ).squeeze(-1)
    active_posterior_variance = (
        equilibrated_variance / diagonal_scale.square()
    ).real.clamp_min(0.0)
    active_data_diagonal = (
        (active_gamma - active_posterior_variance)
        / active_gamma.square()
    ).clamp_min(0.0)

    # Matrix determinant lemma and Woodbury identity:
    #   log|nu I + A Gamma A^H|
    #     = M log(nu) + log|Gamma| + log|S|,
    #   y^H C^-1 y = ||y||^2 / nu - Re(rhs^H S^-1 rhs),
    # where S = Gamma^-1 + A^H A / nu.  ``cholesky`` factors
    # D^-1 S D^-1, hence the two diagonal-scale terms in log|S|.
    log_determinant_system = 2.0 * (
        cholesky.diagonal(dim1=-2, dim2=-1)
        .real.clamp_min(torch.finfo(y.real.dtype).tiny)
        .log()
        .sum(dim=1)
        + diagonal_scale.log().sum(dim=1)
    )
    log_determinant = (
        operator.n_samples * effective_noise.log()
        + active_gamma.log().sum(dim=1)
        + log_determinant_system
    )
    rhs_vector = rhs.squeeze(-1)
    if data_energy is None:
        data_energy = y.abs().square().sum(dim=1)
    quadratic = (
        data_energy / effective_noise
        - (rhs_vector.conj() * active_mean).sum(dim=1).real
    ).clamp_min(0.0)
    negative_log_evidence = log_determinant + quadratic
    inactive_certificate = (
        _inactive_evidence_certificate(
            y,
            columns,
            active_mean,
            effective_noise,
            diagonal_scale,
            cholesky,
            operator,
            active_indices,
        )
        if global_certificate
        else None
    )

    mean = torch.zeros(
        (prior_variance.shape[0], operator.n_grid),
        device=operator.device,
        dtype=y.dtype,
    )
    posterior_variance = prior_variance.clone()
    data_diagonal = torch.zeros_like(prior_variance)
    mean.scatter_(1, active_indices, active_mean)
    posterior_variance.scatter_(
        1, active_indices, active_posterior_variance
    )
    data_diagonal.scatter_(1, active_indices, active_data_diagonal)
    return (
        mean,
        posterior_variance,
        data_diagonal,
        negative_log_evidence,
        inactive_certificate,
    )


def _active_mean_with_evidence(
    y: Tensor,
    active_gamma: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    active_indices: Tensor,
    *,
    columns: Tensor | None = None,
    gram: Tensor | None = None,
    column_data: Tensor | None = None,
    data_energy: Tensor | None = None,
    identity: Tensor | None = None,
    config: SBLConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Return exact zero-tail active means and Type-II evidence only.

    Candidate support screening does not consume posterior covariance
    diagonals.  The general active-posterior routine nevertheless solves
    ``L X = I`` with ``K`` right-hand sides to obtain them.  This specialized
    path retains the identical equilibrated Cholesky objective but performs
    only the one-right-hand-side solve required for the mean.
    """
    if not (
        y.ndim == 2
        and active_gamma.shape == active_indices.shape
        and y.shape[0] == active_indices.shape[0]
        and noise_variance.shape == (y.shape[0],)
    ):
        raise ValueError("Active mean/evidence batch shapes are inconsistent.")
    if columns is None:
        columns = operator.fourier_columns(active_indices).to(dtype=y.dtype)
    elif columns.dtype != y.dtype:
        columns = columns.to(dtype=y.dtype)
    if gram is None:
        gram = columns.mH @ columns

    tiny = torch.finfo(y.real.dtype).tiny
    active_gamma = active_gamma.clamp_min(tiny)
    effective_noise = noise_variance.clamp_min(tiny)
    system = (
        torch.diag_embed(active_gamma.reciprocal()).to(y.dtype)
        + gram / effective_noise[:, None, None]
    )
    system = 0.5 * (system + system.mH)
    diagonal_scale = system.diagonal(
        dim1=-2, dim2=-1
    ).real.clamp_min(tiny).sqrt()
    equilibrated = (
        system
        / diagonal_scale[:, :, None]
        / diagonal_scale[:, None, :]
    )
    equilibrated = 0.5 * (equilibrated + equilibrated.mH)
    if identity is None:
        identity = torch.eye(
            active_indices.shape[1],
            device=operator.device,
            dtype=y.dtype,
        )
    active_config = config or SBLConfig(device=operator.device)
    cholesky = _stable_cholesky(
        equilibrated,
        identity,
        operator,
        active_config,
        fallback_multipliers=(1.0, 10.0, 1e2, 1e3, 1e4, 1e5),
    )
    if cholesky is None:
        raise RuntimeError("Reduced active-set evidence is not positive definite.")

    if column_data is None:
        column_data = columns.mH @ y[:, :, None]
    rhs = column_data / effective_noise[:, None, None]
    scaled_rhs = rhs / diagonal_scale[:, :, None]
    if cholesky.is_cuda:
        forward_solution = torch.linalg.solve_triangular(
            cholesky,
            scaled_rhs,
            upper=False,
        )
        equilibrated_mean = torch.linalg.solve_triangular(
            cholesky.mH,
            forward_solution,
            upper=True,
        )
    else:
        equilibrated_mean = torch.cholesky_solve(
            scaled_rhs, cholesky
        )
    active_mean = (
        equilibrated_mean / diagonal_scale[:, :, None]
    ).squeeze(-1)

    log_determinant_system = 2.0 * (
        cholesky.diagonal(dim1=-2, dim2=-1)
        .real.clamp_min(tiny)
        .log()
        .sum(dim=1)
        + diagonal_scale.log().sum(dim=1)
    )
    log_determinant = (
        operator.n_samples * effective_noise.log()
        + active_gamma.log().sum(dim=1)
        + log_determinant_system
    )
    if data_energy is None:
        data_energy = y.abs().square().sum(dim=1)
    rhs_vector = rhs.squeeze(-1)
    quadratic = (
        data_energy / effective_noise
        - (rhs_vector.conj() * active_mean).sum(dim=1).real
    ).clamp_min(0.0)
    return active_mean, log_determinant + quadratic


def _active_posterior(
    y: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    active_indices: Tensor,
    *,
    columns: Tensor | None = None,
    gram: Tensor | None = None,
    tail_loading: Tensor | None = None,
    config: SBLConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Batched exact posterior on pruned ARD sets with diffuse-tail loading."""
    mean, posterior_variance, data_diagonal, _, _ = (
        _active_posterior_with_evidence(
            y,
            prior_variance,
            noise_variance,
            operator,
            active_indices,
            columns=columns,
            gram=gram,
            tail_loading=tail_loading,
            global_certificate=False,
            config=config,
        )
    )
    return mean, posterior_variance, data_diagonal


def _active_truncation_delta_bound(
    prior_variance: Tensor,
    noise_variance: Tensor,
    active_indices: Tensor,
    prior_floor: Tensor,
    operator: PrefixFourierOperator,
) -> Tensor:
    """Return a BCCB spectral certificate for hard active-set pruning.

    The omitted covariance is Toeplitz/BTTB and is a principal submatrix of
    its Hermitian block-circulant embedding.  Its spectral norm is therefore
    bounded by the maximum magnitude of the embedding eigenvalues, obtained
    by one FFT.  Dividing by the noise lower bound on the retained covariance
    gives a covariance-relative perturbation certificate.  ``prior_floor``
    remains in the signature because finite ARD output precisions use it; the
    certified posterior itself uses the exact zero-tail reduced model.
    """
    del prior_floor
    tail_variance = prior_variance.scatter(
        1,
        active_indices,
        torch.zeros_like(prior_variance.gather(1, active_indices)),
    )
    embedding_spectrum = operator.toeplitz_spectrum(
        tail_variance,
        torch.zeros_like(noise_variance),
    )
    spectral_bound = embedding_spectrum.abs().flatten(1).amax(dim=1)
    reference_loading = noise_variance.clamp_min(
        torch.finfo(operator.real_dtype).tiny
    )
    return spectral_bound / reference_loading


def _uamp_active_em_polish(
    y: Tensor,
    uamp_mean: Tensor,
    uamp_posterior_variance: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    active_indices: Tensor,
    prior_floor: Tensor,
    config: SBLConfig,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor,
    Tensor | None,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
]:
    """Run evidence-gated Type-II updates on a support proposed by UAMP.

    UAMP is only an initializer.  When ``uamp_polish_prune_tail`` is enabled,
    every inactive variance is set to zero and the subsequent posterior,
    support exchanges, coordinate updates, and stopping diagnostics all refer
    to the single Type-II objective

        log|sigma^2 I + A Gamma A^H|
        + y^H (sigma^2 I + A Gamma A^H)^-1 y.

    ``truncation_delta_bound`` measures how far this selected zero-tail model
    can be from the UAMP proposal covariance; it is not required for the
    selected model's exact evidence or coordinate diagnostics.

    With ``uamp_polish_support_prior=True``, support-size changes minimize the
    Bernoulli-Gaussian Type-II MAP objective
    ``negative_log_evidence + lambda * |S|``.  ``lambda`` is the configured
    strength times the log prior odds induced by the initial active capacity.
    """
    need_truncation_diagnostic = (
        config.uamp_polish_prune_tail
        and (
            config.uamp_polish_compute_truncation_diagnostic
            or config.uamp_polish_max_truncation_delta is not None
        )
    )
    truncation_delta_bound = (
        _active_truncation_delta_bound(
            prior_variance,
            noise_variance,
            active_indices,
            prior_floor,
            operator,
        )
        if need_truncation_diagnostic
        else None
    )
    hard_pruning_accepted = (
        (
            torch.zeros_like(noise_variance, dtype=torch.bool)
            if config.uamp_polish_max_truncation_delta is None
            else (
                truncation_delta_bound
                <= config.uamp_polish_max_truncation_delta
            )
        )
        if truncation_delta_bound is not None
        else torch.zeros_like(noise_variance, dtype=torch.bool)
    )
    active_model_selected = torch.full_like(
        noise_variance,
        config.uamp_polish_prune_tail,
        dtype=torch.bool,
    )
    if (
        config.uamp_polish_high_precision
        and prior_variance.dtype == torch.float32
    ):
        y = y.to(torch.complex128)
        uamp_mean = uamp_mean.to(torch.complex128)
        uamp_posterior_variance = uamp_posterior_variance.to(torch.float64)
        prior_variance = prior_variance.to(torch.float64)
        noise_variance = noise_variance.to(torch.float64)
        prior_floor = prior_floor.to(torch.float64)
    working_prior = (
        prior_variance.clone()
        if (
            config.uamp_polish_prune_tail
            or config.uamp_polish_reinitialize
        )
        else prior_variance
    )
    if config.uamp_polish_prune_tail:
        pruned_prior = prior_floor[:, None].expand_as(
            prior_variance
        ).clone()
        selected_prior = prior_variance.gather(1, active_indices)
        pruned_prior.scatter_(1, active_indices, selected_prior)
        working_prior = pruned_prior
    if config.uamp_polish_reinitialize:
        second_moment = (
            uamp_mean.abs().square() + uamp_posterior_variance
        )
        selected_second_moment = second_moment.gather(1, active_indices)
        selected_floor = prior_floor[:, None].expand_as(
            selected_second_moment
        )
        working_prior.scatter_(
            1,
            active_indices,
            torch.maximum(selected_second_moment, selected_floor),
        )

    columns = operator.fourier_columns(active_indices).to(dtype=y.dtype)
    gram = columns.mH @ columns
    column_data = columns.mH @ y[:, :, None]
    data_energy = y.abs().square().sum(dim=1)
    active_identity = torch.eye(
        active_indices.shape[1],
        device=operator.device,
        dtype=y.dtype,
    )
    active_noise = noise_variance
    reduced_tail_loading = (
        torch.zeros_like(noise_variance)
        if config.uamp_polish_prune_tail
        else None
    )
    mean = uamp_mean
    posterior_variance = uamp_posterior_variance
    data_diagonal = torch.zeros_like(prior_variance)
    tiny = torch.finfo(operator.real_dtype).tiny
    best_mean: Tensor | None = None
    best_posterior_variance: Tensor | None = None
    best_data_diagonal: Tensor | None = None
    best_prior: Tensor | None = None
    best_noise: Tensor | None = None
    best_evidence: Tensor | None = None
    best_active_indices: Tensor | None = None
    best_columns: Tensor | None = None
    best_gram: Tensor | None = None
    best_column_data: Tensor | None = None
    best_inactive_certificate: _InactiveEvidenceCertificate | None = None
    accepted_evidence: list[Tensor] = []
    residual_exchange_pending: Tensor | None = None
    residual_exchange_accepted = (
        torch.zeros_like(noise_variance, dtype=torch.int64)
        if config.uamp_polish_residual_exchange_size > 0
        else None
    )

    for polish_iteration in range(config.uamp_polish_iterations):
        (
            candidate_mean,
            candidate_posterior_variance,
            candidate_data_diagonal,
            candidate_evidence,
            candidate_inactive_certificate,
        ) = _active_posterior_with_evidence(
            y,
            working_prior,
            active_noise,
            operator,
            active_indices,
            columns=columns,
            gram=gram,
            column_data=column_data,
            data_energy=data_energy,
            identity=active_identity,
            tail_loading=reduced_tail_loading,
            global_certificate=config.uamp_polish_global_certificate,
            config=config,
        )
        if config.uamp_polish_evidence_gate:
            if best_evidence is None:
                best_mean = candidate_mean
                best_posterior_variance = candidate_posterior_variance
                best_data_diagonal = candidate_data_diagonal
                best_prior = working_prior
                best_noise = active_noise
                best_evidence = candidate_evidence
                best_active_indices = active_indices
                best_columns = columns
                best_gram = gram
                best_column_data = column_data
                best_inactive_certificate = candidate_inactive_certificate
            else:
                evidence_margin = (
                    config.uamp_polish_evidence_margin_factor
                    * torch.finfo(candidate_evidence.dtype).eps
                    * (
                        1.0
                        + candidate_evidence.abs()
                        + best_evidence.abs()
                    )
                )
                accept = torch.isfinite(candidate_evidence) & (
                    ~torch.isfinite(best_evidence)
                    | (
                        candidate_evidence
                        <= best_evidence - evidence_margin
                    )
                )
                vector_mask = accept[:, None]
                best_mean = torch.where(
                    vector_mask, candidate_mean, best_mean
                )
                best_posterior_variance = torch.where(
                    vector_mask,
                    candidate_posterior_variance,
                    best_posterior_variance,
                )
                best_data_diagonal = torch.where(
                    vector_mask,
                    candidate_data_diagonal,
                    best_data_diagonal,
                )
                best_prior = torch.where(
                    vector_mask, working_prior, best_prior
                )
                best_noise = torch.where(
                    accept, active_noise, best_noise
                )
                best_evidence = torch.where(
                    accept, candidate_evidence, best_evidence
                )
                if best_active_indices is None:
                    raise RuntimeError("Missing accepted active indices.")
                best_active_indices = torch.where(
                    vector_mask, active_indices, best_active_indices
                )
                if (
                    best_columns is None
                    or best_gram is None
                    or best_column_data is None
                ):
                    raise RuntimeError("Missing accepted active matrices.")
                matrix_mask = accept[:, None, None]
                best_columns = torch.where(
                    matrix_mask, columns, best_columns
                )
                best_gram = torch.where(
                    matrix_mask, gram, best_gram
                )
                best_column_data = torch.where(
                    matrix_mask, column_data, best_column_data
                )
                if (
                    residual_exchange_pending is not None
                    and residual_exchange_accepted is not None
                ):
                    residual_exchange_accepted = (
                        residual_exchange_accepted
                        + (
                            accept & residual_exchange_pending
                        ).to(torch.int64)
                    )
                    residual_exchange_pending = None
                if (
                    candidate_inactive_certificate is not None
                    and best_inactive_certificate is not None
                ):
                    best_inactive_certificate = (
                        _InactiveEvidenceCertificate(
                            max_ratio=torch.where(
                                accept,
                                candidate_inactive_certificate.max_ratio,
                                best_inactive_certificate.max_ratio,
                            ),
                            best_index=torch.where(
                                accept,
                                candidate_inactive_certificate.best_index,
                                best_inactive_certificate.best_index,
                            ),
                            evidence_gain=torch.where(
                                accept,
                                candidate_inactive_certificate.evidence_gain,
                                best_inactive_certificate.evidence_gain,
                            ),
                            optimal_variance=torch.where(
                                accept,
                                candidate_inactive_certificate.optimal_variance,
                                best_inactive_certificate.optimal_variance,
                            ),
                        )
                    )
            mean = best_mean
            posterior_variance = best_posterior_variance
            data_diagonal = best_data_diagonal
            base_prior = best_prior
            base_noise = best_noise
            if best_active_indices is None:
                raise RuntimeError("Active evidence selection lost support.")
            if (
                best_columns is None
                or best_gram is None
                or best_column_data is None
            ):
                raise RuntimeError("Active evidence selection lost matrices.")
            active_indices = best_active_indices
            columns = best_columns
            gram = best_gram
            column_data = best_column_data
            accepted_evidence.append(best_evidence)
        else:
            mean = candidate_mean
            posterior_variance = candidate_posterior_variance
            data_diagonal = candidate_data_diagonal
            base_prior = working_prior
            base_noise = active_noise
            best_evidence = candidate_evidence
            best_inactive_certificate = candidate_inactive_certificate
            accepted_evidence.append(candidate_evidence)

        if (
            mean is None
            or posterior_variance is None
            or data_diagonal is None
            or base_prior is None
            or base_noise is None
        ):
            raise RuntimeError("Active evidence selection did not initialize.")
        if polish_iteration + 1 == config.uamp_polish_iterations:
            break

        active_gamma = base_prior.gather(1, active_indices)
        active_posterior_variance = posterior_variance.gather(
            1, active_indices
        )
        new_active_gamma = (
            mean.gather(1, active_indices).abs().square()
            + active_posterior_variance
            + config.hyper_rate
        ) / (1.0 + config.hyper_shape)
        selected_floor = prior_floor[:, None].expand_as(new_active_gamma)
        new_active_gamma = torch.maximum(
            new_active_gamma, selected_floor
        ).clamp_max(config.max_precision)
        keep = config.uamp_polish_damping
        updated_active_gamma = (
            keep * active_gamma + (1.0 - keep) * new_active_gamma
        )

        active_prediction_mean = mean.gather(1, active_indices)
        prediction = (
            columns @ active_prediction_mean[:, :, None]
        ).squeeze(-1)
        residual = y - prediction
        residual_energy = residual.abs().square().sum(dim=1)
        inactive_variance = (
            torch.zeros_like(base_noise)
            if reduced_tail_loading is not None
            else (
                base_prior.sum(dim=1)
                - base_prior.gather(1, active_indices).sum(dim=1)
            ).clamp_min(0.0)
        )
        effective_noise = base_noise + inactive_variance
        effective_dof = (
            1.0
            - active_posterior_variance
            / active_gamma.clamp_min(tiny)
        ).clamp(0.0, 1.0).sum(dim=1)
        if config.fixed_noise_variance is not None:
            updated_noise = torch.full_like(
                base_noise, config.fixed_noise_variance
            )
        elif config.noise_update == "em":
            updated_effective_noise = (
                residual_energy
                + effective_noise * effective_dof
                + config.noise_rate
            ) / (operator.n_samples + config.noise_shape)
            # The active posterior absorbs the inactive diagonal tail into
            # nu = sigma^2 + sum(gamma_tail).  EM updates nu exactly; remove
            # the fixed tail loading to recover the physical noise variance.
            updated_noise = updated_effective_noise - inactive_variance
        else:
            denominator = (
                operator.n_samples
                - effective_dof
                + config.noise_shape
            ).clamp_min(torch.finfo(operator.real_dtype).eps)
            updated_effective_noise = (
                residual_energy + config.noise_rate
            ) / denominator
            updated_noise = updated_effective_noise - inactive_variance
        updated_noise = torch.maximum(updated_noise, prior_floor)
        if config.fixed_noise_variance is None:
            active_noise = (
                keep * base_noise + (1.0 - keep) * updated_noise
            )
        else:
            active_noise = updated_noise
        working_prior = base_prior.scatter(
            1, active_indices, updated_active_gamma
        )
        exchange_size = min(
            config.uamp_polish_residual_exchange_size,
            max(active_indices.shape[1] - 1, 0),
            operator.n_grid - active_indices.shape[1],
        )
        if exchange_size > 0 and polish_iteration == 0:
            residual_score = operator.adjoint_flat(
                residual
            ).abs().square()
            residual_score = residual_score.scatter(
                1,
                active_indices,
                torch.full_like(
                    active_indices,
                    -torch.inf,
                    dtype=residual_score.dtype,
                ),
            )
            addition_indices = torch.topk(
                residual_score, exchange_size, dim=1
            ).indices
            addition_energy = residual_score.gather(
                1, addition_indices
            )
            active_certificate = _active_evidence_certificate(
                mean.gather(1, active_indices),
                active_posterior_variance,
                active_gamma,
                active_indices,
            )
            removal_positions = torch.topk(
                active_certificate.deletion_gain,
                exchange_size,
                dim=1,
            ).indices
            removal_indices = active_indices.gather(
                1, removal_positions
            )
            column_energy = float(operator.n_samples)
            addition_ratio = (
                addition_energy
                / (
                    base_noise[:, None] * column_energy
                ).clamp_min(tiny)
            )
            addition_gain = torch.where(
                addition_ratio > 1.0,
                addition_ratio
                - 1.0
                - addition_ratio.clamp_min(tiny).log(),
                torch.zeros_like(addition_ratio),
            )
            deletion_gain = (
                active_certificate.deletion_gain.gather(
                    1, removal_positions
                )
            )
            exchange_mask = (
                addition_gain + deletion_gain
            ) > 0.0
            replacement_indices = torch.where(
                exchange_mask, addition_indices, removal_indices
            )
            exchanged_indices = active_indices.scatter(
                1, removal_positions, replacement_indices
            )
            addition_gamma = (
                (
                    addition_energy
                    - base_noise[:, None] * column_energy
                ).clamp_min(0.0)
                / (column_energy * column_energy)
            )
            addition_gamma = torch.maximum(
                addition_gamma,
                prior_floor[:, None].expand_as(addition_gamma),
            ).clamp_max(config.max_precision)
            retained_gamma = working_prior.gather(
                1, removal_indices
            )
            removal_gamma = torch.where(
                exchange_mask,
                prior_floor[:, None].expand_as(retained_gamma),
                retained_gamma,
            )
            working_prior = working_prior.scatter(
                1,
                removal_indices,
                removal_gamma,
            )
            replacement_gamma = torch.where(
                exchange_mask, addition_gamma, retained_gamma
            )
            working_prior = working_prior.scatter(
                1, replacement_indices, replacement_gamma
            )
            active_indices = exchanged_indices
            columns = operator.fourier_columns(
                active_indices
            ).to(dtype=y.dtype)
            gram = columns.mH @ columns
            column_data = columns.mH @ y[:, :, None]
            residual_exchange_pending = exchange_mask.any(dim=1)

    final_prior = (
        best_prior
        if config.uamp_polish_evidence_gate and best_prior is not None
        else working_prior
    )
    final_noise = (
        best_noise
        if config.uamp_polish_evidence_gate and best_noise is not None
        else active_noise
    )
    final_active_indices = (
        best_active_indices
        if best_active_indices is not None
        else active_indices
    )
    support_repairs_accepted: Tensor | None = residual_exchange_accepted
    proposal_certificate = best_inactive_certificate
    if config.uamp_polish_support_repairs > 0:
        if support_repairs_accepted is None:
            support_repairs_accepted = torch.zeros_like(
                final_noise, dtype=torch.int64
            )
    for _ in range(config.uamp_polish_support_repairs):
        if proposal_certificate is None or best_evidence is None:
            raise RuntimeError("Support repair requires a certificate.")
        current_active_certificate = _active_evidence_certificate(
            mean.gather(1, final_active_indices),
            posterior_variance.gather(1, final_active_indices),
            final_prior.gather(1, final_active_indices),
            final_active_indices,
        )
        repair_index = proposal_certificate.best_index[:, None]
        replaced_position = (
            current_active_certificate.best_deletion_position[:, None]
        )
        replaced_index = final_active_indices.gather(
            1, replaced_position
        )
        repair_indices = final_active_indices.scatter(
            1, replaced_position, repair_index
        )
        repair_variance = torch.maximum(
            proposal_certificate.optimal_variance,
            prior_floor,
        )
        repair_prior = final_prior.scatter(
            1, replaced_index, prior_floor[:, None]
        )
        repair_prior = repair_prior.scatter(
            1, repair_index, repair_variance[:, None]
        )
        repair_columns = operator.fourier_columns(repair_indices).to(
            dtype=y.dtype
        )
        repair_gram = repair_columns.mH @ repair_columns
        repair_tail_loading = (
            torch.zeros_like(final_noise)
            if config.uamp_polish_prune_tail
            else None
        )
        (
            repair_mean,
            repair_posterior_variance,
            repair_data_diagonal,
            repair_evidence,
            repair_certificate,
        ) = _active_posterior_with_evidence(
            y,
            repair_prior,
            final_noise,
            operator,
            repair_indices,
            columns=repair_columns,
            gram=repair_gram,
            tail_loading=repair_tail_loading,
            global_certificate=True,
            config=config,
        )
        repair_accepted = (
            (proposal_certificate.max_ratio > 1.0)
            & torch.isfinite(repair_evidence)
            & (repair_evidence <= best_evidence)
        )
        support_repairs_accepted = (
            support_repairs_accepted + repair_accepted.to(torch.int64)
        )
        repair_mask = repair_accepted[:, None]
        mean = torch.where(repair_mask, repair_mean, mean)
        posterior_variance = torch.where(
            repair_mask,
            repair_posterior_variance,
            posterior_variance,
        )
        data_diagonal = torch.where(
            repair_mask, repair_data_diagonal, data_diagonal
        )
        final_prior = torch.where(
            repair_mask, repair_prior, final_prior
        )
        final_active_indices = torch.where(
            repair_mask, repair_indices, final_active_indices
        )
        best_evidence = torch.where(
            repair_accepted, repair_evidence, best_evidence
        )
        accepted_evidence.append(best_evidence)
        if repair_certificate is not None:
            if best_inactive_certificate is None:
                raise RuntimeError("Missing accepted support certificate.")
            best_inactive_certificate = _InactiveEvidenceCertificate(
                max_ratio=torch.where(
                    repair_accepted,
                    repair_certificate.max_ratio,
                    best_inactive_certificate.max_ratio,
                ),
                best_index=torch.where(
                    repair_accepted,
                    repair_certificate.best_index,
                    best_inactive_certificate.best_index,
                ),
                evidence_gain=torch.where(
                    repair_accepted,
                    repair_certificate.evidence_gain,
                    best_inactive_certificate.evidence_gain,
                ),
                optimal_variance=torch.where(
                    repair_accepted,
                    repair_certificate.optimal_variance,
                    best_inactive_certificate.optimal_variance,
                ),
            )
            proposal_certificate = best_inactive_certificate

    initial_capacity = final_active_indices.shape[1]
    if config.uamp_polish_support_prior and initial_capacity < operator.n_grid:
        prior_odds = max(
            (operator.n_grid - initial_capacity)
            / max(initial_capacity, 1),
            1.0,
        )
        support_penalty_value = (
            config.uamp_polish_support_prior_strength * log(prior_odds)
        )
    else:
        support_penalty_value = 0.0
    support_log_odds_penalty = torch.full_like(
        final_noise, support_penalty_value
    )
    best_penalized_evidence = (
        best_evidence
        + support_log_odds_penalty * final_active_indices.shape[1]
    )
    accepted_penalized_evidence = [
        value
        + support_log_odds_penalty * final_active_indices.shape[1]
        for value in accepted_evidence
    ]

    support_shrink_accepted: Tensor | None = None
    if config.uamp_polish_support_shrink_steps > 0:
        support_shrink_accepted = torch.zeros_like(
            final_noise, dtype=torch.int64
        )
    for _ in range(config.uamp_polish_support_shrink_steps):
        active_size = final_active_indices.shape[1]
        if active_size <= 1 or best_evidence is None:
            break
        active_certificate = _active_evidence_certificate(
            mean.gather(1, final_active_indices),
            posterior_variance.gather(1, final_active_indices),
            final_prior.gather(1, final_active_indices),
            final_active_indices,
        )
        shrink_gain = (
            active_certificate.deletion_gain
            + support_log_odds_penalty[:, None]
        )
        eligible = (
            shrink_gain > config.uamp_polish_coordinate_tolerance
        )
        removal_count = min(
            config.uamp_polish_support_shrink_batch_size,
            active_size - 1,
            int(eligible.sum(dim=1).min().item()),
        )
        if removal_count < 1:
            break
        delete_position = torch.topk(
            shrink_gain, removal_count, dim=1
        ).indices
        position = torch.arange(
            active_size, device=final_active_indices.device
        )[None, :]
        delete_mask = torch.zeros(
            final_active_indices.shape,
            device=final_active_indices.device,
            dtype=torch.bool,
        ).scatter(1, delete_position, True)
        keep = ~delete_mask
        shrink_indices = final_active_indices[keep].reshape(
            final_active_indices.shape[0],
            active_size - removal_count,
        )
        delete_index = final_active_indices.gather(
            1, delete_position
        )
        shrink_prior = final_prior.scatter(
            1,
            delete_index,
            prior_floor[:, None].expand_as(delete_index),
        )
        shrink_columns = operator.fourier_columns(shrink_indices).to(
            dtype=y.dtype
        )
        shrink_gram = shrink_columns.mH @ shrink_columns
        (
            shrink_mean,
            shrink_posterior_variance,
            shrink_data_diagonal,
            shrink_evidence,
            shrink_certificate,
        ) = _active_posterior_with_evidence(
            y,
            shrink_prior,
            final_noise,
            operator,
            shrink_indices,
            columns=shrink_columns,
            gram=shrink_gram,
            tail_loading=torch.zeros_like(final_noise),
            global_certificate=True,
            config=config,
        )
        shrink_penalized_evidence = (
            shrink_evidence
            + support_log_odds_penalty
            * (active_size - removal_count)
        )
        shrink_accepted = (
            torch.isfinite(shrink_penalized_evidence)
            & (shrink_penalized_evidence <= best_penalized_evidence)
        )
        if not bool(shrink_accepted.all()):
            break
        mean = shrink_mean
        posterior_variance = shrink_posterior_variance
        data_diagonal = shrink_data_diagonal
        final_prior = shrink_prior
        final_active_indices = shrink_indices
        best_evidence = shrink_evidence
        best_penalized_evidence = shrink_penalized_evidence
        best_inactive_certificate = shrink_certificate
        proposal_certificate = shrink_certificate
        support_shrink_accepted = (
            support_shrink_accepted
            + removal_count
            * torch.ones_like(support_shrink_accepted)
        )
        accepted_evidence.append(best_evidence)
        accepted_penalized_evidence.append(best_penalized_evidence)

    support_growth_accepted: Tensor | None = None
    if config.uamp_polish_support_growth_steps > 0:
        support_growth_accepted = torch.zeros_like(
            final_noise, dtype=torch.int64
        )
    stationarity_ratio_value = _ratio_threshold_for_evidence_tolerance(
        config.uamp_polish_coordinate_tolerance
    )
    for _ in range(config.uamp_polish_support_growth_steps):
        if proposal_certificate is None or best_evidence is None:
            raise RuntimeError("Support growth requires a certificate.")
        if final_active_indices.shape[1] >= operator.n_grid:
            break
        growth_requested = (
            proposal_certificate.evidence_gain
            > (
                support_log_odds_penalty
                + config.uamp_polish_coordinate_tolerance
            )
        )
        if not bool(growth_requested.all()):
            # Batched active sets have one shared width.  Requiring unanimous
            # growth keeps the representation exact and avoids masking a
            # rejected atom as active for another batch item.  Publication
            # stationarity runs should therefore evaluate independent scenes
            # separately when their support sizes differ.
            break
        growth_index = proposal_certificate.best_index[:, None]
        growth_indices = torch.cat(
            (final_active_indices, growth_index), dim=1
        )
        growth_variance = torch.maximum(
            proposal_certificate.optimal_variance,
            prior_floor,
        )
        growth_prior = final_prior.scatter(
            1, growth_index, growth_variance[:, None]
        )
        growth_columns = operator.fourier_columns(growth_indices).to(
            dtype=y.dtype
        )
        growth_gram = growth_columns.mH @ growth_columns
        (
            growth_mean,
            growth_posterior_variance,
            growth_data_diagonal,
            growth_evidence,
            growth_certificate,
        ) = _active_posterior_with_evidence(
            y,
            growth_prior,
            final_noise,
            operator,
            growth_indices,
            columns=growth_columns,
            gram=growth_gram,
            tail_loading=torch.zeros_like(final_noise),
            global_certificate=True,
            config=config,
        )
        growth_accepted = (
            torch.isfinite(growth_evidence)
            & (
                growth_evidence
                + support_log_odds_penalty
                * growth_indices.shape[1]
                <= best_penalized_evidence
            )
        )
        if not bool(growth_accepted.all()):
            break
        mean = growth_mean
        posterior_variance = growth_posterior_variance
        data_diagonal = growth_data_diagonal
        final_prior = growth_prior
        final_active_indices = growth_indices
        best_evidence = growth_evidence
        best_penalized_evidence = (
            growth_evidence
            + support_log_odds_penalty
            * growth_indices.shape[1]
        )
        best_inactive_certificate = growth_certificate
        proposal_certificate = growth_certificate
        support_growth_accepted = (
            support_growth_accepted
            + torch.ones_like(support_growth_accepted)
        )
        accepted_evidence.append(best_evidence)
        accepted_penalized_evidence.append(best_penalized_evidence)

    inactive_ratio_threshold: Tensor | None = None
    stationarity_ratio_threshold: Tensor | None = None
    familywise_ratio_threshold: Tensor | None = None
    if config.uamp_polish_global_certificate:
        stationarity_ratio_threshold = torch.full_like(
            final_noise,
            stationarity_ratio_value,
        )
        # Backward-compatible name.  This is now always the mathematical
        # Type-II stationarity threshold, never a multiple-testing threshold.
        inactive_ratio_threshold = stationarity_ratio_threshold
        inactive_count = max(
            operator.n_grid - final_active_indices.shape[1], 1
        )
        if config.uamp_polish_familywise_error_rate is not None:
            familywise_ratio_threshold = torch.full_like(
                final_noise,
                log(
                inactive_count
                / config.uamp_polish_familywise_error_rate
                ),
            )

    coordinate_steps_accepted: Tensor | None = None
    if config.uamp_polish_coordinate_steps > 0:
        coordinate_steps_accepted = torch.zeros_like(
            final_noise, dtype=torch.int64
        )
    noise_coordinate_steps_accepted: Tensor | None = None
    joint_coordinate_cycles: Tensor | None = None
    if config.uamp_polish_noise_coordinate_steps > 0:
        noise_coordinate_steps_accepted = torch.zeros_like(
            final_noise, dtype=torch.int64
        )
        joint_coordinate_cycles = torch.zeros_like(
            final_noise, dtype=torch.int64
        )
    gamma_coordinate_attempts = 0
    noise_coordinate_attempts = 0
    joint_noise_blocks = 0
    previous_joint_action = -1
    joint_action_budget = (
        config.uamp_polish_coordinate_steps
        + config.uamp_polish_noise_coordinate_steps
    )
    joint_iteration_limit = (
        joint_action_budget + config.uamp_polish_joint_cycles
        if joint_action_budget > 0
        else 0
    )
    for _ in range(joint_iteration_limit):
        if best_evidence is None or best_inactive_certificate is None:
            raise RuntimeError(
                "Coordinate refinement requires exact evidence certificates."
            )
        current_active_gamma = final_prior.gather(
            1, final_active_indices
        )
        current_active_mean = mean.gather(1, final_active_indices)
        current_active_posterior_variance = posterior_variance.gather(
            1, final_active_indices
        )
        active_certificate = _active_evidence_certificate(
            current_active_mean,
            current_active_posterior_variance,
            current_active_gamma,
            final_active_indices,
        )

        # A MAP coordinate step chooses among all three admissible model
        # changes under one objective
        #
        #   J(S, gamma, sigma2) = L_TypeII + lambda * |S|.
        #
        # Previous versions ran shrink, growth and variance refinement as
        # separate one-way phases.  A growth/update can make a later deletion
        # profitable (and conversely), so that schedule could terminate with a
        # large certified MAP gain still available.  Interleaving the three
        # actions is necessary for the reported coordinate-stationarity test
        # to match the algorithm that produced the result.
        if config.uamp_polish_support_prior:
            update_gain = active_certificate.max_nondeletion_gain
            addition_gain = (
                best_inactive_certificate.evidence_gain
                - support_log_odds_penalty
            )
            deletion_gain = (
                active_certificate.max_deletion_gain
                + support_log_odds_penalty
            )
            negative_infinity = torch.full_like(
                update_gain, -torch.inf
            )
            if final_active_indices.shape[1] >= operator.n_grid:
                addition_gain = negative_infinity
            if final_active_indices.shape[1] <= 1:
                deletion_gain = negative_infinity
            action_gains = torch.stack(
                (update_gain, addition_gain, deletion_gain), dim=1
            )
            action = action_gains.argmax(dim=1)
            action_code_by_batch = action
        else:
            # Fixed-cardinality Type-II uses an exchange instead of a
            # support-size prior.  Code 3 denotes delete-and-add in one exact
            # proposal, preserving the selected model width.
            update_gain = active_certificate.max_gain
            exchange_gain = (
                best_inactive_certificate.evidence_gain
                + active_certificate.max_deletion_gain
            )
            action_gains = torch.stack(
                (update_gain, exchange_gain), dim=1
            )
            action = action_gains.argmax(dim=1)
            action_code_by_batch = torch.where(
                action == 0, action, torch.full_like(action, 3)
            )
        if not bool(
            (
                action_code_by_batch
                == action_code_by_batch[:1]
            ).all()
        ):
            # A rectangular batched tensor has a shared active-set width.
            # Independent research scenes are intentionally evaluated one at
            # a time; MMV scenes share the model order.  Refuse to pad a
            # rejected atom with a fake positive variance merely to keep a
            # common width.
            break
        action_code = int(action_code_by_batch[0].item())
        proposal_gain = action_gains.gather(
            1, action[:, None]
        ).squeeze(1)
        proposal_significant = (
            gamma_coordinate_attempts
            < config.uamp_polish_coordinate_steps
        ) & (
            proposal_gain
            > config.uamp_polish_coordinate_tolerance
        )

        noise_certificate: _NoiseEvidenceCertificate | None = None
        noise_candidate_mean: Tensor | None = None
        noise_candidate_posterior_variance: Tensor | None = None
        noise_candidate_data_diagonal: Tensor | None = None
        noise_candidate_evidence: Tensor | None = None
        noise_candidate_inactive_certificate: (
            _InactiveEvidenceCertificate | None
        ) = None
        noise_candidate_penalized_evidence: Tensor | None = None
        noise_gain = torch.zeros_like(proposal_gain)
        noise_significant = torch.zeros_like(
            proposal_significant
        )
        if (
            noise_coordinate_attempts
            < config.uamp_polish_noise_coordinate_steps
            and not bool(proposal_significant.any())
        ):
            noise_certificate = _noise_evidence_certificate(
                y,
                mean,
                posterior_variance,
                final_prior,
                final_noise,
                final_active_indices,
                prior_floor,
                operator,
                config,
            )
            noise_requires_evaluation = (
                noise_certificate.projected_residual
                > config.uamp_polish_noise_tolerance
            )
            if bool(noise_requires_evaluation.any()):
                noise_columns = operator.fourier_columns(
                    final_active_indices
                ).to(dtype=y.dtype)
                noise_gram = noise_columns.mH @ noise_columns
                (
                    noise_candidate_mean,
                    noise_candidate_posterior_variance,
                    noise_candidate_data_diagonal,
                    noise_candidate_evidence,
                    noise_candidate_inactive_certificate,
                ) = _active_posterior_with_evidence(
                    y,
                    final_prior,
                    noise_certificate.candidate_variance,
                    operator,
                    final_active_indices,
                    columns=noise_columns,
                    gram=noise_gram,
                    tail_loading=torch.zeros_like(final_noise),
                    global_certificate=True,
                    config=config,
                )
                noise_candidate_penalized_evidence = (
                    noise_candidate_evidence
                    + support_log_odds_penalty
                    * final_active_indices.shape[1]
                )
                noise_gain = (
                    best_penalized_evidence
                    - noise_candidate_penalized_evidence
                )
                noise_significant = (
                    noise_requires_evaluation
                    & torch.isfinite(noise_candidate_evidence)
                    & (
                        noise_gain
                        >= config.uamp_polish_noise_objective_tolerance
                    )
                )

        choose_noise = noise_significant & (
            (~proposal_significant) | (noise_gain >= proposal_gain)
        )
        starting_noise_block = previous_joint_action != 4
        if (
            starting_noise_block
            and joint_noise_blocks >= config.uamp_polish_joint_cycles
        ):
            choose_noise = torch.zeros_like(choose_noise)
        if bool(choose_noise.any()):
            if (
                noise_certificate is None
                or noise_candidate_mean is None
                or noise_candidate_posterior_variance is None
                or noise_candidate_data_diagonal is None
                or noise_candidate_evidence is None
                or noise_candidate_penalized_evidence is None
                or noise_candidate_inactive_certificate is None
                or noise_coordinate_steps_accepted is None
            ):
                raise RuntimeError(
                    "Noise-coordinate proposal was not initialized."
                )
            noise_coordinate_attempts += 1
            noise_mask = choose_noise[:, None]
            mean = torch.where(
                noise_mask, noise_candidate_mean, mean
            )
            posterior_variance = torch.where(
                noise_mask,
                noise_candidate_posterior_variance,
                posterior_variance,
            )
            data_diagonal = torch.where(
                noise_mask,
                noise_candidate_data_diagonal,
                data_diagonal,
            )
            final_noise = torch.where(
                choose_noise,
                noise_certificate.candidate_variance,
                final_noise,
            )
            best_evidence = torch.where(
                choose_noise,
                noise_candidate_evidence,
                best_evidence,
            )
            best_penalized_evidence = torch.where(
                choose_noise,
                noise_candidate_penalized_evidence,
                best_penalized_evidence,
            )
            best_inactive_certificate = _InactiveEvidenceCertificate(
                max_ratio=torch.where(
                    choose_noise,
                    noise_candidate_inactive_certificate.max_ratio,
                    best_inactive_certificate.max_ratio,
                ),
                best_index=torch.where(
                    choose_noise,
                    noise_candidate_inactive_certificate.best_index,
                    best_inactive_certificate.best_index,
                ),
                evidence_gain=torch.where(
                    choose_noise,
                    noise_candidate_inactive_certificate.evidence_gain,
                    best_inactive_certificate.evidence_gain,
                ),
                optimal_variance=torch.where(
                    choose_noise,
                    noise_candidate_inactive_certificate.optimal_variance,
                    best_inactive_certificate.optimal_variance,
                ),
            )
            proposal_certificate = best_inactive_certificate
            noise_coordinate_steps_accepted = (
                noise_coordinate_steps_accepted
                + choose_noise.to(torch.int64)
            )
            if starting_noise_block:
                joint_noise_blocks += 1
                if joint_coordinate_cycles is not None:
                    joint_coordinate_cycles = (
                        joint_coordinate_cycles
                        + choose_noise.to(torch.int64)
                    )
            accepted_evidence.append(best_evidence)
            accepted_penalized_evidence.append(
                best_penalized_evidence
            )
            previous_joint_action = 4
            continue

        if (
            config.uamp_polish_coordinate_early_stop
            and not bool(proposal_significant.any())
        ):
            break
        if action_code in (1, 2) and not bool(
            proposal_significant.all()
        ):
            break
        gamma_coordinate_attempts += 1

        if action_code == 0:
            proposal_position = (
                active_certificate.best_nondeletion_position
                if config.uamp_polish_support_prior
                else active_certificate.best_position
            )
            proposal_index = (
                active_certificate.best_nondeletion_index
                if config.uamp_polish_support_prior
                else active_certificate.best_index
            )
            proposal_variance = torch.maximum(
                (
                    active_certificate.best_nondeletion_variance
                    if config.uamp_polish_support_prior
                    else active_certificate.optimal_variance
                ),
                prior_floor,
            )
            coordinate_indices = final_active_indices
            coordinate_prior = final_prior.scatter(
                1, proposal_index[:, None], proposal_variance[:, None]
            )
        elif action_code == 1:
            proposal_index = best_inactive_certificate.best_index
            proposal_variance = torch.maximum(
                best_inactive_certificate.optimal_variance,
                prior_floor,
            )
            coordinate_indices = torch.cat(
                (final_active_indices, proposal_index[:, None]), dim=1
            )
            coordinate_prior = final_prior.scatter(
                1, proposal_index[:, None], proposal_variance[:, None]
            )
        elif action_code == 2:
            proposal_position = (
                active_certificate.best_deletion_position
            )
            proposal_index = final_active_indices.gather(
                1, proposal_position[:, None]
            ).squeeze(1)
            position = torch.arange(
                final_active_indices.shape[1],
                device=final_active_indices.device,
            )[None, :]
            keep = position != proposal_position[:, None]
            coordinate_indices = final_active_indices[keep].reshape(
                final_active_indices.shape[0],
                final_active_indices.shape[1] - 1,
            )
            coordinate_prior = final_prior.scatter(
                1, proposal_index[:, None], prior_floor[:, None]
            )
        else:
            proposal_position = (
                active_certificate.best_deletion_position
            )
            old_index = final_active_indices.gather(
                1, proposal_position[:, None]
            ).squeeze(1)
            proposal_index = best_inactive_certificate.best_index
            proposal_variance = torch.maximum(
                best_inactive_certificate.optimal_variance,
                prior_floor,
            )
            coordinate_indices = final_active_indices.scatter(
                1, proposal_position[:, None], proposal_index[:, None]
            )
            coordinate_prior = final_prior.scatter(
                1, old_index[:, None], prior_floor[:, None]
            )
            coordinate_prior = coordinate_prior.scatter(
                1, proposal_index[:, None], proposal_variance[:, None]
            )

        coordinate_columns = operator.fourier_columns(
            coordinate_indices
        ).to(dtype=y.dtype)
        coordinate_gram = coordinate_columns.mH @ coordinate_columns
        coordinate_tail_loading = (
            torch.zeros_like(final_noise)
            if config.uamp_polish_prune_tail
            else None
        )
        (
            coordinate_mean,
            coordinate_posterior_variance,
            coordinate_data_diagonal,
            coordinate_evidence,
            coordinate_inactive_certificate,
        ) = _active_posterior_with_evidence(
            y,
            coordinate_prior,
            final_noise,
            operator,
            coordinate_indices,
            columns=coordinate_columns,
            gram=coordinate_gram,
            tail_loading=coordinate_tail_loading,
            global_certificate=True,
            config=config,
        )
        coordinate_penalized_evidence = (
            coordinate_evidence
            + support_log_odds_penalty
            * coordinate_indices.shape[1]
        )
        coordinate_accepted = (
            proposal_significant
            & torch.isfinite(coordinate_evidence)
            & (
                best_penalized_evidence
                - coordinate_penalized_evidence
                >= config.uamp_polish_coordinate_tolerance
            )
        )
        if action_code in (1, 2):
            if not bool(coordinate_accepted.all()):
                break
            mean = coordinate_mean
            posterior_variance = coordinate_posterior_variance
            data_diagonal = coordinate_data_diagonal
            final_prior = coordinate_prior
            final_active_indices = coordinate_indices
            best_evidence = coordinate_evidence
            best_penalized_evidence = coordinate_penalized_evidence
            if action_code == 1 and support_growth_accepted is not None:
                support_growth_accepted = (
                    support_growth_accepted
                    + torch.ones_like(support_growth_accepted)
                )
            if action_code == 2 and support_shrink_accepted is not None:
                support_shrink_accepted = (
                    support_shrink_accepted
                    + torch.ones_like(support_shrink_accepted)
                )
        else:
            coordinate_mask = coordinate_accepted[:, None]
            mean = torch.where(
                coordinate_mask, coordinate_mean, mean
            )
            posterior_variance = torch.where(
                coordinate_mask,
                coordinate_posterior_variance,
                posterior_variance,
            )
            data_diagonal = torch.where(
                coordinate_mask,
                coordinate_data_diagonal,
                data_diagonal,
            )
            final_prior = torch.where(
                coordinate_mask, coordinate_prior, final_prior
            )
            final_active_indices = torch.where(
                coordinate_mask,
                coordinate_indices,
                final_active_indices,
            )
            best_evidence = torch.where(
                coordinate_accepted,
                coordinate_evidence,
                best_evidence,
            )
            best_penalized_evidence = torch.where(
                coordinate_accepted,
                coordinate_penalized_evidence,
                best_penalized_evidence,
            )
        coordinate_steps_accepted = (
            coordinate_steps_accepted
            + coordinate_accepted.to(torch.int64)
        )
        if bool(coordinate_accepted.any()):
            previous_joint_action = action_code
        accepted_evidence.append(best_evidence)
        accepted_penalized_evidence.append(best_penalized_evidence)
        if coordinate_inactive_certificate is not None:
            best_inactive_certificate = _InactiveEvidenceCertificate(
                max_ratio=torch.where(
                    coordinate_accepted,
                    coordinate_inactive_certificate.max_ratio,
                    best_inactive_certificate.max_ratio,
                ),
                best_index=torch.where(
                    coordinate_accepted,
                    coordinate_inactive_certificate.best_index,
                    best_inactive_certificate.best_index,
                ),
                evidence_gain=torch.where(
                    coordinate_accepted,
                    coordinate_inactive_certificate.evidence_gain,
                    best_inactive_certificate.evidence_gain,
                ),
                optimal_variance=torch.where(
                    coordinate_accepted,
                    coordinate_inactive_certificate.optimal_variance,
                    best_inactive_certificate.optimal_variance,
                ),
            )
        if (
            config.uamp_polish_coordinate_early_stop
            and not bool(coordinate_accepted.any())
        ):
            break

    final_active_certificate: _ActiveEvidenceCertificate | None = None
    coordinate_stationary: Tensor | None = None
    familywise_support_certified: Tensor | None = None
    familywise_no_discovery: Tensor | None = None
    inactive_penalized_evidence_gain: Tensor | None = None
    active_penalized_evidence_gain: Tensor | None = None
    type2_map_coordinate_stationary: Tensor | None = None
    noise_evidence_gradient: Tensor | None = None
    noise_fixed_point_residual: Tensor | None = None
    noise_evidence_gain: Tensor | None = None
    noise_coordinate_stationary: Tensor | None = None
    joint_map_coordinate_stationary: Tensor | None = None
    if config.uamp_polish_global_certificate:
        final_active_certificate = _active_evidence_certificate(
            mean.gather(1, final_active_indices),
            posterior_variance.gather(1, final_active_indices),
            final_prior.gather(1, final_active_indices),
            final_active_indices,
        )
        if best_inactive_certificate is None:
            raise RuntimeError("Missing final inactive certificate.")
        if familywise_ratio_threshold is not None:
            familywise_no_discovery = (
                best_inactive_certificate.max_ratio
                <= familywise_ratio_threshold
            )
        coordinate_stationary = (
            (
                best_inactive_certificate.evidence_gain
                <= config.uamp_polish_coordinate_tolerance
            )
            & (
                final_active_certificate.max_gain
                <= config.uamp_polish_coordinate_tolerance
            )
        )
        inactive_penalized_evidence_gain = (
            best_inactive_certificate.evidence_gain
            - support_log_odds_penalty
        ).clamp_min(0.0)
        active_penalized_evidence_gain = torch.maximum(
            final_active_certificate.max_nondeletion_gain,
            (
                final_active_certificate.max_deletion_gain
                + support_log_odds_penalty
            ).clamp_min(0.0),
        )
        type2_map_coordinate_stationary = (
            (
                inactive_penalized_evidence_gain
                <= config.uamp_polish_coordinate_tolerance
            )
            & (
                active_penalized_evidence_gain
                <= config.uamp_polish_coordinate_tolerance
            )
        )
        if config.uamp_polish_noise_coordinate_steps > 0:
            final_noise_certificate = _noise_evidence_certificate(
                y,
                mean,
                posterior_variance,
                final_prior,
                final_noise,
                final_active_indices,
                prior_floor,
                operator,
                config,
            )
            noise_evidence_gradient = (
                final_noise_certificate.gradient
            )
            noise_fixed_point_residual = (
                final_noise_certificate.projected_residual
            )
            if bool(
                (
                    noise_fixed_point_residual
                    > config.uamp_polish_noise_tolerance
                ).any()
            ):
                final_noise_columns = operator.fourier_columns(
                    final_active_indices
                ).to(dtype=y.dtype)
                final_noise_gram = (
                    final_noise_columns.mH @ final_noise_columns
                )
                (
                    _,
                    _,
                    _,
                    final_noise_candidate_evidence,
                    _,
                ) = _active_posterior_with_evidence(
                    y,
                    final_prior,
                    final_noise_certificate.candidate_variance,
                    operator,
                    final_active_indices,
                    columns=final_noise_columns,
                    gram=final_noise_gram,
                    tail_loading=torch.zeros_like(final_noise),
                    global_certificate=False,
                    config=config,
                )
                final_noise_candidate_penalized = (
                    final_noise_candidate_evidence
                    + support_log_odds_penalty
                    * final_active_indices.shape[1]
                )
                noise_evidence_gain = (
                    best_penalized_evidence
                    - final_noise_candidate_penalized
                ).clamp_min(0.0)
            else:
                noise_evidence_gain = torch.zeros_like(final_noise)
            noise_coordinate_stationary = (
                noise_fixed_point_residual
                <= config.uamp_polish_noise_tolerance
            )
            joint_map_coordinate_stationary = (
                type2_map_coordinate_stationary
                & noise_coordinate_stationary
            )
        # Deprecated compatibility alias.  The new name deliberately avoids
        # claiming finite-sample coverage after adaptive support selection.
        familywise_support_certified = familywise_no_discovery

    return (
        mean,
        posterior_variance,
        data_diagonal,
        final_prior,
        final_noise,
        best_evidence,
        torch.stack(accepted_evidence, dim=1),
        truncation_delta_bound,
        hard_pruning_accepted,
        (
            best_inactive_certificate.max_ratio
            if best_inactive_certificate is not None
            else None
        ),
        (
            best_inactive_certificate.best_index
            if best_inactive_certificate is not None
            else None
        ),
        (
            best_inactive_certificate.evidence_gain
            if best_inactive_certificate is not None
            else None
        ),
        (
            best_inactive_certificate.optimal_variance
            if best_inactive_certificate is not None
            else None
        ),
        support_repairs_accepted,
        support_growth_accepted,
        (
            final_active_certificate.max_gain
            if final_active_certificate is not None
            else None
        ),
        (
            final_active_certificate.best_index
            if final_active_certificate is not None
            else None
        ),
        (
            final_active_certificate.optimal_variance
            if final_active_certificate is not None
            else None
        ),
        (
            final_active_certificate.max_deletion_gain
            if final_active_certificate is not None
            else None
        ),
        coordinate_steps_accepted,
        coordinate_stationary,
        inactive_ratio_threshold,
        familywise_support_certified,
        final_active_indices,
        stationarity_ratio_threshold,
        familywise_ratio_threshold,
        familywise_no_discovery,
        active_model_selected,
        support_shrink_accepted,
        support_log_odds_penalty,
        best_penalized_evidence,
        torch.stack(accepted_penalized_evidence, dim=1),
        inactive_penalized_evidence_gain,
        active_penalized_evidence_gain,
        type2_map_coordinate_stationary,
        noise_coordinate_steps_accepted,
        noise_evidence_gradient,
        noise_fixed_point_residual,
        noise_evidence_gain,
        noise_coordinate_stationary,
        joint_map_coordinate_stationary,
        joint_coordinate_cycles,
    )


def _pcg(
    right_hand_sides: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    *,
    initial_solution: Tensor | None,
    tolerance: float,
    max_iter: int,
    check_interval: int,
    preconditioner_mode: str,
    preconditioner_rank: int,
    preconditioner_seed: int = 0,
    fixed_iterations: int = 0,
    sync_free_cholesky: bool = True,
    jitter_multiplier: float = 8.0,
    toeplitz_spectrum: Tensor | None,
) -> tuple[Tensor, int]:
    """Batched complex PCG with cached operators and periodic GPU sync."""
    if initial_solution is None:
        solution = torch.zeros_like(right_hand_sides)
        residual = right_hand_sides.clone()
    else:
        if initial_solution.shape != right_hand_sides.shape:
            raise ValueError("initial_solution must match right_hand_sides.")
        solution = initial_solution.clone()
        residual = right_hand_sides - operator.matvec(
            solution,
            prior_variance,
            noise_variance,
            toeplitz_spectrum=toeplitz_spectrum,
        )
    preconditioner = _make_pcg_preconditioner(
        prior_variance,
        noise_variance,
        operator,
        preconditioner_mode,
        preconditioner_rank,
        preconditioner_seed,
        sync_free_cholesky,
        jitter_multiplier,
    )
    preconditioned = _apply_pcg_preconditioner(
        residual, preconditioner, operator
    )
    direction = preconditioned.clone()
    rho = (residual.conj() * preconditioned).sum(dim=1).real
    eps = torch.finfo(operator.real_dtype).eps
    tiny = torch.finfo(operator.real_dtype).tiny
    initial_norm = right_hand_sides.norm(dim=1).clamp_min(
        eps
    )
    best_solution = solution.clone() if fixed_iterations > 0 else None
    best_relative_residual = (
        residual.norm(dim=1) / initial_norm
        if fixed_iterations > 0
        else None
    )
    active = residual.norm(dim=1) > tolerance * initial_norm
    direction = torch.where(active[:, None, :], direction, 0.0)

    iteration_limit = fixed_iterations if fixed_iterations > 0 else max_iter
    for iteration in range(1, iteration_limit + 1):
        q_direction = operator.matvec(
            direction,
            prior_variance,
            noise_variance,
            toeplitz_spectrum=toeplitz_spectrum,
        )
        curvature = (direction.conj() * q_direction).sum(dim=1).real
        valid_step = (
            active
            & torch.isfinite(curvature)
            & torch.isfinite(rho)
            & (curvature > tiny)
            & (rho > tiny)
        )
        alpha = torch.where(valid_step, rho / curvature.clamp_min(tiny), 0.0)
        solution = solution + alpha[:, None, :] * direction
        residual = residual - alpha[:, None, :] * q_direction
        active = valid_step
        if iteration % check_interval == 0 or iteration == iteration_limit:
            # Recursive CG residuals drift badly for ill-conditioned complex64
            # systems.  Recompute the true residual and restart the direction.
            residual = right_hand_sides - operator.matvec(
                solution,
                prior_variance,
                noise_variance,
                toeplitz_spectrum=toeplitz_spectrum,
            )
            relative_residual = residual.norm(dim=1) / initial_norm
            if fixed_iterations > 0:
                if best_solution is None or best_relative_residual is None:
                    raise RuntimeError("Static PCG checkpoint state was not initialized.")
                improved = relative_residual < best_relative_residual
                best_solution = torch.where(
                    improved[:, None, :], solution, best_solution
                )
                best_relative_residual = torch.minimum(
                    best_relative_residual, relative_residual
                )
            residual_active = torch.isfinite(relative_residual)
            residual_active = residual_active & (relative_residual > tolerance)
            # A static CUDA schedule must not revive a right-hand side that
            # has already met the stopping criterion at an earlier checkpoint.
            # The adaptive path exits when all lanes are inactive, so this
            # monotone mask also makes the two paths numerically comparable.
            active = (
                active & residual_active
                if fixed_iterations > 0
                else residual_active
            )
            restarted = _apply_pcg_preconditioner(
                residual, preconditioner, operator
            )
            restarted_rho = (residual.conj() * restarted).sum(dim=1).real
            active = (
                active
                & torch.isfinite(restarted_rho)
                & (restarted_rho > tiny)
            )
            direction = torch.where(active[:, None, :], restarted, 0.0)
            rho = torch.where(active, restarted_rho, rho)
            if fixed_iterations == 0 and not bool(active.any()):
                break
            continue
        new_preconditioned = _apply_pcg_preconditioner(
            residual, preconditioner, operator
        )
        new_rho = (residual.conj() * new_preconditioned).sum(dim=1).real
        valid_update = active & torch.isfinite(new_rho) & (new_rho > tiny)
        beta = torch.where(
            valid_update, new_rho / rho.clamp_min(tiny), 0.0
        )
        direction = new_preconditioned + beta[:, None, :] * direction
        direction = torch.where(valid_update[:, None, :], direction, 0.0)
        rho = torch.where(valid_update, new_rho, rho)
        active = valid_update
    return (
        best_solution if best_solution is not None else solution,
        iteration,
    )


def _make_pcg_preconditioner(
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    mode: str,
    rank: int,
    seed: int,
    sync_free_cholesky: bool = True,
    jitter_multiplier: float = 8.0,
) -> tuple[str, Any]:
    if mode == "none":
        return "none", None
    if mode == "circulant":
        return (
            "circulant",
            operator.circulant_preconditioner_spectrum(
                prior_variance, noise_variance
            ),
        )
    if mode == "nystrom":
        return (
            "nystrom",
            _make_nystrom_preconditioner(
                prior_variance,
                noise_variance,
                operator,
                rank,
                seed,
                sync_free_cholesky,
                jitter_multiplier,
            ),
        )
    return (
        "low_rank",
        _make_low_rank_preconditioner(
            prior_variance,
            noise_variance,
            operator,
            rank,
            sync_free_cholesky,
            jitter_multiplier,
        ),
    )


def _apply_pcg_preconditioner(
    value: Tensor,
    state: tuple[str, Any],
    operator: PrefixFourierOperator,
) -> Tensor:
    mode, payload = state
    if mode == "none":
        return value
    if mode == "circulant":
        return operator.apply_circulant_preconditioner(value, payload)
    if mode == "nystrom":
        return _apply_nystrom_preconditioner(value, payload)
    return _apply_low_rank_preconditioner(value, payload)


def _make_low_rank_preconditioner(
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    requested_rank: int,
    sync_free_cholesky: bool = True,
    jitter_multiplier: float = 8.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Approximate Q by isotropic background plus its strongest atoms."""
    rank = min(requested_rank, operator.n_samples, operator.n_grid)
    if rank == 0:
        background = noise_variance + prior_variance.sum(dim=1)
        empty = torch.empty(
            (prior_variance.shape[0], operator.n_samples, 0),
            device=operator.device,
            dtype=operator.complex_dtype,
        )
        return background, empty, empty

    values, indices = torch.topk(prior_variance, rank, dim=1)
    columns = operator.fourier_columns(indices)
    factors = columns * values.sqrt()[:, None, :]
    omitted = (prior_variance.sum(dim=1) - values.sum(dim=1)).clamp_min(0.0)
    background = noise_variance + omitted
    background = torch.maximum(
        background,
        torch.finfo(operator.real_dtype).eps * prior_variance.sum(dim=1),
    )
    gram = factors.mH @ factors
    small_identity = torch.eye(
        rank, device=operator.device, dtype=operator.complex_dtype
    )
    small = small_identity + gram / background[:, None, None]
    small = 0.5 * (small + small.mH)
    cholesky = _stable_cholesky_raw(
        small,
        small_identity,
        operator,
        sync_free_cholesky=sync_free_cholesky,
        jitter_multiplier=max(10.0, jitter_multiplier),
        fallback_multipliers=(10.0, 1e2, 1e3, 1e4, 1e5),
    )
    if cholesky is None:
        # A scalar preconditioner is slower but cannot destabilise PCG.
        scalar = noise_variance + prior_variance.sum(dim=1)
        empty = factors[:, :, :0]
        return scalar, empty, empty
    return background, factors, torch.cholesky_inverse(cholesky)


def _apply_low_rank_preconditioner(
    value: Tensor,
    state: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    background, factors, small_inverse = state
    base = value / background[:, None, None]
    if factors.shape[2] == 0:
        return base
    projected = factors.mH @ value
    correction_weights = (
        small_inverse @ (projected / background[:, None, None])
    )
    return base - (factors @ correction_weights) / background[:, None, None]


def _make_nystrom_preconditioner(
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    requested_rank: int,
    seed: int,
    sync_free_cholesky: bool = True,
    jitter_multiplier: float = 8.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Randomized Nyström approximation of H Gamma H^H.

    The residual trace is retained as isotropic loading, so the
    preconditioner captures dominant covariance eigendirections without
    underestimating the diffuse Fourier background.
    """
    rank = min(requested_rank, operator.n_samples)
    if rank == 0:
        background = noise_variance + prior_variance.sum(dim=1)
        empty_vectors = torch.empty(
            (prior_variance.shape[0], operator.n_samples, 0),
            device=operator.device,
            dtype=operator.complex_dtype,
        )
        empty_values = torch.empty(
            (prior_variance.shape[0], 0),
            device=operator.device,
            dtype=operator.real_dtype,
        )
        return background, empty_vectors, empty_values

    generator = torch.Generator(device=operator.device)
    generator.manual_seed(seed)
    informed_rank = min(rank, max(1, rank // 2))
    _, strongest_indices = torch.topk(
        prior_variance, informed_rank, dim=1
    )
    informed = operator.fourier_columns(strongest_indices)
    random_rank = rank - informed_rank
    if random_rank:
        real = torch.randn(
            prior_variance.shape[0],
            operator.n_samples,
            random_rank,
            device=operator.device,
            dtype=operator.real_dtype,
            generator=generator,
        )
        imaginary = torch.randn(
            prior_variance.shape[0],
            operator.n_samples,
            random_rank,
            device=operator.device,
            dtype=operator.real_dtype,
            generator=generator,
        )
        random_vectors = torch.complex(real, imaginary).to(
            operator.complex_dtype
        )
        omega = torch.cat((informed, random_vectors), dim=2)
    else:
        omega = informed
    omega = torch.linalg.qr(omega, mode="reduced").Q
    covariance_sample = operator.forward(
        prior_variance[:, :, None] * operator.adjoint(omega)
    )
    core = omega.mH @ covariance_sample
    core = 0.5 * (core + core.mH)
    identity = torch.eye(
        rank, device=operator.device, dtype=operator.complex_dtype
    )
    cholesky = _stable_cholesky_raw(
        core,
        identity,
        operator,
        sync_free_cholesky=sync_free_cholesky,
        jitter_multiplier=max(10.0, jitter_multiplier),
        fallback_multipliers=(10.0, 1e2, 1e3, 1e4, 1e5),
    )
    if cholesky is None:
        background = noise_variance + prior_variance.sum(dim=1)
        empty_values = torch.empty(
            (prior_variance.shape[0], 0),
            device=operator.device,
            dtype=operator.real_dtype,
        )
        return background, omega[:, :, :0], empty_values

    # If C=L L^H, then Y C^-1 Y^H=(Y L^-H)(Y L^-H)^H.
    factor = torch.linalg.solve_triangular(
        cholesky, covariance_sample.mH, upper=False
    ).mH
    eigenvectors, singular_values, _ = torch.linalg.svd(
        factor, full_matrices=False
    )
    eigenvalues = singular_values.square().real
    # Each diagonal of H Gamma H^H equals sum(gamma).  Preserve the trace
    # not represented by the randomized low-rank approximation.
    residual_average = (
        prior_variance.sum(dim=1)
        - eigenvalues.sum(dim=1) / operator.n_samples
    ).clamp_min(0.0)
    background = noise_variance + residual_average
    background = background.clamp_min(
        torch.finfo(operator.real_dtype).tiny
    )
    return background, eigenvectors, eigenvalues


def _apply_nystrom_preconditioner(
    value: Tensor,
    state: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    background, eigenvectors, eigenvalues = state
    if eigenvectors.shape[2] == 0:
        return value / background[:, None, None]
    projected = eigenvectors.mH @ value
    # Frangella--Tropp--Udell spectral flattening: dominant approximate
    # eigenvalues are mapped near lambda_rank + shift, while the unresolved
    # complement is left at one.  A global scalar is irrelevant to PCG.
    scale = eigenvalues[:, -1:] + background[:, None]
    weights = scale / (
        background[:, None] + eigenvalues
    )
    return value + eigenvectors @ (
        (weights - 1.0)[:, :, None] * projected
    )


def _make_probes(
    batch_size: int,
    n_grid: int,
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    probe_type: str = "rademacher",
    probe_shape: Sequence[int] | None = None,
) -> Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if probe_type == "rademacher":
        random_signs = 2 * torch.randint(
            0,
            2,
            (batch_size, n_grid, count),
            device=device,
            generator=generator,
        ) - 1
        return random_signs.to(dtype)

    # Randomized hierarchical probing: D H, where D is one Rademacher
    # diagonal and columns of the Sylvester Hadamard matrix follow Gray-code
    # order.  Every column is marginally Rademacher (hence unbiased), while
    # groups of 2^k columns cancel progressively more off-diagonal structure.
    if probe_shape is None:
        probe_shape = (n_grid,)
    probe_shape = tuple(int(v) for v in probe_shape)
    if prod(probe_shape) != n_grid:
        raise ValueError("probe_shape must have product n_grid.")
    axis_bits = [max(0, (length - 1).bit_length()) for length in probe_shape]
    embedding = 1 << sum(axis_bits)
    if count > embedding:
        raise ValueError("Hadamard probe count cannot exceed its embedding.")
    # A single long hierarchy has low variance when inverse entries decay
    # geometrically, but can be brittle for highly undersampled Fourier
    # systems.  Independent eight-vector hierarchies retain local cancellation
    # while restoring Monte-Carlo averaging.  A complete basis remains one
    # hierarchy and therefore gives the exact diagonal.
    block_size = (
        count
        if count == embedding
        else min(8, 1 << max(0, count.bit_length() - 1))
    )
    block_count = (count + block_size - 1) // block_size
    random_signs = 2 * torch.randint(
        0,
        2,
        (batch_size, n_grid, block_count),
        device=device,
        generator=generator,
    ) - 1
    axes = [
        torch.arange(length, device=device, dtype=torch.long)
        for length in probe_shape
    ]
    coordinates = [
        value.reshape(-1)
        for value in torch.meshgrid(*axes, indexing="ij")
    ]
    # Morton/interleaved-bit ordering makes the hierarchical Hadamard colors
    # respect 2-D sample-grid distance instead of flattened row-major distance.
    row_codes = torch.zeros(n_grid, device=device, dtype=torch.long)
    output_bit = 0
    for input_bit in range(max(axis_bits, default=0)):
        for coordinate, bit_count in zip(coordinates, axis_bits):
            if input_bit < bit_count:
                row_codes = torch.bitwise_or(
                    row_codes,
                    torch.bitwise_left_shift(
                        torch.bitwise_and(
                            torch.bitwise_right_shift(
                                coordinate, input_bit
                            ),
                            1,
                        ),
                        output_bit,
                    ),
                )
                output_bit += 1
    rows = row_codes[:, None]
    probe_numbers = torch.arange(count, device=device, dtype=torch.long)
    columns = torch.remainder(probe_numbers, block_size)
    columns = torch.bitwise_xor(columns, torch.bitwise_right_shift(columns, 1))
    bit_products = torch.bitwise_and(rows, columns[None, :])
    parity = torch.zeros_like(bit_products)
    shift = 0
    while (1 << shift) < embedding:
        parity = torch.bitwise_xor(
            parity,
            torch.bitwise_and(
                torch.bitwise_right_shift(bit_products, shift), 1
            ),
        )
        shift += 1
    hadamard = (1 - 2 * parity).to(dtype)
    block_indices = torch.div(
        probe_numbers, block_size, rounding_mode="floor"
    )
    return (
        random_signs.index_select(2, block_indices).to(dtype)
        * hadamard.unsqueeze(0)
    )


def _cg_schedule(
    iteration: int,
    config: SBLConfig,
    maximum_probes: int,
) -> tuple[int, float]:
    """Use inexpensive inexact solves early and full accuracy near convergence."""
    if not config.adaptive_cg:
        return maximum_probes, config.cg_tolerance
    initial_probes = (
        min(config.initial_hutchinson_probes, maximum_probes)
        if config.adaptive_probes
        else maximum_probes
    )
    if iteration <= 5:
        return (
            initial_probes,
            max(config.cg_initial_tolerance, config.cg_tolerance),
        )
    if iteration <= 15:
        middle_probes = (
            min(
                maximum_probes,
                max(initial_probes, min(24, maximum_probes)),
            )
            if config.adaptive_probes
            else maximum_probes
        )
        middle_tolerance = max(
            config.cg_tolerance,
            (config.cg_initial_tolerance * config.cg_tolerance) ** 0.5,
        )
        return middle_probes, middle_tolerance
    return maximum_probes, config.cg_tolerance


def _resize_warm_start(
    state: Tensor | None,
    rhs_count: int,
    batch_size: int,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    if state is None:
        return None
    if state.shape[2] == rhs_count:
        return state
    resized = torch.zeros(
        (batch_size, n_samples, rhs_count), device=device, dtype=dtype
    )
    shared = min(rhs_count, state.shape[2])
    resized[:, :, :shared] = state[:, :, :shared]
    return resized


def _resolve_device(
    requested: str | torch.device, input_tensor: Tensor
) -> torch.device:
    if str(requested) != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device
    if input_tensor.device.type != "cpu":
        return input_tensor.device
    return (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.complex128, torch.float64):
        return torch.complex128
    return torch.complex64
