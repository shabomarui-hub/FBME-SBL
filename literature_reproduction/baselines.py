"""Independent, equation-level reproductions of accelerated SBL baselines."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import torch
from torch import Tensor

from torch_sbl import SBLConfig, SBLResult, sbl_2d
from torch_sbl.operators import PrefixFourierOperator


@dataclass(slots=True)
class CoFEMResult:
    """Output of the complex partial-Fourier CoFEM adaptation."""

    x: Tensor
    posterior_variance: Tensor
    prior_precision: Tensor
    outer_iterations: int
    e_steps: int
    cg_iterations: tuple[int, ...]
    cg_converged: bool
    probes: int
    noise_variance: Tensor
    arithmetic_dtype: str


@dataclass(slots=True)
class PaperUAMPResult:
    """Output of the equation-level UAMP-SBL Algorithm 2 reproduction."""

    x: Tensor
    posterior_variance: Tensor
    prior_precision: Tensor
    noise_variance: Tensor
    iterations: int
    epsilon: Tensor
    arithmetic_dtype: str


def _complex_rademacher(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tensor:
    """Generate reproducible probes satisfying E[z_i* z_j] = delta_ij."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    signs = torch.randint(
        0,
        2,
        shape,
        device=device,
        generator=generator,
        dtype=torch.int8,
    )
    return (2 * signs.to(torch.float32) - 1).to(dtype=dtype)


def _parallel_pcg(
    operator: PrefixFourierOperator,
    right_hand_sides: Tensor,
    diagonal: Tensor,
    *,
    column_energy: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[Tensor, int, bool]:
    """Author-style parallel PCG, extended to Hermitian complex systems.

    It solves ``(A^H A + diag(diagonal)) X = B``.  Each right-hand side
    has its own CG step length, while the stopping rule follows the author
    implementation's joint Frobenius residual.
    """
    if right_hand_sides.ndim != 2:
        raise ValueError("right_hand_sides must have shape [R, N].")
    if diagonal.ndim != 1:
        raise ValueError("diagonal must have shape [N].")

    def system(value: Tensor) -> Tensor:
        return (
            operator.adjoint_flat(operator.forward_flat(value))
            + diagonal[None, :] * value
        )

    preconditioner = (column_energy + diagonal).reciprocal()
    solution = torch.zeros_like(right_hand_sides)
    residual = right_hand_sides.clone()
    preconditioned = residual * preconditioner[None, :]
    direction = preconditioned.clone()
    residual_preconditioned = (
        residual.conj() * preconditioned
    ).sum(dim=-1, keepdim=True).real
    rhs_energy = right_hand_sides.abs().square().sum().clamp_min(
        torch.finfo(right_hand_sides.real.dtype).tiny
    )

    relative_error = residual.abs().square().sum() / rhs_energy
    if bool(relative_error <= tolerance):
        return solution, 0, True

    for iteration in range(1, max_iterations + 1):
        applied = system(direction)
        curvature = (
            direction.conj() * applied
        ).sum(dim=-1, keepdim=True).real.clamp_min(
            torch.finfo(right_hand_sides.real.dtype).tiny
        )
        step = residual_preconditioned / curvature
        solution = solution + step * direction
        residual = residual - step * applied
        relative_error = residual.abs().square().sum() / rhs_energy
        if bool(relative_error <= tolerance):
            return solution, iteration, True

        next_preconditioned = residual * preconditioner[None, :]
        next_inner = (
            residual.conj() * next_preconditioned
        ).sum(dim=-1, keepdim=True).real
        coefficient = next_inner / residual_preconditioned.clamp_min(
            torch.finfo(right_hand_sides.real.dtype).tiny
        )
        direction = next_preconditioned + coefficient * direction
        preconditioned = next_preconditioned
        residual_preconditioned = next_inner

    return solution, max_iterations, False


def complex_fourier_cofem(
    data: Tensor,
    grid_shape: Sequence[int],
    *,
    noise_variance: float | Tensor,
    outer_iterations: int = 30,
    probes: int = 20,
    cg_tolerance: float = 1e-5,
    max_cg_iterations: int = 400,
    initial_prior_precision: float = 1.0,
    seed: int = 0,
) -> CoFEMResult:
    """Run CoFEM on the same complex partial-Fourier ISAR model.

    This is an independent complex adaptation of Lin et al. (TSP 2022).
    The author repository is real-valued and exposes transpose rather than
    Hermitian-adjoint operators.  The adaptation changes only the required
    complex algebra: conjugate inner products, ``A^H``, ``|mu|^2``, and the
    real part of the Hutchinson diagonal estimate.

    As in the author implementation, noise precision is fixed and supplied
    by the caller.  One initial E-step is followed by ``outer_iterations``
    EM updates, for a total of ``outer_iterations + 1`` PCG solves.
    """
    if not data.is_cuda:
        raise ValueError("The same-device CoFEM benchmark requires CUDA data.")
    if data.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("CoFEM Fourier adaptation requires complex data.")
    if outer_iterations < 0 or probes < 1 or max_cg_iterations < 1:
        raise ValueError("Invalid CoFEM iteration/probe budget.")

    grid_shape = tuple(int(value) for value in grid_shape)
    sample_shape = tuple(int(value) for value in data.shape[-len(grid_shape) :])
    if data.ndim != len(grid_shape):
        raise ValueError("The reproduction currently supports one SMV scene.")
    operator = PrefixFourierOperator(
        grid_shape,
        sample_shape,
        device=data.device,
        complex_dtype=data.dtype,
    )
    y = data.reshape(1, operator.n_samples)
    real_dtype = data.real.dtype
    variance = torch.as_tensor(
        noise_variance,
        device=data.device,
        dtype=real_dtype,
    ).reshape(())
    variance = variance.clamp_min(torch.finfo(real_dtype).tiny)
    beta = variance.reciprocal()
    prior_precision = torch.full(
        (operator.n_grid,),
        initial_prior_precision,
        device=data.device,
        dtype=real_dtype,
    )
    adjoint_data = operator.adjoint_flat(y).squeeze(0)
    cg_iterations: list[int] = []
    converged = True
    expectation_index = 0

    def expectation_step() -> tuple[Tensor, Tensor]:
        nonlocal converged, expectation_index
        # The author implementation draws a new Rademacher matrix in every
        # E-step.  Incrementing the recorded seed preserves that behavior while
        # keeping the benchmark exactly reproducible.
        random_probes = _complex_rademacher(
            (probes, operator.n_grid),
            device=data.device,
            dtype=data.dtype,
            seed=seed + expectation_index,
        )
        expectation_index += 1
        right_hand_sides = torch.cat(
            (adjoint_data[None, :], random_probes),
            dim=0,
        )
        solutions, count, solved = _parallel_pcg(
            operator,
            right_hand_sides,
            prior_precision / beta,
            column_energy=float(operator.n_samples),
            tolerance=cg_tolerance,
            max_iterations=max_cg_iterations,
        )
        cg_iterations.append(count)
        converged = converged and solved
        mean = solutions[0]
        covariance_diagonal = (
            (random_probes.conj() * solutions[1:])
            .mean(dim=0)
            .real
            / beta
        ).clamp_min(0.0)
        return mean, covariance_diagonal

    mean, posterior_variance = expectation_step()
    tiny = torch.finfo(real_dtype).tiny
    for _ in range(outer_iterations):
        prior_precision = (
            mean.abs().square() + posterior_variance
        ).clamp_min(tiny).reciprocal()
        mean, posterior_variance = expectation_step()

    return CoFEMResult(
        x=mean.reshape(grid_shape),
        posterior_variance=posterior_variance.reshape(grid_shape),
        prior_precision=prior_precision.reshape(grid_shape),
        outer_iterations=outer_iterations,
        e_steps=outer_iterations + 1,
        cg_iterations=tuple(cg_iterations),
        cg_converged=converged,
        probes=probes,
        noise_variance=variance,
        arithmetic_dtype=str(data.dtype).removeprefix("torch."),
    )


def dense_em_sbl(
    data: Tensor,
    grid_shape: Sequence[int],
    *,
    iterations: int = 30,
) -> SBLResult:
    """Classical exact EM-SBL through the observation-domain covariance."""
    config = SBLConfig(
        mode="dense",
        max_iter=iterations,
        min_iter=iterations,
        tol=0.0,
        hyper_shape=1e-6,
        hyper_rate=1e-6,
        noise_shape=1e-6,
        noise_rate=1e-6,
        initialization="uniform",
        damping=0.0,
        noise_update="evidence",
        active_set=False,
        device=data.device,
        return_history=False,
        eager_change_tracking=False,
    )
    return sbl_2d(data, grid_shape, config=config)


def paper_uamp_sbl(
    data: Tensor,
    grid_shape: Sequence[int],
    *,
    iterations: int = 30,
) -> PaperUAMPResult:
    """Reproduce Algorithm 2 of Luo et al., IEEE TSP 2021.

    The sensing matrix consists of leading rows of an unnormalised DFT.  Its
    rows are orthogonal, so its unitary left transform is the identity and
    every entry of ``lambda = Lambda Lambda^H 1`` equals ``N``.  Therefore no
    online SVD is needed for this structured case.  The loop follows paper
    Lines 1--13, including its empirical Gamma-shape update.
    """
    if not data.is_cuda:
        raise ValueError("The same-device UAMP-SBL benchmark requires CUDA data.")
    if data.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("UAMP-SBL requires complex data.")
    if iterations < 1:
        raise ValueError("iterations must be positive.")

    grid_shape = tuple(int(value) for value in grid_shape)
    sample_shape = tuple(int(value) for value in data.shape[-len(grid_shape) :])
    if data.ndim != len(grid_shape):
        raise ValueError("The reproduction currently supports one SMV scene.")
    operator = PrefixFourierOperator(
        grid_shape,
        sample_shape,
        device=data.device,
        complex_dtype=data.dtype,
    )
    y = data.reshape(1, operator.n_samples)
    real_dtype = data.real.dtype
    tiny = torch.finfo(real_dtype).tiny

    # Algorithm 2 initialization.
    tau_x = torch.ones((), device=data.device, dtype=real_dtype)
    mean = torch.zeros(
        (1, operator.n_grid), device=data.device, dtype=data.dtype
    )
    epsilon = torch.full((), 1e-3, device=data.device, dtype=real_dtype)
    precision = torch.ones(
        (1, operator.n_grid), device=data.device, dtype=real_dtype
    )
    beta = torch.ones((), device=data.device, dtype=real_dtype)
    s = torch.zeros(
        (1, operator.n_samples), device=data.device, dtype=data.dtype
    )
    lambda_value = torch.as_tensor(
        float(operator.n_grid), device=data.device, dtype=real_dtype
    )

    for _ in range(iterations):
        # Lines 1--7: UAMP output update.
        tau_p = tau_x * lambda_value
        p = operator.forward_flat(mean) - tau_p * s
        denominator = 1.0 + beta * tau_p
        variance_h = tau_p / denominator
        h = (beta * tau_p * y + p) / denominator
        residual_energy = (y - h).abs().square().sum().real
        beta = (
            operator.n_samples
            / (residual_energy + operator.n_samples * variance_h)
        ).clamp_min(tiny)
        tau_s = (tau_p + beta.reciprocal()).reciprocal()
        s = tau_s * (y - p)

        # Lines 8--11: scalar-step UAMP input update.
        inverse_tau_q = (
            operator.n_samples * lambda_value * tau_s / operator.n_grid
        )
        tau_q = inverse_tau_q.clamp_min(tiny).reciprocal()
        q = mean + tau_q * operator.adjoint_flat(s)
        shrink = (1.0 + tau_q * precision).reciprocal()
        tau_x = (tau_q * shrink.mean()).clamp_min(tiny)
        mean = q * shrink

        # Lines 12--13: variational precision and adaptive Gamma shape.
        second_moment = mean.abs().square() + tau_x
        precision = (
            (2.0 * epsilon + 1.0) / second_moment.clamp_min(tiny)
        ).clamp_min(tiny)
        epsilon_argument = (
            torch.log(precision.mean())
            - torch.log(precision).mean()
        ).clamp_min(0.0)
        epsilon = 0.5 * torch.sqrt(epsilon_argument)

    posterior_variance = tau_x.expand(operator.n_grid).reshape(grid_shape)
    return PaperUAMPResult(
        x=mean.reshape(grid_shape),
        posterior_variance=posterior_variance,
        prior_precision=precision.reshape(grid_shape),
        noise_variance=beta.reciprocal(),
        iterations=iterations,
        epsilon=epsilon,
        arithmetic_dtype=str(data.dtype).removeprefix("torch."),
    )


def uamp_sbl(
    data: Tensor,
    grid_shape: Sequence[int],
    *,
    iterations: int = 30,
    backend: str = "torch",
    damping: float = 0.1,
) -> SBLResult:
    """Run the engineering UAMP core without active-set polish.

    This helper is intentionally not labelled as the equation-level literature
    reproduction; use :func:`paper_uamp_sbl` for Luo et al. Algorithm 2.
    """
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or 'triton'.")
    config = SBLConfig.fast_uamp(
        max_iter=iterations,
        min_iter=iterations,
        uamp_min_iter=iterations,
        uamp_auto_iterations=False,
        uamp_damping=damping,
        uamp_adaptive_damping=False,
        uamp_polish_size=0,
        uamp_polish_compute_truncation_diagnostic=False,
        uamp_backend=backend,
        device=data.device,
    )
    return sbl_2d(data, grid_shape, config=config)
