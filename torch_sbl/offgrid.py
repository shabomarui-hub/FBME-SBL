from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import torch
from torch import Tensor


@dataclass(slots=True)
class OffGridResult:
    """Continuous Fourier parameters refined from an on-grid SBL estimate.

    Positions are expressed in grid-bin units.  For example, position 10.25
    on a length-128 1-D grid corresponds to normalized frequency 10.25 / 128.
    """

    initial_positions: Tensor
    positions: Tensor
    amplitudes: Tensor
    position_std: Tensor
    initial_residual_norm: Tensor
    residual_norm: Tensor
    iterations: Tensor
    converged: Tensor


def continuous_fourier_dictionary(
    positions: Tensor,
    grid_shape: Sequence[int],
    sample_shape: Sequence[int],
) -> Tensor:
    """Construct Fourier atoms at continuous 1-D/2-D grid positions.

    ``positions`` has shape ``[K, D]`` and the output has shape ``[M, K]``.
    The Fourier convention matches ``torch.fft.fftn`` and
    :class:`PrefixFourierOperator`.
    """
    grid_shape = tuple(int(v) for v in grid_shape)
    sample_shape = tuple(int(v) for v in sample_shape)
    if len(grid_shape) not in (1, 2):
        raise ValueError("Only 1-D and 2-D Fourier grids are supported.")
    if len(grid_shape) != len(sample_shape):
        raise ValueError("grid_shape and sample_shape must have equal rank.")
    if positions.ndim != 2 or positions.shape[1] != len(grid_shape):
        raise ValueError("positions must have shape [K, len(grid_shape)].")

    real_dtype = (
        torch.float64 if positions.dtype == torch.float64 else torch.float32
    )
    axes = [
        torch.arange(length, device=positions.device, dtype=real_dtype)
        for length in sample_shape
    ]
    coordinates = torch.stack(
        [
            value.reshape(-1)
            for value in torch.meshgrid(*axes, indexing="ij")
        ],
        dim=1,
    )
    grid_scale = torch.tensor(
        grid_shape, device=positions.device, dtype=real_dtype
    )
    phase = (coordinates / grid_scale) @ positions.to(real_dtype).mT
    complex_dtype = (
        torch.complex128 if real_dtype == torch.float64 else torch.complex64
    )
    return torch.exp((-2j * torch.pi * phase).to(complex_dtype))


def refine_offgrid_fourier(
    data: Tensor,
    coefficients: Tensor,
    grid_shape: Sequence[int],
    *,
    component_count: int | None = None,
    max_components: int = 64,
    relative_threshold: float = 0.1,
    minimum_separation: float = 1.5,
    max_iter: int = 15,
    tolerance: float = 1e-4,
    max_step: float = 0.35,
    damping: float = 1e-3,
    ridge: float = 1e-7,
) -> OffGridResult:
    """Refine on-grid SBL peaks using projected Gauss--Newton.

    Complex amplitudes are eliminated by regularized least squares at every
    nonlinear step (variable projection).  Only the real continuous
    coordinates are updated by Gauss--Newton, with Levenberg damping,
    per-coordinate trust bounds and monotone backtracking.
    """
    grid_shape = tuple(int(v) for v in grid_shape)
    if len(grid_shape) not in (1, 2):
        raise ValueError("grid_shape must be one- or two-dimensional.")
    if coefficients.ndim < len(grid_shape):
        raise ValueError("coefficients has fewer dimensions than grid_shape.")
    if tuple(coefficients.shape[-len(grid_shape) :]) != grid_shape:
        raise ValueError("The trailing coefficient shape must equal grid_shape.")
    if data.ndim < len(grid_shape):
        raise ValueError("data has fewer dimensions than grid_shape.")
    sample_shape = tuple(int(v) for v in data.shape[-len(grid_shape) :])
    if any(m > n for m, n in zip(sample_shape, grid_shape)):
        raise ValueError("Observed dimensions cannot exceed grid dimensions.")
    if component_count is not None and component_count < 1:
        raise ValueError("component_count must be positive.")
    if max_components < 1:
        raise ValueError("max_components must be positive.")
    if not 0.0 < relative_threshold < 1.0:
        raise ValueError("relative_threshold must be in (0, 1).")
    if minimum_separation < 0.0:
        raise ValueError("minimum_separation cannot be negative.")
    if max_iter < 1 or tolerance <= 0.0 or max_step <= 0.0:
        raise ValueError("Refinement iteration settings must be positive.")
    if damping <= 0.0 or ridge < 0.0:
        raise ValueError("damping must be positive and ridge nonnegative.")

    batch_shape = tuple(coefficients.shape[: -len(grid_shape)])
    if tuple(data.shape[: -len(grid_shape)]) != batch_shape:
        raise ValueError("data and coefficients must have the same batch shape.")
    batch_size = prod(batch_shape) if batch_shape else 1
    if batch_size > 1 and component_count is None:
        raise ValueError(
            "Batch off-grid refinement requires a fixed component_count."
        )

    complex_dtype = (
        torch.complex128
        if data.dtype in (torch.complex128, torch.float64)
        else torch.complex64
    )
    real_dtype = (
        torch.float64 if complex_dtype == torch.complex128 else torch.float32
    )
    device = data.device
    y = data.to(dtype=complex_dtype).reshape(batch_size, -1)
    x = coefficients.to(device=device).reshape(batch_size, -1)
    maximum = min(max_components, y.shape[1] - 1, x.shape[1])
    if component_count is not None:
        maximum = min(maximum, component_count)
    if maximum < 1:
        raise ValueError("At least two measurements are required.")

    all_positions: list[Tensor] = []
    all_initial_positions: list[Tensor] = []
    all_amplitudes: list[Tensor] = []
    all_standard_deviations: list[Tensor] = []
    initial_norms: list[Tensor] = []
    final_norms: list[Tensor] = []
    iteration_counts: list[int] = []
    convergence_flags: list[bool] = []

    for batch_index in range(batch_size):
        initial_positions = _select_peak_positions(
            x[batch_index].abs(),
            grid_shape,
            component_count=component_count,
            max_components=maximum,
            relative_threshold=relative_threshold,
            minimum_separation=minimum_separation,
            real_dtype=real_dtype,
        )
        if component_count is None and initial_positions.shape[0] > 1:
            initial_positions = _select_bic_model_order(
                y[batch_index],
                initial_positions,
                grid_shape,
                sample_shape,
            )
        (
            refined_positions,
            amplitudes,
            position_std,
            initial_norm,
            final_norm,
            iterations,
            converged,
        ) = _refine_single(
            y[batch_index],
            initial_positions,
            grid_shape,
            sample_shape,
            max_iter=max_iter,
            tolerance=tolerance,
            max_step=max_step,
            damping=damping,
            ridge=ridge,
        )
        all_initial_positions.append(initial_positions)
        all_positions.append(refined_positions)
        all_amplitudes.append(amplitudes)
        all_standard_deviations.append(position_std)
        initial_norms.append(initial_norm)
        final_norms.append(final_norm)
        iteration_counts.append(iterations)
        convergence_flags.append(converged)

    # Automatic component selection is only allowed for B=1, so all stacked
    # tensors necessarily share their component dimension.
    positions = torch.stack(all_positions)
    initial_positions = torch.stack(all_initial_positions)
    amplitudes = torch.stack(all_amplitudes)
    standard_deviations = torch.stack(all_standard_deviations)
    output_batch_shape = batch_shape if batch_shape else ()
    position_shape = (*output_batch_shape, positions.shape[1], len(grid_shape))
    amplitude_shape = (*output_batch_shape, amplitudes.shape[1])
    return OffGridResult(
        initial_positions=initial_positions.reshape(position_shape),
        positions=positions.reshape(position_shape),
        amplitudes=amplitudes.reshape(amplitude_shape),
        position_std=standard_deviations.reshape(position_shape),
        initial_residual_norm=torch.stack(initial_norms).reshape(
            output_batch_shape
        ),
        residual_norm=torch.stack(final_norms).reshape(output_batch_shape),
        iterations=torch.tensor(
            iteration_counts, device=device, dtype=torch.long
        ).reshape(output_batch_shape),
        converged=torch.tensor(
            convergence_flags, device=device, dtype=torch.bool
        ).reshape(output_batch_shape),
    )


def _select_peak_positions(
    magnitude: Tensor,
    grid_shape: tuple[int, ...],
    *,
    component_count: int | None,
    max_components: int,
    relative_threshold: float,
    minimum_separation: float,
    real_dtype: torch.dtype,
) -> Tensor:
    candidate_count = min(
        magnitude.numel(),
        max(64, 16 * max_components),
    )
    values, indices = torch.topk(magnitude, candidate_count)
    cutoff = relative_threshold * values[0]
    selected: list[Tensor] = []
    selected_indices: set[int] = set()
    grid = torch.tensor(grid_shape, device=magnitude.device, dtype=real_dtype)

    for candidate_offset in range(candidate_count):
        if (
            component_count is None
            and len(selected) > 0
            and bool(values[candidate_offset] < cutoff)
        ):
            break
        flat_index = int(indices[candidate_offset].item())
        coordinate = _flat_to_coordinate(
            flat_index, grid_shape, magnitude.device, real_dtype
        )
        if selected:
            existing = torch.stack(selected)
            difference = (existing - coordinate).abs()
            periodic_difference = torch.minimum(
                difference, grid[None, :] - difference
            )
            distance = periodic_difference.square().sum(dim=1).sqrt()
            if bool((distance < minimum_separation).any()):
                continue
        selected.append(coordinate)
        selected_indices.add(flat_index)
        if len(selected) >= (
            component_count if component_count is not None else max_components
        ):
            break

    target_count = component_count if component_count is not None else None
    if target_count is not None and len(selected) < target_count:
        # Extremely close peaks can defeat nonmaximum suppression.  Preserve
        # the requested model order by filling from the remaining strongest
        # distinct grid points.
        for flat_tensor in indices:
            flat_index = int(flat_tensor.item())
            if flat_index in selected_indices:
                continue
            selected.append(
                _flat_to_coordinate(
                    flat_index, grid_shape, magnitude.device, real_dtype
                )
            )
            selected_indices.add(flat_index)
            if len(selected) == target_count:
                break
    if not selected:
        selected.append(
            _flat_to_coordinate(
                int(indices[0].item()),
                grid_shape,
                magnitude.device,
                real_dtype,
            )
        )
    return torch.stack(selected)


def _flat_to_coordinate(
    flat_index: int,
    grid_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if len(grid_shape) == 1:
        values = (flat_index,)
    else:
        values = (
            flat_index // grid_shape[1],
            flat_index % grid_shape[1],
        )
    return torch.tensor(values, device=device, dtype=dtype)


def _select_bic_model_order(
    y: Tensor,
    candidate_positions: Tensor,
    grid_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
) -> Tensor:
    """Choose the number of ordered peaks with a complex-data BIC score."""
    dictionary = continuous_fourier_dictionary(
        candidate_positions, grid_shape, sample_shape
    ).to(y.dtype)
    orthogonal = torch.linalg.qr(dictionary, mode="reduced").Q
    projected_energy = (orthogonal.mH @ y).abs().square()
    residual_energy = (
        y.abs().square().sum() - projected_energy.cumsum(dim=0)
    ).real.clamp_min(torch.finfo(y.real.dtype).eps)
    count = torch.arange(
        1,
        candidate_positions.shape[0] + 1,
        device=y.device,
        dtype=y.real.dtype,
    )
    # Each component has one complex amplitude (two real parameters) and D
    # continuous position coordinates.
    parameter_count = (2 + len(grid_shape)) * count
    sample_count = float(y.numel())
    bic = (
        sample_count * torch.log(residual_energy / sample_count)
        + parameter_count * torch.log(
            torch.tensor(sample_count, device=y.device, dtype=y.real.dtype)
        )
        # Extended-BIC correction for selecting support locations from a
        # high-dimensional grid, rather than fitting a fixed set of regressors.
        + 2.0
        * count
        * torch.log(
            torch.tensor(
                float(prod(grid_shape)),
                device=y.device,
                dtype=y.real.dtype,
            )
        )
    )
    selected_count = int(torch.argmin(bic).item()) + 1
    return candidate_positions[:selected_count]


def _refine_single(
    y: Tensor,
    initial_positions: Tensor,
    grid_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
    *,
    max_iter: int,
    tolerance: float,
    max_step: float,
    damping: float,
    ridge: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, bool]:
    positions = initial_positions.clone()
    grid = torch.tensor(
        grid_shape, device=y.device, dtype=positions.dtype
    )
    current_damping = damping
    dictionary = continuous_fourier_dictionary(
        positions, grid_shape, sample_shape
    ).to(y.dtype)
    amplitudes, cholesky = _fit_amplitudes(dictionary, y, ridge)
    residual = y - dictionary @ amplitudes
    initial_norm = residual.norm()
    objective = residual.abs().square().sum().real
    converged = False

    for iteration in range(1, max_iter + 1):
        jacobian = _projected_position_jacobian(
            dictionary,
            amplitudes,
            residual,
            cholesky,
            grid_shape,
            sample_shape,
        )
        normal = (jacobian.mH @ jacobian).real
        right_hand_side = (jacobian.mH @ residual).real
        diagonal = normal.diagonal().clamp_min(
            torch.finfo(normal.dtype).eps
        )
        regularized = normal + current_damping * torch.diag(diagonal)
        step = torch.linalg.solve(regularized, right_hand_side)
        step = step.reshape(positions.shape).clamp(-max_step, max_step)
        if not bool(torch.isfinite(step).all()):
            break

        accepted = False
        accepted_step = step
        previous_objective = objective
        for line_scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial_positions = torch.remainder(
                positions + line_scale * step, grid
            )
            trial_dictionary = continuous_fourier_dictionary(
                trial_positions, grid_shape, sample_shape
            ).to(y.dtype)
            trial_amplitudes, trial_cholesky = _fit_amplitudes(
                trial_dictionary, y, ridge
            )
            trial_residual = y - trial_dictionary @ trial_amplitudes
            trial_objective = trial_residual.abs().square().sum().real
            if bool(trial_objective < objective):
                positions = trial_positions
                dictionary = trial_dictionary
                amplitudes = trial_amplitudes
                cholesky = trial_cholesky
                residual = trial_residual
                objective = trial_objective
                accepted_step = line_scale * step
                accepted = True
                current_damping = max(current_damping * 0.5, 1e-8)
                break

        if not accepted:
            current_damping = min(current_damping * 10.0, 1e8)
            if current_damping >= 1e8:
                converged = True
                break
            continue
        relative_decrease = (
            (previous_objective - objective)
            / previous_objective.clamp_min(
                torch.finfo(objective.dtype).eps
            )
        )
        if (
            float(accepted_step.abs().max()) <= tolerance
            or float(relative_decrease) <= tolerance**2
        ):
            converged = True
            break

    final_norm = residual.norm()
    position_std = _position_standard_deviation(
        dictionary,
        amplitudes,
        residual,
        cholesky,
        grid_shape,
        sample_shape,
    )
    return (
        positions,
        amplitudes,
        position_std,
        initial_norm,
        final_norm,
        iteration,
        converged,
    )


def _fit_amplitudes(
    dictionary: Tensor,
    y: Tensor,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    gram = dictionary.mH @ dictionary
    diagonal_mean = gram.diagonal().real.mean().clamp_min(
        torch.finfo(gram.real.dtype).tiny
    )
    regularization = max(ridge, torch.finfo(gram.real.dtype).eps)
    gram = gram + regularization * diagonal_mean * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    gram = 0.5 * (gram + gram.mH)
    cholesky = torch.linalg.cholesky(gram)
    amplitudes = torch.cholesky_solve(
        (dictionary.mH @ y)[:, None], cholesky
    ).squeeze(-1)
    return amplitudes, cholesky


def _projected_position_jacobian(
    dictionary: Tensor,
    amplitudes: Tensor,
    residual: Tensor,
    amplitude_cholesky: Tensor,
    grid_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
) -> Tensor:
    del residual  # Reserved for a future full Golub--Pereyra correction.
    real_dtype = dictionary.real.dtype
    axes = [
        torch.arange(length, device=dictionary.device, dtype=real_dtype)
        for length in sample_shape
    ]
    coordinates = torch.stack(
        [
            value.reshape(-1)
            for value in torch.meshgrid(*axes, indexing="ij")
        ],
        dim=1,
    )
    grid = torch.tensor(
        grid_shape, device=dictionary.device, dtype=real_dtype
    )
    derivative_factor = (
        -2j * torch.pi * coordinates / grid[None, :]
    ).to(dictionary.dtype)
    derivatives = (
        dictionary[:, :, None]
        * derivative_factor[:, None, :]
        * amplitudes[None, :, None]
    )
    jacobian = derivatives.reshape(dictionary.shape[0], -1)
    projection_coefficients = torch.cholesky_solve(
        dictionary.mH @ jacobian, amplitude_cholesky
    )
    return jacobian - dictionary @ projection_coefficients


def _position_standard_deviation(
    dictionary: Tensor,
    amplitudes: Tensor,
    residual: Tensor,
    amplitude_cholesky: Tensor,
    grid_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
) -> Tensor:
    jacobian = _projected_position_jacobian(
        dictionary,
        amplitudes,
        residual,
        amplitude_cholesky,
        grid_shape,
        sample_shape,
    )
    fisher = (jacobian.mH @ jacobian).real
    dof = max(
        dictionary.shape[0] - dictionary.shape[1],
        1,
    )
    noise_variance = residual.abs().square().sum().real / dof
    covariance = 0.5 * noise_variance * torch.linalg.pinv(
        fisher, hermitian=True
    )
    return covariance.diagonal().clamp_min(0.0).sqrt().reshape(
        amplitudes.shape[0], len(grid_shape)
    )
