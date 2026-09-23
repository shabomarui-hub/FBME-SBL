from __future__ import annotations

from dataclasses import replace
from math import log, prod
from typing import Any, Sequence

import torch
from torch import Tensor

from .operators import PrefixFourierOperator
from .solver import (
    SBLConfig,
    SBLResult,
    _complex_dtype,
    _make_probes,
    _pcg,
    _resolve_device,
)


def autofocus_sbl_2d(
    data: Any,
    grid_shape: Sequence[int],
    *,
    config: SBLConfig,
    observation_mask: Any | None = None,
) -> SBLResult:
    """Evidence-gated minimum-entropy autofocus followed by 2-D SBL.

    A low-dimensional smooth phase candidate is estimated only through a
    differentiable zero-filled Fourier backprojection.  SBL is then evaluated
    for both the original and corrected data.  The candidate is accepted only
    if its Type-II negative log evidence improves after a BIC-style parameter
    penalty, preventing minimum-entropy over-focusing on already clean data.
    """
    config.validate()
    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 2:
        raise ValueError("Autofocus requires a 2-D reconstruction grid.")
    input_tensor = torch.as_tensor(data)
    if input_tensor.ndim != 2:
        raise ValueError(
            "Autofocus currently requires one 2-D observation array."
        )
    if config.learn_snapshot_covariance:
        raise ValueError(
            "Autofocus and snapshot-covariance learning cannot yet be "
            "enabled in the same call."
        )
    if config.joint_sparsity:
        raise ValueError(
            "Autofocus currently supports one SMV reconstruction, not MMV."
        )
    if config.offgrid_refinement:
        raise ValueError(
            "Run off-grid refinement after autofocus in a separate call."
        )

    from .solver import sbl_2d

    solver_config = replace(config, phase_error_correction=False)
    observed = input_tensor.to(
        device=_resolve_device(config.device, input_tensor),
        dtype=_complex_dtype(input_tensor.dtype),
    )
    sample_shape = tuple(int(value) for value in observed.shape)
    if any(m > n for m, n in zip(sample_shape, shape)):
        raise ValueError("Observed dimensions cannot exceed grid dimensions.")

    mask = None
    if observation_mask is not None:
        mask = torch.as_tensor(
            observation_mask, device=observed.device, dtype=torch.bool
        )
        if tuple(mask.shape) != sample_shape:
            raise ValueError(
                "observation_mask must match the 2-D observation shape."
            )
        if not bool(mask.any()):
            raise ValueError("observation_mask must retain at least one sample.")
        observed = torch.where(mask, observed, torch.zeros_like(observed))

    candidate_phase, candidate_migration = _minimum_entropy_motion(
        observed,
        shape,
        config=config,
        observation_mask=mask,
    )
    corrected_data = _apply_motion_correction(
        observed,
        candidate_phase,
        candidate_migration,
        config.phase_error_axis,
        mask,
    )
    batched = sbl_2d(
        torch.stack((observed, corrected_data), dim=0),
        shape,
        config=solver_config,
        observation_mask=mask,
    )
    baseline = _slice_batch_result(batched, 0)
    corrected = _slice_batch_result(batched, 1)

    baseline_evidence = _negative_log_evidence(
        observed,
        baseline,
        shape,
        solver_config,
        observation_mask=mask,
    )
    corrected_evidence = _negative_log_evidence(
        corrected_data,
        corrected,
        shape,
        solver_config,
        observation_mask=mask,
    )
    selected = corrected
    selected_evidence = corrected_evidence
    selected_phase = candidate_phase
    selected_migration = candidate_migration
    if (
        config.motion_refinement_steps > 0
        and (
            config.range_dependent_phase
            or config.range_migration_correction
        )
    ):
        complete_operator = PrefixFourierOperator(
            shape,
            sample_shape,
            device=observed.device,
            complex_dtype=observed.dtype,
        )
        predicted = complete_operator.forward(
            corrected.x.reshape(1, -1, 1)
        ).reshape(sample_shape)
        refined_phase, refined_migration = _refine_motion_from_prediction(
            observed,
            predicted,
            candidate_phase,
            candidate_migration,
            config=config,
            observation_mask=mask,
        )
        refined_data = _apply_motion_correction(
            observed,
            refined_phase,
            refined_migration,
            config.phase_error_axis,
            mask,
        )
        refined = sbl_2d(
            refined_data,
            shape,
            config=solver_config,
            observation_mask=mask,
        )
        refined_evidence = _negative_log_evidence(
            refined_data,
            refined,
            shape,
            solver_config,
            observation_mask=mask,
        )
        if refined_evidence < selected_evidence:
            selected = refined
            selected_evidence = refined_evidence
            selected_phase = refined_phase
            selected_migration = refined_migration
    aperture_parameter_count = min(
        config.phase_basis_rank,
        sample_shape[config.phase_error_axis] - 1,
    )
    range_parameter_count = (
        1
        + min(
            config.phase_range_basis_rank,
            sample_shape[1 - config.phase_error_axis] - 1,
        )
        if config.range_dependent_phase
        else 1
    )
    migration_parameter_count = (
        min(
            config.migration_basis_rank,
            sample_shape[config.phase_error_axis] - 1,
        )
        if config.range_migration_correction
        else 0
    )
    parameter_count = (
        aperture_parameter_count * range_parameter_count
        + migration_parameter_count
    )
    observed_count = (
        int(mask.sum().item()) if mask is not None else prod(sample_shape)
    )
    complexity_penalty = (
        config.phase_evidence_penalty
        * parameter_count
        * log(observed_count)
    )
    evidence_gain = (
        baseline_evidence - selected_evidence - complexity_penalty
    )
    acceptance_margin = (
        config.phase_min_evidence_gain_per_parameter
        * parameter_count
        * prod(sample_shape)
        / observed_count
    )
    accepted = evidence_gain > acceptance_margin
    result = selected if accepted else baseline
    result.phase_candidate = selected_phase
    result.phase_error = (
        selected_phase
        if accepted
        else torch.zeros_like(selected_phase)
    )
    result.phase_correction_accepted = accepted
    result.phase_evidence_gain = float(evidence_gain)
    result.range_migration_candidate = selected_migration
    result.range_migration = (
        selected_migration
        if accepted
        else torch.zeros_like(selected_migration)
    )
    return result


def _slice_batch_result(result: SBLResult, index: int) -> SBLResult:
    coupling = result.learned_spatial_coupling
    return SBLResult(
        x=result.x[index],
        posterior_variance=result.posterior_variance[index],
        precision=result.precision[index],
        noise_variance=result.noise_variance[index],
        effective_dof=result.effective_dof[index],
        iterations=result.iterations,
        converged=result.converged,
        relative_change=result.relative_change,
        mode=result.mode,
        active_set_size=result.active_set_size,
        offgrid=None,
        learned_spatial_coupling=(
            coupling[index] if coupling is not None else None
        ),
        history=result.history,
    )


def _minimum_entropy_motion(
    observed: Tensor,
    grid_shape: tuple[int, int],
    *,
    config: SBLConfig,
    observation_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Estimate a smooth range-dependent phase field and migration trajectory."""
    aperture_axis = config.phase_error_axis
    range_axis = 1 - aperture_axis
    aperture_length = observed.shape[aperture_axis]
    range_length = observed.shape[range_axis]
    aperture_rank = min(config.phase_basis_rank, aperture_length - 1)
    migration_rank = min(config.migration_basis_rank, aperture_length - 1)
    if (
        not config.range_dependent_phase
        and not config.range_migration_correction
        and observation_mask is None
    ):
        phase = _minimum_entropy_phase(
            observed,
            grid_shape,
            axis=aperture_axis,
            rank=aperture_rank,
            steps=config.phase_optimization_steps,
            learning_rate=config.phase_learning_rate,
            smoothness=config.phase_smoothness,
        )
        return phase, torch.zeros_like(phase)

    real_dtype = observed.real.dtype
    aperture_basis = _dct_basis(
        aperture_length,
        aperture_rank,
        observed.device,
        real_dtype,
        include_constant=False,
    )
    range_basis = _dct_basis(
        range_length,
        (
            min(config.phase_range_basis_rank, range_length - 1)
            if config.range_dependent_phase
            else 0
        ),
        observed.device,
        real_dtype,
        include_constant=True,
    )
    phase_coefficients = torch.zeros(
        (range_basis.shape[1], aperture_basis.shape[1]),
        device=observed.device,
        dtype=real_dtype,
        requires_grad=True,
    )
    parameters: list[Tensor] = [phase_coefficients]
    migration_basis = _dct_basis(
        aperture_length,
        migration_rank,
        observed.device,
        real_dtype,
        include_constant=False,
    )
    migration_coefficients = torch.zeros(
        migration_rank,
        device=observed.device,
        dtype=real_dtype,
        requires_grad=config.range_migration_correction,
    )
    if config.range_migration_correction:
        parameters.append(migration_coefficients)
    optimizer = torch.optim.Adam(parameters, lr=config.phase_learning_rate)
    crop = tuple(slice(0, value) for value in observed.shape)
    tiny = torch.finfo(real_dtype).tiny

    with torch.enable_grad():
        for _ in range(config.phase_optimization_steps):
            optimizer.zero_grad(set_to_none=True)
            phase_canonical = (
                range_basis @ phase_coefficients @ aperture_basis.mT
            )
            phase = (
                phase_canonical
                if range_axis == 0
                else phase_canonical.mT
            )
            migration = (
                config.migration_max_shift
                * torch.tanh(migration_basis @ migration_coefficients)
                if config.range_migration_correction
                else torch.zeros(
                    aperture_length,
                    device=observed.device,
                    dtype=real_dtype,
                )
            )
            corrected = _apply_motion_correction(
                observed,
                phase,
                migration,
                aperture_axis,
                observation_mask,
            )
            padded = torch.zeros(
                grid_shape,
                device=observed.device,
                dtype=observed.dtype,
            )
            padded[crop] = corrected
            backprojection = torch.fft.ifft2(padded) * prod(grid_shape)
            power = backprojection.abs().square()
            probability = power / power.sum().clamp_min(tiny)
            entropy = -(
                probability * probability.clamp_min(tiny).log()
            ).sum()
            phase_penalty = _curvature_energy(phase)
            migration_penalty = _curvature_energy(migration)
            migration_amplitude = (
                migration.square().mean()
                / config.migration_max_shift**2
            )
            objective = (
                entropy
                + config.phase_smoothness * phase_penalty
                + config.migration_smoothness * migration_penalty
                + config.motion_phase_amplitude_penalty
                * phase.square().mean()
                + config.motion_migration_amplitude_penalty
                * migration_amplitude
            )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()

    phase_canonical = (
        range_basis @ phase_coefficients.detach() @ aperture_basis.mT
    )
    phase = (
        phase_canonical if range_axis == 0 else phase_canonical.mT
    )
    phase = torch.nan_to_num(phase)
    phase = phase - phase.mean(dim=aperture_axis, keepdim=True)
    migration = (
        config.migration_max_shift
        * torch.tanh(migration_basis @ migration_coefficients.detach())
        if config.range_migration_correction
        else torch.zeros(
            aperture_length,
            device=observed.device,
            dtype=real_dtype,
        )
    )
    migration = torch.nan_to_num(migration)
    migration = migration - migration.mean()
    if not config.range_dependent_phase:
        phase = phase.mean(dim=range_axis)
    return phase, migration


def _dct_basis(
    length: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    include_constant: bool,
) -> Tensor:
    positions = (
        torch.arange(length, device=device, dtype=dtype) + 0.5
    ) / length
    columns: list[Tensor] = []
    if include_constant:
        columns.append(torch.ones(length, device=device, dtype=dtype))
    if rank:
        frequencies = torch.arange(
            1, rank + 1, device=device, dtype=dtype
        )
        columns.extend(
            torch.cos(
                torch.pi
                * positions[:, None]
                * frequencies[None, :]
            ).unbind(dim=1)
        )
    return torch.stack(columns, dim=1)


def _curvature_energy(value: Tensor) -> Tensor:
    energies: list[Tensor] = []
    for axis, length in enumerate(value.shape):
        if length < 3:
            continue
        first = value.narrow(axis, 0, length - 2)
        middle = value.narrow(axis, 1, length - 2)
        last = value.narrow(axis, 2, length - 2)
        energies.append((last - 2.0 * middle + first).square().mean())
    if energies:
        return torch.stack(energies).mean()
    return value.square().mean() * 0.0


def _refine_motion_from_prediction(
    observed: Tensor,
    predicted: Tensor,
    initial_phase: Tensor,
    initial_migration: Tensor,
    *,
    config: SBLConfig,
    observation_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Generalized-EM motion update using the SBL-predicted complex echo."""
    aperture_axis = config.phase_error_axis
    range_axis = 1 - aperture_axis
    aperture_length = observed.shape[aperture_axis]
    range_length = observed.shape[range_axis]
    real_dtype = observed.real.dtype
    aperture_basis = _dct_basis(
        aperture_length,
        min(config.phase_basis_rank, aperture_length - 1),
        observed.device,
        real_dtype,
        include_constant=False,
    )
    range_basis = _dct_basis(
        range_length,
        (
            min(config.phase_range_basis_rank, range_length - 1)
            if config.range_dependent_phase
            else 0
        ),
        observed.device,
        real_dtype,
        include_constant=True,
    )
    initial_field = (
        initial_phase
        if initial_phase.ndim == 2
        else (
            initial_phase[None, :].expand(range_length, -1)
            if range_axis == 0
            else initial_phase[:, None].expand(-1, range_length)
        )
    )
    canonical_initial = (
        initial_field if range_axis == 0 else initial_field.mT
    )
    phase_initial_coefficients = (
        torch.linalg.pinv(range_basis)
        @ canonical_initial
        @ torch.linalg.pinv(aperture_basis.mT)
    )
    phase_coefficients = (
        phase_initial_coefficients.detach().clone().requires_grad_(True)
    )
    migration_basis = _dct_basis(
        aperture_length,
        min(config.migration_basis_rank, aperture_length - 1),
        observed.device,
        real_dtype,
        include_constant=False,
    )
    normalized_migration = (
        initial_migration / config.migration_max_shift
    ).clamp(-0.999, 0.999)
    migration_target = torch.atanh(normalized_migration)
    migration_initial_coefficients = (
        torch.linalg.pinv(migration_basis) @ migration_target
    )
    migration_coefficients = (
        migration_initial_coefficients.detach().clone().requires_grad_(
            config.range_migration_correction
        )
    )
    parameters: list[Tensor] = [phase_coefficients]
    if config.range_migration_correction:
        parameters.append(migration_coefficients)
    optimizer = torch.optim.Adam(
        parameters, lr=config.motion_refinement_learning_rate
    )
    valid = (
        observation_mask
        if observation_mask is not None
        else torch.ones_like(observed, dtype=torch.bool)
    )
    observed_power = observed[valid].abs().square().mean().clamp_min(
        torch.finfo(real_dtype).tiny
    )
    for _ in range(config.motion_refinement_steps):
        optimizer.zero_grad(set_to_none=True)
        phase_canonical = (
            range_basis @ phase_coefficients @ aperture_basis.mT
        )
        phase = (
            phase_canonical
            if range_axis == 0
            else phase_canonical.mT
        )
        migration = (
            config.migration_max_shift
            * torch.tanh(migration_basis @ migration_coefficients)
            if config.range_migration_correction
            else torch.zeros_like(initial_migration)
        )
        model = fractional_range_shift(
            predicted,
            migration,
            aperture_axis=aperture_axis,
        ) * torch.exp(1j * phase)
        model_valid = model[valid]
        observed_valid = observed[valid]
        scale = (
            (model_valid.conj() * observed_valid).sum()
            / model_valid.abs().square().sum().clamp_min(
                torch.finfo(real_dtype).tiny
            )
        )
        residual = scale * model_valid - observed_valid
        objective = (
            residual.abs().square().mean() / observed_power
            + 1e-3
            * config.phase_smoothness
            * _curvature_energy(phase)
            + 1e-3
            * config.migration_smoothness
            * _curvature_energy(migration)
            + 1e-3
            * config.motion_phase_amplitude_penalty
            * phase.square().mean()
            + 1e-3
            * config.motion_migration_amplitude_penalty
            * (
                migration.square().mean()
                / config.migration_max_shift**2
            )
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
    phase_canonical = (
        range_basis @ phase_coefficients.detach() @ aperture_basis.mT
    )
    phase = phase_canonical if range_axis == 0 else phase_canonical.mT
    phase = torch.nan_to_num(phase)
    phase = phase - phase.mean(dim=aperture_axis, keepdim=True)
    migration = (
        config.migration_max_shift
        * torch.tanh(migration_basis @ migration_coefficients.detach())
        if config.range_migration_correction
        else torch.zeros_like(initial_migration)
    )
    migration = torch.nan_to_num(migration)
    migration = migration - migration.mean()
    if not config.range_dependent_phase:
        phase = phase.mean(dim=range_axis)
    return phase, migration


def _minimum_entropy_phase(
    observed: Tensor,
    grid_shape: tuple[int, int],
    *,
    axis: int,
    rank: int,
    steps: int,
    learning_rate: float,
    smoothness: float,
) -> Tensor:
    phase_length = observed.shape[axis]
    rank = min(rank, phase_length - 1)
    if rank < 1:
        return torch.zeros(
            phase_length,
            device=observed.device,
            dtype=observed.real.dtype,
        )
    real_dtype = observed.real.dtype
    positions = (
        torch.arange(
            phase_length,
            device=observed.device,
            dtype=real_dtype,
        )
        + 0.5
    ) / phase_length
    frequencies = torch.arange(
        1,
        rank + 1,
        device=observed.device,
        dtype=real_dtype,
    )
    basis = (2.0 / phase_length) ** 0.5 * torch.cos(
        torch.pi * positions[:, None] * frequencies[None, :]
    )
    coefficients = torch.zeros(
        rank,
        device=observed.device,
        dtype=real_dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        (coefficients,), lr=learning_rate
    )
    crop = (
        slice(0, observed.shape[0]),
        slice(0, observed.shape[1]),
    )
    tiny = torch.finfo(real_dtype).tiny
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            phase = basis @ coefficients
            corrected = _apply_phase_correction(
                observed, phase, axis
            )
            padded = torch.zeros(
                grid_shape,
                device=observed.device,
                dtype=observed.dtype,
            )
            padded[crop] = corrected
            backprojection = (
                torch.fft.ifft2(padded) * prod(grid_shape)
            )
            power = backprojection.abs().square()
            probability = power / power.sum().clamp_min(tiny)
            entropy = -(
                probability
                * probability.clamp_min(tiny).log()
            ).sum()
            if phase_length >= 3:
                curvature = (
                    phase[2:] - 2.0 * phase[1:-1] + phase[:-2]
                )
                penalty = smoothness * curvature.square().mean()
            else:
                penalty = torch.zeros_like(entropy)
            objective = entropy + penalty
            objective.backward()
            torch.nn.utils.clip_grad_norm_((coefficients,), 10.0)
            optimizer.step()
    phase = torch.nan_to_num((basis @ coefficients).detach())
    return phase - phase.mean()


def _apply_phase_correction(
    data: Tensor,
    phase: Tensor,
    axis: int,
) -> Tensor:
    if phase.ndim == 2:
        if tuple(phase.shape) != tuple(data.shape):
            raise ValueError("A 2-D phase field must match the data shape.")
        return data * torch.exp(-1j * phase)
    view_shape = [1, 1]
    view_shape[axis] = phase.numel()
    return data * torch.exp(-1j * phase.reshape(view_shape))


def _apply_motion_correction(
    data: Tensor,
    phase: Tensor,
    migration: Tensor,
    aperture_axis: int,
    observation_mask: Tensor | None = None,
) -> Tensor:
    """Invert phase corruption followed by a fractional range migration."""
    corrected = _apply_phase_correction(data, phase, aperture_axis)
    corrected = fractional_range_shift(
        corrected,
        -migration,
        aperture_axis=aperture_axis,
    )
    if observation_mask is not None:
        corrected = torch.where(
            observation_mask, corrected, torch.zeros_like(corrected)
        )
    return corrected


def fractional_range_shift(
    data: Tensor,
    shifts: Tensor,
    *,
    aperture_axis: int = 1,
) -> Tensor:
    """Apply differentiable per-aperture fractional shifts in range bins."""
    if data.ndim != 2 or aperture_axis not in (0, 1):
        raise ValueError("fractional_range_shift requires 2-D data.")
    range_axis = 1 - aperture_axis
    if shifts.ndim != 1 or shifts.numel() != data.shape[aperture_axis]:
        raise ValueError("shifts must match the aperture dimension.")
    frequencies = torch.fft.fftfreq(
        data.shape[range_axis],
        device=data.device,
        dtype=data.real.dtype,
    )
    frequency_shape = [1, 1]
    frequency_shape[range_axis] = frequencies.numel()
    shift_shape = [1, 1]
    shift_shape[aperture_axis] = shifts.numel()
    ramp = torch.exp(
        -2j
        * torch.pi
        * frequencies.reshape(frequency_shape)
        * shifts.reshape(shift_shape)
    )
    spectrum = torch.fft.fft(data, dim=range_axis)
    return torch.fft.ifft(spectrum * ramp, dim=range_axis)


def _negative_log_evidence(
    data: Tensor,
    result: SBLResult,
    grid_shape: tuple[int, int],
    config: SBLConfig,
    observation_mask: Tensor | None = None,
) -> float:
    operator = PrefixFourierOperator(
        grid_shape,
        data.shape,
        device=data.device,
        complex_dtype=data.dtype,
        observation_mask=observation_mask,
    )
    prior_variance = (
        result.precision.reshape(1, -1).reciprocal()
    )
    noise_variance = result.noise_variance.reshape(1)
    if result.mode.startswith("dense"):
        covariance = operator.covariance(
            prior_variance, noise_variance
        )
        scale = covariance.diagonal(
            dim1=-2, dim2=-1
        ).real.mean(dim=1)
        identity = operator.eye
        cholesky = None
        for multiplier in (1.0, 10.0, 100.0, 1000.0):
            candidate, info = torch.linalg.cholesky_ex(
                covariance
                + (
                    multiplier
                    * torch.finfo(operator.real_dtype).eps
                    * scale
                )[:, None, None]
                * identity
            )
            if bool((info == 0).all()):
                cholesky = candidate
                break
        if cholesky is None:
            return float("inf")
        vector = operator.restrict(
            data.reshape(1, -1, 1)
        )
        solved = torch.cholesky_solve(vector, cholesky)
        quadratic = (vector.conj() * solved).sum().real
        log_determinant = 2.0 * cholesky.diagonal(
            dim1=-2, dim2=-1
        ).real.log().sum()
        return float((quadratic + log_determinant).item())
    return _slq_negative_log_evidence(
        operator.restrict(data.reshape(1, -1, 1)).reshape(-1),
        prior_variance,
        noise_variance,
        operator,
        config,
    )


def _slq_negative_log_evidence(
    data: Tensor,
    prior_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    config: SBLConfig,
) -> float:
    probes = _make_probes(
        1,
        operator.n_samples,
        config.phase_slq_probes,
        device=operator.device,
        dtype=operator.complex_dtype,
        seed=config.seed + 7919,
        probe_type="rademacher",
        probe_shape=(
            operator.sample_shape if operator.is_complete else None
        ),
    )
    toeplitz_spectrum = (
        operator.toeplitz_spectrum(
            prior_variance, noise_variance
        )
        if config.cg_matvec == "tbt"
        else None
    )
    tolerance = max(
        min(config.cg_tolerance, 1e-5),
        3.0 * torch.finfo(operator.real_dtype).eps**0.5,
    )
    vector = data.reshape(1, -1, 1)
    solved, _ = _pcg(
        vector,
        prior_variance,
        noise_variance,
        operator,
        initial_solution=None,
        tolerance=tolerance,
        max_iter=config.cg_max_iter,
        check_interval=config.cg_check_interval,
        preconditioner_mode=config.cg_preconditioner,
        preconditioner_rank=config.cg_preconditioner_rank,
        preconditioner_seed=config.seed,
        toeplitz_spectrum=toeplitz_spectrum,
    )
    quadratic = (vector.conj() * solved).sum().real

    lanczos_vector = probes / operator.n_samples**0.5
    previous = torch.zeros_like(lanczos_vector)
    previous_beta = torch.zeros(
        (1, config.phase_slq_probes),
        device=operator.device,
        dtype=operator.real_dtype,
    )
    alphas: list[Tensor] = []
    betas: list[Tensor] = []
    tiny = torch.finfo(operator.real_dtype).tiny
    for step in range(config.phase_slq_steps):
        product = operator.matvec(
            lanczos_vector,
            prior_variance,
            noise_variance,
            toeplitz_spectrum=toeplitz_spectrum,
        )
        alpha = (
            lanczos_vector.conj() * product
        ).sum(dim=1).real
        remainder = (
            product
            - alpha[:, None, :] * lanczos_vector
            - previous_beta[:, None, :] * previous
        )
        beta = remainder.norm(dim=1)
        alphas.append(alpha[0])
        if step + 1 < config.phase_slq_steps:
            betas.append(beta[0])
        previous = lanczos_vector
        lanczos_vector = remainder / beta.clamp_min(tiny)[:, None, :]
        previous_beta = beta

    diagonal = torch.stack(alphas, dim=1)
    off_diagonal = torch.stack(betas, dim=1)
    probe_count, steps = diagonal.shape
    tridiagonal = torch.zeros(
        (probe_count, steps, steps),
        device=operator.device,
        dtype=operator.real_dtype,
    )
    indices = torch.arange(steps, device=operator.device)
    tridiagonal[:, indices, indices] = diagonal
    off_indices = torch.arange(steps - 1, device=operator.device)
    tridiagonal[:, off_indices, off_indices + 1] = off_diagonal
    tridiagonal[:, off_indices + 1, off_indices] = off_diagonal
    eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
    eigenvalues = eigenvalues.clamp_min(
        torch.finfo(operator.real_dtype).tiny
    )
    weights = eigenvectors[:, 0, :].square()
    log_determinant = operator.n_samples * (
        weights * eigenvalues.log()
    ).sum(dim=1).mean()
    return float((quadratic + log_determinant).item())
