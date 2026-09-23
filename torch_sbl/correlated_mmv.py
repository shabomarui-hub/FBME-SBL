from __future__ import annotations

from math import prod
from typing import Any, Sequence

import torch
from torch import Tensor

from .offgrid import refine_offgrid_fourier
from .operators import PrefixFourierOperator
from .solver import (
    SBLConfig,
    SBLResult,
    _anisotropic_pattern_precision_update,
    _cg_schedule,
    _complex_dtype,
    _effective_anisotropic_pattern_precision,
    _effective_pattern_precision,
    _initial_anisotropic_pattern_precision,
    _initial_pattern_precision,
    _learn_anisotropic_coupling,
    _make_probes,
    _neighbor_sum_2d,
    _project_relevance,
    _resolve_device,
    _shared_mmv_posterior,
)


def correlated_mmv_sbl_2d(
    data: Any,
    grid_shape: Sequence[int],
    *,
    config: SBLConfig,
) -> SBLResult:
    """Recover a 2-D MMV field while learning inter-snapshot covariance.

    For each spatial row ``x_i`` across ``L`` snapshots, the prior is
    ``CN(0, gamma_i B)``.  The robust T-MSBL fixed point keeps the E-step
    in the original MMV space and replaces Euclidean row energy with a
    learned Mahalanobis energy.  Consequently the existing dense or
    covariance-free TBT-PCG posterior remains directly applicable.
    """
    config.validate()
    shape = tuple(int(v) for v in grid_shape)
    if len(shape) != 2:
        raise ValueError("correlated_mmv_sbl_2d requires a 2-D grid.")

    input_tensor = torch.as_tensor(data)
    if input_tensor.ndim != 3:
        raise ValueError(
            "Snapshot-covariance learning requires data shaped "
            "[n_snapshots, M1, M2]."
        )
    snapshot_count = int(input_tensor.shape[0])
    if snapshot_count < 2:
        raise ValueError("At least two snapshots are required.")
    sample_shape = tuple(int(v) for v in input_tensor.shape[-2:])
    if any(m > n for m, n in zip(sample_shape, shape)):
        raise ValueError("Observed dimensions cannot exceed grid dimensions.")
    if config.offgrid_refinement and config.offgrid_components is None:
        raise ValueError(
            "Correlated MMV off-grid refinement requires "
            "offgrid_components to be specified."
        )

    device = _resolve_device(config.device, input_tensor)
    complex_dtype = _complex_dtype(input_tensor.dtype)
    y = input_tensor.to(device=device, dtype=complex_dtype).reshape(
        snapshot_count, prod(sample_shape)
    )
    operator = PrefixFourierOperator(
        shape,
        sample_shape,
        device=device,
        complex_dtype=complex_dtype,
    )
    if config.mode == "uamp":
        raise ValueError(
            "Correlated MMV snapshot-covariance learning does not yet "
            "support the UAMP backend; use mode='dense' or mode='cg'."
        )
    auto_dense_threshold = (
        config.dense_threshold
        if device.type == "cuda"
        else min(config.dense_threshold, 512)
    )
    mode = (
        "dense"
        if config.mode == "auto"
        and operator.n_samples <= auto_dense_threshold
        else "cg"
        if config.mode == "auto"
        else config.mode
    )

    real_dtype = operator.real_dtype
    eps = torch.finfo(real_dtype).eps
    data_power = y.abs().square().mean().real.clamp_min(eps)
    variance_floor = torch.maximum(
        data_power * config.min_variance,
        torch.as_tensor(
            torch.finfo(real_dtype).tiny,
            device=device,
            dtype=real_dtype,
        ),
    )
    uniform_variance = data_power / operator.n_grid
    if config.initialization == "backprojection":
        backprojection = operator.adjoint_flat(y)
        backprojection_energy = (
            backprojection.abs().square().mean(dim=0).real
        )
        structured_variance = (
            backprojection_energy
            / backprojection_energy.sum().clamp_min(eps)
            * data_power
        )
        support_variance = (
            config.initialization_blend * structured_variance
            + (1.0 - config.initialization_blend) * uniform_variance
        )
    else:
        support_variance = torch.full(
            (operator.n_grid,),
            uniform_variance,
            device=device,
            dtype=real_dtype,
        )
    support_variance = support_variance.clamp_min(variance_floor)

    pattern_precision = None
    directional_coupling = None
    spatial_enabled = (
        config.spatial_coupling > 0.0
        or config.learn_spatial_coupling
    )
    if config.learn_spatial_coupling:
        initial_coupling = (
            config.spatial_coupling
            if config.spatial_coupling > 0.0
            else min(0.1, config.spatial_coupling_max)
        )
        directional_coupling = torch.full(
            (1, 2),
            initial_coupling,
            device=device,
            dtype=real_dtype,
        )
        pattern_precision = _initial_anisotropic_pattern_precision(
            support_variance[None, :],
            shape,
            directional_coupling,
            config.spatial_boundary,
        )
        support_variance = _effective_anisotropic_pattern_precision(
            pattern_precision,
            shape,
            directional_coupling,
            config.spatial_boundary,
        ).clamp(
            min=1.0 / config.max_precision,
            max=config.max_precision,
        ).reciprocal().squeeze(0)
    elif config.spatial_coupling > 0.0:
        pattern_precision = _initial_pattern_precision(
            support_variance[None, :],
            shape,
            config.spatial_coupling,
            config.spatial_boundary,
        )
        support_variance = _effective_pattern_precision(
            pattern_precision,
            shape,
            config.spatial_coupling,
            config.spatial_boundary,
        ).clamp(
            min=1.0 / config.max_precision,
            max=config.max_precision,
        ).reciprocal().squeeze(0)

    if config.fixed_noise_variance is None:
        noise_variance = (
            config.initial_noise_fraction * data_power
        ).clamp_min(variance_floor)
    else:
        noise_variance = torch.as_tensor(
            config.fixed_noise_variance,
            device=device,
            dtype=real_dtype,
        )

    sample_covariance = y @ y.mH / operator.n_samples
    (
        observed_snapshot_correlation,
        estimated_snapshot_snr_db,
    ) = _snapshot_reliability_statistics(
        sample_covariance,
        eps,
    )
    covariance_is_reliable = bool(
        observed_snapshot_correlation
        >= config.snapshot_covariance_min_correlation
    ) and bool(
        estimated_snapshot_snr_db
        >= config.snapshot_covariance_min_snr_db
    ) and (
        operator.n_grid / operator.n_samples
        >= config.snapshot_covariance_min_undersampling
    )
    identity_covariance = torch.eye(
        snapshot_count, device=device, dtype=complex_dtype
    )
    snapshot_covariance = (
        _stabilize_snapshot_covariance(
            sample_covariance,
            shrinkage=max(
                config.snapshot_covariance_shrinkage, 0.1
            ),
            eigenvalue_floor=config.snapshot_covariance_floor,
            model=config.snapshot_covariance_model,
        )
        if covariance_is_reliable
        else identity_covariance.clone()
    )

    probes = None
    lifted_probes = None
    maximum_probes = config.hutchinson_probes
    if mode == "cg":
        if config.auto_probe_scaling:
            undersampling = operator.n_grid / operator.n_samples
            recommended_float = 8.0 * undersampling**0.5
            if undersampling > 16.0:
                recommended_float = max(
                    recommended_float, 2.0 * undersampling
                )
            recommended = int(recommended_float + 0.999999)
            maximum_probes = min(
                config.max_hutchinson_probes,
                max(config.hutchinson_probes, recommended),
            )
        probes = _make_probes(
            1,
            operator.n_samples,
            maximum_probes,
            device=device,
            dtype=complex_dtype,
            seed=config.seed,
            probe_type=config.probe_type,
            probe_shape=operator.sample_shape,
        )
        lifted_probes = operator.adjoint(probes)

    history_tensors: dict[str, list[Tensor]] = {
        "relative_change": [],
        "noise_variance": [],
        "effective_dof": [],
        "cg_iterations": [],
        "snapshot_condition": [],
        "snapshot_effective_rank": [],
    }
    if directional_coupling is not None:
        history_tensors["spatial_vertical_coupling"] = []
        history_tensors["spatial_horizontal_coupling"] = []
    previous_mean = torch.zeros(
        (snapshot_count, operator.n_grid),
        device=device,
        dtype=complex_dtype,
    )
    last_change = torch.full((), float("inf"), device=device, dtype=real_dtype)
    total_dof = torch.zeros((), device=device, dtype=real_dtype)
    convergence_streak = 0
    converged = False
    relative_change = float("inf")
    minimum_iterations = (
        max(config.min_iter, config.cg_min_iter)
        if mode == "cg"
        else config.min_iter
    )

    with torch.no_grad():
        for iteration in range(1, config.max_iter + 1):
            shared_prior = support_variance[None, :]
            shared_noise = noise_variance.reshape(1)

            probe_count, cg_tolerance = _cg_schedule(
                iteration, config, maximum_probes
            )
            cg_tolerance = max(
                cg_tolerance,
                3.0 * torch.finfo(real_dtype).eps**0.5,
            )
            iteration_probes = (
                probes[:, :, :probe_count] if probes is not None else None
            )
            iteration_lifted = (
                lifted_probes[:, :, :probe_count]
                if lifted_probes is not None
                else None
            )
            (
                mean,
                posterior_variance,
                data_diagonal,
                cg_iterations,
                _,
            ) = _shared_mmv_posterior(
                y,
                shared_prior,
                shared_noise,
                operator,
                mode,
                config,
                iteration_probes,
                iteration_lifted,
                cg_tolerance=cg_tolerance,
                initial_solution=None,
                active_indices=None,
                active_columns=None,
                active_gram=None,
            )
            relevance = _project_relevance(
                shared_prior,
                data_diagonal,
                operator.n_samples,
                eps,
            )
            total_dof = relevance.sum()

            # T-MSBL fixed point: replace Euclidean row energy by its
            # Mahalanobis counterpart while retaining the covariance-free
            # M-SBL posterior in the original problem space.
            continuation_weight = min(
                1.0,
                max(
                    0.0,
                    (
                        iteration
                        - config.snapshot_covariance_start
                    )
                    / config.snapshot_covariance_warmup,
                ),
            )
            if not covariance_is_reliable:
                continuation_weight = 0.0
            effective_snapshot_covariance = (
                (1.0 - continuation_weight) * identity_covariance
                + continuation_weight * snapshot_covariance
            )
            whitened_mean = _stable_snapshot_solve(
                effective_snapshot_covariance,
                mean,
                eigenvalue_floor=config.snapshot_covariance_floor,
            )
            mahalanobis_moment = (
                (mean.conj() * whitened_mean).sum(dim=0).real
                / snapshot_count
                + posterior_variance.mean(dim=0)
            )
            keep = (
                max(config.damping, config.cg_damping)
                if mode == "cg"
                else config.damping
            )
            if spatial_enabled:
                if pattern_precision is None:
                    raise RuntimeError("Pattern precision was not initialized.")
                if config.learn_spatial_coupling:
                    if directional_coupling is None:
                        raise RuntimeError(
                            "Directional coupling was not initialized."
                        )
                    structure_moment = mahalanobis_moment[None, :]
                    if (
                        (iteration - 1)
                        % config.spatial_coupling_update_interval
                        == 0
                    ):
                        learned_coupling = _learn_anisotropic_coupling(
                            pattern_precision,
                            structure_moment,
                            shape,
                            directional_coupling,
                            config.spatial_boundary,
                            maximum=config.spatial_coupling_max,
                            steps=config.spatial_coupling_steps,
                            magnitude_penalty=(
                                config.spatial_coupling_penalty
                            ),
                            anisotropy_penalty=(
                                config.spatial_anisotropy_penalty
                            ),
                        )
                        directional_coupling = (
                            config.spatial_coupling_damping
                            * directional_coupling
                            + (
                                1.0
                                - config.spatial_coupling_damping
                            )
                            * learned_coupling
                        )
                    new_pattern_precision = (
                        _anisotropic_pattern_precision_update(
                            structure_moment,
                            shape,
                            directional_coupling,
                            config.spatial_boundary,
                            hyper_shape=config.hyper_shape,
                            hyper_rate=config.hyper_rate,
                            max_precision=config.max_precision,
                        )
                    )
                else:
                    coupled_moment = (
                        mahalanobis_moment[None, :]
                        + config.spatial_coupling
                        * _neighbor_sum_2d(
                            mahalanobis_moment[None, :],
                            shape,
                            config.spatial_boundary,
                        )
                    )
                    new_pattern_precision = (
                        (1.0 + config.hyper_shape)
                        / (coupled_moment + config.hyper_rate)
                    ).clamp(
                        min=1.0 / config.max_precision,
                        max=config.max_precision,
                    )
                pattern_precision = (
                    keep * pattern_precision
                    + (1.0 - keep) * new_pattern_precision
                )
                effective_precision = (
                    _effective_anisotropic_pattern_precision(
                        pattern_precision,
                        shape,
                        directional_coupling,
                        config.spatial_boundary,
                    )
                    if directional_coupling is not None
                    else _effective_pattern_precision(
                        pattern_precision,
                        shape,
                        config.spatial_coupling,
                        config.spatial_boundary,
                    )
                ).clamp(
                    min=1.0 / config.max_precision,
                    max=config.max_precision,
                )
                support_variance = effective_precision.reciprocal().squeeze(0)
            else:
                new_support_variance = (
                    mahalanobis_moment + config.hyper_rate
                ) / (1.0 + config.hyper_shape)
                new_support_variance = new_support_variance.clamp(
                    min=variance_floor,
                    max=config.max_precision,
                )
                support_variance = (
                    keep * support_variance
                    + (1.0 - keep) * new_support_variance
                )
            support_variance = support_variance.clamp_min(variance_floor)

            if (
                covariance_is_reliable
                and iteration >= config.snapshot_covariance_start
            ):
                covariance_candidate = _snapshot_covariance_update(
                    mean,
                    support_variance,
                    shrinkage=config.snapshot_covariance_shrinkage,
                    eigenvalue_floor=config.snapshot_covariance_floor,
                    model=config.snapshot_covariance_model,
                )
                covariance_keep = config.snapshot_covariance_damping
                snapshot_covariance = _stabilize_snapshot_covariance(
                    covariance_keep * snapshot_covariance
                    + (1.0 - covariance_keep) * covariance_candidate,
                    shrinkage=0.0,
                    eigenvalue_floor=config.snapshot_covariance_floor,
                    model=config.snapshot_covariance_model,
                )

            residual = y - operator.forward(mean[:, :, None]).squeeze(-1)
            residual_energy = residual.abs().square().sum().real
            if config.fixed_noise_variance is None:
                new_noise_variance = (
                    residual_energy
                    + noise_variance * total_dof
                    + config.noise_rate
                ) / (
                    snapshot_count * operator.n_samples
                    + config.noise_shape
                )
                new_noise_variance = new_noise_variance.clamp_min(
                    variance_floor
                )
                noise_variance = (
                    keep * noise_variance
                    + (1.0 - keep) * new_noise_variance
                )

            denominator = mean.norm().clamp_min(eps)
            last_change = (mean - previous_mean).norm() / denominator
            previous_mean = mean
            covariance_eigenvalues = torch.linalg.eigvalsh(
                effective_snapshot_covariance
            ).real
            normalized_eigenvalues = (
                covariance_eigenvalues
                / covariance_eigenvalues.sum().clamp_min(eps)
            )
            effective_rank = torch.exp(
                -(
                    normalized_eigenvalues
                    * normalized_eigenvalues.clamp_min(eps).log()
                ).sum()
            )
            condition = (
                covariance_eigenvalues.max()
                / covariance_eigenvalues.min().clamp_min(eps)
            )
            if config.return_history:
                history_tensors["relative_change"].append(
                    last_change.detach()
                )
                history_tensors["noise_variance"].append(
                    noise_variance.detach()
                )
                history_tensors["effective_dof"].append(total_dof.detach())
                history_tensors["cg_iterations"].append(
                    torch.as_tensor(
                        cg_iterations, device=device, dtype=real_dtype
                    )
                )
                history_tensors["snapshot_condition"].append(
                    condition.detach()
                )
                history_tensors["snapshot_effective_rank"].append(
                    effective_rank.detach()
                )
                if directional_coupling is not None:
                    history_tensors["spatial_vertical_coupling"].append(
                        directional_coupling[0, 0].detach()
                    )
                    history_tensors["spatial_horizontal_coupling"].append(
                        directional_coupling[0, 1].detach()
                    )

            should_check = (
                iteration >= minimum_iterations
                and (
                    (iteration - minimum_iterations)
                    % config.convergence_check_interval
                    == 0
                    or iteration == config.max_iter
                )
            )
            if should_check:
                relative_change = float(last_change.item())
                convergence_streak = (
                    convergence_streak + 1
                    if relative_change <= config.tol
                    else 0
                )
                if convergence_streak >= config.convergence_patience:
                    converged = True
                    break
            if config.verbose and (iteration == 1 or iteration % 10 == 0):
                print(
                    f"[CMMV-SBL] iter={iteration:4d} "
                    f"change={float(last_change):.3e} "
                    f"noise={float(noise_variance):.3e} "
                    f"rank={float(effective_rank):.2f} "
                    f"cond={float(condition):.2f} cg={cg_iterations:3d}"
                )

        (
            mean,
            posterior_variance,
            total_dof,
        ) = _final_correlated_posterior(
            y,
            support_variance,
            noise_variance,
            operator,
            mode,
            config,
            probes,
            lifted_probes,
            eps,
        )
        relative_change = float(last_change.item())

    reconstructed = mean.reshape(snapshot_count, *shape)
    offgrid_result = None
    if config.offgrid_refinement:
        offgrid_result = refine_offgrid_fourier(
            input_tensor.to(device=device, dtype=complex_dtype),
            reconstructed,
            shape,
            component_count=config.offgrid_components,
            max_components=config.offgrid_max_components,
            relative_threshold=config.offgrid_relative_threshold,
            minimum_separation=config.offgrid_minimum_separation,
            max_iter=config.offgrid_max_iter,
            tolerance=config.offgrid_tolerance,
            max_step=config.offgrid_max_step,
            damping=config.offgrid_damping,
            ridge=config.offgrid_ridge,
        )
    history = (
        {
            key: torch.stack(values).cpu().tolist()
            for key, values in history_tensors.items()
        }
        if config.return_history
        else {}
    )
    shared_precision = support_variance.reciprocal().expand(
        snapshot_count, -1
    )
    mean_dof = (total_dof / snapshot_count).expand(snapshot_count)
    return SBLResult(
        x=reconstructed,
        posterior_variance=posterior_variance.reshape(
            snapshot_count, *shape
        ),
        precision=shared_precision.reshape(snapshot_count, *shape),
        noise_variance=noise_variance.expand(snapshot_count),
        effective_dof=mean_dof,
        iterations=iteration,
        converged=converged,
        relative_change=relative_change,
        mode=f"{mode}-correlated-mmv",
        active_set_size=0,
        offgrid=offgrid_result,
        snapshot_covariance=snapshot_covariance,
        snapshot_covariance_active=covariance_is_reliable,
        snapshot_snr_db=float(estimated_snapshot_snr_db.item()),
        learned_spatial_coupling=(
            directional_coupling.squeeze(0)
            if directional_coupling is not None
            else None
        ),
        history=history,
    )


def _snapshot_covariance_update(
    mean: Tensor,
    support_variance: Tensor,
    *,
    shrinkage: float,
    eigenvalue_floor: float,
    model: str,
) -> Tensor:
    """Regularized T-MSBL update of B from posterior row means.

    The exact EM covariance term is dominated by inactive rows when the
    dictionary is highly overcomplete.  The robust T-MSBL fixed-point update
    instead uses the reconstructed row outer products and diagonal loading.
    """
    relative_scale = support_variance / support_variance.max().clamp_min(
        torch.finfo(support_variance.dtype).tiny
    )
    active = relative_scale >= 1e-4
    minimum_rows = min(mean.shape[0], support_variance.numel())
    if int(active.sum()) < minimum_rows:
        indices = torch.topk(
            support_variance, minimum_rows, sorted=False
        ).indices
        active = torch.zeros_like(active)
        active[indices] = True
    selected_scale = support_variance[active].clamp_min(
        torch.finfo(support_variance.dtype).tiny
    ).rsqrt()
    scaled_mean = mean[:, active] * selected_scale[None, :]
    covariance = scaled_mean @ scaled_mean.mH
    return _stabilize_snapshot_covariance(
        covariance,
        shrinkage=shrinkage,
        eigenvalue_floor=eigenvalue_floor,
        model=model,
    )


def _snapshot_reliability_statistics(
    sample_covariance: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Estimate observed correlation and a conservative covariance SNR."""
    diagonal_power = sample_covariance.diagonal().real.mean().clamp_min(
        eps
    )
    first_lag = sample_covariance.diagonal(offset=-1).mean().abs()
    observed_correlation = (first_lag / diagonal_power).clamp(
        0.0, 1.0
    )
    if sample_covariance.shape[0] >= 3:
        second_lag = (
            sample_covariance.diagonal(offset=-2).mean().abs()
        )
        latent_correlation = (
            second_lag / first_lag.clamp_min(eps)
        ).clamp(
            min=observed_correlation,
            max=0.999,
        )
        signal_power = first_lag / latent_correlation.clamp_min(eps)
    else:
        # With two snapshots the latent AR coefficient and white noise are
        # not separately identifiable; rho≈1 gives a conservative estimate.
        signal_power = first_lag
    signal_power = signal_power.clamp(max=diagonal_power)
    noise_power = (diagonal_power - signal_power).clamp_min(
        eps * diagonal_power
    )
    estimated_snr_db = 10.0 * torch.log10(
        signal_power.clamp_min(eps * diagonal_power) / noise_power
    )
    return observed_correlation, estimated_snr_db


def _stable_snapshot_solve(
    covariance: Tensor,
    right_hand_side: Tensor,
    *,
    eigenvalue_floor: float,
) -> Tensor:
    """Solve a small Hermitian system safely in complex32/complex64."""
    covariance = 0.5 * (covariance + covariance.mH)
    identity = torch.eye(
        covariance.shape[0],
        device=covariance.device,
        dtype=covariance.dtype,
    )
    if not bool(torch.isfinite(covariance).all()):
        covariance = identity
    eps = torch.finfo(covariance.real.dtype).eps
    base_jitter = max(eigenvalue_floor * 1e-3, eps)
    for multiplier in (0.0, 1.0, 10.0, 100.0, 1000.0):
        cholesky, info = torch.linalg.cholesky_ex(
            covariance + multiplier * base_jitter * identity
        )
        if int(info.item()) == 0:
            return torch.cholesky_solve(
                right_hand_side, cholesky
            )
    return torch.linalg.pinv(
        covariance,
        rtol=eps**0.5,
        hermitian=True,
    ) @ right_hand_side


def _stabilize_snapshot_covariance(
    covariance: Tensor,
    *,
    shrinkage: float,
    eigenvalue_floor: float,
    model: str = "full",
) -> Tensor:
    """Hermitian projection, trace identification and eigenvalue clipping."""
    size = covariance.shape[0]
    covariance = 0.5 * (covariance + covariance.mH)
    real_dtype = covariance.real.dtype
    eps = torch.finfo(real_dtype).eps
    trace = covariance.diagonal().real.sum().clamp_min(eps)
    covariance = covariance * (size / trace)
    if model == "ar1":
        diagonal_mean = covariance.diagonal().real.mean().clamp_min(eps)
        coefficient = (
            covariance.diagonal(offset=-1).mean() / diagonal_mean
        )
        magnitude = coefficient.abs().clamp_max(0.99)
        coefficient = torch.where(
            coefficient.abs() > eps,
            coefficient / coefficient.abs() * magnitude,
            torch.zeros_like(coefficient),
        )
        coefficient = (1.0 - shrinkage) * coefficient
        indices = torch.arange(size, device=covariance.device)
        difference = indices[:, None] - indices[None, :]
        lower_power = difference.clamp_min(0)
        upper_power = (-difference).clamp_min(0)
        covariance = torch.where(
            difference >= 0,
            coefficient**lower_power,
            coefficient.conj() ** upper_power,
        ).to(covariance.dtype)
    elif model != "full":
        raise ValueError("Snapshot covariance model must be 'ar1' or 'full'.")
    elif shrinkage > 0.0:
        identity = torch.eye(
            size, device=covariance.device, dtype=covariance.dtype
        )
        covariance = (
            (1.0 - shrinkage) * covariance + shrinkage * identity
        )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.real.clamp_min(eigenvalue_floor)
    eigenvalues = eigenvalues * (size / eigenvalues.sum().clamp_min(eps))
    return (eigenvectors * eigenvalues[None, :]) @ eigenvectors.mH


def _final_correlated_posterior(
    y: Tensor,
    support_variance: Tensor,
    noise_variance: Tensor,
    operator: PrefixFourierOperator,
    mode: str,
    config: SBLConfig,
    probes: Tensor | None,
    lifted_probes: Tensor | None,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    shared_prior = support_variance[None, :]
    (
        mean,
        posterior_variance,
        data_diagonal,
        _,
        _,
    ) = _shared_mmv_posterior(
        y,
        shared_prior,
        noise_variance.reshape(1),
        operator,
        mode,
        config,
        probes,
        lifted_probes,
        cg_tolerance=max(
            config.cg_tolerance,
            3.0 * torch.finfo(operator.real_dtype).eps**0.5,
        ),
        initial_solution=None,
        active_indices=None,
        active_columns=None,
        active_gram=None,
    )
    total_dof = _project_relevance(
        shared_prior.expand(y.shape[0], -1),
        data_diagonal,
        operator.n_samples,
        eps,
    ).sum()
    return mean, posterior_variance, total_dof
