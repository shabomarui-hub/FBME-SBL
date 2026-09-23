"""Optional Triton fusion kernels for the FFT-UAMP execution path.

cuFFT remains responsible for Fourier transforms.  These kernels fuse the
complex elementwise arithmetic and short row reductions around the FFTs.
The module is safe to import when Triton is absent or cannot load: callers
can query :func:`triton_is_available` and retain the pure PyTorch path.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

_TRITON_IMPORT_ERROR: Exception | None = None
_TRITON_RUNTIME_ERROR: Exception | None = None

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - depends on optional runtime
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc


_ELEMENT_BLOCK: Final = 256
_MESSAGE_BLOCK: Final = 256
_MAX_REDUCTION_BLOCK: Final = 8192
_REDUCTION_TILE: Final = 1024


if triton is not None:

    @triton.jit
    def _uamp_message_kernel(
        y_ptr,
        prediction_ptr,
        old_message_ptr,
        data_variance_ptr,
        noise_variance_ptr,
        output_ptr,
        pseudo_variance_ptr,
        n_samples: tl.constexpr,
        damping: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)
        sample = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = sample < n_samples
        index = batch_index * n_samples + sample
        complex_index = 2 * index

        y_real = tl.load(y_ptr + complex_index, mask=mask, other=0.0)
        y_imag = tl.load(y_ptr + complex_index + 1, mask=mask, other=0.0)
        prediction_real = tl.load(
            prediction_ptr + complex_index, mask=mask, other=0.0
        )
        prediction_imag = tl.load(
            prediction_ptr + complex_index + 1, mask=mask, other=0.0
        )
        old_real = tl.load(
            old_message_ptr + complex_index, mask=mask, other=0.0
        )
        old_imag = tl.load(
            old_message_ptr + complex_index + 1, mask=mask, other=0.0
        )
        data_variance = tl.load(data_variance_ptr + batch_index)
        noise_variance = tl.load(noise_variance_ptr + batch_index)
        precision = 1.0 / (data_variance + noise_variance)

        pseudo_real = prediction_real - data_variance * old_real
        pseudo_imag = prediction_imag - data_variance * old_imag
        candidate_real = precision * (y_real - pseudo_real)
        candidate_imag = precision * (y_imag - pseudo_imag)
        keep = damping
        update = 1.0 - keep
        output_real = keep * old_real + update * candidate_real
        output_imag = keep * old_imag + update * candidate_imag

        tl.store(output_ptr + complex_index, output_real, mask=mask)
        tl.store(output_ptr + complex_index + 1, output_imag, mask=mask)
        tl.store(
            pseudo_variance_ptr + batch_index,
            (data_variance + noise_variance) / n_samples,
            mask=block_index == 0,
        )


    @triton.jit
    def _uamp_denoise_kernel(
        prior_variance_ptr,
        pseudo_variance_ptr,
        lifted_message_ptr,
        old_mean_ptr,
        old_variance_ptr,
        output_mean_ptr,
        output_variance_ptr,
        effective_dof_ptr,
        coefficient_variance_sum_ptr,
        prior_floor_ptr,
        output_prior_ptr,
        n_grid: tl.constexpr,
        damping: tl.constexpr,
        tiny: tl.constexpr,
        update_prior: tl.constexpr,
        hyper_shape: tl.constexpr,
        hyper_rate: tl.constexpr,
        prior_damping: tl.constexpr,
        prior_maximum: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)
        coefficient = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = coefficient < n_grid
        index = batch_index * n_grid + coefficient
        complex_index = 2 * index

        prior = tl.load(prior_variance_ptr + index, mask=mask, other=1.0)
        pseudo_variance = tl.load(pseudo_variance_ptr + batch_index)
        old_real = tl.load(
            old_mean_ptr + complex_index, mask=mask, other=0.0
        )
        old_imag = tl.load(
            old_mean_ptr + complex_index + 1, mask=mask, other=0.0
        )
        lifted_real = tl.load(
            lifted_message_ptr + complex_index, mask=mask, other=0.0
        )
        lifted_imag = tl.load(
            lifted_message_ptr + complex_index + 1, mask=mask, other=0.0
        )
        old_variance = tl.load(
            old_variance_ptr + index, mask=mask, other=0.0
        )

        denominator = tl.maximum(prior + pseudo_variance, tiny)
        gain = prior / denominator
        pseudo_real = old_real + pseudo_variance * lifted_real
        pseudo_imag = old_imag + pseudo_variance * lifted_imag
        candidate_real = gain * pseudo_real
        candidate_imag = gain * pseudo_imag
        candidate_variance = prior * pseudo_variance / denominator

        keep = damping
        update = 1.0 - keep
        output_real = keep * old_real + update * candidate_real
        output_imag = keep * old_imag + update * candidate_imag
        variance = tl.maximum(
            keep * old_variance + update * candidate_variance,
            tiny,
        )
        relevance = 1.0 - variance / tl.maximum(prior, tiny)
        relevance = tl.maximum(0.0, tl.minimum(1.0, relevance))
        relevance = tl.where(mask, relevance, 0.0)

        tl.store(output_mean_ptr + complex_index, output_real, mask=mask)
        tl.store(output_mean_ptr + complex_index + 1, output_imag, mask=mask)
        tl.store(output_variance_ptr + index, variance, mask=mask)
        if update_prior:
            floor = tl.load(prior_floor_ptr + batch_index)
            second_moment = (
                output_real * output_real
                + output_imag * output_imag
                + variance
            )
            candidate_prior = (
                second_moment + hyper_rate
            ) / (1.0 + hyper_shape)
            candidate_prior = tl.maximum(
                floor, tl.minimum(prior_maximum, candidate_prior)
            )
            next_prior = (
                prior_damping * prior
                + (1.0 - prior_damping) * candidate_prior
            )
            tl.store(
                output_prior_ptr + index,
                tl.maximum(next_prior, floor),
                mask=mask,
            )
        tl.atomic_add(
            effective_dof_ptr + batch_index,
            tl.sum(relevance, axis=0),
        )
        tl.atomic_add(
            coefficient_variance_sum_ptr + batch_index,
            tl.sum(tl.where(mask, variance, 0.0), axis=0),
        )


    @triton.jit
    def _uamp_prior_kernel(
        mean_ptr,
        posterior_variance_ptr,
        old_prior_ptr,
        prior_floor_ptr,
        output_ptr,
        n_grid: tl.constexpr,
        hyper_shape: tl.constexpr,
        hyper_rate: tl.constexpr,
        damping: tl.constexpr,
        maximum: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)
        coefficient = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = coefficient < n_grid
        index = batch_index * n_grid + coefficient
        complex_index = 2 * index

        mean_real = tl.load(mean_ptr + complex_index, mask=mask, other=0.0)
        mean_imag = tl.load(
            mean_ptr + complex_index + 1, mask=mask, other=0.0
        )
        posterior_variance = tl.load(
            posterior_variance_ptr + index, mask=mask, other=0.0
        )
        old_prior = tl.load(old_prior_ptr + index, mask=mask, other=0.0)
        floor = tl.load(prior_floor_ptr + batch_index)

        second_moment = (
            mean_real * mean_real
            + mean_imag * mean_imag
            + posterior_variance
        )
        candidate = (second_moment + hyper_rate) / (1.0 + hyper_shape)
        candidate = tl.maximum(floor, tl.minimum(maximum, candidate))
        updated = damping * old_prior + (1.0 - damping) * candidate
        updated = tl.maximum(updated, floor)
        tl.store(output_ptr + index, updated, mask=mask)


    @triton.jit
    def _complex_residual_energy_kernel(
        data_ptr,
        prediction_ptr,
        output_ptr,
        n_samples: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        sample = tl.arange(0, BLOCK_SIZE)
        mask = sample < n_samples
        index = batch_index * n_samples + sample
        complex_index = 2 * index

        residual_real = tl.load(
            data_ptr + complex_index, mask=mask, other=0.0
        ) - tl.load(
            prediction_ptr + complex_index, mask=mask, other=0.0
        )
        residual_imag = tl.load(
            data_ptr + complex_index + 1, mask=mask, other=0.0
        ) - tl.load(
            prediction_ptr + complex_index + 1, mask=mask, other=0.0
        )
        energy = residual_real * residual_real + residual_imag * residual_imag
        tl.store(output_ptr + batch_index, tl.sum(energy, axis=0))


    @triton.jit
    def _complex_residual_noise_kernel(
        data_ptr,
        prediction_ptr,
        old_noise_ptr,
        effective_dof_ptr,
        variance_floor_ptr,
        output_ptr,
        n_samples: tl.constexpr,
        noise_shape: tl.constexpr,
        noise_rate: tl.constexpr,
        damping: tl.constexpr,
        evidence_update: tl.constexpr,
        epsilon: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        sample = tl.arange(0, BLOCK_SIZE)
        mask = sample < n_samples
        index = batch_index * n_samples + sample
        complex_index = 2 * index

        residual_real = tl.load(
            data_ptr + complex_index, mask=mask, other=0.0
        ) - tl.load(
            prediction_ptr + complex_index, mask=mask, other=0.0
        )
        residual_imag = tl.load(
            data_ptr + complex_index + 1, mask=mask, other=0.0
        ) - tl.load(
            prediction_ptr + complex_index + 1, mask=mask, other=0.0
        )
        energy = tl.sum(
            residual_real * residual_real
            + residual_imag * residual_imag,
            axis=0,
        )
        old_noise = tl.load(old_noise_ptr + batch_index)
        effective_dof = tl.load(effective_dof_ptr + batch_index)
        floor = tl.load(variance_floor_ptr + batch_index)
        if evidence_update:
            denominator = tl.maximum(
                n_samples - effective_dof + noise_shape,
                epsilon,
            )
            candidate = (energy + noise_rate) / denominator
        else:
            candidate = (
                energy + old_noise * effective_dof + noise_rate
            ) / (n_samples + noise_shape)
        candidate = tl.maximum(candidate, floor)
        updated = damping * old_noise + (1.0 - damping) * candidate
        tl.store(output_ptr + batch_index, updated)


    @triton.jit
    def _complex_residual_partial_kernel(
        data_ptr,
        prediction_ptr,
        partial_ptr,
        n_samples: tl.constexpr,
        n_blocks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)
        sample = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = sample < n_samples
        index = batch_index * n_samples + sample
        complex_index = 2 * index
        residual_real = tl.load(
            data_ptr + complex_index, mask=mask, other=0.0
        ) - tl.load(
            prediction_ptr + complex_index, mask=mask, other=0.0
        )
        residual_imag = tl.load(
            data_ptr + complex_index + 1, mask=mask, other=0.0
        ) - tl.load(
            prediction_ptr + complex_index + 1, mask=mask, other=0.0
        )
        energy = tl.sum(
            residual_real * residual_real
            + residual_imag * residual_imag,
            axis=0,
        )
        tl.store(
            partial_ptr + batch_index * n_blocks + block_index,
            energy,
        )


    @triton.jit
    def _partial_sum_kernel(
        partial_ptr,
        output_ptr,
        n_blocks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < n_blocks
        values = tl.load(
            partial_ptr + batch_index * n_blocks + block,
            mask=mask,
            other=0.0,
        )
        tl.store(output_ptr + batch_index, tl.sum(values, axis=0))


    @triton.jit
    def _partial_noise_update_kernel(
        partial_ptr,
        old_noise_ptr,
        effective_dof_ptr,
        variance_floor_ptr,
        output_ptr,
        n_blocks: tl.constexpr,
        n_samples: tl.constexpr,
        noise_shape: tl.constexpr,
        noise_rate: tl.constexpr,
        damping: tl.constexpr,
        evidence_update: tl.constexpr,
        epsilon: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < n_blocks
        energy = tl.sum(
            tl.load(
                partial_ptr + batch_index * n_blocks + block,
                mask=mask,
                other=0.0,
            ),
            axis=0,
        )
        old_noise = tl.load(old_noise_ptr + batch_index)
        effective_dof = tl.load(effective_dof_ptr + batch_index)
        floor = tl.load(variance_floor_ptr + batch_index)
        if evidence_update:
            denominator = tl.maximum(
                n_samples - effective_dof + noise_shape,
                epsilon,
            )
            candidate = (energy + noise_rate) / denominator
        else:
            candidate = (
                energy + old_noise * effective_dof + noise_rate
            ) / (n_samples + noise_shape)
        candidate = tl.maximum(candidate, floor)
        updated = damping * old_noise + (1.0 - damping) * candidate
        tl.store(output_ptr + batch_index, updated)


def triton_is_available() -> bool:
    """Whether Triton imported and has not failed during a kernel launch."""
    return (
        triton is not None
        and _TRITON_IMPORT_ERROR is None
        and _TRITON_RUNTIME_ERROR is None
        and torch.cuda.is_available()
    )


def triton_error() -> Exception | None:
    """Return the import or most recent launch error, if any."""
    return _TRITON_RUNTIME_ERROR or _TRITON_IMPORT_ERROR


def _record_runtime_error(exc: Exception) -> None:
    global _TRITON_RUNTIME_ERROR
    _TRITON_RUNTIME_ERROR = exc


def reset_triton_runtime_error() -> None:
    """Clear a recorded launch failure after the caller repairs its runtime.

    Import failures are immutable for the current module import.  Launch
    failures are resettable so one transient driver/cache error does not
    permanently disable Triton for every later plan in the process.
    """
    global _TRITON_RUNTIME_ERROR
    _TRITON_RUNTIME_ERROR = None


def _validate_tensor_contract(
    name: str,
    tensor: Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | None = None,
) -> None:
    """Validate the raw-pointer contract assumed by the Triton kernels."""
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}.")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(tensor.shape)}."
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be C-contiguous.")
    if device is not None and tensor.device != device:
        raise ValueError(
            f"{name} must be on {device}, got {tensor.device}."
        )


def uamp_message_update(
    y: Tensor,
    prediction: Tensor,
    old_message: Tensor,
    data_variance: Tensor,
    noise_variance: Tensor,
    *,
    damping: float,
) -> tuple[Tensor, Tensor]:
    """Fused complex UAMP data-message update and pseudo variance."""
    if triton is None:
        raise RuntimeError("Triton is not importable.") from _TRITON_IMPORT_ERROR
    if y.ndim != 2:
        raise ValueError("y must have shape [B, M].")
    batch_size, n_samples = y.shape
    reference_device = y.device
    _validate_tensor_contract(
        "y",
        y,
        dtype=torch.complex64,
        shape=(batch_size, n_samples),
    )
    for name, tensor in (
        ("prediction", prediction),
        ("old_message", old_message),
    ):
        _validate_tensor_contract(
            name,
            tensor,
            dtype=torch.complex64,
            shape=(batch_size, n_samples),
            device=reference_device,
        )
    for name, tensor in (
        ("data_variance", data_variance),
        ("noise_variance", noise_variance),
    ):
        _validate_tensor_contract(
            name,
            tensor,
            dtype=torch.float32,
            shape=(batch_size,),
            device=reference_device,
        )
    output = torch.empty_like(old_message)
    pseudo_variance = torch.empty(
        old_message.shape[0],
        device=old_message.device,
        dtype=data_variance.dtype,
    )
    grid = (
        batch_size,
        triton.cdiv(n_samples, _MESSAGE_BLOCK),
    )
    try:
        _uamp_message_kernel[grid](
            torch.view_as_real(y),
            torch.view_as_real(prediction),
            torch.view_as_real(old_message),
            data_variance,
            noise_variance,
            torch.view_as_real(output),
            pseudo_variance,
            n_samples=n_samples,
            damping=float(damping),
            BLOCK_SIZE=_MESSAGE_BLOCK,
            num_warps=4,
        )
    except Exception as exc:
        _record_runtime_error(exc)
        raise
    return output, pseudo_variance


def uamp_denoise(
    prior_variance: Tensor,
    pseudo_variance: Tensor,
    lifted_message: Tensor,
    old_mean: Tensor,
    old_variance: Tensor,
    *,
    damping: float,
    tiny: float,
    prior_floor: Tensor | None = None,
    hyper_shape: float = 0.0,
    hyper_rate: float = 0.0,
    prior_damping: float = 0.0,
    prior_maximum: float = 1e12,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Fuse denoising, reductions, and the optional ARD prior update."""
    if triton is None:
        raise RuntimeError("Triton is not importable.") from _TRITON_IMPORT_ERROR
    if prior_variance.ndim != 2:
        raise ValueError("prior_variance must have shape [B, N].")
    batch_size, n_grid = prior_variance.shape
    reference_device = prior_variance.device
    for name, tensor in (
        ("prior_variance", prior_variance),
        ("old_variance", old_variance),
    ):
        _validate_tensor_contract(
            name,
            tensor,
            dtype=torch.float32,
            shape=(batch_size, n_grid),
            device=reference_device,
        )
    _validate_tensor_contract(
        "pseudo_variance",
        pseudo_variance,
        dtype=torch.float32,
        shape=(batch_size,),
        device=reference_device,
    )
    for name, tensor in (
        ("lifted_message", lifted_message),
        ("old_mean", old_mean),
    ):
        _validate_tensor_contract(
            name,
            tensor,
            dtype=torch.complex64,
            shape=(batch_size, n_grid),
            device=reference_device,
        )
    output_mean = torch.empty_like(old_mean)
    output_variance = torch.empty_like(old_variance)
    output_prior = (
        torch.empty_like(prior_variance)
        if prior_floor is not None
        else None
    )
    if prior_floor is not None:
        _validate_tensor_contract(
            "prior_floor",
            prior_floor,
            dtype=torch.float32,
            shape=(batch_size,),
            device=reference_device,
        )
    effective_dof = torch.zeros(
        old_mean.shape[0],
        device=old_mean.device,
        dtype=prior_variance.dtype,
    )
    coefficient_variance_sum = torch.zeros_like(effective_dof)
    grid = (
        batch_size,
        triton.cdiv(n_grid, _ELEMENT_BLOCK),
    )
    try:
        _uamp_denoise_kernel[grid](
            prior_variance,
            pseudo_variance,
            torch.view_as_real(lifted_message),
            torch.view_as_real(old_mean),
            old_variance,
            torch.view_as_real(output_mean),
            output_variance,
            effective_dof,
            coefficient_variance_sum,
            (
                prior_floor
                if prior_floor is not None
                else pseudo_variance
            ),
            (
                output_prior
                if output_prior is not None
                else output_variance
            ),
            n_grid=n_grid,
            damping=float(damping),
            tiny=float(tiny),
            update_prior=prior_floor is not None,
            hyper_shape=float(hyper_shape),
            hyper_rate=float(hyper_rate),
            prior_damping=float(prior_damping),
            prior_maximum=float(prior_maximum),
            BLOCK_SIZE=_ELEMENT_BLOCK,
            num_warps=4,
        )
    except Exception as exc:
        _record_runtime_error(exc)
        raise
    return (
        output_mean,
        output_variance,
        effective_dof,
        coefficient_variance_sum,
        output_prior,
    )


def uamp_prior_update(
    mean: Tensor,
    posterior_variance: Tensor,
    old_prior: Tensor,
    prior_floor: Tensor,
    *,
    hyper_shape: float,
    hyper_rate: float,
    damping: float,
    maximum: float,
) -> Tensor:
    """Fuse the independent-scene variational ARD update."""
    if triton is None:
        raise RuntimeError("Triton is not importable.") from _TRITON_IMPORT_ERROR
    if old_prior.ndim != 2:
        raise ValueError("old_prior must have shape [B, N].")
    batch_size, n_grid = old_prior.shape
    reference_device = old_prior.device
    _validate_tensor_contract(
        "mean",
        mean,
        dtype=torch.complex64,
        shape=(batch_size, n_grid),
        device=reference_device,
    )
    for name, tensor in (
        ("posterior_variance", posterior_variance),
        ("old_prior", old_prior),
    ):
        _validate_tensor_contract(
            name,
            tensor,
            dtype=torch.float32,
            shape=(batch_size, n_grid),
            device=reference_device,
        )
    _validate_tensor_contract(
        "prior_floor",
        prior_floor,
        dtype=torch.float32,
        shape=(batch_size,),
        device=reference_device,
    )
    output = torch.empty_like(old_prior)
    grid = (
        batch_size,
        triton.cdiv(n_grid, _ELEMENT_BLOCK),
    )
    try:
        _uamp_prior_kernel[grid](
            torch.view_as_real(mean),
            posterior_variance,
            old_prior,
            prior_floor,
            output,
            n_grid=n_grid,
            hyper_shape=float(hyper_shape),
            hyper_rate=float(hyper_rate),
            damping=float(damping),
            maximum=float(maximum),
            BLOCK_SIZE=_ELEMENT_BLOCK,
            num_warps=4,
        )
    except Exception as exc:
        _record_runtime_error(exc)
        raise
    return output


def complex_residual_energy(
    data: Tensor,
    prediction: Tensor,
) -> Tensor | None:
    """Return a one-kernel row reduction, or ``None`` for very wide rows."""
    if triton is None:
        raise RuntimeError("Triton is not importable.") from _TRITON_IMPORT_ERROR
    if data.ndim != 2:
        raise ValueError("data must have shape [B, M].")
    batch_size, n_samples = data.shape
    reference_device = data.device
    for name, tensor in (("data", data), ("prediction", prediction)):
        _validate_tensor_contract(
            name,
            tensor,
            dtype=torch.complex64,
            shape=(batch_size, n_samples),
            device=reference_device,
        )
    block_size = triton.next_power_of_2(n_samples)
    output = torch.empty(
        batch_size,
        device=data.device,
        dtype=data.real.dtype,
    )
    try:
        if block_size <= _MAX_REDUCTION_BLOCK:
            _complex_residual_energy_kernel[(batch_size,)](
                torch.view_as_real(data),
                torch.view_as_real(prediction),
                output,
                n_samples=n_samples,
                BLOCK_SIZE=block_size,
                num_warps=8 if block_size >= 2048 else 4,
            )
        else:
            n_blocks = triton.cdiv(n_samples, _REDUCTION_TILE)
            final_block = triton.next_power_of_2(n_blocks)
            if final_block > _MAX_REDUCTION_BLOCK:
                return None
            partial = torch.empty(
                (batch_size, n_blocks),
                device=data.device,
                dtype=data.real.dtype,
            )
            _complex_residual_partial_kernel[(batch_size, n_blocks)](
                torch.view_as_real(data),
                torch.view_as_real(prediction),
                partial,
                n_samples=n_samples,
                n_blocks=n_blocks,
                BLOCK_SIZE=_REDUCTION_TILE,
                num_warps=4,
            )
            _partial_sum_kernel[(batch_size,)](
                partial,
                output,
                n_blocks=n_blocks,
                BLOCK_SIZE=final_block,
                num_warps=4,
            )
    except Exception as exc:
        _record_runtime_error(exc)
        raise
    return output


def uamp_noise_update(
    data: Tensor,
    prediction: Tensor,
    old_noise: Tensor,
    effective_dof: Tensor,
    variance_floor: Tensor,
    *,
    noise_shape: float,
    noise_rate: float,
    damping: float,
    evidence_update: bool,
    epsilon: float,
) -> Tensor | None:
    """Fuse residual reduction, scalar noise update, clipping, and damping."""
    if triton is None:
        raise RuntimeError("Triton is not importable.") from _TRITON_IMPORT_ERROR
    if data.ndim != 2:
        raise ValueError("data must have shape [B, M].")
    batch_size, n_samples = data.shape
    reference_device = data.device
    for name, tensor in (("data", data), ("prediction", prediction)):
        _validate_tensor_contract(
            name,
            tensor,
            dtype=torch.complex64,
            shape=(batch_size, n_samples),
            device=reference_device,
        )
    for name, tensor in (
        ("old_noise", old_noise),
        ("effective_dof", effective_dof),
        ("variance_floor", variance_floor),
    ):
        _validate_tensor_contract(
            name,
            tensor,
            dtype=torch.float32,
            shape=(batch_size,),
            device=reference_device,
        )
    block_size = triton.next_power_of_2(n_samples)
    output = torch.empty_like(old_noise)
    try:
        if block_size <= _MAX_REDUCTION_BLOCK:
            _complex_residual_noise_kernel[(batch_size,)](
                torch.view_as_real(data),
                torch.view_as_real(prediction),
                old_noise,
                effective_dof,
                variance_floor,
                output,
                n_samples=n_samples,
                noise_shape=float(noise_shape),
                noise_rate=float(noise_rate),
                damping=float(damping),
                evidence_update=bool(evidence_update),
                epsilon=float(epsilon),
                BLOCK_SIZE=block_size,
                num_warps=8 if block_size >= 2048 else 4,
            )
        else:
            n_blocks = triton.cdiv(n_samples, _REDUCTION_TILE)
            final_block = triton.next_power_of_2(n_blocks)
            if final_block > _MAX_REDUCTION_BLOCK:
                return None
            partial = torch.empty(
                (batch_size, n_blocks),
                device=data.device,
                dtype=data.real.dtype,
            )
            _complex_residual_partial_kernel[(batch_size, n_blocks)](
                torch.view_as_real(data),
                torch.view_as_real(prediction),
                partial,
                n_samples=n_samples,
                n_blocks=n_blocks,
                BLOCK_SIZE=_REDUCTION_TILE,
                num_warps=4,
            )
            _partial_noise_update_kernel[(batch_size,)](
                partial,
                old_noise,
                effective_dof,
                variance_floor,
                output,
                n_blocks=n_blocks,
                n_samples=n_samples,
                noise_shape=float(noise_shape),
                noise_rate=float(noise_rate),
                damping=float(damping),
                evidence_update=bool(evidence_update),
                epsilon=float(epsilon),
                BLOCK_SIZE=final_block,
                num_warps=4,
            )
    except Exception as exc:
        _record_runtime_error(exc)
        raise
    return output
