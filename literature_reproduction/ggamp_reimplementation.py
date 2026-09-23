"""Paper-based complex GGAMP-SBL reimplementation.

This follows Algorithm 1 of Al-Shoukairi, Schniter, and Rao (2018).
The published equations are real-valued in the presentation; this module
uses Hermitian Fourier adjoints and ``abs(x)**2`` in the complex M-step.
That adaptation is recorded in the experiment metadata and is not claimed
to be an author-supplied implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import torch
from torch import Tensor

from torch_sbl.operators import PrefixFourierOperator


@dataclass(slots=True)
class GGAMPResult:
    x: Tensor
    posterior_variance: Tensor
    precision: Tensor
    prior_precision: Tensor
    noise_variance: Tensor
    iterations: int
    message_iterations: int
    damping: float
    arithmetic_dtype: str


def ggamp_sbl(
    data: Tensor,
    grid_shape: Sequence[int],
    *,
    noise_variance: Tensor | float,
    outer_iterations: int = 15,
    max_messages: int = 15,
    damping: float = 0.35,
    noise_scale: float = 3.0,
    update_noise: bool = False,
    tolerance: float = 1e-5,
    mean_remove: bool = False,
) -> GGAMPResult:
    """Run a fixed-budget complex SMV GGAMP-SBL adaptation.

    ``data`` may be ``[m1,m2]`` or ``[B,m1,m2]``.  The current project uses
    the leading partial 2-D Fourier block, so the implicit operator provides
    the matrix-vector products and the exact ``abs(A)**2`` reductions.

    The published paper treats the noise variance as known for its main
    GGAMP-SBL experiments and reports using ``3*sigma**2``.  Accordingly,
    ``update_noise=False`` (the default) keeps that oracle variance fixed.
    Set ``update_noise=True`` only for the optional EM noise-update variant;
    this is recorded separately because Eq. (13) is explicitly described as
    potentially inaccurate in the paper.
    """
    grid_shape = tuple(int(v) for v in grid_shape)
    if data.ndim == len(grid_shape):
        data = data.unsqueeze(0)
    if data.ndim != len(grid_shape) + 1:
        raise ValueError("data rank does not match grid_shape")
    batch = data.shape[0]
    sample_shape = tuple(int(v) for v in data.shape[1:])
    if len(sample_shape) != len(grid_shape):
        raise ValueError("data and grid_shape must have equal rank")
    if data.dtype not in (torch.complex64, torch.complex128):
        data = data.to(torch.complex64)
    device = data.device
    dtype = data.dtype
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    n_grid = prod(grid_shape)
    n_samples = prod(sample_shape)
    operator = PrefixFourierOperator(
        grid_shape,
        sample_shape,
        device=device,
        complex_dtype=dtype,
    )
    y = data.reshape(batch, n_samples)
    # The paper uses a known-noise experiment and reports that 3*sigma^2
    # was used for the SBL/GGAMP family. Keep that explicit here.
    sigma = torch.as_tensor(
        noise_variance, device=device, dtype=real_dtype
    ).reshape(1).expand(batch).clone()
    sigma = (noise_scale * sigma).clamp_min(torch.finfo(real_dtype).tiny)

    # Algorithm 1 initialization: gamma=1, x=s=0, tau_x=1.
    gamma = torch.ones((batch, n_grid), device=device, dtype=real_dtype)
    xhat = torch.zeros((batch, n_grid), device=device, dtype=dtype)
    stilde = torch.zeros((batch, n_samples), device=device, dtype=dtype)
    tau_x = torch.ones((batch, n_grid), device=device, dtype=real_dtype)
    posterior_variance = tau_x.clone()
    total_messages = 0
    outer_done = 0

    # For the partial Fourier matrix, each entry has unit magnitude, hence
    # |A|^2 @ tau_x and |A|^2.T @ tau_s are scalar reductions broadcast over
    # the corresponding axis. This is exactly the S matrix operation in the
    # published algorithm without materialising M-by-N.
    for outer in range(int(outer_iterations)):
        x_old_outer = xhat
        gamma_old = gamma
        x_inner = xhat
        s = stilde
        tau_inner = tau_x
        for _ in range(int(max_messages)):
            total_messages += 1
            tau_p = 1.0 / tau_inner.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(real_dtype).tiny
            )
            # Paper notation: p = s + tau_p * A x.
            p = s + tau_p.to(dtype) * operator.forward_flat(x_inner)
            tau_s = tau_p / (1.0 + sigma[:, None] * tau_p)
            s_new = (p / tau_p.to(dtype) - y) / (
                sigma[:, None].to(dtype) + 1.0 / tau_p.to(dtype)
            )
            s_new = (1.0 - damping) * s + damping * s_new
            tau_r = 1.0 / (
                tau_s.sum(dim=1, keepdim=True).clamp_min(
                    torch.finfo(real_dtype).tiny
                )
            )
            r = x_inner - tau_r.to(dtype) * operator.adjoint_flat(s_new)
            tau_next = tau_r * gamma / (gamma + tau_r)
            x_next = gamma.to(dtype) * r / (gamma + tau_r).to(dtype)
            x_next = (1.0 - damping) * x_inner + damping * x_next
            denominator = x_next.abs().square().sum(dim=1).sqrt().clamp_min(
                torch.finfo(real_dtype).tiny
            )
            delta = (x_next - x_inner).abs().square().sum(dim=1).sqrt() / denominator
            x_inner, s, tau_inner = x_next, s_new, tau_next
            if bool(torch.all(delta < tolerance)):
                break

        xhat = x_inner
        stilde = s
        tau_x = tau_inner
        posterior_variance = tau_x
        # Complex extension of the published real-valued M-step.
        gamma = xhat.abs().square() + tau_x
        residual = y - operator.forward_flat(xhat)
        correction = (1.0 - tau_x / gamma_old).sum(dim=1)
        if update_noise:
            sigma = (
                residual.abs().square().sum(dim=1) + sigma * correction
            ) / float(n_samples)
            sigma = sigma.clamp_min(torch.finfo(real_dtype).tiny)
        outer_done = outer + 1
        denom = xhat.abs().square().sum(dim=1).sqrt().clamp_min(
            torch.finfo(real_dtype).tiny
        )
        if bool(torch.all((xhat - x_old_outer).abs().square().sum(dim=1).sqrt() / denom < tolerance)):
            break

    return GGAMPResult(
        x=xhat.reshape(batch, *grid_shape).squeeze(0),
        posterior_variance=posterior_variance.reshape(batch, *grid_shape).squeeze(0),
        precision=gamma.reciprocal().reshape(batch, *grid_shape).squeeze(0),
        prior_precision=gamma.reciprocal().reshape(batch, *grid_shape).squeeze(0),
        noise_variance=sigma,
        iterations=outer_done,
        message_iterations=total_messages,
        damping=float(damping),
        arithmetic_dtype=str(dtype),
    )
