"""Independent complex64 implementation of the published 2D-AFCIFSBL updates.

The implementation follows Wang et al., Remote Sensing 16(14), 2521 (2024),
Eqs. (41)--(45), (47), and (52).  It is kept in the isolated comparison copy;
the original FBME solver and historical result directories are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class AFCIFSBLResult:
    x: torch.Tensor
    posterior_variance: torch.Tensor
    precision: torch.Tensor
    noise_variance: torch.Tensor
    iterations: int
    relative_change: float
    phase_error: torch.Tensor


def _fourier_factors(side: Tuple[int, int], sample: Tuple[int, int], *,
                     device: torch.device, dtype: torch.dtype):
    """Return unnormalised partial DFT factors matching torch.fft.fft2."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    p, q = int(side[0]), int(side[1])
    m, h = int(sample[0]), int(sample[1])
    rows_r = torch.arange(m, device=device, dtype=real_dtype)[:, None]
    cols_r = torch.arange(p, device=device, dtype=real_dtype)[None, :]
    rows_a = torch.arange(h, device=device, dtype=real_dtype)[:, None]
    cols_a = torch.arange(q, device=device, dtype=real_dtype)[None, :]
    two_pi = torch.tensor(2.0 * torch.pi, device=device, dtype=real_dtype)
    fr = torch.exp((-1j * two_pi * rows_r * cols_r / p)).to(dtype)
    fa = torch.exp((-1j * two_pi * rows_a * cols_a / q)).to(dtype)
    return fr, fa


@torch.no_grad()
def afcifsbl_2d(
    data: torch.Tensor,
    grid_shape: Tuple[int, int] | list[int],
    *,
    outer_iterations: int = 30,
    tolerance: float = 1e-4,
    a: float = 1e-6,
    b: float = 1e-6,
    c: float = 1e-6,
    d: float = 1e-6,
) -> AFCIFSBLResult:
    """Run the published matrix-form 2D-AFCIFSBL updates.

    ``data`` is an ``M x H`` complex tensor containing the partial 2-D Fourier
    observation.  The current project uses the first half of each FFT axis,
    so this routine builds the matching partial DFT factors and returns a
    ``P x Q`` image estimate.
    """
    if data.ndim != 2 or not data.is_complex():
        raise ValueError("2D-AFCIFSBL expects a 2-D complex observation matrix")
    p, q = (int(grid_shape[0]), int(grid_shape[1]))
    m, h = int(data.shape[0]), int(data.shape[1])
    if m > p or h > q:
        raise ValueError("observation shape cannot exceed reconstruction grid")
    dtype = data.dtype
    device = data.device
    fr, fa = _fourier_factors((p, q), (m, h), device=device, dtype=dtype)
    fr_h = fr.conj().transpose(0, 1)
    fa_h = fa.conj().transpose(0, 1)

    # T is slightly larger than the Lipschitz constant 2*lambda_max(F_r^H F_r)
    # *lambda_max(F_a^H F_a).  For an unnormalised partial DFT, P*Q is a valid
    # upper bound, avoiding a dense eigendecomposition at 512x512.
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    T = torch.tensor(2.0 * p * q + 1e-5, device=device, dtype=real_dtype)
    ones = torch.ones((p, q), device=device, dtype=real_dtype)
    delta = torch.tensor(1.0, device=device, dtype=real_dtype)
    D = ones.clone()
    E = torch.ones_like(data)

    # Eq. (1) in Algorithm 1: X^(0) = F_r^H S F_a^*.
    Z = fr_h @ data @ fa.conj()
    X = Z.clone()
    Sigma = (T * delta / 2.0 + D).reciprocal()
    rel = float("inf")
    completed = 0

    for iteration in range(1, int(outer_iterations) + 1):
        # Eqs. (40)--(42): VBE update for the image.
        forward_z = fr @ Z @ fa.transpose(0, 1)
        residual_for_grad = E.conj() * data - forward_z
        grad = fr_h @ residual_for_grad @ fa.conj()
        Sigma = (T * delta / 2.0 + D).reciprocal()
        U = delta * (T * Z / 2.0 + grad) * Sigma

        # Eqs. (43)--(45): noise precision and hyper-precision updates.
        pred_z = E * forward_z
        mismatch = (data - pred_z).abs().square().sum().real
        relaxed_residual = forward_z - E.conj() * data
        relaxed_grad = fr_h @ relaxed_residual @ fa.conj()
        cross = 2.0 * ((U - Z).conj() * relaxed_grad).real.sum()
        g = mismatch + cross + (U - Z).abs().square().sum().real
        g = g + (T / 2.0) * Sigma.sum().real
        g = g.clamp_min(torch.finfo(real_dtype).tiny)
        delta = ((a + m * h) / (b + g)).clamp_min(torch.finfo(real_dtype).tiny)
        D = (c + 1.0) / (d + U.abs().square() + Sigma).clamp_min(
            torch.finfo(real_dtype).tiny
        )

        # Eq. (47), followed by Eq. (52): VBM image and MLE phase update.
        X_prev = X
        Z = U
        X = Z
        forward_x = fr @ X @ fa.transpose(0, 1)
        E = torch.exp(1j * torch.angle(data * forward_x.conj()))

        denom = X_prev.abs().square().sum().real.clamp_min(
            torch.finfo(real_dtype).tiny
        )
        rel_tensor = (X - X_prev).abs().square().sum().real / denom
        rel = float(rel_tensor.item())
        completed = iteration
        if rel <= tolerance:
            break

    return AFCIFSBLResult(
        x=X,
        posterior_variance=Sigma,
        precision=D,
        noise_variance=(1.0 / delta).reshape(()),
        iterations=completed,
        relative_change=rel,
        phase_error=E,
    )

