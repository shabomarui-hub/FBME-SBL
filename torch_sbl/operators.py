from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from torch import Tensor


def _next_smooth_length(minimum: int) -> int:
    """Return the first length >= minimum with only 2/3/5/7 prime factors."""
    candidate = int(minimum)
    while True:
        remainder = candidate
        for factor in (2, 3, 5, 7):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


class PrefixFourierOperator:
    """Implicit 1-D/2-D DFT dictionary sampled from its leading block.

    Internally, coefficient and data arrays have shapes ``[B, N, R]`` and
    ``[B, M, R]``.  ``B`` is the batch size and ``R`` is the number of
    right-hand sides.  The DFT is deliberately unnormalised so that the
    convention agrees with MATLAB's ``dftmtx`` and ``fft``.
    """

    def __init__(
        self,
        grid_shape: Sequence[int],
        sample_shape: Sequence[int],
        *,
        device: torch.device,
        complex_dtype: torch.dtype,
        observation_mask: Tensor | None = None,
    ) -> None:
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.sample_shape = tuple(int(v) for v in sample_shape)
        if len(self.grid_shape) not in (1, 2):
            raise ValueError("Only 1-D and 2-D Fourier grids are supported.")
        if len(self.grid_shape) != len(self.sample_shape):
            raise ValueError("grid_shape and sample_shape must have the same rank.")
        if any(m <= 0 or m > n for m, n in zip(self.sample_shape, self.grid_shape)):
            raise ValueError("Each sample dimension must satisfy 0 < M <= N.")

        self.device = device
        self.complex_dtype = complex_dtype
        self.real_dtype = (
            torch.float64 if complex_dtype == torch.complex128 else torch.float32
        )
        self.n_grid = prod(self.grid_shape)
        self.full_n_samples = prod(self.sample_shape)
        if observation_mask is None:
            self.observation_mask = None
            self.observed_indices = None
            self.n_samples = self.full_n_samples
        else:
            mask = torch.as_tensor(
                observation_mask, device=device, dtype=torch.bool
            )
            if tuple(mask.shape) != self.sample_shape:
                raise ValueError(
                    "observation_mask must have shape "
                    f"{self.sample_shape}, got {tuple(mask.shape)}."
                )
            indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                raise ValueError("observation_mask must retain at least one sample.")
            self.observation_mask = mask
            self.observed_indices = indices
            self.n_samples = int(indices.numel())
        self._fft_dims = tuple(range(1, 1 + len(self.grid_shape)))
        self._crop = (slice(None),) + tuple(
            slice(0, m) for m in self.sample_shape
        ) + (slice(None),)

        # Dense O(M^2) helpers are deliberately lazy.  Large covariance-free
        # problems must not allocate an M-by-M lag table or identity matrix.
        axes = [
            torch.arange(m, device=device, dtype=torch.long)
            for m in self.sample_shape
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        coordinates = torch.stack([v.reshape(-1) for v in mesh], dim=1)
        if self.observed_indices is not None:
            coordinates = coordinates.index_select(0, self.observed_indices)
        self.sample_coordinates = coordinates
        self.sample_coordinates_real = coordinates.to(self.real_dtype)
        self.grid_scale = torch.tensor(
            self.grid_shape,
            device=device,
            dtype=self.real_dtype,
        )
        self._lag_index: Tensor | None = None
        self._eye: Tensor | None = None

        # Exact Toeplitz/BTTB matvec through a BCCB embedding.  Positive and
        # negative lags occupy the beginning and end of each padded axis.
        self.embedding_shape = tuple(
            _next_smooth_length(2 * m - 1) for m in self.sample_shape
        )
        embedding_positions = []
        signed_lags = []
        for m, length in zip(self.sample_shape, self.embedding_shape):
            embedding_positions.append(
                torch.cat(
                    (
                        torch.arange(m, device=device, dtype=torch.long),
                        torch.arange(
                            length - m + 1,
                            length,
                            device=device,
                            dtype=torch.long,
                        ),
                    )
                )
            )
            signed_lags.append(
                torch.cat(
                    (
                        torch.arange(m, device=device, dtype=torch.long),
                        torch.arange(
                            -(m - 1), 0, device=device, dtype=torch.long
                        ),
                    )
                )
            )
        position_mesh = torch.meshgrid(*embedding_positions, indexing="ij")
        lag_mesh = torch.meshgrid(*signed_lags, indexing="ij")
        if len(self.grid_shape) == 1:
            self.embedding_positions_flat = position_mesh[0].reshape(-1)
            self.embedding_lag_indices_flat = torch.remainder(
                lag_mesh[0], self.grid_shape[0]
            ).reshape(-1)
        else:
            self.embedding_positions_flat = (
                position_mesh[0] * self.embedding_shape[1] + position_mesh[1]
            ).reshape(-1)
            self.embedding_lag_indices_flat = (
                torch.remainder(lag_mesh[0], self.grid_shape[0])
                * self.grid_shape[1]
                + torch.remainder(lag_mesh[1], self.grid_shape[1])
            ).reshape(-1)

        # Signed lags used by a Strang BCCB/circulant approximation to Q.
        circulant_axes = []
        for n, m in zip(self.grid_shape, self.sample_shape):
            k = torch.arange(m, device=device, dtype=torch.long)
            circulant_axes.append(torch.where(k <= m // 2, k, n - (m - k)))
        circulant_mesh = torch.meshgrid(*circulant_axes, indexing="ij")
        if len(self.grid_shape) == 1:
            self.circulant_index = circulant_mesh[0]
        else:
            self.circulant_index = (
                circulant_mesh[0] * self.grid_shape[1] + circulant_mesh[1]
            )

    @property
    def eye(self) -> Tensor:
        if self._eye is None:
            self._eye = torch.eye(
                self.n_samples,
                device=self.device,
                dtype=self.complex_dtype,
            )
        return self._eye

    @property
    def lipschitz_upper_bound(self) -> float:
        """Exact ``||H||_2^2`` for a subset of unnormalised DFT rows."""
        return float(self.n_grid)

    @property
    def column_norms(self) -> Tensor:
        """Exact Euclidean norms of all sampled Fourier atoms."""
        return torch.full(
            (self.n_grid,),
            float(self.n_samples) ** 0.5,
            device=self.device,
            dtype=self.real_dtype,
        )

    def materialize_columns(self, flat_indices: Tensor) -> Tensor:
        """Return selected Fourier atoms for generic active-set solvers."""
        return self.fourier_columns(flat_indices)

    @property
    def lag_index(self) -> Tensor:
        if self._lag_index is None:
            coordinates = self.sample_coordinates
            lag = coordinates[:, None, :] - coordinates[None, :, :]
            grid = torch.tensor(
                self.grid_shape, device=self.device, dtype=torch.long
            )
            lag = torch.remainder(lag, grid)
            if len(self.grid_shape) == 1:
                self._lag_index = lag[..., 0]
            else:
                self._lag_index = (
                    lag[..., 0] * self.grid_shape[1] + lag[..., 1]
                )
        return self._lag_index

    def forward(self, coefficients: Tensor) -> Tensor:
        """Apply H using FFT: ``[B,N,R] -> [B,M,R]``."""
        self._check_vector_shape(coefficients, self.n_grid, "coefficients")
        batch, _, rhs = coefficients.shape
        field = coefficients.reshape(batch, *self.grid_shape, rhs)
        spectrum = torch.fft.fftn(field, dim=self._fft_dims)
        cropped = spectrum[self._crop].reshape(
            batch, self.full_n_samples, rhs
        )
        return self.restrict(cropped)

    def adjoint(self, data: Tensor) -> Tensor:
        """Apply H^H using FFT-native zero extension."""
        self._check_vector_shape(data, self.n_samples, "data")
        batch, _, rhs = data.shape
        full_data = self.expand(data)
        field = full_data.reshape(
            batch, *self.sample_shape, rhs
        )
        coefficients = (
            torch.fft.ifftn(
                field,
                s=self.grid_shape,
                dim=self._fft_dims,
            )
            * self.n_grid
        )
        return coefficients.reshape(batch, self.n_grid, rhs)

    def forward_flat(self, coefficients: Tensor) -> Tensor:
        """Apply H to ``[B, N]`` without a singleton RHS dimension."""
        if coefficients.ndim != 2 or coefficients.shape[1] != self.n_grid:
            raise ValueError(
                f"coefficients must have shape [B, {self.n_grid}]."
            )
        batch = coefficients.shape[0]
        field = coefficients.reshape(batch, *self.grid_shape)
        dimensions = tuple(range(1, 1 + len(self.grid_shape)))
        spectrum = torch.fft.fftn(field, dim=dimensions)
        crop = (slice(None),) + tuple(
            slice(0, sample) for sample in self.sample_shape
        )
        full_data = spectrum[crop].reshape(batch, self.full_n_samples)
        if self.observed_indices is None:
            return full_data
        return full_data.index_select(1, self.observed_indices)

    def adjoint_flat(self, data: Tensor) -> Tensor:
        """Apply H^H to ``[B, M]`` using FFT-native end padding.

        Passing ``s=grid_shape`` lets cuFFT perform the zero extension and
        avoids materialising and filling a ``[B, *grid_shape, 1]`` buffer.
        """
        if data.ndim != 2 or data.shape[1] != self.n_samples:
            raise ValueError(f"data must have shape [B, {self.n_samples}].")
        batch = data.shape[0]
        if self.observed_indices is None:
            full_data = data
        else:
            full_data = torch.zeros(
                (batch, self.full_n_samples),
                device=self.device,
                dtype=self.complex_dtype,
            )
            full_data.index_copy_(1, self.observed_indices, data)
        field = full_data.reshape(batch, *self.sample_shape)
        dimensions = tuple(range(1, 1 + len(self.grid_shape)))
        coefficients = torch.fft.ifftn(
            field,
            s=self.grid_shape,
            dim=dimensions,
        )
        return (coefficients * self.n_grid).reshape(batch, self.n_grid)

    def covariance(self, prior_variance: Tensor, noise_variance: Tensor) -> Tensor:
        """Build exact Q = noise*I + H*Gamma*H^H from Fourier lags.

        No ``M x N`` dictionary is materialised.  For 1-D Q is Toeplitz; for
        2-D it is block Toeplitz with Toeplitz blocks (BTTB).
        """
        if prior_variance.ndim != 2 or prior_variance.shape[1] != self.n_grid:
            raise ValueError("prior_variance must have shape [B, N].")
        batch = prior_variance.shape[0]
        field = prior_variance.to(self.complex_dtype).reshape(
            batch, *self.grid_shape
        )
        lags = torch.fft.fftn(field, dim=tuple(range(1, field.ndim)))
        flat_lags = lags.reshape(batch, self.n_grid)
        q = flat_lags[:, self.lag_index]
        return q + noise_variance[:, None, None] * self.eye

    def toeplitz_spectrum(
        self,
        prior_variance: Tensor,
        noise_variance: Tensor,
    ) -> Tensor:
        """Build and FFT the exact BTTB covariance embedding once per EM step."""
        if prior_variance.ndim != 2 or prior_variance.shape[1] != self.n_grid:
            raise ValueError("prior_variance must have shape [B, N].")
        batch = prior_variance.shape[0]
        field = prior_variance.to(self.complex_dtype).reshape(
            batch, *self.grid_shape
        )
        covariance_lags = torch.fft.fftn(
            field, dim=tuple(range(1, field.ndim))
        ).reshape(batch, self.n_grid)
        kernel_flat = torch.zeros(
            (batch, prod(self.embedding_shape)),
            device=self.device,
            dtype=self.complex_dtype,
        )
        kernel_flat[:, self.embedding_positions_flat] = covariance_lags[
            :, self.embedding_lag_indices_flat
        ]
        kernel_flat[:, 0] += noise_variance
        kernel = kernel_flat.reshape(batch, *self.embedding_shape)
        return torch.fft.fftn(
            kernel, dim=tuple(range(1, kernel.ndim))
        )

    def toeplitz_matvec(self, data: Tensor, spectrum: Tensor) -> Tensor:
        """Apply an exact Toeplitz/BTTB covariance using its cached spectrum."""
        self._check_vector_shape(data, self.n_samples, "data")
        batch, _, rhs = data.shape
        expected = (batch, *self.embedding_shape)
        if spectrum.shape != expected:
            raise ValueError(
                f"spectrum must have shape {expected}, got {tuple(spectrum.shape)}."
            )
        sample_crop = (slice(None),) + tuple(
            slice(0, m) for m in self.sample_shape
        ) + (slice(None),)
        full_data = self.expand(data)
        field = full_data.reshape(
            batch, *self.sample_shape, rhs
        )
        transformed = torch.fft.fftn(
            field,
            s=self.embedding_shape,
            dim=self._fft_dims,
        )
        product = torch.fft.ifftn(
            transformed * spectrum.unsqueeze(-1), dim=self._fft_dims
        )
        full_product = product[sample_crop].reshape(
            batch, self.full_n_samples, rhs
        )
        return self.restrict(full_product)

    def matvec(
        self,
        data: Tensor,
        prior_variance: Tensor,
        noise_variance: Tensor,
        *,
        toeplitz_spectrum: Tensor | None = None,
    ) -> Tensor:
        """Matrix-free product with Q for PCG."""
        if toeplitz_spectrum is not None:
            return self.toeplitz_matvec(data, toeplitz_spectrum)
        return (
            noise_variance[:, None, None] * data
            + self.forward(prior_variance[:, :, None] * self.adjoint(data))
        )

    def circulant_preconditioner_spectrum(
        self,
        prior_variance: Tensor,
        noise_variance: Tensor,
    ) -> Tensor:
        """Build a stable Strang circulant/BCCB preconditioner spectrum."""
        batch = prior_variance.shape[0]
        gamma_field = prior_variance.to(self.complex_dtype).reshape(
            batch, *self.grid_shape
        )
        covariance_lags = torch.fft.fftn(
            gamma_field, dim=tuple(range(1, gamma_field.ndim))
        ).reshape(batch, self.n_grid)
        circulant_column = covariance_lags[:, self.circulant_index]
        eigenvalues = torch.fft.fftn(
            circulant_column,
            dim=tuple(range(1, circulant_column.ndim)),
        ).real + noise_variance.reshape(batch, *([1] * len(self.sample_shape)))
        scale_floor = torch.finfo(self.real_dtype).eps * eigenvalues.abs().amax(
            dim=tuple(range(1, eigenvalues.ndim)), keepdim=True
        )
        noise_floor = noise_variance.reshape(
            batch, *([1] * len(self.sample_shape))
        )
        return eigenvalues.clamp_min(torch.maximum(scale_floor, noise_floor))

    def apply_circulant_preconditioner(
        self,
        data: Tensor,
        spectrum: Tensor,
    ) -> Tensor:
        """Apply a cached Strang circulant/BCCB approximation of Q^-1."""
        self._check_vector_shape(data, self.n_samples, "data")
        batch, _, rhs = data.shape
        field = self.expand(data).reshape(
            batch, *self.sample_shape, rhs
        )
        transformed = torch.fft.fftn(field, dim=self._fft_dims)
        solved = torch.fft.ifftn(
            transformed / spectrum.unsqueeze(-1), dim=self._fft_dims
        )
        return self.restrict(
            solved.reshape(batch, self.full_n_samples, rhs)
        )

    def posterior_data_diagonal(self, q_inverse: Tensor) -> Tensor:
        """Compute diag(H^H Q^-1 H) exactly using inverse diagonal sums.

        Accumulating entries that share the same Fourier lag reduces an
        otherwise O(M^2*N) diagonal computation to O(M^2 + N log N).
        """
        if q_inverse.ndim != 3 or q_inverse.shape[1:] != (
            self.n_samples,
            self.n_samples,
        ):
            raise ValueError("q_inverse must have shape [B, M, M].")
        batch = q_inverse.shape[0]
        lag_sums = torch.zeros(
            (batch, self.n_grid),
            device=self.device,
            dtype=self.complex_dtype,
        )
        indices = self.lag_index.reshape(-1).unsqueeze(0).expand(batch, -1)
        lag_sums.scatter_add_(1, indices, q_inverse.reshape(batch, -1))
        field = lag_sums.reshape(batch, *self.grid_shape)
        diagonal = self.n_grid * torch.fft.ifftn(
            field, dim=tuple(range(1, field.ndim))
        )
        # The exact result is real and nonnegative; discard roundoff only.
        return diagonal.real.reshape(batch, self.n_grid)

    def fourier_columns(self, flat_indices: Tensor) -> Tensor:
        """Materialise only selected DFT columns for a low-rank preconditioner."""
        if flat_indices.ndim != 2:
            raise ValueError("flat_indices must have shape [B, K].")
        if len(self.grid_shape) == 1:
            grid_coordinates = flat_indices.unsqueeze(-1)
        else:
            grid_coordinates = torch.stack(
                (
                    torch.div(
                        flat_indices, self.grid_shape[1], rounding_mode="floor"
                    ),
                    torch.remainder(flat_indices, self.grid_shape[1]),
                ),
                dim=-1,
            )
        grid_coordinates = grid_coordinates.to(self.real_dtype)
        phase = torch.einsum(
            "md,bkd->bmk",
            self.sample_coordinates_real / self.grid_scale,
            grid_coordinates,
        )
        return torch.exp((-2j * torch.pi * phase).to(self.complex_dtype))

    @property
    def is_complete(self) -> bool:
        """Whether every point in the rectangular sample block is observed."""
        return self.observed_indices is None

    def restrict(self, full_data: Tensor) -> Tensor:
        """Gather observed entries from ``[B, prod(sample_shape), R]``."""
        if full_data.ndim != 3 or full_data.shape[1] != self.full_n_samples:
            raise ValueError(
                "full_data must have shape "
                f"[B, {self.full_n_samples}, R]."
            )
        if self.observed_indices is None:
            return full_data
        return full_data.index_select(1, self.observed_indices)

    def expand(self, observed_data: Tensor) -> Tensor:
        """Scatter observed entries into the complete rectangular sample block."""
        self._check_vector_shape(observed_data, self.n_samples, "observed_data")
        if self.observed_indices is None:
            return observed_data
        batch, _, rhs = observed_data.shape
        full = torch.zeros(
            (batch, self.full_n_samples, rhs),
            device=self.device,
            dtype=self.complex_dtype,
        )
        full.index_copy_(1, self.observed_indices, observed_data)
        return full

    @staticmethod
    def _check_vector_shape(value: Tensor, length: int, name: str) -> None:
        if value.ndim != 3 or value.shape[1] != length:
            raise ValueError(f"{name} must have shape [B, {length}, R].")
