# FBME-SBL

Fourier-structured sparse Bayesian learning in PyTorch, with FBME-Fast,
FBME-Stable, and independent implementations of selected SBL methods.

This repository contains the FBME-SBL solvers, CUDA Graph execution wrappers, and independent implementations of selected SBL methods. It includes installation, usage, and source documentation, together with an archived RTX 4090 example for regenerating the reported Table 2 and Fig. 3 from saved results. It does not include manuscripts, internal project records, or the complete original experimental environment. The numerical solver implementations are preserved from the project code.

## Available methods

| Method | Entry point | Execution requirements |
|---|---|---|
| FBME-Fast | `torch_sbl_fastest_stable.make_fastest_plan` | CUDA, complex64, Triton, fixed-shape CUDA Graph |
| FBME-Stable | `torch_sbl_fastest_stable.make_stable_plan` | CUDA, complex64, Triton, fixed-shape CUDA Graph |
| General FBME solver | `torch_sbl.sbl_2d` / `fast_sbl` | Computation and execution configured through `SBLConfig` |
| Dense EM-SBL | `literature_reproduction.dense_em_sbl` | PyTorch on the input tensor's CPU or CUDA device |
| UAMP-SBL (paper updates) | `literature_reproduction.paper_uamp_sbl` | CUDA; a single scene |
| UAMP (engineering variant) | `literature_reproduction.uamp_sbl` | `backend="torch"` or `"triton"`; the latter requires CUDA and Triton |
| CoFEM (complex Fourier adaptation) | `literature_reproduction.complex_fourier_cofem` | CUDA; a noise variance supplied by the caller |
| GGAMP-SBL (complex adaptation) | `literature_reproduction.ggamp_sbl` | PyTorch; a noise variance supplied by the caller |
| 2D-AFCIFSBL (partial Fourier adaptation) | `literature_reproduction.afcifsbl_2d` | PyTorch; a single two-dimensional scene |

Dense EM uses the same implementation on CPU and GPU. Likewise, the Torch and Triton paths of the engineering UAMP variant are execution options, not separate methods from the literature. This repository does not contain an independent FFD-SBL implementation. See [Sources and citations](SOURCES.md) for differences between these implementations and the cited methods.

## Installation

Python 3.10 or later is required. Dependency ranges follow the project's PyTorch 2.6 and Triton 3.2 environment. Linux or WSL2 is recommended for CUDA execution. First install PyTorch 2.6.0 with a build compatible with your driver, following the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/). Fast and Stable require a CUDA-enabled build.

Clone the repository:

```bash
git clone https://github.com/ybb3663/FBME-SBL.git
cd FBME-SBL
```

From the repository root, install the package:

```bash
python -m pip install .
```

For Fast, Stable, or the Triton UAMP variant, install the optional Triton dependency:

```bash
python -m pip install ".[triton]"
```

On native Windows, this optional dependency uses `triton-windows`. Providing that installation option does not mean that all native Windows driver combinations have been validated. CPU-only systems can use methods such as Dense EM, but cannot run Fast or Stable. The algorithm packages do not require NumPy, SciPy, or dataset packages to import.

The following command checks installation and package exports without running a solver:

```bash
python -c "import torch_sbl, torch_sbl_fastest_stable, literature_reproduction; print(torch_sbl.__version__)"
```

## Input model

- For a single two-dimensional scene, `data` is a complex observation tensor of shape `[M1, M2]`. The reconstruction grid is specified by `grid_shape=(N1, N2)`, with `0 < Mi <= Ni`.
- The default model uses a **prefix observation block of the unnormalized two-dimensional DFT**: `data = fft2(x)[:M1, :M2] + noise`. DFT ordering, amplitude scaling, and phase conventions must agree. A grayscale image, magnitude image, or arbitrary raw radar file is not a direct substitute for these complex observations.
- `complex64` represents each complex value using two float32 components; it does not specify the grid size. Fast and Stable require complex64. Other interfaces may support different types; see their source code.
- The general `sbl_2d` interface supports batched input and some mask configurations. Batching and mask support differ across the literature implementations, so their interfaces are not interchangeable in every configuration.
- For CoFEM and GGAMP, `noise_variance` is the complex noise power `E[|noise|²]`, in the same amplitude scale as the observations. The caller must supply or estimate it; do not pass an SNR value in decibels directly.

## FBME-Fast and FBME-Stable

The following functions accept observations prepared by the caller. They do not load or generate data.

```python
import torch
from torch_sbl_fastest_stable import make_fastest_plan, make_stable_plan


def build_fbme_plans(data: torch.Tensor, grid_shape: tuple[int, int]):
    if not data.is_cuda or data.dtype != torch.complex64:
        raise ValueError("data must be a CUDA complex64 tensor")
    fast_plan = make_fastest_plan(data, grid_shape, polish_size=32)
    stable_plan = make_stable_plan(data, grid_shape, polish_size=32)
    return fast_plan, stable_plan


def reconstruct_pair(fast_plan, stable_plan, data):
    # Each replay returns graph-owned storage. Clone outputs to retain them.
    fast_x = fast_plan.replay(data).x.clone()
    stable_x = stable_plan.replay(data).x.clone()
    return fast_x, stable_x
```

Plan construction includes warm-up and CUDA Graph capture. To reuse a plan, keep the input shape, dtype, device, and reconstruction grid unchanged. By default, `replay()` submits work asynchronously; use `synchronize=True` when an explicit wait is needed. Do not share one plan between concurrent calls. The plan also owns the returned posterior state; copy the fields you need to retain before the next call.

The default configurations differ as follows:

| Configuration | Active-set capacity (`polish_size`) | Active-set evaluations | Residual candidate exchanges | Fixed-budget diagnostics |
|---|---:|---:|---:|---|
| FBME-Fast | 32 | 2 | 0 | Disabled |
| FBME-Stable | 32 | 3 | 12 | Enabled |

The caller can set `polish_size`; its default remains 32. This is the active-set capacity, not the true number of targets, and increasing it does not necessarily improve accuracy. Fast and Stable are fixed-budget configurations. Their names do not guarantee reconstruction quality, convergence, or global optimality for arbitrary input. `SBLResult` provides the reconstruction, posterior information, and applicable diagnostics; see the source for the exact fields.

## Calling the literature implementations

The following example also expects existing CUDA complex64 observations for a single scene and a noise variance supplied by the caller. Each method retains its own parameters and update budget.

```python
from literature_reproduction import (
    dense_em_sbl, paper_uamp_sbl, uamp_sbl,
    complex_fourier_cofem, ggamp_sbl, afcifsbl_2d,
)


def reconstruct_with_method(name, data, grid_shape, noise_variance):
    if name == "dense":
        return dense_em_sbl(data, grid_shape)
    if name == "uamp_paper":
        return paper_uamp_sbl(data, grid_shape)
    if name == "uamp_triton":
        return uamp_sbl(data, grid_shape, backend="triton")
    if name == "cofem":
        return complex_fourier_cofem(
            data, grid_shape, noise_variance=noise_variance
        )
    if name == "ggamp":
        return ggamp_sbl(
            data, grid_shape, noise_variance=noise_variance, mean_remove=False
        )
    if name == "afcifsbl":
        return afcifsbl_2d(data, grid_shape)
    raise ValueError(f"Unknown method: {name}")
```

Each result object's `.x` field contains the reconstruction. The two-dimensional single-scene calls above return `[N1, N2]`. GGAMP retains a batch dimension when the batch size is greater than one and removes it when the batch size is one. Its noise-variance argument accepts only a scalar or a single-element tensor. Posterior field names and shapes are not necessarily identical across methods. GGAMP does not implement `mean_remove=True`; use `False`. Its default `noise_scale=3.0` internally scales the supplied variance; see [Sources and citations](SOURCES.md).

## Typical RTX 4090 example

The [typical RTX 4090 example](RTX4090_EXAMPLE.md) is available as a [downloadable archive](rtx4090_typical_example.zip). It contains saved results and a plotting workflow for regenerating the GPU rows of Table 2 and Fig. 3. The archived setting uses seed 200, a 10 dB SNR, a 40 × 32 reconstruction grid, and 20 × 16 observations.

This example regenerates reported results from saved records; it does not rerun the original solver comparison or establish independent reproduction of its timings. The original RTX 4090 runner and input arrays are not included in the available archive. See the [example guide](RTX4090_EXAMPLE.md) for extraction instructions; the archive contains a detailed README, protocol, and provenance record. Its plotting requirements are separate from the solver installation above.

## Repository layout

```text
torch_sbl/                  # Core solvers, Fourier operators, Triton, CUDA Graph
torch_sbl_fastest_stable/   # FBME-Fast and FBME-Stable configuration wrappers
literature_reproduction/   # Independent literature implementations
RTX4090_EXAMPLE.md         # Guide to the archived RTX 4090 example
rtx4090_typical_example.zip # Archived Table 2 GPU rows / Fig. 3 materials
rtx4090_typical_example_SHA256.txt # Archive integrity checksum
pyproject.toml             # Package installation and dependencies
README.md
SOURCES.md                 # Method provenance and citations
NOTICE.md                  # Licensing status and third-party dependencies
.gitignore
```

This directory is prepared for publication as a GitHub repository. No MIT, Apache, or other open-source license has been granted by this release. The rights holder must determine the licensing terms; see [Licensing notice](NOTICE.md).
