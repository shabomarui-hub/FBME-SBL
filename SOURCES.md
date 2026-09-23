# Sources and citations

This repository contains the project's FBME-SBL implementation and independent implementations based on published methods. The latter are not software released by the original authors. Interface adaptations and complex-valued extensions do not constitute a complete reproduction of the original papers' acquisition systems, processing pipelines, or results.

## FBME-SBL

`torch_sbl` contains the numerical solvers, implicit partial Fourier operators, Triton kernels, and CUDA Graph wrappers. `torch_sbl_fastest_stable` provides the FBME-Fast and FBME-Stable configurations. The core identifier `FBME-SBL-TAES-v1.1` and configuration identifier `FBME-SBL-FASTEST-STABLE-v1.0` are source compatibility identifiers; they do not indicate publication status.

The existing numerical solver implementations are unchanged. The algorithm package streamlines module exports, installation dependencies, and documentation. The repository also provides an [archived RTX 4090 example](RTX4090_EXAMPLE.md) as a [downloadable ZIP](rtx4090_typical_example.zip), containing saved result records and a workflow for regenerating the GPU rows of Table 2 and Fig. 3. This is result regeneration from the archive, not a complete rerun of the original solver experiment: the original RTX 4090 runner and input arrays are not available in that archive. Manuscripts and internal project records are not distributed.

## Literature implementations

| Entry point | Source and adaptation |
|---|---|
| `dense_em_sbl` | Implements a classical EM-SBL baseline using the project's exact observation-domain posterior calculation. CPU and GPU share the same code. This is not FFD-SBL. |
| `paper_uamp_sbl` | Follows Algorithm 2 of Luo et al., using the row orthogonality of the prefix partial DFT to avoid an online SVD while retaining the paper's precision and Gamma-shape updates. |
| `uamp_sbl` | The project's engineering UAMP variant, with Torch and Triton execution paths. It is not an alias for the equation-based implementation above. |
| `complex_fourier_cofem` | Follows the CoFEM algorithm structure of Lin et al., adapting random probes and parallel PCG to complex Hermitian Fourier systems. It neither imports nor bundles the authors' repository code. |
| `ggamp_sbl` | Based on Algorithm 1 of Al-Shoukairi et al., with Hermitian transposes and complex second moments replacing the real-valued formulation. Noise variance is supplied as a fixed input by default, `noise_scale=3.0`, and noise updates are disabled by default. The `mean_remove` argument is retained, but mean removal is not implemented; calls must keep it `False`. |
| `afcifsbl_2d` | Based on Eqs. (41)–(45), (47), and (52) of Wang et al., adapted to prefix two-dimensional Fourier observations with the spectral bound `T=2PQ+1e-5`. It is not equivalent to the complete original RSF-ISAR acquisition and processing pipeline. |

The repository does not include mechanism prototypes such as LCDRD-SBL, FNUBSBL, or LaAPC-SBL that have not completed equation-by-equation verification against the full papers. It also does not bundle original source code released by third-party authors.

## References

1. Maher Al-Shoukairi, Philip Schniter, Bhaskar D. Rao. **A GAMP-Based Low Complexity SBL Algorithm.** IEEE Transactions on Signal Processing, 66(2):294–308, 2018. [DOI: 10.1109/TSP.2017.2764855](https://doi.org/10.1109/TSP.2017.2764855).
2. Man Luo, Qinghua Guo, Ming Jin, Yonina C. Eldar, Defeng Huang, Xiangming Meng. **Unitary Approximate Message Passing for Sparse Bayesian Learning.** IEEE Transactions on Signal Processing, 69:6023–6039, 2021. [DOI: 10.1109/TSP.2021.3114985](https://doi.org/10.1109/TSP.2021.3114985).
3. Alexander Lin, Andrew H. Song, Berkin Bilgic, Demba Ba. **Covariance-Free Sparse Bayesian Learning.** IEEE Transactions on Signal Processing, 70:3818–3831, 2022. [DOI: 10.1109/TSP.2022.3186185](https://doi.org/10.1109/TSP.2022.3186185). Authors' code: [al5250/sparse-bayes-learn](https://github.com/al5250/sparse-bayes-learn); that code is not included in this release.
4. Yiding Wang, Yuanhao Li, Jiongda Song, Guanghui Zhao. **Random Stepped Frequency ISAR 2D Joint Imaging and Autofocusing by Using 2D-AFCIFSBL.** Remote Sensing, 16(14):2521, 2024. [DOI: 10.3390/rs16142521](https://doi.org/10.3390/rs16142521).

When using a literature method, cite its original paper and identify this repository as an independent implementation, including the adaptation settings used. The authors should add the formal FBME-SBL citation once its publication details are confirmed. No unverified publication record is supplied here.
