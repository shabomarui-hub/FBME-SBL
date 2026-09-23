# FBME-SBL：算法代码

Fourier-structured sparse Bayesian learning in PyTorch, with FBME-Fast,
FBME-Stable, and independent implementations of selected SBL methods.

本仓库提供本项目的 FBME-SBL 求解器、CUDA Graph 执行入口，以及部分 SBL 方法的独立复现代码。仅包含算法及安装、调用和来源说明，不包含数据集、数据生成器、实验脚本、计时结果、论文或项目记忆。算法数值主体保留原实现。

## 包含的方法

| 方法 | 调用入口 | 当前实现的执行条件 |
|---|---|---|
| FBME-Fast | `torch_sbl_fastest_stable.make_fastest_plan` | CUDA、complex64、Triton、固定形状 CUDA Graph |
| FBME-Stable | `torch_sbl_fastest_stable.make_stable_plan` | CUDA、complex64、Triton、固定形状 CUDA Graph |
| FBME 通用求解入口 | `torch_sbl.sbl_2d` / `fast_sbl` | 由 `SBLConfig` 指定计算与执行方式 |
| Dense EM-SBL | `literature_reproduction.dense_em_sbl` | PyTorch；输入张量所在的 CPU 或 CUDA 设备 |
| UAMP-SBL（论文更新式） | `literature_reproduction.paper_uamp_sbl` | 当前入口要求 CUDA，单场景 |
| UAMP（工程版本） | `literature_reproduction.uamp_sbl` | `backend="torch"` 或 `"triton"`；后者要求 CUDA/Triton |
| CoFEM（复数 Fourier 适配） | `literature_reproduction.complex_fourier_cofem` | 当前入口要求 CUDA，需提供噪声方差 |
| GGAMP-SBL（复数适配） | `literature_reproduction.ggamp_sbl` | PyTorch；需提供噪声方差 |
| 2D-AFCIFSBL（部分 Fourier 适配） | `literature_reproduction.afcifsbl_2d` | PyTorch，二维单场景 |

Dense 的 CPU/GPU 路径共用同一实现；工程 UAMP 的 Torch/Triton 路径也不是两种独立文献方法。这里没有 FFD-SBL 的独立实现。文献复现与原论文的差异见[来源与引用](来源与引用.md)。

## 安装

Python 3.10 或以上。依赖范围围绕项目使用的 PyTorch 2.6 / Triton 3.2 设置；推荐 Linux 或 WSL2 使用 CUDA 路径。先按 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/) 安装匹配本机驱动的 PyTorch 2.6.0；需要 Fast/Stable 时应安装 CUDA 版本。

在解压后的仓库根目录运行：

```bash
python -m pip install .
```

需要 Fast/Stable 或 Triton 工程 UAMP 时：

```bash
python -m pip install ".[triton]"
```

Windows 原生额外依赖使用 `triton-windows`；该安装选项不代表已经验证所有 Windows 原生驱动组合。仅有 CPU 时可使用 Dense 等入口，不能运行 Fast/Stable。项目自身不需要 NumPy、SciPy 或数据集依赖来导入这些算法包。

可用以下命令确认安装与导出入口，不会执行求解器：

```bash
python -c "import torch_sbl, torch_sbl_fastest_stable, literature_reproduction; print(torch_sbl.__version__)"
```

## 输入的含义

- 二维单场景输入 `data` 是复数观测张量 `[M1, M2]`；重构网格由 `grid_shape=(N1, N2)` 指定，满足 `0 < Mi <= Ni`。
- 默认模型使用**未归一化二维 DFT 的前缀观测块**：`data = fft2(x)[:M1, :M2] + noise`。DFT 顺序、幅度和相位约定必须一致；不要直接把灰度图、幅度图或任意雷达原始文件当作这里的复数观测输入。
- `complex64` 指每个复数由两个 float32 表示，与网格大小不同。Fast/Stable 固定使用 complex64；其他接口的支持范围见源码。
- 通用 `sbl_2d` 支持批量输入和部分掩码配置；文献入口的批量/掩码能力不同，不能假定同一接口全部兼容。
- CoFEM/GGAMP 所需的 `noise_variance` 指与观测幅度尺度一致的复数噪声功率 `E[|noise|²]`，由调用方提供或估计；不要把 dB 数值直接传入。

## FBME-Fast 与 FBME-Stable

下面的函数接收调用方已经准备好的观测，不加载或生成数据：

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

计划构建包含预热和 CUDA Graph 捕获；后续复用计划的输入形状、dtype、设备和网格必须保持不变。`replay()` 默认异步提交；需要显式等待时使用 `synchronize=True`。同一个计划不应被多个并发调用共享。返回的后验状态也由计划持有，需要跨下次调用保存时逐字段复制。

两种默认配置的区别：

| 配置 | 活动集容量 `polish_size` | 活动集评估次数 | 残差候选交换数 | 固定预算诊断 |
|---|---:|---:|---:|---|
| FBME-Fast | 32 | 2 | 0 | 关闭 |
| FBME-Stable | 32 | 3 | 12 | 开启 |

`polish_size` 可由调用方指定，默认值保持 32；它是活动集容量，不是真实目标个数，也不表示增大后必然更准确。Fast/Stable 是固定预算配置，名称不构成任意输入下的质量、收敛或全局最优保证。`SBLResult` 中提供重构、后验及相应诊断；具体字段以源码为准。

## 文献方法调用

以下示例同样由调用方传入已有的 CUDA complex64 单场景观测和噪声方差；各方法保留自己的参数和更新预算：

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

各结果对象的 `.x` 是重构；本例二维单场景入口均返回 `[N1, N2]`。GGAMP 在批量数大于 1 时保留批量维，批量数为 1 时会去掉该维；其噪声方差参数只接受标量或单元素张量。不要依赖所有后验字段名称和形状完全相同。GGAMP 的 `mean_remove=True` 未实现，必须使用 `False`；默认 `noise_scale=3.0` 会对传入方差作内部缩放，参见来源说明。

## 目录

```text
torch_sbl/                  # 本项目核心、Fourier 算子、Triton、CUDA Graph
torch_sbl_fastest_stable/   # FBME-Fast / FBME-Stable 配置入口
literature_reproduction/   # 独立文献实现
pyproject.toml             # 安装与依赖
README.md
来源与引用.md
授权说明.md
.gitignore
```

这是一份代码发布目录，可将目录内的文件放到 GitHub 仓库根目录。发布前由权利人确定开源许可证；本目录没有擅自授予 MIT、Apache 或其他许可证，见[授权说明](授权说明.md)。
