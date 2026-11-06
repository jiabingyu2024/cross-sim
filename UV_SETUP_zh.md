# CrossSim：Windows + uv 环境与命令

本目录按 **Windows x64、Python 3.11、CUDA Toolkit 12.8、NVIDIA RTX 50 系列**配置。
`pyproject.toml` 中的 `lab` 依赖组默认安装；`uv.lock` 固定解析结果。
CrossSim 以 editable 模式安装，修改 `simulator/` 源码后无需重新安装。

## 1. 这次配置的内容

| 组件 | 版本 / 用途 |
| --- | --- |
| Python | 3.11（项目内 `.venv`） |
| CrossSim | 本地源码 3.2.1 |
| NumPy / SciPy | 1.26.4 / 1.11.4 |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128，GPU |
| CuPy | cupy-cuda12x 13.6.0，CrossSim GPU 后端 |
| TensorFlow / Keras | 2.17.1 / 3.6.0，原生 Windows CPU |
| 教程工具 | JupyterLab、IPython、ipykernel、Matplotlib、Pillow |
| 常见应用依赖 | OpenCV、scikit-learn、pandas、tqdm |

2026-11-06 本机实测：uv 0.13.7，`.venv` 使用 Python 3.11.13，JupyterLab 4.6.5。
创建虚拟环境时复用了已有的 `D:\AppMajor\miniconda3\python.exe`；未向 Conda base 安装项目依赖。
已通过以下检查：

- `uv sync --locked`：锁文件与配置一致，环境已同步。
- `uv pip check`：146 个已安装包，依赖关系全部兼容。
- `verify_environment.py`：CrossSim CPU/GPU MVM 与 NumPy 结果一致。
- CrossSim + PyTorch：CPU 与 CUDA 的前向计算、反向梯度均通过。
- CrossSim + TensorFlow/Keras：`AnalogDense` CPU 输出与 Keras Dense 一致。
- `tutorial/IISWC2026/Pt0-Basics.ipynb`：使用 `crosssim-uv` 内核完整执行 12 个代码单元通过；在临时目录执行，没有改写原 Notebook。
- Matplotlib、OpenCV、pandas、scikit-learn、Pillow 均成功导入。

`CrossSim (uv, Python 3.11)` 内核已经注册在项目 `.venv` 内。

RTX 50 系列采用较新的 GPU 架构，因此这里使用 CUDA 12.8 版 PyTorch，而没有照搬上游 README 中旧的 PyTorch 2.2.1。
本机已存在 NVIDIA 驱动和 CUDA Toolkit 12.8。`nvidia-smi` 显示的 CUDA 13.0 是驱动支持上限，不表示项目必须使用 CUDA 13。

**TensorFlow 的限制：**2.10 是最后一个支持原生 Windows CUDA GPU 的 TensorFlow 版本。这里的 TensorFlow 2.17.1 使用 CPU；PyTorch、CuPy、CrossSim GPU 不受此限制。若要 TensorFlow CUDA GPU，需要另建 WSL2/Linux 环境，不能直接复用 Windows 的 `.venv` 和本目录的 Windows 专用锁文件。本次未安装或修改 WSL2。

## 2. 人类手动搭建：复现本目录配置

以下命令在 **PowerShell** 中执行。

### 安装 uv（每台机器通常只做一次）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

# 让当前窗口立刻找到 uv；以后新开的终端通常无需这行
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv --version
```

### 进入已有项目并安装

```powershell
cd 'E:\Resources\01_lessons\class3_LS\09-chip-cim\work\sim1\cross-sim'

# 已有 Python 3.11 时可跳过；没有时让 uv 安装
uv python install 3.11

# 第一次复现已提供的配置：创建 .venv 并安装锁定的依赖
uv sync --locked

# 检查依赖关系和实际运算
uv pip check
uv run --locked python verify_environment.py
```

如果 `uv python install` 在本机遇到下载或链接错误，但已经有 Python 3.11，可直接指定已有解释器创建环境：

```powershell
# 只在尚未创建 .venv 时使用；这里的路径换成自己的 Python 3.11
uv venv --python 'D:\AppMajor\miniconda3\python.exe'
uv sync --locked
```

`.venv` 的包与原来的 Conda / 系统环境隔离；借用已有 Python 创建虚拟环境不等于向 Conda base 安装依赖。

### 日常启动

```powershell
cd 'E:\Resources\01_lessons\class3_LS\09-chip-cim\work\sim1\cross-sim'
uv run --locked jupyter lab tutorial
```

浏览器打开后，建议从 `IISWC2026/Pt0-Basics.ipynb` 或 `tutorial_pt1.ipynb` 开始。
Jupyter 界面选择 `CrossSim (uv, Python 3.11)` 内核。第一次 GPU 运算可能需要编译 CuPy 内核。
按 `Ctrl+C` 结束 Jupyter 服务。

如需注册项目内的 Jupyter 内核（已配置的本机无需重复）：

```powershell
uv run --locked python -m ipykernel install --sys-prefix --name crosssim-uv --display-name 'CrossSim (uv, Python 3.11)'
```

用脚本时：

```powershell
uv run --locked python your_script.py
```

`your_script.py` 代表你自己编写的脚本文件名。VS Code / PyCharm 解释器选择：

```text
E:\Resources\01_lessons\class3_LS\09-chip-cim\work\sim1\cross-sim\.venv\Scripts\python.exe
```

## 3. 这些命令分别做什么

| 命令 | 含义 |
| --- | --- |
| `uv python install 3.11` | 准备 Python 解释器；已有合适版本时不一定需要 |
| `uv python pin 3.11` | 在 `.python-version` 写入项目默认 Python；本目录已写好 |
| `uv venv --python 3.11` | 仅创建虚拟环境；`uv sync` 也会自动创建 |
| `uv sync` | 根据项目配置解析、更新锁文件并安装；首次配置或修改依赖后使用 |
| `uv sync --locked` | 按现有锁文件安装；配置与锁不一致时报错，适合复现 |
| `uv run --locked ...` | 使用项目 `.venv` 执行命令，并检查锁文件 |
| `uv pip check` | 检查已安装包之间的版本依赖 |
| `uv add --group lab 包名` | 添加本地学习环境依赖并更新锁文件 |

**无需先激活环境。**`uv run` 会选择本项目的 `.venv`。
如果希望像传统 Python 一样运行，也可以选择激活：

```powershell
.\.venv\Scripts\Activate.ps1
python verify_environment.py
deactivate
```

若 PowerShell 禁止执行激活脚本，继续使用 `uv run` 即可。

## 4. 从一个未经配置的开源仓库开始，一般怎么做

通常先读 `README` 和 `pyproject.toml`，确认 Python、框架与硬件要求，再决定依赖。
本项目上游只声明了 NumPy 和 SciPy，直接 `uv sync` 不会自动装齐 PyTorch、CuPy 和全部教程工具。
本目录已经补充了 `lab` 依赖组及 PyTorch 专用索引，因此你现在只需要前面的复现命令。

对于一般的、暂不需要锁文件的源码项目，常见 uv 工作流如下（解释用，不要再对当前已配置环境重复安装）：

```powershell
cd '你的项目目录'
uv venv --python 3.11
uv pip install -e .
# 按 README 再安装需要的应用依赖，例如：
uv pip install matplotlib jupyterlab ipykernel
# 不激活也能直接使用虚拟环境解释器
.\.venv\Scripts\python.exe -c "import simulator"
```

有明确版本要求时固定版本；要长期复现，则像本目录一样，把依赖保存在 `pyproject.toml` 与 `uv.lock`。
当前项目已有 `pyproject.toml`，**不用执行 `uv init`**。

## 5. 在自己的代码里启用 GPU

安装 CUDA 依赖不会让所有仿真自动使用 GPU。需要在创建第一个核心前设置参数：

```python
import cupy as cp
from simulator import AnalogCore
from applications.mvm_params import set_params

params = set_params(useGPU=True, gpu_id=0)
weights = cp.asarray([[0.2, -0.4], [0.6, 0.1]], dtype=cp.float32)
inputs = cp.asarray([1.0, 2.0], dtype=cp.float32)
core = AnalogCore(weights, params=params)
print(cp.asnumpy(core @ inputs))
```

CrossSim 的计算后端在首次初始化后固定。若已运行过 CPU 核心，再改成 GPU，最稳妥的做法是重启 Python 进程 / Jupyter 内核，再从头运行并启用 GPU。
PyTorch 模型和输入也要放到 `cuda`，并使用 `useGPU=True` 的 CrossSim 参数；验证脚本包含可运行的完整例子。

不要执行 `uv sync --all-extras` 或额外安装上游 `gpu` extra：上游依赖名 `cupy` 会尝试源码构建，本环境已经使用预编译的 `cupy-cuda12x`。两种 CuPy 发行包不应同时安装，本地配置已声明二者冲突。

## 6. 数据集与预训练模型

环境安装与数据下载是两件事。基础教程和本地验证不需要下载全部模型。
若某个 DNN 示例需要上游子模块，可在项目根目录执行：

```powershell
git submodule update --init --recursive --progress
```

上游 README 提示这部分下载约 1.2 GB，解压后还会占用额外空间。本次未下载这两个可选子模块。
部分较新教程还会自行下载 CIFAR、语音数据、模型，或要求 `norse` 等专用依赖；这些按具体教程另行配置，不代表基础环境安装失败。

## 7. 官方资料

- [uv 安装](https://docs.astral.sh/uv/getting-started/installation/)
- [uv 项目、锁文件与运行命令](https://docs.astral.sh/uv/guides/projects/)
- [uv 配置 PyTorch 索引](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [PyTorch 历史版本与 CUDA 安装组合](https://pytorch.org/get-started/previous-versions/)
- [CuPy 安装与 CUDA 对应关系](https://docs.cupy.dev/en/stable/install.html)
- [TensorFlow 原生 Windows / WSL2 支持](https://www.tensorflow.org/install/pip)
- [CrossSim 上游仓库](https://github.com/sandialabs/cross-sim)
