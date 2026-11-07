# ResNet14 + CIFAR-10：正式实验操作汇总

本文汇总 [GUIDE.md](GUIDE.md) 中的实验流程。依据课程 [指导书与记录表](../../../src/软硬件协同设计项目-指导书与记录表.docx)、[实验 PPT](../../../src/PPT软硬件协同设计项目实验.pdf) 以及论文 §V / Figs.8–10 确定配置；**图 10 及其独立误差归因按算力限制改用固定 1,000 张测试图，保留 10 次 MC**。这与原先全测试集规模不同，报告中须注明；图 8、图 9 和 ADC 校准规模保持不变。

所有命令在 **PowerShell、CrossSim 仓库根目录、同一个终端窗口**中按顺序执行。命令使用现有 `tools/run_experiment.py` 与 `tools/plot_results.py`，不要求新增脚本或执行器。

## 1. 本次正式实验采用的配置

| 实验 | 本汇总采用的配置 | 依据 |
|---|---|---|
| 网络 / 数据集 | 预训练 ResNet14 + CIFAR-10 | 课程指定，替换论文的 ResNet50 + ImageNet |
| 图 10 的 FP、w8、w8a8、q8 基线 | 固定同一组 1,000 张测试图，每条件 1 次 | 与图 10 随机条件保持相同图片；确定性条件无需随机误差棒 |
| 理想 CrossSim 转换检查 | CIFAR-10 全部 10,000 张测试图，1 次 | 独立检查，不进入图 10 |
| 图 10 的编程、读噪声、保持及独立归因 | 固定同一组 1,000 张测试图，每条件 **10 次 MC** | 按用户要求缩减图数；保留论文 10 次误差重采样，满足课程 Nruns≥5 |
| 保持时间 | 1、2、3、4、5 天 | 课程要求，补齐原论文未单独展示的 3、4 天 |
| ADC 校准 | CIFAR-10 训练集固定 500 张；L1 / 12 位优化 | GUIDE 第 7 节，对应论文 500 张校准及 12 位优化方法 |
| 类 Fig.8 | 30 张测试图片推理中的全部 MVM，统计 LSB 正/负 BL 电流 | 论文 Fig.8 与课程电流统计要求 |
| 类 Fig.9(a) | 固定 500 张测试图，扫描 BL 相邻单元段电阻；器件误差关闭 | 论文 Fig.9(a) 的正式评估图数，课程要求隔离器件误差 |
| 类 Fig.9(b) | 固定 1,000 张测试图，ADC 从 2 到 10 位；器件误差与 Rp 关闭 | 论文 Fig.9(b) 的正式评估图数及课程位宽范围 |
| 随机种子 | 器件为 `20261106 + run_index`；数据索引为 `314159` | 现有脚本的固定默认值；各对照条件保持一致 |
| batch size | 按下节实测建议分场景设置 | 图数按本表；显存不足时仅降低 batch，同组噪声对照保持一致 |

**500 张 Rp、1,000 张 ADC 是论文图 9 的正式协议。图 10 的 1,000 张则是本次为节省时间采用的缩减评估规模。** 图 10 全部阶梯点使用相同的数据种子和图片索引，统一按 1,000 张重新运行，不保留旧版 10,000 张图 10 结果。1,000 张结果的采样波动更大，报告注明样本量，避免将小幅精度差异解释为确定结论。

硬件设置沿用 GUIDE 第 5 节：BALANCED / ONE_SIDED 差分映射，权重与输入 8 位，主配置 ADC 8 位，权重不切片，输入逐位串行、模拟累积后 ADC；`Rmin=62500 Ω`、`Rmax=6.25e11 Ω`、`Vread=0.1 V`、最大行数 1152、BN 折叠、数字偏置。Rp 主实验只扫 `Rp_col`。

模型边界沿用 GUIDE 第 4 节：当前内置 SONOS 编程/保持拟合来源与目标 TCAS-I 论文不同；本流程读噪声采用论文描述的 0.87% 比例模型。本文统一正式实验规模与流程，不把这一已知差异表述成已解决。

### 1.1 本机耗时口径与 8GB 显存的 batch 建议

以下针对 **RTX 5060 Laptop GPU 8GB**、修复后的 BL 寄生求解器、单进程运行。2026-11-07 用真实 CIFAR-10 的少量批次测量稳态速度，再按本页正式 N 和 MC 次数外推；测速不是正式准确率实验，也不表示整套正式实验已跑完。下文估计均对应命令中已填写的推荐 batch。

| 实验 | 正式工作量 | 推荐 batch | 预计该项总耗时 | 负责人 |
|---|---|---:|---|---|
| ADC 校准 | 500 张训练图，含量程优化 | **8** | **0.5–2 分钟** |
| 图 10 FP 基线 | 1,000 张 × 1 次 | **64** | **10–30 秒** |
| 理想 CrossSim | 10,000 张 × 1 次 | **16** | **0.5–2 分钟** |
| 仅权重量化 w8 | 1,000 张 × 1 次 | **16** | **10–30 秒** |
| 权重+输入量化 w8a8 | 1,000 张 × 1 次 | **16** | **20–50 秒** |
| 权重+输入+ADC 量化 q8 | 1,000 张 × 1 次 | **16** | **20–50 秒** |
| Fig.8 BL 电流 | 30 张图全部窗口 | **1** | **10–30 秒** |
| Fig.9(a) Rp 扫描 | 10 点，每点 500 张 | **4** | **整组约 1–1.5 小时**；正 Rp 每点约 6–10 分钟，Rp=0 通常几十秒 |
| Fig.9(b) ADC 扫描 | 9 个位宽，每点 1,000 张 | **16** | **整组约 2–6 分钟** |
| 仅编程误差 program | 1,000 张 × 10 次 MC | **16** | **约 2–5 分钟** |  ***蔡依凡*** |
| 编程+读噪声 write-read | 1,000 张 × 10 次 MC | **8** | **约 40–60 分钟** | ***蔡依凡*** |
| 保持实验，每个天数 | 1,000 张 × 10 次 MC，含读噪声 | **8** | **每个天数约 40–60 分钟** |   ***但美帛***  |
| 保持 1–5 天全部完成 | 上述条件共 5 组 | **8** | **合计约 3.5–5 小时** |
| 仅读噪声 read | 1,000 张 × 10 次 MC | **8** | **约 40–60 分钟** |  ***蔡依凡*** |
| 四幅图绘制 | 读取已保存的 JSON | 不涉及推理 batch | **约 10–30 秒** |

**整套已给出命令的实验，建议安排约 6–9 小时连续 GPU 时间。** 其中图 10 主阶梯（含 write-read 和 5 个保持时间）约 4–6 小时，另有 only-read 约 40–60 分钟、Rp 扫描约 1–1.5 小时及其他短实验；未实现的独立漂移、联合配置与 QAT 未计入。笔记本长期降频时可能更久。

#### 为什么不统一设置 batch，也不追求占满 8GB

本机含读噪声条件的测量如下。每个配置预热 1 批，再计时 2 批；数值只用于估计速度与显存余量。

| batch | 前向耗时，秒/张 | 批次结束后 CUDA 报告的已占用显存 |
|---:|---:|---:|
| 4 | 约 0.289 | 约 1.6 GiB |
| 8 | 约 0.281 | 约 1.9 GiB |
| 16 | 约 0.279 | 约 2.6 GiB |
| 32 | 约 0.282 | 约 4.1 GiB |

从 8 增到 32 没有明显提速，因此 read/write-read/retention 推荐 **8**。Rp 测速中，batch=4 约 0.82 秒/张、占用约 2.2 GiB；batch=8 约 0.78 秒/张、占用约 3.1 GiB，提速有限，推荐 **4**。普通量化 batch=16 观测占用约 1.5 GiB，纯 PyTorch batch=64 约 1.3 GiB。

这些显存值包含上下文、内存池缓存及设备其他占用，**不是精确瞬时峰值或模型独占量**。8GB 建议保留至少 1–2GB 余量，不同时运行多个 GPU 实验；显存不足时，重负载实验按 `8→4→2→1` 降 batch，不减少 N 或 MC。

实际 batch 记录在结果 JSON 中。同组噪声条件统一设置；若已有同组结果采用 batch=1，应保持对照设置一致，或在新目录统一重跑。改变 batch 可能改变噪声随机数的消费顺序，不能简单认为相同 seed 就会得到完全相同的逐图误差。

#### 怎样理解耗时数字

估时公式是 `单张秒数 × N × Nruns`。例如本次读噪声约 `0.28×1000×10≈2800 秒≈47 分钟`。仅编程误差在建立阵列时采样，推理时没有逐 MVM 读噪声展开，因此远快于 write-read。

JSON 中 `runs[].seconds` 主要记录推理，不含全部进程启动、模型转换、校准优化、下载或绘图。历史 500 张、batch=8 校准记录的推理约 6.7 秒，从结果中的创建时间到保存时间约 13.8 秒，说明不能把 6.7 秒当成整个校准耗时。表中已留出启动和其他开销余量；首次编译、量程、供电、散热、后台程序仍会使实际时间变化。首次数据下载不计入。

## 2. 环境、入口与输出目录

```powershell
cd 'E:\Resources\01_lessons\class3_LS\09-chip-cim\work\sim1\cross-sim'
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync --locked

$tool = 'docs/sim-work/tools/run_experiment.py'
$plot = 'docs/sim-work/tools/plot_results.py'
$out = 'docs/sim-work/result'
$adc = "$out/calibration.adc.npy"
$MC = 10
```

数据默认放在 `scratch/sim-work/data/cifar10/`。第一次数据准备由下一节的 `--download` 完成；脚本会创建结果文件所需的目录。

图 10 的基线、随机条件、独立误差归因及图表继续保存到原目录 `$out`。若其中已有旧版图 10 结果，先删除对应的 `ladder-*.json`、`only-read.json` 和 `fig10.png/csv`，再按下面命令以同一组 1,000 张重新运行。已有 `$out/calibration.adc.npy` 可直接复用，图 8、图 9 的结果无需重跑。

同一条件只运行一次命令。已有输出 JSON 时，现有脚本会拒绝覆盖；中断后从尚未完成的条件继续，不重复执行已经完成的条件。每个条件内的 MC 全部结束后才写 JSON，因此不能从某次未结束的 MC 中间恢复。

## 3. ADC 逐层量程校准

**建议 batch=8；500 张训练图连同量程优化预计 0.5–2 分钟。首次数据下载另计。**

```powershell
uv run --locked python $tool --case calibrate --gpu --n 500 --nruns 1 --batch-size 8 --download --out "$out/calibration.json"
if ($LASTEXITCODE -ne 0) { throw 'ADC 校准失败' }
```

使用训练集 500 张，关闭器件误差、寄生及 ADC 量化，采集最终 SIR 累积信号，进行 L1 / 12 位量程优化。

输出 `calibration.json` 与 `calibration.adc.npy`。之后所有相关实验统一使用 `$adc`，不在 ADC 扫描中按位数更换量程。输入量程沿用 GUIDE 中固定的仓库 `input_limits_ResNet14.npy`。

## 4. 图 10 的 1,000 张基线、量化阶段与转换检查

### 4.1 原始 PyTorch 浮点基线

**建议 batch=64；1,000 张预计 10–30 秒，含启动开销的规划余量。**

```powershell
uv run --locked python $tool --case fp --gpu --n 1000 --nruns 1 --batch-size 64 --out "$out/ladder-fp.json"
if ($LASTEXITCODE -ne 0) { throw 'FP 基线失败' }
```

本结果是图 10 及其误差归因、精度预算的 FP 参考，与其他阶梯点使用同一组 1,000 张。

### 4.2 理想 CrossSim 对照

**建议 batch=16；全测试集预计 0.5–2 分钟。**

```powershell
uv run --locked python $tool --case ideal --gpu --n 10000 --nruns 1 --batch-size 16 --out "$out/ideal.json"
if ($LASTEXITCODE -ne 0) { throw '理想 CrossSim 对照失败' }
```

此项仍用全测试集做独立转换检查，不放入误差阶梯主图。若需逐图核对与 FP 的一致性，应再以相同 1,000 张、相同数据种子运行 ideal，不能将不同样本数产生的小幅精度差异归因于转换误差。

### 4.3 8 位权重 → 8 位输入 → 8 位 ADC

**三组均用 1,000 张、batch=16；w8 约 10–30 秒，w8a8、q8 各约 20–50 秒，三组共约 1–3 分钟。**

```powershell
foreach ($c in @('w8', 'w8a8', 'q8')) {
    uv run --locked python $tool --case $c --gpu --n 1000 --nruns 1 --batch-size 16 --adc-limits $adc --out "$out/ladder-$c.json"
    if ($LASTEXITCODE -ne 0) { throw "$c 失败" }
}
```

| case | 配置 |
|---|---|
| w8 | 仅量化权重，输入与 ADC 不量化 |
| w8a8 | 8 位权重 + 8 位输入，ADC 不量化 |
| q8 | 8 位权重 + 8 位输入 + 8 位 ADC，作为器件误差公共基线 |

这三个结果与 FP 一起构成累积误差阶梯的前四点。课程表中可补充“ADC 量化”一行，单独记录 ADC 的损失。

## 5. 类 Fig.8：阵列行数—正/负 BL 电流

**保持 batch=1；30 张图预计 10–30 秒，含启动和绘图余量。** 已有统计的推理部分约 5.7 秒，无需为这一步增加批量。

```powershell
uv run --locked python $tool --case bl --gpu --n 30 --nruns 1 --batch-size 1 --adc-limits $adc --out "$out/fig8.json"
if ($LASTEXITCODE -ne 0) { throw 'BL 电流统计失败' }

uv run --locked python $plot --kind fig8 --input "$out/fig8.json" --out "$out/fig8"
if ($LASTEXITCODE -ne 0) { throw 'Fig.8 绘图失败' }
```

按实际阵列输入行数分组，统计 30 张图片全部窗口的 LSB 正、负 BL 电流。使用默认 8/8/8 位，关闭器件误差与寄生。最大容量仍为 1152 行，横轴显示网络实际用到的行数，不人为生成 1152 行的柱子。

保存 `fig8.png`、`fig8.csv`、`fig8.json`。报告回答：是否单调、为何远低于 `1.6 μA × 行数`、权重分布与高 On/Off 比各起什么作用。

## 6. 类 Fig.9(a)：BL 寄生电阻扫描

**建议 batch=4；正 Rp 每点预计 6–10 分钟，Rp=0 通常几十秒，10 点整组约 1–1.5 小时。** 估计采用修复后的直接求解器，不采用旧迭代计时；补点另计。

```powershell
$rpValues = @(0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)

foreach ($r in $rpValues) {
    uv run --locked python $tool --case rp --gpu --n 500 --nruns 1 --batch-size 4 --rp $r --adc-limits $adc --out "$out/rp-$r.json"
    if ($LASTEXITCODE -ne 0) { throw "Rp=$r 仿真失败，请检查收敛日志" }
}

uv run --locked python $plot --kind rp --input "$out/rp-*.json" --out "$out/fig9a"
if ($LASTEXITCODE -ne 0) { throw 'Fig.9(a) 绘图失败' }
```

每个点使用同一组 500 张测试图。固定权重/输入/ADC 为 8/8/8 位、固定量程，关闭编程误差、读噪声及漂移，只改变相邻单元间 BL 电阻。

上面的离散电阻网格来自 GUIDE 的扫描设置；课程没有指定这些离散值。`Rp=0` 是无寄生参考，绘图时画为基准线，正电阻点采用对数横轴。若转折落在相邻两个点之间，在该区间补点时仍使用 `--n 500` 和相同量程。

保存 `fig9a.png/csv` 与逐点 JSON。报告给出拐点或容忍区间，并与 `Rmin=62.5 kΩ` 比较。数值不收敛的点应注明未测得；扫描范围内没有拐点则如实说明，不能填写虚构值。

## 7. 类 Fig.9(b)：ADC 2–10 位扫描

**建议 batch=16；每个位宽约 10–40 秒，9 个位宽合计约 2–6 分钟。** 本场景关闭器件噪声及寄生。

```powershell
foreach ($b in 2..10) {
    uv run --locked python $tool --case adc --gpu --n 1000 --nruns 1 --batch-size 16 --bits $b --adc-limits $adc --out "$out/adc-$b.json"
    if ($LASTEXITCODE -ne 0) { throw "ADC=$b 仿真失败" }
}

uv run --locked python $plot --kind adc --input "$out/adc-*.json" --out "$out/fig9b"
if ($LASTEXITCODE -ne 0) { throw 'Fig.9(b) 绘图失败' }
```

每个点使用同一组 1,000 张测试图。固定 8 位权重、8 位输入、同一量程，关闭器件误差与寄生，仅改变 ADC 位数。

保存 `fig9b.png/csv` 与逐点 JSON。报告标出转折区、收益递减区，并据实说明选定 ADC 位数及安全裕量。

## 8. 类 Fig.10：1,000 张累积误差与 1–5 天保持

### 8.1 加入编程误差，再加入读噪声

**program 用 batch=16，10 次 MC 预计 2–5 分钟；write-read 用 batch=8，10 次 MC 预计 40–60 分钟。** 每次 MC 均为同一组 1,000 张，两者不能按相同速度估计。

```powershell
foreach ($c in @('program', 'write-read')) {
    $batch = if ($c -eq 'program') { 16 } else { 8 }
    uv run --locked python $tool --case $c --gpu --n 1000 --nruns $MC --batch-size $batch --noise-model paper --adc-limits $adc --out "$out/ladder-$c.json"
    if ($LASTEXITCODE -ne 0) { throw "$c 仿真失败" }
}
```

`program` 在 q8 上仅加 SONOS 编程误差；`write-read` 再加论文描述的 0.87% 比例读噪声。每条件 10 次 MC，每次重新建立、编程阵列。

### 8.2 保持 1、2、3、4、5 天

**统一 batch=8；每个天数的 10 次 MC 预计 40–60 分钟，五个天数合计约 3.5–5 小时。** 主要耗时来自读噪声，不需要现实中等待相应天数；这里是吞吐外推，尚未跑完全部保持实验。

```powershell
foreach ($d in 1..5) {
    uv run --locked python $tool --case retention --gpu --n 1000 --nruns $MC --batch-size 8 --days $d --noise-model paper --adc-limits $adc --out "$out/ladder-day$d.json"
    if ($LASTEXITCODE -ne 0) { throw "保持 $d 天仿真失败" }
}
```

每个时间点均为同一组 1,000 张测试图 × 10 次 MC，使用相应时刻的完整 SONOS 保持状态分布及读噪声；不在 day1 已扰动矩阵上继续叠加 day2 误差。整条阶梯关闭 Rp，符合论文 Fig.10 的隔离设置。

### 8.3 汇总前面全部阶梯点

**预计耗时：数秒至约 1 分钟。** 只读已有 JSON 并绘图，不重新推理。

```powershell
uv run --locked python $plot --kind ladder --input "$out/ladder-*.json" --out "$out/fig10"
if ($LASTEXITCODE -ne 0) { throw 'Fig.10 绘图失败' }
```

主图顺序：FP → w8 → w8a8 → q8 → program → write-read → day1 → day2 → day3 → day4 → day5。

保存到 `$out` 下的 `fig10.png/csv` 和逐条件 JSON。随机条件报告均值±样本标准差；确定性条件单次运行的标准差为不可估计/不适用，不人为增加误差棒。`fig10.csv` 同时保留 top-1、top-5 及其标准差，主图展示 top-1。图注注明“CIFAR-10 固定 1,000 张测试图，随机条件 10 次 MC”；MC 误差棒反映器件重采样波动，不包含从完整测试集抽取 1,000 张带来的采样不确定性。

## 9. 课程要求的独立误差归因

### 9.1 补跑“仅读噪声”

**建议 batch=8；1,000 张 × 10 次 MC 预计 40–60 分钟。** 使用与图 10 相同的图片，与 write-read 同量级，不能用无噪声 q8 的速度估算。

```powershell
uv run --locked python $tool --case read --gpu --n 1000 --nruns $MC --batch-size 8 --noise-model paper --adc-limits $adc --out "$out/only-read.json"
if ($LASTEXITCODE -ne 0) { throw '仅读噪声实验失败' }
```

“仅编程误差”已由 `ladder-program.json` 得到，不需要重复运行。用 q8 作公共基线：

| 量 | 从哪些结果计算 |
|---|---|
| 仅编程误差的独立损失 | q8 − program |
| 仅读噪声的独立损失 | q8 − only-read |
| 编程误差已存在时，增加读噪声的增量损失 | program − write-read |
| 加入 t 天保持后的增量损失 | write-read − dayt |

### 9.2 “仅漂移”一行的现有实现边界

**耗时未估计：当前没有可执行的独立 case。** 不能把包含读噪声的保持实验耗时直接称为“纯漂移实验耗时”。

当前脚本没有“只开启保持模型、关闭读噪声”的独立 case；`retention` 会同时开启读噪声，不能直接将它填成“仅漂移”。GUIDE 第 12.1 节给出的操作是另设该配置；本文只汇总现有命令，不改动脚本或列出不存在的命令。

即使另设配置，当前 SONOS 的保持分布也包含状态散布/编程成分，不能称为已完全解耦的“纯漂移”。上表最后一项是可从现有结果计算的**保持增量归因**，不能冒充已完成了独立漂移配置。

## 10. 协同设计与记录表整理

**已有结果的读取、预算计算只需秒级；参数解释与记录表撰写是人工工作。** 联合配置当前未提供执行入口，也没有本机实测耗时，不据现有单变量结果给出确定时长。

完成上述实验后，按 GUIDE 第 12.3 节进行决策：

1. 以本次相同 1,000 张的 FP 精度为基准，分别计算 `FP−1.50 pp` 和 `FP−0.50 pp` 门槛，注明缩减评估规模；候选配置应使用相同图片比较。
2. 从量化阶梯、ADC 曲线、Rp 曲线及保持时间曲线中选择候选参数，并记录数据依据。
3. 联合配置需要把选定 ADC、Rp、器件状态及时间同时启用后再评估；单因素损失相加不能代替联合仿真。当前脚本没有 `joint` case，不能通过给 `retention` 加 `--rp` 或给 `rp` 加 `--days` 假装启用了联合效应；这些参数只在对应 case 中生效。
4. 对 0.50 pp 目标，依据免重训练结果讨论能否达到；若不能，写出 QAT/硬件感知训练思路。课程此项要求给出判断与设计理由，不要求凭空填写训练提升结果。

“仅漂移”和“联合配置”均是 GUIDE 已指出需要另设配置的部分；本操作汇总没有宣称现有 CLI 已覆盖这两项。若记录表要报告其独立实测值，必须实际补齐对应配置后测量。

| 记录表位置 | 本流程对应的文件或工作 |
|---|---|
| 器件累积误差 | `fig10.png/csv` 与 `ladder-*.json` |
| 独立归因 | `ladder-q8.json`、`ladder-program.json`、`only-read.json`；保持部分注明采用增量归因或另设配置 |
| BL 电流统计 | `fig8.png/csv/json` |
| 寄生电阻 | `fig9a.png/csv` 与 `rp-*.json` |
| ADC 位宽 | `fig9b.png/csv` 与 `adc-*.json` |
| 协同设计参数清单 | 引用上述实测数据，写明选择、精度预算与联合验证是否完成 |
| 代码与可复现设置 | 本文命令、校准文件、JSON 内的参数、索引、种子和文件哈希 |

最终整理四张图及其 CSV、全部原始 JSON、`calibration.adc.npy`，并按课程要求完成结果解释、讨论与 AI 使用说明。本文中的命令是正式运行入口，不表示这些长时实验已执行完成。
