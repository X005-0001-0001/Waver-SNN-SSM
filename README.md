# Waver-SNN-SSM：从脉冲到状态空间的统一序列建模框架

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22309186.svg)](https://doi.org/10.5281/zenodo.22309186)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--4332--1295-green)](https://orcid.org/0009-0006-4332-1295)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**A complex-valued SSM turned "complex-free": every complex operator is replaced by a real 2×2 block isomorphism.**
**把复值 SSM 的复数算子完整替换为实值 2×2 块同构 —— 代数层面实现"无复数"的复数 SSM。**

> **当前最新 / Latest：`8.1.0`（2026-10-12）** —— 新增 **009elitelinux** 变体：把 Win11 上无法启用的 **Linux 单卡编译加速**（`torch.compile` / Inductor-Triton 融合标准算子）与 DataLoader 多进程做成**独立可关**开关（默认开启）；**全关时与对应 009 逐位一致**，超小 A/B 验证收敛一致、无正确性回归。

> 上一版 / Previous：`8.0.0`（2026-10-11）—— 009 变体：注入端**遗忘平衡** `B̄ = B̃·(1−|A|)`，解耦时峰值增益恒等 `G_peak ≡ |B̃|`、耦合时有界（实测 ≲3.5×），消除慢极点（ν→0）直流增益爆炸（承接 7.2.0 的 008 变体）。

**语言 / Language：** [中文文档](#中文文档) · [English Documentation](#english-documentation)

---

<a name="中文文档"></a>

# 中文文档

## 📌 项目简介

Waver-SNN-SSM 是一个从脉冲神经网络（SNN）出发、最终与 Mamba-3 理念殊途同归的高效状态空间模型（SSM）实现。核心创新在于把复值 SSM 的复数算子完整替换为**实值 2×2 块同构**，在代数层面实现"无复数"的复数 SSM，从而安全启用 BF16 混合精度训练，大幅降低显存占用并加速训练。

## 🧭 架构总览

```mermaid
flowchart TD
    A["Token IDs"] --> B["Embedding"]
    B --> C["ParallelComplexWaveLayer × L<br/>complex SSM → real 2×2 blocks"]
    C --> D["MoE-FFN<br/>SwiGLU experts, Top-2 routing"]
    D --> E["Global Summarizer<br/>read / write head"]
    E --> F["LM Head"]
```

核心状态方程（唯一路径）：

```
h_t = A_t @ h_{t-1} + b_t          # 并行前缀扫描
A_t = matrix_exp(δ_t · Ω_t)        # 实值 2×2 块矩阵指数
Ω_t = M + diag(-ν + i·(θ_base + θ_delta_t))
```

- `M`：可学习子通道耦合（斜埃尔米特结构），比对角 / 标量矩阵更具表达力
- 稳定性由 **Bendixson 定理**结构性保证（`Re Ω < 0`），不依赖任何阈值
- `iθ` 项天然提供位置编码（等价 RoPE），无需额外的正弦位置编码

## 🗂 仓库结构

```
Waver-SNN-SSM/
├── v1/          # 早期 1.6B 训练快照（312002proplus_elite）
├── v2/          # 312003elite + 与 Transformer 的受控对照
├── v3.1/        # 统一 matrix_exp 实值矩阵扫描（312003elite）
├── v4/          # Loop 机制 + SSM 分块扫描（312004elite）
├── v4.1/        # 稳定性更新与受控对照（310004elite）
├── v4.2/        # jsonl 显存碎片修复（312004elite）
├── v5/          # v5 数据格式层（312005elite / 312505elite）
├── v6/          # 闭式矩阵指数 + 批量分词（312006elite / 312506elite）
├── v7/          # 训练步 CUDA Graph + 8-bit AdamW + θ 结构性上界 / δ 下限（312007elite / 312008elite / 312017elite / 312507elite / 3125008elite / 3125017elite）
├── v8/          # 遗忘平衡注入 · 峰值增益结构性有界（312009elite / 3125009elite）+ Linux 单卡加速（312009elitelinux / 3125009elitelinux）
├── compare/     # 对照实验：round3 全套图 + v6 验证报告 + 看板工具
├── count_tokens.py / plot_training.py / plot_watch.py / overview_viewer.html
├── tokenizer_incremental.bin / tokenizer_incremental_tokenizer.json
└── README.md / CHANGELOG.md / LICENSE / CITATION.cff / DOI.txt
```

**命名规则**：`312005elite.py` 为主族（层内耦合），`312505elite.py` 为 x25 族（层外耦合）。**v5 起两族同时发布**；v4 及以前仅发布 `31200x`。每个 `.py` 均为**单文件**、自包含全流程（训练 / 推理 / 强化学习）。v7.1.0 起主族 / x25 族各增设 `017` 变体（θ 结构性上界，默认开启）；v7.2.0 起再增设 `008` 变体（回退 θ 上界 + 新增 δ 下限，默认开启）；v8.0.0 起增设 `009` 变体（遗忘平衡注入，峰值增益结构性有界，默认开启）；v8.1.0 起增设 `009elitelinux` 变体（Linux 单卡编译加速 + 多进程数据加载，独立可关，默认开启；全关时与对应 009 逐位一致）。

## 📜 版本历史

| 版本 | 日期 | 核心变更 |
|---|---|---|
| 1.0.0 | 2026-09-05 | 首次公开发布：复值 SSM 实值化 2×2 块同构、MoE-FFN、全局摘要器、自定义 Blelloch 并行前缀扫描 |
| 1.0.1 | 2026-09-05 | FastBPE 逐 token 性能修复（`_vocab_size_cache`）|
| 1.0.2 | 2026-09-05 | `brain.pt` 惰性加载，聊天不再触发 OOM |
| 2.0.0 | 2026-09-09 | 性能优化与工程加速：Blelloch 填充修复、Ω 对角复扫描、分块线性注意力、并行 Prefill、CUDA Graph 解码、MoE 独立 Copy Stream / BF16 常驻 |
| 3.1.0 | 2026-09-17 | 架构重构：统一 `A_t = matrix_exp(δ_t·Ω_t)` 实值矩阵扫描，删除 eig 路径，δ 逐通道化，新增数据依赖旋转 `theta_delta` |
| 4.0.0 | 2026-09-23 | Loop 机制（权重共享循环层）+ SSM 分块扫描 `SSM_CHUNK_SIZE` |
| 4.1.0 | 2026-09-28 | 稳定性更新 + 与经典 Transformer 的受控对照（层内耦合开 / 关）|
| 4.2.0 | 2026-10-02 | 修复 jsonl 路径显存碎片 |
| 5.0.0 | 2026-10-04 | 数据格式层（messages / reasoning_content / tool_calls）+ 词表 / 分词修复 + 配置精简 |
| 6.0.0 | 2026-10-10 | 闭式矩阵指数（K=2）+ 批量分词 + 解码 CUDA Graph 默认启用 |
| **8.1.0** | **2026-10-12** | **009elitelinux 变体：Linux 单卡编译加速（`torch.compile`/Inductor-Triton）+ DataLoader 多进程，独立可关、全关逐位一致** |
| 8.0.0 | 2026-10-11 | 009 变体：遗忘平衡注入 `B̄ = B̃·(1−|A|)`，峰值增益解耦恒等 `G_peak ≡ |B̃|`、耦合有界（≲3.5×） |
| 7.2.0 | 2026-10-11 | 008 变体：回退 θ 结构性上界 + 新增 δ 下限（治无遗忘积分器 / amp 自激） |
| 7.1.0 | 2026-10-11 | 017 变体：相位基频 θ 结构性上界 `θ = π·sigmoid(theta_raw)` |
| 7.0.3 | 2026-10-11 | 修复 CUDA Graph 解码 × AMP 生成崩溃（autocast 缓存冲突） |
| 7.0.2 | 2026-10-10 | 菜单 11 切分保留 system 段 + 切分长度对齐 + 日志降噪 |
| 7.0.1 | 2026-10-10 | 修复 system 段丢失 + 以 user 结尾样本护栏 |
| 7.0.0 | 2026-10-10 | 训练步 CUDA Graph + 8-bit AdamW + 矩阵指数开关集中化 |

## 🚀 8.1.0 核心变更

> 定位：v8 的**Linux 单卡加速变体** —— 在 009 基础上派生 **009elitelinux**：把 Win11 上因 Triton / Inductor 支持受限而无法启用的编译加速（`torch.compile` / Inductor-Triton 融合）与 DataLoader 多进程等**适用于单卡**的优化做成**独立可关**的开关（默认开启）。不改模型结构 / 优化器数学；**全部开关关闭时与对应 009 逐位一致**。

- **文件头集中开关区**（每个独立可关；全关 ⇒ 与对应 009 **逐位一致**）：
  - `LINUX_USE_TORCH_COMPILE`（**默认 True**）：用 Inductor + Triton 融合**标准算子** —— `RMSNorm.forward`、FFN / SwiGLU 专家块、LM Head 分块交叉熵。
  - `LINUX_COMPILE_MODE`（默认 `max-autotune-no-cudagraphs`）：显式不用 Inductor 自身的 cudagraph trees，避免与手写训练步 CUDA Graph 叠加冲突。
  - `LINUX_COMPILE_DYNAMIC`（默认 `False`）：静态形状（batch=1、T 固定）利于编译。
  - `LINUX_NUM_WORKERS`（默认 `4`）：DataLoader 多进程 + `persistent_workers`（覆盖文件内 `NUM_WORKERS`）；Windows 曾被迫设 0，Linux 侧稳定。
  - `LINUX_OPTIMIZER`（默认 `'bnb8bit'`）：`'bnb8bit'`（= 009 原样）\| `'fused_adamw'`（`torch.optim.AdamW(fused=True)`）\| `'torchao8bit'`（可选，会改数值）。
  - `LINUX_USE_TF32`（默认 `False`）：可选开 `allow_tf32`，会改数值，默认关。
- **自定义算子隔离**：对 `BlellochScanFn` 扫描核心与矩阵指数闭式实现（x25 族为 `_capturable_expm_skew`，其余为 `_closed_form_expm_2x2` / `_capturable_expm`）加 `torch.compiler.disable`，避免 graph break 破坏其余融合。
- **失败兜底**：`try/except` 包裹编译，编译不可用 / 失败 ⇒ 自动回退 eager 并写日志（与 8-bit 优化器双重兜底同风格）。
- **009elitelinux 变体**：ultra3 的 6 个 009 文件各派生一个 linux 变体；本次 release 发布主族 `312009elitelinux.py` 与 x25 族 `3125009elitelinux.py`，其余 4 个仅留私有开发目录。

### 验证与边界
- **数值等价**：固定输入前向，开关全关时 logits 最大绝对误差 **0.0**（逐位一致）；全开（WSL, sm_120）时 logits 最大绝对误差 **9.537e-07 < 1e-6**。
- **收敛一致性（超小 A/B，同一张 5080）**：Win11 未优化 `312009elite.py` vs WSL 优化 `312009elitelinux.py`（VOCAB≈256 / H=128 / L=2 / T=256，均 534 步）：最终 Loss 1.0182 vs 1.0178、PPL 2.77 vs 2.77，逐步 Loss 曲线走势一致、**无正确性回归**。证据见 [`compare/linux/`](./compare/linux/)。
- **边界**：SSM 核心（自定义 autograd 扫描 + 闭式矩阵指数）被**隔离为 eager**，编译融合收益只在 norm / FFN / CE 三处标准算子；**超小实验只证明「收敛一致 + 无正确性回归」，不能代表真实规模的吞吐收益** —— 本版本只声称「Linux 单卡加速路径可用 + 收敛一致性已验」，**不声称**具体加速比。单卡场景，**不含** NCCL / DDP / FSDP 等多卡项。

## 🚀 8.0.0 核心变更

> 定位：v8 的**动力学稳定性分支（注入端结构性修法）** —— 在 007 基础上派生 **009 变体**：把注入项从 **ZOH 连续积分** 改为 **遗忘平衡注入** `B̄ = B̃·(1−|A|)`，使**峰值增益**在**解耦极限（M=0）下恒等** `G_peak ≡ |B̃|`、在**耦合（M≠0）下保持有界**（实测 ≲3.5×），从结构上消除慢极点（ν→0）的直流增益爆炸。不改动模型结构 / 优化器 / 数值路径（`GAIN_NORM_ENABLED=False` 时与 007 逐位一致）。

- **峰值增益病灶**：标量复递推 `h_t = A·h_{t−1} + B̄·x_t` 的峰值增益为 `G_peak = |B̄|/(1−|A|)`（单位圆到 A 最近距离的倒数）。旧 ZOH 注入 `B̄ = B·(e^{δγ}−1)/γ = B·(1−A)/γ` 给出 `G_peak = |B|/|γ| · |1−A|/(1−|A|)`，在 `ν→0`（`1−|A|→0`）时**无界** —— 慢极点会把直流增益放大到任意大。实测更正：固定 ν 单独令 `|γ|→0` 增益不变，故无界入口**仅** `ν→0` 一条（此前“双入口”表述已收窄）。
- **遗忘平衡注入开关** `GAIN_NORM_ENABLED`（**默认开启**）：新规则 `B̄ = B̃·(1−|A|)`，其中 `B̃ = B_real·B_scale`（原可学习系数，不动），`|A| = e = exp(−δν)`。于是**解耦（M=0）时 `G_peak ≡ |B̃|`（精确恒等）**；**耦合（M≠0）时** `A` 非正规，`G_peak` 偏离 `|B̃|`、可放大到同量级倍数（实测 K≤8、ν≥1e-4 时 ≲3.5×），但**始终有界**。数值验证：小 ν 通道旧式 `G_peak` 随 ν→0 无界（实测 ν=1e-3→~962、ν=1e-4→~9546），新式保持 O(1)（解耦 ≈1，耦合 ≲2.4）。
- **重参数化而非截断**：任意目标 `B̄` 可由 `B̃ = B̄/(1−|A|)` 精确表示（δ>0、ν>0 严格 ⇒ `0 < 1−|A| ≤ 1`，无除零），**可达集合与 007 完全相同**；实现为纯乘加、无开方无除法，梯度安全。
- **009 变体**：ultra3 的 6 个 007 文件（`210/212/2125/310/312/3125`）各派生一个 009；本次 release 发布主族 `312009elite.py` 与 x25 族 `3125009elite.py`，同时保留 007 供对照（009 = 007 + 遗忘平衡注入）。
- **旗舰版同步**：旗舰版 312007 派生 `312009elite.py`（= 旗舰版 312007 + 遗忘平衡注入；梯度累积 / 裁剪 / LR 全部不动）。

### 破坏性变更与边界
- 开关开启后，加载旧 007 checkpoint 的前向输出**不再逐位一致**（注入公式已改变）；如需与 007 严格对照，置 `GAIN_NORM_ENABLED=False` 即可**逐位回退**。**009 不是 007 的逐位兼容版本。**
- **对训练收敛的影响待验证**：本变体由「峰值增益结构性有界」支撑（已数值验证：**解耦 `G_peak ≡ |B̃|` 精确，耦合有界 ≲3.5×、非严格上界**）；但其对病态 PPL 收敛的长期效果**尚未经实验证实** —— 先前对照实验中把增益压制 ~160×，在 200 步续训窗口内均未观察到 PPL 同步改善。**请勿解读为「已修复 PPL 震荡」。**

## 🚀 7.2.0 核心变更

> 定位：v7 的**动力学稳定性分支（修正 7.1.0 的抓手）** —— 在 007 基础上派生 **008 变体**：**回退 017 的 θ 结构性上界**（回到原生 `θ = exp(theta_raw)`），并新增 **δ 下限** 软约束，直击 0.6B 训练中 δ→0 导致的无遗忘积分器与 amp 自激。不改动模型结构 / 优化器 / 数值路径（`DELTA_MIN_ENABLED=False` 时与 007 逐位一致）。

- **修正：017 的 θ 上界抓错了变量**。旋转本身**模 2π**，「转过头」不存在；真正的现象是**采样混叠**（`δ·θ > π`）。实测 δ~0.05，混叠起点 `θ ≈ π/δ ≈ 30`，而 π 上限远未触界；且 `π·sigmoid` 在边界附近梯度趋零、反害优化。故 008 **回退原生 `θ = exp(theta_raw)`**，与 007 逐位一致（`THETA_STRUCTURAL_BOUND_ENABLED=False`，开关保留供对照）。
- **δ 下限开关** `DELTA_MIN_ENABLED`（**默认开启**）与 `DELTA_MIN`（默认 `0.01`）：病根是 `δ = softplus(delta_raw)·softplus(delta_scale)`，训练可把 `delta_scale` 推向 −∞ 使 δ→0；此时 `A = exp(δ·Ω) → I`，该通道退化为**无遗忘积分器**（记忆无界）→ BPTT 梯度沿时间不衰减而爆掉 → amp 自激。旗舰版 0.6B 实测：`amp_std` 从 0.0001 涨到 1231，`PPL` 长期 2000+ 震荡；而 `|A| = exp(−δν)` 与 θ 无关，故卡 θ 治不了自激，真正的抓手是 δ。
- **加性软下限** `δ ← DELTA_MIN + δ`：保证 `δ ≥ DELTA_MIN` 且**处处可导、无梯度死区**（梯度 1:1 透传）；最坏情形 `|A| ≤ exp(−DELTA_MIN·ν_min) < 1`，记忆时域有界。`DELTA_MIN_ENABLED=False`（或 `DELTA_MIN=0`）时与 007 逐位一致。
- **008 变体**：ultra3 的 6 个 007 文件（`210/212/2125/310/312/3125`）各派生一个 008；本次 release 发布主族 `312008elite.py` 与 x25 族 `3125008elite.py`，同时保留 `312007elite.py` / `312507elite.py` 供对照（008 = 007 + δ 下限）。

## 🚀 7.1.0 核心变更

> 定位：v7 的**结构性稳定性分支** —— 在 007 基础上派生 **017 变体**，把相位基频 θ 的 `exp` 参数化改为**结构性上界** `θ = π·sigmoid(theta_raw)`；不改动模型结构 / 优化器 / 数值路径（开关关闭时与 007 逐位一致）。

- **θ 结构性上界开关** `THETA_STRUCTURAL_BOUND_ENABLED`（**默认开启**）：物理来源是离散旋转的非混叠（Nyquist）约束 `δ·θ < π` ⇒ 基频 θ 应落在 `(0, π)`。参数化 `θ = π·sigmoid(theta_raw)` **处处可导、无梯度死区、渐近逼近 π 而不真正触界**；替代原生 `θ = exp(theta_raw)`（无上界，θ 可漂移至任意大 → 相位自激）。置 `False` 回退原生 `exp` 参数化，与 007 **逐位一致**。
- **初始化一致**：结构性上界下反解 `theta_raw`，使初始 θ 分布与原生 log-uniform 完全一致（往返误差 ~1e-16）。
- **017 变体**：ultra3 的 6 个 007 文件（`210/212/2125/310/312/3125`）各派生一个 017；本次 release 发布主族 `312017elite.py` 与 x25 族 `3125017elite.py`，同时保留 `312007elite.py` / `312507elite.py` 供对照。

### 7.0.3 生成路径正确性补丁

> 定位：v7 的**生成路径正确性补丁** —— 修复启用 `USE_CUDA_GRAPH_DECODE` 时，菜单 2 对话在 AMP（bf16）下生成若干步后崩溃的问题；不改动模型结构 / 优化器 / 数值路径。

- **CUDA Graph 解码 × autocast 缓存冲突**：`generate` 原先让 `prefill` 与每一步解码各自单独进出 `torch.amp.autocast`。抓图前的预热在 autocast 内执行，会把 bf16 权重副本写入 autocast 缓存；随后的抓图直接复用这些缓存张量、未记录 cast 算子；而退出最外层 autocast 会触发 `clear_autocast_cache()` 释放这些副本 —— 图回放便读到已被复用的显存，数值发散产生 NaN/Inf 非法分布，最终 `torch.multinomial` 触发设备端断言、进程硬崩。
- **修复方式**：把整个 `prefill` + 逐 token 解码循环保持在**同一个** autocast 上下文内，不再每步单独进出。autocast 缓存全程存活，抓图烘焙的 bf16 权重始终有效。
- **验证**：同一 checkpoint（seed=0）下，7.0.2 开启图解码必崩；修复后正常生成无异常；固定 token 序列上「图解码 vs eager」logits 最大绝对误差为 **0.0**（逐位一致）。

### 7.0.2 数据预处理与日志降噪

> 定位：v7 的**数据预处理与日志降噪补丁** —— 修复菜单 11 切分后 system 段丢失、对齐切分长度口径，并消除无 triton 环境下的无害告警；不改动模型结构 / 优化器 / 数值路径。

- **菜单 11 切分保留 system 段**：`p4_segment_dataset` 按 `(user, assistant)` 对切分时会把对话开头的连续 `system` 段丢弃，与训练路径 `_truncate_pairs` 语义不一致。现改为先摘出开头连续 `system` 段并计入 token 预算，切分后拼接到每条样本前（`{'messages': head + list(pair)}`），保证 system 在多轮切分后仍可见。
- **切分长度口径对齐**：`pair_tokens` 补上真实样本长度的 `+1`（`input_ids = tokens[:-1]`），避免边界样本多留一条，与训练侧长度口径一致。
- **日志降噪**：`bitsandbytes` 导入会顺带导入 `torch.utils.flop_counter`，在无 triton 的 CUDA 环境下打一条 `triton not found; flop counting will not work for triton kernels` 的 WARNING。现将其 logger 降至 ERROR（纯降噪，不影响任何功能）。

### 7.0.1 数据管线修复

> 定位：v7 的**数据管线正确性补丁** —— 修复训练样本中 system 段丢失、以及残缺样本污染生成行为两处问题；不改动模型结构 / 优化器 / 数值路径。

- **修复 system 消息丢失**：`_truncate_pairs` 只识别相邻 `(user, assistant)` 对，会把对话开头的连续 `system` 段整段丢弃（`render_messages` 支持 system、训练语料里却永远看不到 system）。现改为先把开头连续 system 段原样摘出并计入 token 预算，再从尾部取 `(user, assistant)` 对合并返回。
- **以 user 结尾样本的护栏**：仅含 user（或 system + user）、无任何 assistant 回复的残缺样本，其唯一训练目标只剩结尾的 `<|eos|>`，会污染生成行为。新增 `_has_assistant_target` 护栏，`StreamingJSONLDataset` 迭代时跳过此类样本。

### 7.0.0 基础（无损加速 + 降显存）

> 定位：v7 是 v6 的**训练侧「无损加速 + 降显存」版本** —— 所有新增开关默认启用、失败自动回退、数学逐位一致。

- **训练步 CUDA Graph** `USE_TRAIN_STEP_CUDA_GRAPH`：把训练步的 **fwd + bwd** 抓成 CUDA Graph（按序列长度分桶缓存，`TRAIN_GRAPH_BUCKET` / `TRAIN_GRAPH_MAX_GRAPHS`），消除每步约 10 万次 kernel 启动开销。`optimizer.step()` / `scheduler.step()` / 梯度裁剪留在图外；仅 CUDA + 非 MoE + batch=1 生效，抓图失败自动回退 eager（**仅性能回退，数学不变**）。
- **8-bit AdamW** `USE_ADAMW8BIT`：主训练优化器改用 bitsandbytes `AdamW8bit`（8-bit 动量状态），约省 **36%** 优化器显存；非 CUDA 或未安装 bitsandbytes 时自动回退本文件原生优化器（Adafactor / AdamW）。
- **矩阵指数 / 抓图开关集中化**：统一放在文件头部。新增 `USE_CAPTURABLE_EXPM`（K>2 走「缩放-平方 + Taylor 级数」，纯 matmul / 逐元素、无 LAPACK，可被 CUDA Graph 捕获；**默认关闭**，K>2 默认回退 `torch.linalg.matrix_exp`）。
- **梯度检查点默认开启**；loss 缩放与 `zero_grad(set_to_none=False)` 适配训练图。
- 精简对照实验纳入 `compare/`；README 重写 + 新增 `CHANGELOG.md`。

## 📖 核心特性

- **彻底的复数化（Complex-to-Real 2×2 块同构）**
  每个复矩阵元 `a+bi` 被替换为实矩阵 `[[a, -b], [b, a]]`，实虚通道自然交替，**无任何 complex 中间张量**，可安全开启 BF16 AMP。
- **可选的全局摘要器（读写头）**
  `Mean` / `LinearAttention` / `MiniWave` 三种全局摘要器，为 SSM 提供显式全局信息，弥补其固有的上下文压缩局限。
- **更灵活的层内耦合（Intra-layer Coupling）**
  内置可学习的子通道间耦合矩阵（斜埃尔米特结构），比 Mamba 的对角 / 标量矩阵更具表达力。
- **完整的 MoE 集成与工业级优化**
  MoE-FFN（SwiGLU 专家 + Top-K 路由），含 Expert Offload（CPU 卸载）、独立 Copy Stream 异步搬运、FP8 / NVFP4 / BF16 量化、Weight Normalization。
- **单文件 + 全配置**
  训练、推理、强化学习（GRPO / RL）全流程单文件封装，所有超参数由文件头部常量集中控制。
- **跨平台稳定**
  Win11 / Linux / macOS 全平台支持，DataLoader 多 worker 在 spawn 模式下可安全使用。

## ⚙️ 核心组件

### ParallelComplexWaveLayer
模型核心，将复值 SSM 的所有算子替换为实值 2×2 块：

- 状态转移：`h_t = A_t @ h_{t-1} + b_t`
- 唯一路径：`A_t = matrix_exp(δ_t · Ω_t)`，其中 `Ω_t = M + diag(-ν + i·(θ_base + θ_delta_t))`
- 并行扫描：自定义 `BlellochScanFn` 实现 O(log T) 并行前缀扫描，前向 / 反向均可并行
- 稳定性由 **Bendixon 定理**结构性保证（`Re Ω < 0`），不依赖任何阈值

### 全局摘要器（Summarizer）
通过 `SUMMARY_TYPE` 切换：
- `mean`：全局平均池化（无参数基线）
- `linear_attention`：可学习的线性注意力摘要，支持分块（`SUMMARY_CHUNK`）避免大张量物化
- `mini_wave`：迷你 `ParallelComplexWaveLayer` 实例，独立参数

### MoE-FFN
SwiGLU 混合专家：
- `MOE_OFFLOAD`：专家权重可卸载到 CPU，按需搬移
- `_copy_stream`：独立 CUDA 流异步搬运，与计算重叠
- `MOE_EXPERT_QUANT`：支持 FP8（E4M3）训练量化、NVFP4（E2M1）推理量化、BF16 常驻推理
- `MOE_EXPERT_WNORM`：权重归一化，防止训练发散

## 📊 对照实验（含图）

全部结果由 `对照测试/` 的原始训练日志**重新解析生成**；**共 60 组实验，每条一行、附数据结论的完整对照矩阵**见 [`compare/README.md`](./compare/README.md)，图集见 [`compare/`](./compare/)。
统一条件：**同词表 / 同优化器 / 同超参数**，`H=256, L=3, T=2048, batch=1, lr=1e-3`，约 8.9 万步。

### 1. 第 3 轮受控对照：Waver / x25 vs 经典 Transformer

三联图（训练 Loss · 验证 PPL · 相对基线 Loss 差值）：

![round3 main](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_main.png)

> Waver v4.2 / x25 的训练 Loss 整体**略低于** Transformer 基线，验证 PPL 高度接近。

**多 seed 稳定性**（各 5 个 seed：42 / 43 / 114514 / 1919810 / 31415926）：

![waver seeds](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_seeds_waver.png)

![x25 seeds](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_seeds_x25.png)

![transformer seeds](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_seeds_transformer.png)

> Waver 的验证 PPL 在 5 个 seed 上**高度稳定**（31.1–31.6），Transformer 波动明显更大（22.3–30.0）。

**消融与扫描**（完整 15 图见 [`compare/round3/`](./compare/round3/)）：

| 学习率扫描（Waver） | 层内耦合系数消融 | 状态维度 K=4 |
|---|---|---|
| ![lr scan](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_lr_waver.png) | ![coupling](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_coupling.png) | ![k4](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_state_dim_k4.png) |

> `lr=8e-3` 时明显恶化并提前中断（印证复值 SSM 对过高学习率的敏感性）；耦合系数消融的 ValPPL 极差仅 0.65（过大或完全无耦合略差）。
> 其余消融：Loop 循环层、h368 更宽隐藏层、Warmup、摘要器 dk、10 万条数据 —— 详见对照矩阵。

**全实验总览**（全部 55 个 round3/100k run 的训练曲线 + 最优 ValPPL 排序）：

| 训练 Loss 小图总览 | 最优验证 PPL 排序 | 多 seed 稳定性 |
|---|---|---|
| ![matrix loss](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/matrix/matrix_loss_all.png) | ![matrix valppl](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/matrix/matrix_valppl.png) | ![matrix seed](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/matrix/matrix_seed_variance.png) |

> 多 seed 实测：最优 ValPPL 的跨 seed 标准差 —— Transformer H256 **±2.89**、Waver H256 **±0.19**、x25 H256 **±0.24**（Waver/x25 方差低约一个数量级，但绝对 PPL 水平更高）。

### 2. v1 1.6B 模型训练总览

![v1 overview](https://x005-0001-0001.github.io/Waver-SNN-SSM/v1/plots/overview.png)

> 1.6B 参数模型在约 17.9M tokens（远低于 Chinchilla 最优）下，Loss 从 ~10 稳定降至 ~5，PPL 从 ~49000 降至 ~200 —— 架构有效性的有力证据。

### 3. v6 低精度 + 训练侧加速验证（RTX 5080 16GB）

| 结论 | 数据 |
|---|---|
| 8-bit AdamW 稳定性 | 全程 MON 指标相对偏差 < 3.4e-3（bf16 量化噪声量级），无漂移 |
| 8-bit AdamW 显存 | 优化器显存 **-35.7%**（421.7MB → 271.1MB）|
| FP8 / NVFP4 专家量化 | NVFP4 推理省 **45.6%** 显存，logits 相对误差仅 2e-4 |
| 训练步 CUDA Graph | fwd+bwd 抓图 **2.4–2.6×**；关梯度检查点 + 抓图合计约 **3.4×** |

完整报告与探针脚本见 [`compare/v6/`](./compare/v6/) 与 [`compare/README.md`](./compare/README.md)。

## 🔧 使用

每个版本目录下的 `.py` 即为完整程序，直接运行进入菜单：

```bash
python v8/312009elite.py
```

所有关键参数（模型结构、优化器、数据路径、MoE、显存优化等）均在文件头部集中定义。

---

## 📜 引用 / Citation

```bibtex
@misc{zeng2026waversnnssm,
  author = {Zeng, Naiqi},
  title = {Waver-SNN-SSM: From Spiking Neurons to State Space Models},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.22309186},
  url = {https://doi.org/10.5281/zenodo.22309186}
}
```

## 📧 联系方式 / Contact
- ORCID: [0009-0006-4332-1295](https://orcid.org/0009-0006-4332-1295)
- Email: X005_0001_0001@163.com

## 📄 许可证 / License
本项目采用 **GNU General Public License v3.0**。任何使用、修改或分发本代码的行为，都必须遵守 GPL-3.0 的条款，即任何衍生作品也必须以 GPL-3.0 开源。

## ⚠️ 版权声明 / Copyright
Copyright © 2026 Naiqi Zeng. All rights reserved.
本代码及相关文档的首次公开发布时间由 Zenodo DOI 10.5281/zenodo.22309186 锁定。任何未经授权的抄袭、洗稿或未注明来源的引用，均视为侵犯版权。

---

<a name="english-documentation"></a>

# English Documentation

## 📌 Overview

Waver-SNN-SSM starts from spiking neural networks (SNN) and evolves into an efficient state-space model (SSM) that converges, by a different route, with the ideas behind Mamba-3. Its core innovation is replacing every complex operator of a complex-valued SSM with a **real 2×2 block isomorphism** — a "complex-free" complex SSM at the algebraic level — which safely unlocks BF16 mixed-precision training and substantially reduces memory usage while accelerating training.

## 🧭 Architecture

```mermaid
flowchart TD
    A["Token IDs"] --> B["Embedding"]
    B --> C["ParallelComplexWaveLayer × L<br/>complex SSM → real 2×2 blocks"]
    C --> D["MoE-FFN<br/>SwiGLU experts, Top-2 routing"]
    D --> E["Global Summarizer<br/>read / write head"]
    E --> F["LM Head"]
```

Core state dynamics (single unified path):

```
h_t = A_t @ h_{t-1} + b_t          # parallel prefix scan
A_t = matrix_exp(δ_t · Ω_t)        # real 2×2 block matrix exponential
Ω_t = M + diag(-ν + i·(θ_base + θ_delta_t))
```

- `M`: learnable inter-subchannel coupling (skew-Hermitian), more expressive than diagonal/scalar matrices
- Stability is structurally guaranteed by the **Bendixson theorem** (`Re Ω < 0`), with no thresholds
- The `iθ` term natively provides positional encoding (equivalent to RoPE); no separate sinusoidal PE is needed

## 🗂 Repository Layout

```
Waver-SNN-SSM/
├── v1/          # early 1.6B training snapshots (312002proplus_elite)
├── v2/          # 312003elite + controlled comparison vs. Transformer
├── v3.1/        # unified matrix_exp real-matrix scan (312003elite)
├── v4/          # Loop mechanism + chunked SSM scan (312004elite)
├── v4.1/        # stability update + controlled comparison (310004elite)
├── v4.2/        # jsonl memory-fragmentation fix (312004elite)
├── v5/          # v5 data-format layer (312005elite / 312505elite)
├── v6/          # closed-form matrix exponential + batched tokenization (312006elite / 312506elite)
├── v7/          # training-step CUDA Graph + 8-bit AdamW + θ structural bound / δ floor (312007elite / 312008elite / 312017elite / 312507elite / 3125008elite / 3125017elite)
├── v8/          # forget-balanced injection · structurally bounded peak gain (312009elite / 3125009elite) + Linux single-GPU accel (312009elitelinux / 3125009elitelinux)
├── compare/     # comparison: round3 figure set + v6 report + dashboards
├── count_tokens.py / plot_training.py / plot_watch.py / overview_viewer.html
├── tokenizer_incremental.bin / tokenizer_incremental_tokenizer.json
└── README.md / CHANGELOG.md / LICENSE / CITATION.cff / DOI.txt
```

**Naming**: `312005elite.py` is the main family (intra-layer coupling); `312505elite.py` is the x25 family (inter-layer coupling). **Both families ship from v5 onward**; v4 and earlier ship only `31200x`. Every `.py` is a **single self-contained file** covering training / inference / RL. From v7.1.0 each family also ships an `017` variant (θ structural bound, on by default); from v7.2.0 it additionally ships an `008` variant (θ bound reverted + δ floor, on by default); from v8.0.0 it adds a `009` variant (forget-balanced injection, structurally bounded peak gain, on by default); from v8.1.0 it adds a `009elitelinux` variant (Linux single-GPU compile acceleration + multiprocess data loading, independently switchable, on by default; with all switches off it is bit-identical to the corresponding 009).

## 📜 Version History

| Version | Date | Highlights |
|---|---|---|
| 1.0.0 | 2026-09-05 | First public release: complex SSM as real 2×2 blocks, MoE-FFN, global summarizers, custom Blelloch parallel prefix scan |
| 1.0.1 | 2026-09-05 | FastBPE per-token performance fix (`_vocab_size_cache`) |
| 1.0.2 | 2026-09-05 | Lazy `brain.pt` loading; chat no longer triggers OOM |
| 2.0.0 | 2026-09-09 | Performance & engineering: Blelloch padding fix, Ω eigendecomposition → diagonal complex scan, chunked linear attention, parallel prefill, CUDA Graph decoding, MoE dedicated copy stream / BF16 residency |
| 3.1.0 | 2026-09-17 | Architecture refactor: unified `A_t = matrix_exp(δ_t·Ω_t)` real-matrix scan, removed the eig path, per-channel δ, data-dependent rotation `theta_delta` |
| 4.0.0 | 2026-09-23 | Loop mechanism (weight-shared recurrent layers) + chunked SSM scan `SSM_CHUNK_SIZE` |
| 4.1.0 | 2026-09-28 | Stability update + controlled comparison vs. a classic Transformer (coupling on/off) |
| 4.2.0 | 2026-10-02 | jsonl memory-fragmentation fix |
| 5.0.0 | 2026-10-04 | v5 data-format layer (messages / reasoning_content / tool_calls) + tokenizer fixes + config cleanup |
| 6.0.0 | 2026-10-10 | Closed-form matrix exponential (K=2) + batched tokenization + CUDA Graph decoding on by default |
| **8.1.0** | **2026-10-12** | **009elitelinux variant: Linux single-GPU compile acceleration (`torch.compile`/Inductor-Triton) + multiprocess DataLoader, independently switchable, bit-identical when all off** |
| 8.0.0 | 2026-10-11 | 009 variant: forget-balanced injection `B̄ = B̃·(1−|A|)`, peak gain `G_peak ≡ |B̃|` structurally bounded |
| 7.2.0 | 2026-10-11 | 008 variant: θ structural bound reverted + new δ floor (fixes the forgetless integrator / amp self-excitation) |
| 7.1.0 | 2026-10-11 | 017 variant: structural upper bound on the base phase θ, `θ = π·sigmoid(theta_raw)` |
| 7.0.3 | 2026-10-11 | Fix CUDA Graph decoding × AMP generation crash (autocast-cache conflict) |
| 7.0.2 | 2026-10-10 | Menu 11 segmentation keeps the system span + aligned segmentation length + log-noise cleanup |
| 7.0.1 | 2026-10-10 | Fix dropped system span + guard for user-ending samples |
| 7.0.0 | 2026-10-10 | Training-step CUDA Graph + 8-bit AdamW + centralized matrix-exponential switches |

## 🚀 What's New in 8.1.0

> v8.1.0 is the **Linux single-GPU acceleration variant** of v8 — it derives **009elitelinux** from 009: it turns the compile acceleration (`torch.compile` / Inductor-Triton fusion) and multiprocess data loading that were unavailable on Win11 (limited Triton / Inductor support) into **independently switchable** flags for a **single GPU** (on by default). The model architecture and optimizer math are untouched; **with all switches off it is bit-identical to the corresponding 009**.

- **Centralized switch block at the top of the file** (each independently switchable; all off ⇒ **bit-identical** to the corresponding 009):
  - `LINUX_USE_TORCH_COMPILE` (**on by default**): Inductor + Triton fusion of **standard ops** — `RMSNorm.forward`, the FFN / SwiGLU expert block, and the chunked LM-head cross-entropy.
  - `LINUX_COMPILE_MODE` (default `max-autotune-no-cudagraphs`): explicitly avoids Inductor's own cudagraph trees so it can coexist with the hand-written training-step CUDA Graph.
  - `LINUX_COMPILE_DYNAMIC` (default `False`): static shapes (batch=1, fixed T) favor compilation.
  - `LINUX_NUM_WORKERS` (default `4`): multiprocess DataLoader + `persistent_workers` (overrides the in-file `NUM_WORKERS`); Windows was forced to 0, Linux is stable.
  - `LINUX_OPTIMIZER` (default `'bnb8bit'`): `'bnb8bit'` (= 009 as-is) \| `'fused_adamw'` (`torch.optim.AdamW(fused=True)`) \| `'torchao8bit'` (optional, changes numerics).
  - `LINUX_USE_TF32` (default `False`): optionally enables `allow_tf32`; changes numerics, off by default.
- **Custom-op isolation**: `torch.compiler.disable` on the `BlellochScanFn` scan core and the closed-form matrix-exponential implementations (x25 family: `_capturable_expm_skew`; the rest: `_closed_form_expm_2x2` / `_capturable_expm`) to avoid graph breaks breaking the rest of the fusion.
- **Failure fallback**: the compile is wrapped in `try/except`; unavailable / failed compilation ⇒ automatically falls back to eager with a log line (same style as the 8-bit optimizer double fallback).
- **009elitelinux variant**: each of the 6 ultra3 009 files derives a linux variant; this release ships the main family `312009elitelinux.py` and the x25 family `3125009elitelinux.py`, the other 4 stay in the private dev directory only.

### Verification & scope
- **Numerical equivalence**: on a fixed forward input, with all switches off the logits max absolute error is **0.0** (bit-identical); with all on (WSL, sm_120) it is **9.537e-07 < 1e-6**.
- **Convergence consistency (tiny A/B, same 5080)**: Win11 unoptimized `312009elite.py` vs. WSL optimized `312009elitelinux.py` (VOCAB≈256 / H=128 / L=2 / T=256, both 534 steps): final Loss 1.0182 vs. 1.0178, PPL 2.77 vs. 2.77, step-wise Loss curves agree, **no correctness regression**. Evidence in [`compare/linux/`](./compare/linux/).
- **Scope**: the SSM core (custom autograd scan + closed-form matrix exponential) is **isolated as eager**, so the fusion gains live only in norm / FFN / CE; **the tiny experiment only proves "consistent convergence + no correctness regression", not real-scale throughput** — this release only claims "the Linux single-GPU acceleration path works and convergence consistency is verified", and makes **no** claim about a specific speedup. Single-GPU scope, **excluding** NCCL / DDP / FSDP.

## 🚀 What's New in 8.0.0

> v8.0.0 is the **dynamics-stability branch** of v8 (an injection-side structural fix) — it derives an **009 variant** from 007: it replaces the injection term from **ZOH continuous integration** with **forget-balanced injection** `B̄ = B̃·(1−|A|)`, so that the **peak gain** `G_peak ≡ |B̃|` (constant and bounded for any δ/ν/θ), structurally eliminating the DC-gain blow-up of slow poles (ν→0). The model architecture, optimizers and numerical paths are untouched (with `GAIN_NORM_ENABLED=False` it is bit-identical to 007).

- **The peak-gain pathology**: for the scalar complex recursion `h_t = A·h_{t−1} + B̄·x_t`, the peak gain is `G_peak = |B̄|/(1−|A|)` (the reciprocal of the unit circle's distance to A). The old ZOH injection `B̄ = B·(e^{δγ}−1)/γ = B·(1−A)/γ` gives `G_peak = |B|/|γ| · |1−A|/(1−|A|)`, which is **unbounded** as `ν→0` (`1−|A|→0`) or `|γ|→0` — slow poles can amplify the DC gain without limit.
- **Forget-balanced injection switch** `GAIN_NORM_ENABLED` (**on by default**): the new rule `B̄ = B̃·(1−|A|)`, with `B̃ = B_real·B_scale` (the original learnable coefficients, untouched) and `|A| = e = exp(−δν)`. Hence `G_peak ≡ |B̃|`, **constant and bounded for any δ/ν/θ**. Numerically verified: for small-ν channels the old form reaches `G_peak ≈ 175` while the new form exactly equals `|B̃|` (error ~1e-7, float32).
- **Reparameterization, not truncation**: any target `B̄` is exactly representable via `B̃ = B̄/(1−|A|)` (δ>0 and ν>0 strictly ⇒ `0 < 1−|A| ≤ 1`, no division by zero), so the **reachable set is identical to 007**; it is pure multiply-add with no sqrt and no division, hence gradient-safe.
- **009 variant**: each of the 6 ultra3 007 files (`210/212/2125/310/312/3125`) derives an 009; this release ships the main family `312009elite.py` and the x25 family `3125009elite.py`, while keeping 007 for comparison (009 = 007 + forget-balanced injection).
- **Flagship sync**: the flagship 312007 derives `312009elite.py` (= flagship 312007 + forget-balanced injection; gradient accumulation / clipping / LR all untouched).

### Breaking change & scope
- With the switch on, loading an old 007 checkpoint gives forward outputs that are **no longer bit-identical** (the injection formula changed); set `GAIN_NORM_ENABLED=False` for a **bit-exact fallback** to 007.
- **Effect on training convergence is unverified**: this variant is backed by the **mathematical identity** of "structurally bounded peak gain" (verified numerically as `G_peak ≡ |B̃|`); its long-term effect on the pathological PPL has **not** been established experimentally — in earlier controlled runs the gain was suppressed ~160× yet no concurrent PPL improvement was observed within a 200-step continuation window. **Do not read this as "the PPL oscillation is fixed".**

## 🚀 What's New in 7.2.0

> v7.2.0 is the **dynamics-stability branch** of v7 (it corrects the lever chosen in 7.1.0) — it derives an **008 variant** from 007: it **reverts the 017 θ structural bound** (back to the native `θ = exp(theta_raw)`) and adds a **δ floor** soft constraint that directly targets the forgetless integrator and amp self-excitation caused by δ→0 in 0.6B training. The model architecture, optimizers and numerical paths are untouched (with `DELTA_MIN_ENABLED=False` it is bit-identical to 007).

- **Correction: the 017 θ bound constrained the wrong variable.** Rotation is **mod 2π**, so "turning past half a turn" does not exist; the real phenomenon is **sampling aliasing** (`δ·θ > π`). With measured δ~0.05, aliasing starts at `θ ≈ π/δ ≈ 30`, far beyond the π cap; moreover `π·sigmoid` has a vanishing gradient near the boundary, which hurts optimization. Hence 008 **reverts to the native `θ = exp(theta_raw)`**, bit-identical to 007 (`THETA_STRUCTURAL_BOUND_ENABLED=False`, the switch is kept for comparison).
- **δ-floor switch** `DELTA_MIN_ENABLED` (**on by default**) and `DELTA_MIN` (default `0.01`): the root cause is `δ = softplus(delta_raw)·softplus(delta_scale)` — training can push `delta_scale` toward −∞ so that δ→0; then `A = exp(δ·Ω) → I`, and the channel degenerates into a **forgetless integrator** (unbounded memory) → the BPTT gradient no longer decays over time and blows up → amp self-excitation. Measured on the flagship 0.6B: `amp_std` grows from 0.0001 to 1231 and `PPL` oscillates around 2000+ for a long time; since `|A| = exp(−δν)` is independent of θ, capping θ cannot fix the self-excitation — the real lever is δ.
- **Additive soft floor** `δ ← DELTA_MIN + δ`: guarantees `δ ≥ DELTA_MIN` and is **everywhere differentiable with no gradient dead zone** (gradient passes through 1:1); worst case `|A| ≤ exp(−DELTA_MIN·ν_min) < 1`, so the memory horizon is bounded. With `DELTA_MIN_ENABLED=False` (or `DELTA_MIN=0`) it is bit-identical to 007.
- **008 variant**: each of the 6 ultra3 007 files (`210/212/2125/310/312/3125`) derives an 008; this release ships the main family `312008elite.py` and the x25 family `3125008elite.py`, while keeping `312007elite.py` / `312507elite.py` for comparison (008 = 007 + δ floor).

## 🚀 What's New in 7.1.0

> v7.1.0 is the **structural-stability branch** of v7 — it derives an **017 variant** from 007, replacing the base-phase `exp` parameterization with a **structural upper bound** `θ = π·sigmoid(theta_raw)`; the model architecture, optimizers and numerical paths are untouched (with the switch off it is bit-identical to 007).

- **θ structural-bound switch** `THETA_STRUCTURAL_BOUND_ENABLED` (**on by default**): its physical origin is the anti-aliasing (Nyquist) condition of the discrete rotation `δ·θ < π` ⇒ the base phase θ should live in `(0, π)`. The parameterization `θ = π·sigmoid(theta_raw)` is **everywhere differentiable, has no gradient dead zone, and approaches π asymptotically without ever reaching it**; it replaces the native `θ = exp(theta_raw)` (unbounded — θ can drift to arbitrarily large values → phase self-excitation). Set it to `False` to fall back to the native `exp` parameterization, **bit-identical to 007**.
- **Consistent initialization**: under the structural bound, `theta_raw` is inverted so the initial θ distribution matches the native log-uniform exactly (round-trip error ~1e-16).
- **017 variant**: each of the 6 ultra3 007 files (`210/212/2125/310/312/3125`) derives an 017; this release ships the main family `312017elite.py` and the x25 family `3125017elite.py`, while keeping `312007elite.py` / `312507elite.py` for comparison.

### 7.0.3 generation-path correctness patch

> v7.0.3 is a **generation-path correctness patch** over v7.0.2 — it fixes a crash in menu 2 chat under AMP (bf16) when `USE_CUDA_GRAPH_DECODE` is enabled; the model architecture, optimizers and numerical paths are untouched.

- **CUDA Graph decoding × autocast-cache conflict**: `generate` used to enter/exit `torch.amp.autocast` separately for `prefill` and for every decode step. The pre-capture warmup ran inside autocast and cached bf16 weight copies; the subsequent capture then aliased those cached tensors instead of recording cast ops; exiting the outermost autocast triggered `clear_autocast_cache()`, freeing them — so graph replay read already-reused memory, diverged to NaN/Inf distributions, and `torch.multinomial` hit a device-side assert that hard-crashed the process.
- **Fix**: keep the entire `prefill` + per-token decode loop inside **one** autocast context instead of entering/exiting per step. The autocast cache stays alive throughout, so the bf16 weights baked into the graph remain valid.
- **Verification**: with the same checkpoint (seed=0), 7.0.2 crashes deterministically with graph decoding on; the fixed build generates normally. On a fixed token sequence, the max absolute logit difference between graph decoding and eager is **0.0** (bit-identical).

### 7.0.2 data-preprocessing and log-noise

> v7.0.2 is a **data-preprocessing and log-noise patch** over v7.0.1 — it fixes the dropped `system` span in menu 11 segmentation, aligns the segmentation length accounting, and silences a harmless warning in triton-less environments; the model architecture, optimizers and numerical paths are untouched.

- **Menu 11 segmentation keeps the `system` span**: `p4_segment_dataset` split conversations into `(user, assistant)` pairs and discarded the leading contiguous `system` span, inconsistent with the training path `_truncate_pairs`. It now lifts the leading `system` span out first, counts it into the token budget, and prepends it to every segmented sample (`{'messages': head + list(pair)}`), so the system prompt survives multi-turn segmentation.
- **Aligned segmentation length**: `pair_tokens` now includes the `+1` of the real sample length (`input_ids = tokens[:-1]`), preventing an extra boundary sample from being kept, matching the training-side length accounting.
- **Log-noise cleanup**: importing `bitsandbytes` transitively imports `torch.utils.flop_counter`, which logs a `triton not found; flop counting will not work for triton kernels` WARNING on CUDA without triton. Its logger is now raised to ERROR (pure de-noising; no functional change).

### 7.0.1 data-pipeline fix

> v7.0.1 is a **data-pipeline correctness patch** over v7.0.0 — it fixes the dropped `system` span and the pollution of generation behavior by incomplete samples; the model architecture, optimizers and numerical paths are untouched.

- **Fix dropped `system` messages**: `_truncate_pairs` only recognized adjacent `(user, assistant)` pairs and discarded any leading contiguous `system` span (`render_messages` supported system, yet training data never saw it). It now lifts the leading `system` span out first, counts it into the token budget, then takes `(user, assistant)` pairs from the tail and merges.
- **Guard for user-ending samples**: incomplete samples that only contain user (or system + user) with no assistant reply leave the trailing `<|eos|>` as the sole training target, which pollutes generation behavior. A new `_has_assistant_target` guard makes `StreamingJSONLDataset` skip such samples during iteration.

### 7.0.0 base (lossless speedup + memory reduction)

> v7 is the **training-side "lossless speedup + memory reduction"** release over v6 — every new switch is on by default, falls back automatically on failure, and stays mathematically bit-identical.

- **Training-step CUDA Graph** `USE_TRAIN_STEP_CUDA_GRAPH`: captures the training step's **forward + backward** into a CUDA Graph (bucketed by sequence length via `TRAIN_GRAPH_BUCKET` / `TRAIN_GRAPH_MAX_GRAPHS`), eliminating ~100k kernel launches per step. `optimizer.step()` / `scheduler.step()` / gradient clipping stay outside the graph; active only on CUDA + non-MoE + batch=1, with automatic eager fallback on capture failure (**performance-only fallback; math unchanged**).
- **8-bit AdamW** `USE_ADAMW8BIT`: switches the main optimizer to bitsandbytes `AdamW8bit` (8-bit momentum states), saving ~**36%** optimizer memory; automatically falls back to the file's native optimizer (Adafactor / AdamW) when unavailable or non-CUDA.
- **Centralized matrix-exponential / graph switches** at the top of each file. Adds `USE_CAPTURABLE_EXPM` (K>2 uses "scaling-and-squaring + Taylor series", pure matmul/elementwise, no LAPACK, CUDA-Graph-capturable; **off by default**, K>2 falls back to `torch.linalg.matrix_exp`).
- **Gradient checkpointing on by default**; loss scaling and `zero_grad(set_to_none=False)` adapted to the training graph.
- Curated comparison experiments moved into `compare/`; README rewritten + new `CHANGELOG.md`.

## 📖 Key Features

- **Truly complex-free (Complex-to-Real 2×2 block isomorphism)**
  Each complex entry `a+bi` becomes the real matrix `[[a, -b], [b, a]]`, so real/imaginary channels alternate naturally with **no complex intermediate tensors** — BF16 AMP is safe.
- **Optional global summarizers (read/write heads)**
  `Mean` / `LinearAttention` / `MiniWave` summarizers give the SSM explicit global information, compensating for its inherent context-compression limits.
- **Flexible intra-layer coupling**
  A learnable inter-subchannel coupling matrix (skew-Hermitian) is more expressive than Mamba's diagonal/scalar matrices.
- **Full MoE integration with production-grade optimizations**
  MoE-FFN (SwiGLU experts + Top-K routing) with expert offload, a dedicated copy stream, FP8 / NVFP4 / BF16 quantization, and weight normalization.
- **Single file + all-config**
  Training, inference and RL (GRPO) in one self-contained file; every hyperparameter is a top-of-file constant.
- **Cross-platform stability**
  Win11 / Linux / macOS; DataLoader with multiple workers works safely under the spawn start method.

## ⚙️ Core Components

### ParallelComplexWaveLayer
The model core, replacing all complex-SSM operators with real 2×2 blocks:

- State transition: `h_t = A_t @ h_{t-1} + b_t`
- Single path: `A_t = matrix_exp(δ_t · Ω_t)`, where `Ω_t = M + diag(-ν + i·(θ_base + θ_delta_t))`
- Parallel scan: a custom `BlellochScanFn` gives an O(log T) parallel prefix scan (parallel in both forward and backward)
- Stability structurally guaranteed by the **Bendixson theorem** (`Re Ω < 0`), with no thresholds

### Global Summarizer
Selected via `SUMMARY_TYPE`:
- `mean`: global average pooling (parameter-free baseline)
- `linear_attention`: learnable linear-attention summary, chunked (`SUMMARY_CHUNK`) to avoid materializing large tensors
- `mini_wave`: a miniature `ParallelComplexWaveLayer` with its own parameters

### MoE-FFN
SwiGLU mixture-of-experts:
- `MOE_OFFLOAD`: expert weights can be offloaded to CPU and moved on demand
- `_copy_stream`: a dedicated CUDA stream overlaps the transfer with compute
- `MOE_EXPERT_QUANT`: FP8 (E4M3) training quantization, NVFP4 (E2M1) inference quantization, BF16 resident inference
- `MOE_EXPERT_WNORM`: weight normalization to prevent training divergence

## 📊 Comparison Experiments (with plots)

All results are **re-parsed from the raw training logs** under `对照测试/`; the **full comparison matrix — 60 runs, one row each with a data-backed conclusion — is in [`compare/README.md`](./compare/README.md)**, figure set in [`compare/`](./compare/).
Unified setting: **same vocab / same optimizer / same hyperparameters**, `H=256, L=3, T=2048, batch=1, lr=1e-3`, ~89k steps.

### 1. Round-3 controlled comparison: Waver / x25 vs. a classic Transformer

Three-panel figure (training loss · validation PPL · loss diff vs. baseline):

![round3 main](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_main.png)

> Waver v4.2 / x25 reach a slightly **lower** training loss than the Transformer baseline, with validation PPL very close.

**Multi-seed stability** (5 seeds each: 42 / 43 / 114514 / 1919810 / 31415926):

![waver seeds](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_seeds_waver.png)

![x25 seeds](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_seeds_x25.png)

![transformer seeds](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_seeds_transformer.png)

> Waver's validation PPL is **highly stable** across 5 seeds (31.1–31.6), while the Transformer swings much more (22.3–30.0).

**Ablations & sweeps** (full 15 figures in [`compare/round3/`](./compare/round3/)):

| LR sweep (Waver) | Intra-layer coupling | State dim K=4 |
|---|---|---|
| ![lr scan](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_lr_waver.png) | ![coupling](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_coupling.png) | ![k4](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_state_dim_k4.png) |

> `lr=8e-3` clearly degrades and aborts early (confirming the complex-valued SSM's sensitivity to too-high LR); the coupling ablation spans only 0.65 PPL (over-large or fully-removed coupling is slightly worse).
> Other ablations: Loop layers, wider h368, Warmup, summarizer dk, 100k data — see the comparison matrix.

**All-experiment overview** (train curves of all 55 round3/100k runs + best-ValPPL ranking):

| Train-loss small multiples | Best validation PPL ranking | Multi-seed stability |
|---|---|---|
| ![matrix loss](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/matrix/matrix_loss_all.png) | ![matrix valppl](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/matrix/matrix_valppl.png) | ![matrix seed](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/matrix/matrix_seed_variance.png) |

> Multi-seed (measured): std of best ValPPL — Transformer H256 **±2.89**, Waver H256 **±0.19**, x25 H256 **±0.24** (Waver/x25 variance is ~one order of magnitude lower, at a higher absolute PPL level).

### 2. v1 1.6B model — training overview

![v1 overview](https://x005-0001-0001.github.io/Waver-SNN-SSM/v1/plots/overview.png)

> With only ~17.9M tokens (far below the Chinchilla optimum), the 1.6B model brings loss from ~10 down to ~5 and PPL from ~49000 down to ~200 — strong evidence for the architecture.

### 3. v6 low-precision + training-side speedup validation (RTX 5080 16GB)

| Finding | Result |
|---|---|
| 8-bit AdamW stability | Max relative deviation of MON metrics < 3.4e-3 (bf16 quantization-noise level), no drift |
| 8-bit AdamW memory | Optimizer memory **-35.7%** (421.7MB → 271.1MB) |
| FP8 / NVFP4 expert quantization | NVFP4 inference saves **45.6%** memory with a 2e-4 relative logit error |
| Training-step CUDA Graph | fwd+bwd capture **2.4–2.6×**; checkpointing off + capture ≈ **3.4×** total |

Full report and probe scripts: [`compare/v6/`](./compare/v6/) and [`compare/README.md`](./compare/README.md).

## 🔧 Usage

The `.py` inside each version directory is the complete program — just run it and use the menu:

```bash
python v8/312009elite.py
```

All key parameters (model architecture, optimizer, data paths, MoE, memory optimizations, etc.) are defined as constants at the top of the file.

---

## 📜 Citation

```bibtex
@misc{zeng2026waversnnssm,
  author = {Zeng, Naiqi},
  title = {Waver-SNN-SSM: From Spiking Neurons to State Space Models},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.22309186},
  url = {https://doi.org/10.5281/zenodo.22309186}
}
```

## 📧 Contact
- ORCID: [0009-0006-4332-1295](https://orcid.org/0009-0006-4332-1295)
- Email: X005_0001_0001@163.com

## 📄 License
This project is licensed under the **GNU General Public License v3.0**. Any use, modification or distribution of this code must comply with GPL-3.0, i.e. any derivative work must also be open-sourced under GPL-3.0.

## ⚠️ Copyright
Copyright © 2026 Naiqi Zeng. All rights reserved.
The first public release date of this code and its documentation is locked by Zenodo DOI 10.5281/zenodo.22309186. Any unauthorized plagiarism, rewording or citation without attribution is considered copyright infringement.

Version: 8.1.0
Last Updated: 2026-10-12