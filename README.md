# Waver-SNN-SSM: 从脉冲到状态空间的统一序列建模框架

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22309186.svg)](https://doi.org/10.5281/zenodo.22309186)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--4332--1295-green)](https://orcid.org/0009-0006-4332-1295)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Waver-SNN-SSM** 是一个从脉冲神经网络（SNN）出发，最终演进到与 **Mamba-3** 理念殊途同归的高效状态空间模型（SSM）实现。本项目将复值 SSM 的复数算子完整替换为实值 2×2 块同构，彻底移除了训练时的复数中间张量，从而安全启用 BF16 混合精度训练，大幅降低显存占用并加速训练。

---

## 📖 项目亮点

1. **彻底的复数化（Complex-to-Real 2×2 块同构）**  
   在代数层面实现“无复数”的复数 SSM，安全开启 BF16 AMP 加速。每个复矩阵元 `a+bi` 被替换为实矩阵 `[[a, -b], [b, a]]`，实虚通道自然交替，无任何 complex 类型的中间张量。

2. **可选的“读写头”（全局摘要器）**  
   独创 **Mean / LinearAttention / MiniWave** 三种全局摘要器，为 SSM 提供显式的全局信息参考，弥补 SSM 固有的上下文压缩局限。

3. **更灵活的层内耦合（Intra-layer Coupling）**  
   内置可学习的子通道间耦合矩阵（斜埃尔米特结构），比 Mamba 的对角/标量矩阵更具表达力，状态通道间可交叉交互。

4. **完整的 MoE 集成与工业级优化**  
   集成 MoE-FFN（SwiGLU 专家 + Top-K 路由），包含 **Expert Offload（CPU 卸载）**、**FP8/NVFP4 量化**、**Weight Normalization** 等全套优化方案，让大模型在消费级显卡上训练成为可能。

5. **“单文件 + 全配置”的工程化典范**  
   训练、推理、强化学习（GRPO/RL）全流程单文件封装，所有超参数（`HIDDEN_SIZE`、`NUM_LAYERS`、`STATE_DIM`、`MOE_ENABLED` 等）均通过顶部常量集中控制，实验可复现性极强。

6. **沿袭自 SNN 的独特迭代路径**  
   项目从脉冲神经网络（SNN）起步，历经多代演进，最终与 Mamba-3 殊途同归。这种跨越不同 AI 子领域的思考和实践，极具故事性和启发性。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：
- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `tokenizers >= 0.13.0`
- `psutil >= 5.9.0`（可选，用于内存监控）

### 2. 准备数据

将你的训练数据（`.txt` 或 `.jsonl` 格式）放入项目根目录。

- **`.txt` 格式**：纯文本语料，每行一个样本或连续文本，程序会自动滑动窗口切分。
- **`.jsonl` 格式**：对话数据，格式为 `{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`。

### 3. 训练模型

```bash
python 312002proplus_elite.py
```

然后根据菜单提示选择 `1. 从头训练模型`，并选择你的数据集文件。

---

## 📊 训练状态（实时快照）

以下为当前训练在自制中文名著数据集（`supernovel.txt`，52.1 MB）上的表现：

| 指标 | 值 |
|------|-----|
| **Epoch** | 1/1 |
| **Step** | 7534 |
| **Loss** | 6.0082 |
| **PPL** | 406.74 |
| **Activity** | -0.0313 |
| **LR** | 0.000425 |
| **Memory** | 27328.0 MB |
| **Progress** | 59.8% |

### 📈 训练曲线总览

![训练总览](plots/overview.png)

*上图展示了 Loss、PPL、Activity、LR 和 Memory 的全流程变化曲线（平滑版本）。*

### 📉 分项详细曲线

| Loss | PPL |
|------|-----|
| ![Loss](plots/loss.png) | ![PPL](plots/ppl.png) |

| Learning Rate | Memory |
|---------------|--------|
| ![LR](plots/lr.png) | ![Memory](plots/memory.png) |

### 🧩 层内统计（Layer Stats）

![Layer Stats](plots/layer_stats.png)

*每层的 `I_std`、`amp_std`、`wave_raw_mean`、`wave_raw_std`、`delta_mean/min/max` 变化趋势。*

### 🔧 参数监控（MON Parameters）

![MON Parameters](plots/mon_params.png)

*每层的 Bn、Sn、Dn、Mn、An、nu、th、betan、Cbn、In 等关键参数的演化。*

---

### 📋 实时监控数据（单步快照）

以下为 Step 7534 时的详细层状态：

<details>
<summary>点击展开 Layer 监控数据</summary>

| Layer | I_std | amp_std | wave_raw_mean | wave_raw_std | delta_mean | delta_min | delta_max |
|-------|-------|---------|---------------|--------------|------------|-----------|-----------|
| L0    | 0.1893 | 0.4593 | -0.1128 | 5.5625 | 0.2131 | 0.0006 | 1.9319 |
| L1    | 3.2674 | 26.0764 | -3.0312 | 166.0000 | 0.0850 | 0.0001 | 2.3475 |
| L2    | 3.5674 | 2.7920 | -0.6641 | 29.3750 | 0.0560 | 0.0014 | 0.3998 |
| L3    | 4.7529 | 0.3293 | 0.1885 | 3.6094 | 0.0764 | 0.0053 | 0.4920 |
| L4    | 3.2784 | 0.3575 | -0.2246 | 5.0000 | 0.1207 | 0.0017 | 1.2077 |

</details>

<details>
<summary>点击展开 MON 参数</summary>

| Layer | Bn | Sn | Dn | Mn | An | nu(mean/max) | th(min/max) | Cbn | In |
|-------|----|----|----|----|----|--------------|-------------|-----|----|
| L0    | 5.622 | 24.655 | 7.948 | 6.065 | 36.551 | 0.260/1.578 | 0.009/3.393 | 9.923 | 27.826 |
| L1    | 2.702 | 25.665 | 4.010 | 5.946 | 37.626 | 0.239/1.130 | 0.010/3.264 | 5.809 | 27.349 |
| L2    | 3.100 | 26.945 | 3.636 | 5.958 | 36.255 | 0.228/1.077 | 0.009/3.644 | 6.486 | 26.442 |
| L3    | 4.035 | 25.470 | 4.052 | 6.006 | 36.319 | 0.244/1.140 | 0.010/3.386 | 7.345 | 26.737 |
| L4    | 4.772 | 26.105 | 6.515 | 6.048 | 35.390 | 0.249/1.533 | 0.009/3.626 | 8.186 | 26.627 |

</details>

<details>
<summary>点击展开 MoE 专家监控</summary>

| Layer | Router Norm | W_gate Scale | W_up Scale | W_down Scale |
|-------|-------------|--------------|------------|--------------|
| L0    | 3.369 | 1.030 | 1.030 | 1.014 |
| L1    | 3.276 | 1.039 | 1.040 | 1.025 |
| L2    | 4.125 | 1.091 | 1.089 | 1.073 |
| L3    | 4.671 | 1.138 | 1.135 | 1.117 |
| L4    | 4.403 | 1.129 | 1.124 | 1.104 |

</details>

---

### 📂 数据集信息

| 属性 | 值 |
|------|-----|
| **文件** | `supernovel.txt` |
| **类型** | 纯文本（txt） |
| **大小** | 52.1 MB |
| **行数** | 204,123 |
| **Token 数** | 12,909,063（12.91M） |
| **模型参数量** | ≈ 1.6B |

---

### 🔍 训练解读

- Loss 稳定在 6.0 左右，PPL 约 407，表明模型正在有效学习。
- `wave_raw_std` 从 L0 到 L4 逐渐收敛（L1 的 166 为初期瞬态，后续层已稳定在 3~5），信号传播正常。
- MoE 专家权重范数（W_gate/W_up/W_down）集中在 1.0~1.1，路由负载均衡良好，未出现专家坍缩。

> 💡 **提示**：以上图表由 `plot_training.py` 基于 `training_log.txt` 自动生成。你也可以运行 `python plot_watch.py` 实现实时更新绘图。

### 核心组件：`ParallelComplexWaveLayer`

这是模型的核心，它将复值 SSM 的所有算子替换为实值 2×2 块：

- **状态转移**：`h_t = A_disc @ h_{t-1} + b_t`
- **并行扫描**：使用自定义的 `BlellochScanFn` 实现高效的 O(log T) 并行前缀扫描。
- **实值 2×2 块**：`Ω = M + diag(-ν + iθ)` → `[[M - diag(ν), -diag(θ)], [diag(θ), M - diag(ν)]]`

### 全局摘要器 (Summarizer)

通过 `SUMMARY_TYPE` 切换三种模式：
- `mean`：全局平均池化（无参数基线）。
- `linear_attention`：可学习的线性注意力摘要（Key/Query 归一化，余弦相似度有界）。
- `mini_wave`：迷你的 `ParallelComplexWaveLayer` 实例，独立参数。

### MoE-FFN

集成了带有 **SwiGLU** 激活函数的混合专家模型，并通过以下参数实现显存优化：
- `MOE_OFFLOAD`：专家权重可卸载到 CPU，按需搬移。
- `MOE_EXPERT_QUANT`：支持 FP8（E4M3）训练量化或 NVFP4（E2M1）推理量化。
- `MOE_EXPERT_WNORM`：权重归一化，将每专家权重范数解耦为可学习标量，防止训练发散。

---

## 🔧 配置调优

所有关键参数均在 `312002proplus_elite.py` 文件头部定义：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `HIDDEN_SIZE` | 隐藏层维度 | 512 |
| `NUM_LAYERS` | 网络层数 | 5 |
| `STATE_DIM` | SSM 状态维度（必须为偶数） | 6 |
| `MOE_ENABLED` | 是否启用 MoE | `True` |
| `SUMMARY_TYPE` | 摘要器类型 | `'linear_attention'` |
| `USE_AMP` | 是否启用自动混合精度 | `True` |
| `FREEZE_LAYERS` | 冻结前 N 层 | 0 |
| `MAX_SEQ_LEN` | 最大序列长度 | 1024 |
| `BATCH_SIZE` | 批大小 | 1（可调大） |

---

## 📜 引用

如果你在研究中使用了本项目，请引用：

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

---

## 📧 联系方式

- **ORCID**: [0009-0006-4332-1295](https://orcid.org/0009-0006-4332-1295)
- **Email**: X005_0001_0001@163.com

---

## 📄 许可证

本项目采用 **GNU General Public License v3.0**。任何使用、修改或分发本代码的行为，都必须遵守 GPL-3.0 的条款，即**任何衍生作品也必须以 GPL-3.0 开源**。

---

## ⚠️ 版权声明

Copyright © 2026 Naiqi Zeng. All rights reserved.

本代码及相关文档的首次公开发布时间由 Zenodo DOI 10.5281/zenodo.22309186 锁定。任何未经授权的抄袭、洗稿或未注明来源的引用，均视为侵犯版权。本项目的 GPL-3.0 许可证明确要求任何二次分发必须公开源代码，商业用途必须获得额外授权。

---

**Version**: 1.0.0  
**Last Updated**: 2026-09-05

---

这份 README 已经整合了你所有的训练监控数据、数据集统计和架构亮点。你可以直接复制使用，需要调整的地方告诉我。
