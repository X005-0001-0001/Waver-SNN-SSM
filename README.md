# Waver-SNN-SSM：从脉冲到状态空间的统一序列建模框架

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22309186.svg)](https://doi.org/10.5281/zenodo.22309186)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--4332--1295-green)](https://orcid.org/0009-0006-4332-1295)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Waver-SNN-SSM 是一个从脉冲神经网络（SNN）出发、最终与 Mamba-3 理念殊途同归的高效状态空间模型（SSM）实现。核心创新在于把复值 SSM 的复数算子完整替换为**实值 2×2 块同构**，在代数层面实现“无复数”的复数 SSM，从而安全启用 BF16 混合精度训练，大幅降低显存占用并加速训练。

> **当前最新：v6.0.0（2026-10-10）** —— 闭式矩阵指数（K=2）+ 批量分词 + 解码 CUDA Graph 默认启用。

---

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
├── compare/     # 精简对照实验（核心图 + 验证报告 + 看板工具）
├── count_tokens.py / plot_training.py / plot_watch.py / overview_viewer.html
├── tokenizer_incremental.bin / tokenizer_incremental_tokenizer.json
└── README.md / CHANGELOG.md / LICENSE / CITATION.cff / DOI.txt
```

**命名规则**：`312005elite.py` 为主族（层内耦合），`312505elite.py` 为 x25 族（层外耦合）。**v5 起两族同时发布**；v4 及以前仅发布 `31200x`。每个 `.py` 均为**单文件**、自包含全流程（训练 / 推理 / 强化学习）。

## 📜 版本历史

| 版本 | 日期 | 核心变更 |
|---|---|---|
| v1.0.0 | 2026-09-05 | 首次公开发布：复值 SSM 实值化 2×2 块同构、MoE-FFN、全局摘要器、自定义 Blelloch 并行前缀扫描 |
| v1.0.1 | 2026-09-05 | FastBPE 逐 token 性能修复（`_vocab_size_cache`）|
| v1.0.2 | 2026-09-05 | `brain.pt` 惰性加载，聊天不再触发 OOM |
| 2.0.0 | 2026-09-09 | 性能优化与工程加速：Blelloch 填充修复、Ω 对角复扫描、分块线性注意力、并行 Prefill、CUDA Graph 解码、MoE 独立 Copy Stream / BF16 常驻 |
| 3.1.0 | 2026-09-17 | 架构重构：统一 `A_t = matrix_exp(δ_t·Ω_t)` 实值矩阵扫描，删除 eig 路径，δ 逐通道化，新增数据依赖旋转 `theta_delta` |
| 4.0.0 | 2026-09-23 | Loop 机制（权重共享循环层）+ SSM 分块扫描 `SSM_CHUNK_SIZE` |
| 4.1.0 | 2026-09-28 | 稳定性更新 + 与经典 Transformer 的受控对照（层内耦合开 / 关）|
| 4.2.0 | 2026-10-02 | 修复 jsonl 路径显存碎片 |
| 5.0.0 | 2026-10-04 | 数据格式层（messages / reasoning_content / tool_calls）+ 词表 / 分词修复 + 配置精简 |
| **6.0.0** | **2026-10-10** | **闭式矩阵指数（K=2）+ 批量分词 + 解码 CUDA Graph 默认启用** |

## 🚀 v6.0.0 核心变更

- **闭式矩阵指数（K=2）**：`USE_CLOSED_FORM_EXPM` —— 2×2 复矩阵 `exp(δ·Ω_t)` 走 Cayley–Hamilton 闭式，配级数小量保护；与 `torch.linalg.matrix_exp` **数学恒等**（fp64 相对差 ~1e-15），且**无 LAPACK 调用、可被 CUDA Graph 捕获**。
- **解码 CUDA Graph 默认启用**：`USE_CUDA_GRAPH_DECODE=True`，抓图后对输出 `detach().clone()`，与 eager 逐位一致（12/12 抓图成功）。
- **批量分词** `_encode_batch_with_offsets`：一次 `encode_batch` 一批，**保持 shuffle 时机与 rng 调用序列完全一致**（T0 排序哈希不变）。
- **`_split_reasoning`** 同时支持 `think` / `thinking` 双标签。
- **分词缓存键**纳入归一化模式 + `tokenizers` 库版本，避免跨配置误复用。
- BlellochScanFn 去掉多余 clone；梯度检查点 `preserve_rng_state=False`；保留 `R_cpl` 缓存键（修复抓图失败）。

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
- 稳定性由 **Bendixson 定理**结构性保证（`Re Ω < 0`），不依赖任何阈值

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

## 📊 对照实验

历史对照结果与复现脚本集中在 [`compare/`](./compare/)：

- **v4.1** —— 与经典 Transformer 的受控对照（同词表 / 同优化器 / 参数量对齐）
- **v6** —— 低精度方案（8-bit AdamW / FP8 / NVFP4）与训练侧加速验证报告 + 复现脚本
- **看板工具** —— 只读训练日志、无第三方依赖的本地交互看板

## 🔧 使用

每个版本目录下的 `.py` 即为完整程序，直接运行进入菜单：

```bash
python v5/312005elite.py
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

Version: 6.0.0
Last Updated: 2026-10-10