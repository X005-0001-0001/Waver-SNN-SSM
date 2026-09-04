# Waver-SNN-SSM: 从脉冲到状态空间的统一序列建模框架

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22309186.svg)](https://doi.org/10.5281/zenodo.22309186)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--4332--1295-green)](https://orcid.org/0009-0006-4332-1295)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Waver-SNN-SSM** 是一个从脉冲神经网络（SNN）出发，最终演进到与 **Mamba-3** 理念殊途同归的高效状态空间模型（SSM）实现。

## 📖 项目亮点

1.  **彻底的复数化（Complex-to-Real 2×2 块同构）**：在代数层面实现“无复数”的复数 SSM，安全开启 BF16 AMP 加速。
2.  **可选的“读写头”（全局摘要器）**：独创 Mean/LinearAttention/MiniWave 三种全局摘要器，为 SSM 提供显式的全局信息参考。
3.  **更灵活的层内耦合（Intra-layer Coupling）**：内置可学习的子通道间耦合矩阵，比 Mamba 的对角/标量矩阵更具表达力。
4.  **完整的 MoE 集成与工业级优化**：集成 MoE-FFN，包含 Offload、FP8/NVFP4 量化等全套优化方案。
5.  **“单文件 + 全配置”的工程化典范**：训练、推理、RL 全流程单文件封装，所有超参数顶部常量集中控制。
6.  **沿袭自 SNN 的独特迭代路径**：从 SNN 到 SSM 的独特演进视角，极具启发性。

## 🚀 快速开始

1.  **安装依赖**
    ```bash
    pip install -r requirements.txt
准备数据
将你的训练数据（.txt 或 .jsonl 格式）放入项目根目录。

训练模型

bash
python 312002proplus_elite.py
然后根据菜单提示选择 1. 从头训练模型。

🏗️ 架构细节
核心组件：ParallelComplexWaveLayer
这是模型的核心，它将复值 SSM 的所有算子替换为实值 2×2 块，实现了：

状态转移：h_t = A_disc @ h_{t-1} + b_t

并行扫描：使用自定义的 BlellochScanFn 实现高效的 O(log T) 并行前缀扫描。

全局摘要器 (Summarizer)
通过 SUMMARY_TYPE 切换三种模式：

mean：全局平均池化。

linear_attention：可学习的线性注意力摘要。

mini_wave：一个迷你的 ParallelComplexWaveLayer 实例。

MoE-FFN
集成了带有 SwiGLU 激活函数的混合专家模型，并通过 MOE_OFFLOAD 等参数实现了显存优化。

🔧 配置调优
所有关键参数均在 312002proplus_elite.py 文件头部定义：

参数	描述	默认值
HIDDEN_SIZE	隐藏层维度	512
NUM_LAYERS	网络层数	5
STATE_DIM	SSM 状态维度（必须为偶数）	6
MOE_ENABLED	是否启用 MoE	True
SUMMARY_TYPE	摘要器类型	'linear_attention'
USE_AMP	是否启用自动混合精度	True
FREEZE_LAYERS	冻结前 N 层	0
📜 引用
如果你在研究中使用了本项目，请引用：

bibtex
@misc{zeng2026waversnnssm,
  author = {Zeng, Naiqi},
  title = {Waver-SNN-SSM: From Spiking Neurons to State Space Models},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.22309186},
  url = {https://doi.org/10.5281/zenodo.22309186}
}
📧 联系方式
ORCID: 0009-0006-4332-1295

Email: X005_0001_0001@163.com