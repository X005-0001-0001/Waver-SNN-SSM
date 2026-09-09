Waver-SNN-SSM: 从脉冲到状态空间的统一序列建模框架
https://zenodo.org/badge/DOI/10.5281/zenodo.22309186.svg
https://img.shields.io/badge/ORCID-0009--0006--4332--1295-green
https://img.shields.io/badge/License-GPLv3-blue.svg
https://img.shields.io/badge/python-3.9+-blue.svg
https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg

Waver-SNN-SSM 是一个从脉冲神经网络（SNN）出发，最终演进到与 Mamba-3 理念殊途同归的高效状态空间模型（SSM）实现。本项目将复值 SSM 的复数算子完整替换为实值 2×2 块同构，彻底移除了训练时的复数中间张量，从而安全启用 BF16 混合精度训练，大幅降低显存占用并加速训练。

🔥 最新更新 (v2.0 — 2026-09-09)
完整更新日志见 CHANGELOG。

🚀 核心性能优化
优化项	效果
Ω 特征分解 → 对角复扫描	SSM 显存 755MB/层 → 126MB/层，提速 5-20×
分块线性注意力	摘要器显存 ~1GB/层 → <1MB
并行 Prefill	prompt 处理 1024 次 step → 1 次 forward，提速 10-50×
CUDA Graph 解码	decode 提速 2-5×，与 eager 逐位 0 误差
MoE 独立 Copy Stream	训练步时 16s → 3-5s
MoE BF16 常驻	推理内存 2× 压缩，零反量化开销
Blelloch 填充修复	扫描算量/显存直接减半
🐛 关键修复
Win11 DataLoader spawn 死锁：collate_fn/worker_init_fn 改为顶层可 pickle，num_workers>0 不再必崩。

AMP + linear_attention 生成崩溃：显式 fp32 上转 + autocast 块内计算 S_final，AMP+CUDA Graph 组合稳定运行。

COUPLING + MoE 既有 bug：pin 循环容错 + 补 MOE_EXPERT_WNORM=False，COUPLING 家族 MoE 全路径可用。

📖 项目亮点
彻底的复数化（Complex-to-Real 2×2 块同构）
在代数层面实现“无复数”的复数 SSM，安全开启 BF16 AMP 加速。每个复矩阵元 a+bi 被替换为实矩阵 [[a, -b], [b, a]]，实虚通道自然交替，无任何 complex 类型的中间张量。

可选的“读写头”（全局摘要器）
独创 Mean / LinearAttention / MiniWave 三种全局摘要器，为 SSM 提供显式的全局信息参考，弥补 SSM 固有的上下文压缩局限。

更灵活的层内耦合（Intra-layer Coupling）
内置可学习的子通道间耦合矩阵（斜埃尔米特结构），比 Mamba 的对角/标量矩阵更具表达力，状态通道间可交叉交互。

完整的 MoE 集成与工业级优化
集成 MoE-FFN（SwiGLU 专家 + Top-K 路由），包含 Expert Offload（CPU 卸载）、独立 Copy Stream 异步搬运、FP8/NVFP4/BF16 量化、Weight Normalization 等全套优化方案。

“单文件 + 全配置”的工程化典范
训练、推理、强化学习（GRPO/RL）全流程单文件封装，所有超参数均通过顶部常量集中控制，实验可复现性极强。

跨平台稳定
Win11 / Linux / macOS 全平台支持，DataLoader 多 worker 在 spawn 模式下可安全使用（需 if __name__ == "__main__" 守卫）。

🚀 快速开始
1. 安装依赖
bash
pip install -r requirements.txt
依赖项：

torch >= 2.0.0

numpy >= 1.24.0

tokenizers >= 0.13.0

psutil >= 5.9.0（可选，用于内存监控）

2. 准备数据
将你的训练数据（.txt 或 .jsonl 格式）放入项目根目录。

.txt 格式：纯文本语料，每行一个样本或连续文本，程序会自动滑动窗口切分。

.jsonl 格式：对话数据，格式为 {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}。

3. 训练模型
bash
python 312003elite.py
然后根据菜单提示选择 1. 训练模型（继续训练），并选择你的数据集文件。

📊 训练状态
312002proplus_elite（旧版本）已完成：
数据集为 52.1MB 的中文名著语料，约 204,123 行，12.9M token，epoch1；
deepseek r1蒸馏集一万条（无思维链）5M tokens，epoch1；
在百度百科上额外训练了几百步中断；

312003elite。py（最近更新）
暂未训练

📌 为什么 Loss 看起来偏高？
这是一个 1.6B 参数模型在 约17.9M token 数据上训练了 1 个 epoch 的状态。参考 Chinchilla 定律，1.6B 模型的最优训练数据量约为 320 亿 token，当前数据量仅为最优值的 ~4%。在如此悬殊的数据-参数比下，模型能稳定将 Loss 从初始的 ~10 降到 ~5，PPL 从 ~49000 降到 ~200，本身就是架构有效性的有力证据。

📈 312002训练曲线三段合集总览
![训练总览](v1/log合集看原图/把所有日志拼一块plots/overview.png)

上图展示了 Loss、PPL、Activity、LR 和 Memory 的全流程变化曲线。

完整日志：详见v1文件夹内各个阶段的traininglog

⚙️ 核心组件
ParallelComplexWaveLayer
模型的核心，将复值 SSM 的所有算子替换为实值 2×2 块：

状态转移：h_t = A_disc @ h_{t-1} + b_t

并行扫描：使用自定义的 BlellochScanFn 实现高效的 O(log T) 并行前缀扫描。

特征分解加速（USE_EIG_SCAN=True）：将 exp(δΩ) 拆解为 P·diag(exp(δλ))·P⁻¹，矩阵扫描退化为复标量扫描。

全局摘要器 (Summarizer)
通过 SUMMARY_TYPE 切换三种模式：

mean：全局平均池化（无参数基线）。

linear_attention：可学习的线性注意力摘要，支持分块（SUMMARY_CHUNK）避免大张量物化。

mini_wave：迷你的 ParallelComplexWaveLayer 实例，独立参数。

MoE-FFN
集成了带有 SwiGLU 激活函数的混合专家模型：

MOE_OFFLOAD：专家权重可卸载到 CPU，按需搬移。

_copy_stream：独立 CUDA 流异步搬运，与计算重叠。

MOE_EXPERT_QUANT：支持 FP8（E4M3）训练量化、NVFP4（E2M1）推理量化、BF16 常驻推理。

MOE_EXPERT_WNORM：权重归一化，防止训练发散。

🔧 配置调优
所有关键参数均在文件头部定义：
📜 引用
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

📄 许可证
本项目采用 GNU General Public License v3.0。任何使用、修改或分发本代码的行为，都必须遵守 GPL-3.0 的条款，即任何衍生作品也必须以 GPL-3.0 开源。

⚠️ 版权声明
Copyright © 2026 Naiqi Zeng. All rights reserved.

本代码及相关文档的首次公开发布时间由 Zenodo DOI 10.5281/zenodo.22309186 锁定。任何未经授权的抄袭、洗稿或未注明来源的引用，均视为侵犯版权。本项目的 GPL-3.0 许可证明确要求任何二次分发必须公开源代码，商业用途必须获得额外授权。

Version: 2.0.0
Last Updated: 2026-09-08
