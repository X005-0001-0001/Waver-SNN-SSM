# Linux 单卡加速对照 · Linux Single-GPU Acceleration Comparison

本目录收录 **8.1.0 / 009elitelinux** 的收敛一致性对照证据：**同一张 RTX 5080**，只换软件栈（Win11 未优化 vs. WSL 优化），不换硬件。

- 图：[`ab_curve.png`](./ab_curve.png)（左：A/B 逐步 Loss；右：绝对差 |Δ|）
- 数据：[`ab_data.csv`](./ab_data.csv)（逐步 `win11_loss` / `linux_loss` / `abs_diff`，共 534 步）

![A/B loss curve](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/linux/ab_curve.png)

## 实验条件

| 项 | 值 |
|---|---|
| 硬件 | 单卡 RTX 5080 16GB（Blackwell sm_120） |
| A 组 | Win11 + `312009elite.py`（无 Linux 编译加速） |
| B 组 | WSL(Ubuntu) + `312009elitelinux.py`（编译加速 + 多进程全开） |
| 规模 | VOCAB≈256 / H=128 / L=2 / T=256 / batch=1 |
| 步数 | 534（两组一致） |
| 优化器 | 两组均 `bnb8bit`（一致，以隔离编译影响） |

## 结果（实测）

| 指标 | A（Win11 未优化） | B（WSL 优化） |
|---|---|---|
| 最终 Loss | 1.0182 | 1.0178 |
| PPL（末值） | 2.77 | 2.77 |

逐步 Loss 走势一致；|Δ| 落在超小规模的数值噪声范围内。

## 结论边界

- **只证明**：009elitelinux 开启全部优化后**收敛一致、无正确性回归**。
- **不证明**：真实规模下的**吞吐加速比** —— 超小规模下 `torch.compile` 的编译预热开销占主导，本文**不给出任何加速比声明**。
- 数值等价另见 release 说明：全部开关关闭时前向 logits 与对应 009 **逐位一致**（maxdiff `0.0`）；全部开启时 maxdiff **9.537e-07 < 1e-6**（WSL, sm_120）。

---

# Linux Single-GPU Acceleration Comparison

This directory holds the convergence-consistency evidence for **8.1.0 / 009elitelinux**: the **same RTX 5080**, only the software stack changes (Win11 unoptimized vs. WSL optimized), not the hardware.

- Figure: [`ab_curve.png`](./ab_curve.png) (left: A/B step-wise Loss; right: absolute difference |Δ|)
- Data: [`ab_data.csv`](./ab_data.csv) (step-wise `win11_loss` / `linux_loss` / `abs_diff`, 534 steps)

## Setup

| Item | Value |
|---|---|
| Hardware | Single RTX 5080 16GB (Blackwell sm_120) |
| A | Win11 + `312009elite.py` (no Linux compile acceleration) |
| B | WSL(Ubuntu) + `312009elitelinux.py` (compile acceleration + multiprocessing fully on) |
| Scale | VOCAB≈256 / H=128 / L=2 / T=256 / batch=1 |
| Steps | 534 (identical for both) |
| Optimizer | `bnb8bit` for both (identical, to isolate the compile effect) |

## Results (measured)

| Metric | A (Win11 unoptimized) | B (WSL optimized) |
|---|---|---|
| Final Loss | 1.0182 | 1.0178 |
| PPL (last) | 2.77 | 2.77 |

The step-wise Loss curves agree; |Δ| stays within the numerical-noise range of the tiny scale.

## Scope

- **Proves only**: with all optimizations on, 009elitelinux yields **consistent convergence and no correctness regression**.
- **Does not prove**: a **throughput speedup at real scale** — at tiny scale the `torch.compile` warm-up dominates, so this document makes **no speedup claim**.
- Numerical equivalence (see the release notes): with all switches off the forward logits are **bit-identical** to the corresponding 009 (maxdiff `0.0`); with all on the maxdiff is **9.537e-07 < 1e-6** (WSL, sm_120).