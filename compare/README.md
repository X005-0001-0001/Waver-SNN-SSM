# 对照实验矩阵 · Comparison Matrix

本目录收录 **60 组训练实验**的统一对照矩阵：**每条实验一行**，给出条件、可复现指标与**基于原始日志的数据结论**。
所有曲线均由 `对照测试/` 下各 run 的 `training_log.txt` **重新解析生成**（EMA 平滑 + 均匀降采样），不使用任何过时旧图。

- 机器可读结果：[`matrix/matrix.csv`](./matrix/matrix.csv)、[`matrix/summary.json`](./matrix/summary.json)
- 图集：[`round3/`](./round3/)（15 张分维度图）、[`matrix/`](./matrix/)（3 张总览图）、[`v6/`](./v6/)（低精度与加速）、[`linux/`](./linux/)（单卡 Linux 加速收敛一致性）

> **结论边界声明**：下列结论中，凡标注「实测」者为日志原始数据直接读出；标注「本窗口内」者为该实验窗口（约 8.9 万步 / 10 epoch）内的观察，不外推到更长训练；标注「未验证」者为尚无实验支撑的推断。

---

## 0. 统一实验条件（round3，共 53 条）

| 项 | 值 |
|---|---|
| 数据集 | `dialogues_seg.jsonl`（9886 样本/epoch，验证集 988，种子 42） |
| 序列长 / batch | `T=2048` / `batch=1` |
| 层数 / 隐藏维 | `L=3` / `H=256`（另有 H=368、K4-H196 消融） |
| 默认学习率 | `1e-3`，warmup 1977 步，共约 **88 980 步 / 10 epoch** |
| 基线 | Transformer H256 seed42；Waver 基线 = `v4.2 ... loop0 seed42` |
| 参数量 | Transformer H256 **4.16M**；Waver/x25 H256 **7.24M** |

---

## 1. 跨架构主对照（seed42, lr=1e-3）

![round3 main](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_main.png)

| 目录 | 条件 | 参数 | 末 Loss(EMA) | 最优 ValPPL | 末轮 ValPPL |
|---|---|---|---|---|---|
| `第3轮transofrmer2048h256` | Transformer H256 | 4.16M | 2.155 | 29.61 | 29.85 |
| `第3轮v4.2waverbat12048loop0seed42` | Waver v4.2 H256 | 7.24M | **1.952** | 31.57 | 33.23 |
| `第3轮x25waverbat12048loop0h256clean50seed42` | x25 H256 | 7.24M | 2.001 | 31.67 | 33.12 |

**数据结论（实测）**：Waver / x25 的训练 Loss（EMA 末值）比 Transformer 低约 **0.15–0.20**；但验证 PPL 略高（31.5–31.7 vs 29.6），即 **Waver 存在更大的 train/val 间隔**。本窗口内两种指标方向不一致，不构成任何一方的整体优劣判定。

---

## 2. 多随机种子稳定性（4 组 × 5 seed）

![seed variance](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/matrix/matrix_seed_variance.png)

| 组 | 目录（seed 42/43/114514/1919810/31415926） | 最优 ValPPL（5 seed） | 末轮 ValPPL |
|---|---|---|---|
| Transformer H256 | `transofrmer2048h256`(+`seedXX`) | 22.28 / 27.46 / 29.61 / 29.52 / 30.01 | 22.28–30.26 |
| Waver v4.2 H256 | `v4.2...loop0seed42`, `...h256clean50seedXX` | 31.09 / 31.14 / 31.18 / 31.45 / 31.57 | 32.30–33.23 |
| x25 H256 | `x25...h256clean50seedXX` | 30.95 / 31.32 / 31.38 / 31.49 / 31.67 | 32.45–33.12 |
| Transformer H368 | `transofrmer2048h368`(+`seedXX`) | 24.47 / 29.13 / 29.36 / 29.42 / 29.68 | 25.05–30.81 |

**汇总（实测）**：最优 ValPPL 的跨 seed 标准差 —— Transformer H256 **±2.89**、H368 **±1.98**；Waver H256 **±0.19**、x25 **±0.24**。**Waver/x25 的 seed 方差比 Transformer 低约一个数量级**，但绝对 PPL 水平更高。二者不是同一件事，须分开陈述。

图：`round3_seeds_transformer.png` / `round3_seeds_waver.png` / `round3_seeds_x25.png` / `round3_h368_seeds.png`。

---

## 3. 学习率扫描（seed42）

| 目录 | 架构 | lr | 末 Loss(EMA) | 最优 ValPPL | 结论 |
|---|---|---|---|---|---|
| `transofrmer2048h256lr0.0005seed42` | T H256 | 5e-4 | 2.080 | 27.58 | Transformer 最优 |
| `transofrmer2048h256` | T H256 | 1e-3 | 2.155 | 29.61 | — |
| `transofrmer2048h256lr0.002seed42` | T H256 | 2e-3 | 2.291 | 30.62 | 变差 |
| `transofrmer2048h256lr0.004seed42` | T H256 | 4e-3 | 2.834 | 35.81 | 明显变差 |
| `v4.2...seed42lr0.0005` | W H256 | 5e-4 | 1.964 | 32.37 | — |
| `v4.2...seed42` | W H256 | 1e-3 | 1.952 | 31.57 | — |
| `v4.2...seed42lr0.002` | W H256 | 2e-3 | 2.031 | **30.53** | Waver 最优 |
| `v4.2...seed42lr0.008` | W H256 | 8e-3 | 4.292 | 49.38 | **崩溃**（第 6 epoch 中断，末轮 ValPPL 105） |

**数据结论（实测）**：Transformer 的最优 ValPPL 随 lr 单调变差（5e-4 最好）；Waver 在 **lr=2e-3** 处最好，但 **lr=8e-3 训练发散**（Loss 抬升至 4.29、ValPPL 飙至 105 并提前中断）。这**印证复值 SSM 对过高学习率的敏感性**（实测），但不代表 2e-3 是全局最优。

图：`round3_lr_transformer.png` / `round3_lr_waver.png`；Waver lr=2e-3 的多 seed 见 `round3_lr_waver_seeds.png`（4 seed：30.37–30.63，稳定）。

---

## 4. 层内耦合系数消融（Waver H256, seed42）

![coupling](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_coupling.png)

| 目录 | 耦合 | 末 Loss(EMA) | 最优 ValPPL | 末轮 ValPPL |
|---|---|---|---|---|
| `v4.2...seed42` | 默认 | 1.952 | 31.57 | 33.23 |
| `v4.2...seed42coup0.368` | 0.368 | 1.979 | 31.54 | 32.97 |
| `v4.2...seed42coup1.0` | 1.0 | 1.963 | **31.43** | 33.18 |
| `v4.2...seed42coup2.718` | 2.718 | 2.040 | 32.08 | 33.56 |
| `v4.2...无耦合` | 无（M=0） | 2.033 | 31.91 | 33.42 |

**数据结论（实测）**：五档耦合的 ValPPL 极差仅 **0.65**，**落在 Waver seed 噪声（±0.19）之上、但幅度很小**。可观察到的弱趋势是「耦合系数过大（2.718）或完全去掉耦合都略差，1.0 附近略好」；**不宜据此宣称「适度耦合最优」是强结论**。

---

## 5. 状态维度 K 消融（K4-H196 vs K2-H256）

![k4](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_state_dim_k4.png)

| 目录 | 配置 | 参数 | 末 Loss(EMA) | 最优 ValPPL | 峰值显存 |
|---|---|---|---|---|---|
| `v4.2...seed42k4h196` | K=4, H=196 | 7.14M | 1.947 | 32.34 | **11.9 GB** |
| `v4.2...seed42`（基准） | K=2, H=256 | 7.24M | 1.952 | 31.57 | ~3.0 GB |

**数据结论（实测）**：在**近似等参数量**下，K=4 (H=196) 的训练 Loss 与 K=2 持平（1.947 vs 1.952），但 **ValPPL 更差 0.77**，且峰值显存约为基准的 **4×**（K=4 的 `matrix_exp` 开销）。本窗口内 **未观察到增大 K 的收益**。

---

## 6. Loop 循环层消融（Waver, seed42）

| 目录 | 结构 | 末 Loss(EMA) | 最优 ValPPL | 末轮 ValPPL |
|---|---|---|---|---|
| `v4.2...seed42` | loop0（基准） | 1.952 | 31.57 | 33.23 |
| `v4.2...loop1-10-1` | loop 1-10-1 | 2.034 | **30.71** | 31.81 |
| `v4.2...loop121` | loop 121 | 2.053 | 31.34 | 32.52 |

**数据结论（实测）**：`loop 1-10-1` 的 ValPPL 比基准低 **0.86**（30.71），是本节唯一的正向信号；`loop 121` 基本持平。差异仍接近 seed 噪声量级，**属于「本窗口内值得注意但未达显著」**。

---

## 7. 隐藏层宽度 H=368

![h368](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_h368.png)

| 目录 | 架构 | 参数 | 末 Loss(EMA) | 最优 ValPPL |
|---|---|---|---|---|
| `transofrmer2048h368`（+4 seed） | Transformer H368 | 7.46M | 1.877 | 24.47（seed42），5 seed 28.41±1.98 |
| `v4.2...loop0h368clean50` | Waver H368 | **13.74M** | 1.964 | 29.97（第 9 epoch 中断） |

**数据结论（实测）**：加宽到 H=368 后 Transformer 训练 Loss 进一步下降（1.877）；Waver H368 参数膨胀到 13.74M（约 Transformer H368 的 1.8×），ValPPL 仍未突破 30，**未体现宽度换容量的收益**。

---

## 8. Warmup 消融（Waver, seed42）

| 目录 | warmup | 末 Loss(EMA) | 最优 ValPPL |
|---|---|---|---|
| `v4.2...seed42` | 1977（基准） | 1.952 | 31.57 |
| `v4.2...seed42warm6000` | 6000 | 1.954 | **31.07** |
| `v4.2...seed42warm6000lr0.0005` | 6000 + lr5e-4 | 2.001 | 32.28 |

**数据结论（实测）**：延长 warmup 使起始 Loss 更高（5.002 vs 4.758），最终 ValPPL 略好（31.07）；组合更低 lr 反而更差。差异均在噪声量级。

---

## 9. 全局摘要器 dk 消融（line128 / line256）

| 目录 | dk | 参数 | 末 Loss(EMA) | 最优 ValPPL |
|---|---|---|---|---|
| `v4.2...seed42` | 基准 | 7.24M | 1.952 | 31.57 |
| `v4.2...line128` | 128 | 7.33M | 2.006 | 31.64 |
| `v4.2...line256` | 256 | 7.53M | 1.986 | **31.09** |

**数据结论（实测）**：line128/line256 与基准差异 ≤0.55 PPL，**在本窗口内为中性结果**。

---

## 10. 10 万条数据对照（单 epoch，不验证早停）

| 目录 | 架构 | 参数 | lr | 最优/末轮 ValPPL（同点，仅 1 epoch） |
|---|---|---|---|---|
| `第3轮100k条transformer...h368` | Transformer H368 | 8.94M | 3e-4 | **22.73** |
| `第3轮100k条waver...h268clean50` | Waver v4.2 H268 | 8.91M | 3e-4 | 30.52 |

**数据结论（实测）**：在约 89 500 步的单 epoch 窗口内，Transformer h368 的验证 PPL（22.73）低于 Waver h268（30.52）。**注意该 run 无多 epoch 早停点，仅一个验证读数**，不可与 round3 的 10-epoch 曲线直接横比。

![100k](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/round3/round3_100k.png)

---

## 11. v6 · 低精度与训练侧加速（Windows 11 / RTX 5080 16GB）

![v6 loss](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/v6/v6_loss_all.png)

![v6 mon](https://x005-0001-0001.github.io/Waver-SNN-SSM/compare/v6/v6_mon_drift.png)

| 目录 | 内容 | 步数 | 末 Loss(EMA) | 结论 |
|---|---|---|---|---|
| `v6` | 基线 fp32 AdamW（supernovel.txt, 3393 步/epoch） | 1006 | 7.462 | 参照 |
| `v6测试8bit` | 8-bit AdamW | 399 | 7.596 | 与对照同量级 |
| `v6测试8bit_ctrl` | 8-bit 对照 | 451 | 7.596 | **与 8bit 曲线重合** |
| `v6测试fp8` | FP8 专家 | 299 | 7.727 | 可用 |
| `v6测试moe_ctrl` | MoE 对照 | 299 | 7.713 | 参照 |

**数据结论（实测）**：`8bit` 与 `8bit_ctrl` 的训练 Loss 曲线**重合**（末值均 7.596），MON 指标漂移同量级（详见 `v6量化与训练侧加速验证报告.md`：相对偏差 <3.4e-3、无漂移）。FP8 专家量化可用。**这些 run 无验证集读数，仅作训练侧稳定性判断。**
既有结论：8-bit AdamW 省约 **35.7%** 优化器显存；NVFP4 推理省 **45.6%**；关梯度检查点 + 训练步 CUDA Graph 合计提速约 **3.4×**。

---

## 附：总览图与数据表

| 文件 | 说明 |
|---|---|
| [`matrix/matrix_loss_all.png`](./matrix/matrix_loss_all.png) | **全部 55 个 round3/100k run** 的训练 Loss(EMA) 小图总览 |
| [`matrix/matrix_valppl.png`](./matrix/matrix_valppl.png) | **全部含验证的 run** 的最优 ValPPL 排序条形图 |
| [`matrix/matrix_seed_variance.png`](./matrix/matrix_seed_variance.png) | 4 组多 seed 的稳定性（mean±1σ） |
| [`matrix/matrix.csv`](./matrix/matrix.csv) | 每 run 一行的全部指标（机器可读） |
| [`matrix/summary.json`](./matrix/summary.json) | 每 run 的完整汇总（含验证曲线） |

> 复现：解析+绘图脚本位于工作副本 `对照测试/_matrix_gen/`（只读解析日志，约 3 GB 文本 → 11 MB 结构化缓存），**不入库**以免携带绝对路径；本目录只保留**图与数据表**这两类可追溯产物。

---

## `dashboard/` —— 对照实验数据看板（只读）

只读训练日志、仅用 Python 标准库的本地交互看板（SVG 绘制，可悬停 / 缩放 / 开关曲线）。

```bash
python compare/dashboard/server.py 8777
# 浏览器打开 http://127.0.0.1:8777/
```

## `live_dashboard/` —— 实时训练看板

增量解析 `training_log.txt`、SSE 推送，面向「训练进行中」的观察。启动：

```bash
python compare/live_dashboard/server.py 8778
```

---
---

# Comparison Matrix (English)

This directory holds a unified **60-run comparison matrix**: **one row per experiment**, with conditions, reproducible metrics and **data-backed conclusions** derived from the raw logs.
All curves are **re-parsed from each run's `training_log.txt`** under `对照测试/` (EMA smoothing + uniform downsampling); no stale figures are reused.

- Machine-readable: [`matrix/matrix.csv`](./matrix/matrix.csv), [`matrix/summary.json`](./matrix/summary.json)
- Figures: [`round3/`](./round3/) (15 per-dimension), [`matrix/`](./matrix/) (3 overview), [`v6/`](./v6/)

> **Scope disclaimer**: "measured" = read directly from logs; "within this window" = observed in ~89k steps / 10 epochs only, not extrapolated; "unverified" = inference without experimental support.

## 0. Unified conditions (round3, 53 runs)

| Item | Value |
|---|---|
| Dataset | `dialogues_seg.jsonl` (9886 samples/epoch, val 988, seed 42) |
| Seq / batch | `T=2048` / `batch=1` |
| Layers / hidden | `L=3` / `H=256` (plus H=368 and K4-H196 ablations) |
| Default lr | `1e-3`, warmup 1977, ~**88 980 steps / 10 epochs** |
| Params | Transformer H256 **4.16M**; Waver/x25 H256 **7.24M** |

## 1. Cross-architecture main comparison (seed42, lr=1e-3)

| Dir | Condition | Params | Final Loss(EMA) | Best ValPPL | Last ValPPL |
|---|---|---|---|---|---|
| `transofrmer2048h256` | Transformer H256 | 4.16M | 2.155 | 29.61 | 29.85 |
| `v4.2...loop0seed42` | Waver v4.2 H256 | 7.24M | **1.952** | 31.57 | 33.23 |
| `x25...h256clean50seed42` | x25 H256 | 7.24M | 2.001 | 31.67 | 33.12 |

**Measured**: Waver/x25 train loss (EMA) is ~**0.15–0.20 lower** than Transformer, while val PPL is slightly higher (31.5–31.7 vs 29.6) — i.e. **Waver shows a larger train/val gap**. The two metrics disagree within this window; no overall superiority is implied.

## 2. Multi-seed stability (4 groups × 5 seeds)

| Group | Best ValPPL (5 seeds) | Std |
|---|---|---|
| Transformer H256 | 22.28 / 27.46 / 29.52 / 29.61 / 30.01 | **±2.89** |
| Waver H256 | 31.09 / 31.14 / 31.18 / 31.45 / 31.57 | **±0.19** |
| x25 H256 | 30.95 / 31.32 / 31.38 / 31.49 / 31.67 | **±0.24** |
| Transformer H368 | 24.47 / 29.13 / 29.36 / 29.42 / 29.68 | **±1.98** |

**Measured**: Waver/x25 seed variance is ~**one order of magnitude lower** than Transformer, but at a higher absolute PPL level. These are two different facts and must not be conflated.

## 3. Learning-rate sweep (seed42)

| Arch | lr | Final Loss(EMA) | Best ValPPL |
|---|---|---|---|
| T H256 | 5e-4 | 2.080 | 27.58 |
| T H256 | 1e-3 | 2.155 | 29.61 |
| T H256 | 2e-3 | 2.291 | 30.62 |
| T H256 | 4e-3 | 2.834 | 35.81 |
| W H256 | 5e-4 | 1.964 | 32.37 |
| W H256 | 1e-3 | 1.952 | 31.57 |
| W H256 | 2e-3 | 2.031 | **30.53** |
| W H256 | 8e-3 | 4.292 | 49.38 (**diverged**, aborted at epoch 6, last ValPPL 105) |

**Measured**: Transformer best ValPPL degrades monotonically with lr (5e-4 best); Waver is best at **lr=2e-3** but **diverges at 8e-3**. This substantiates the complex SSM's sensitivity to high lr, but does not prove 2e-3 is globally optimal.

## 4. Intra-layer coupling ablation (Waver H256, seed42)

| Coupling | Final Loss(EMA) | Best ValPPL |
|---|---|---|
| default | 1.952 | 31.57 |
| 0.368 | 1.979 | 31.54 |
| 1.0 | 1.963 | **31.43** |
| 2.718 | 2.040 | 32.08 |
| none (M=0) | 2.033 | 31.91 |

**Measured**: the spread is only **0.65** PPL — above Waver seed noise (±0.19) but small. A weak trend: too-large or zero coupling is slightly worse, ~1.0 slightly better. Not a strong "moderate coupling is optimal" claim.

## 5. State-dim K ablation (K4-H196 vs K2-H256)

| Config | Params | Final Loss(EMA) | Best ValPPL | Peak mem |
|---|---|---|---|---|
| K=4, H=196 | 7.14M | 1.947 | 32.34 | **11.9 GB** |
| K=2, H=256 | 7.24M | 1.952 | 31.57 | ~3.0 GB |

**Measured**: at near-equal params, K=4 train loss matches K=2 but val PPL is 0.77 worse and peak memory is ~**4×**. **No benefit from larger K was observed in this window.**

## 6. Loop ablation (Waver, seed42)

| Structure | Final Loss(EMA) | Best ValPPL |
|---|---|---|
| loop0 (baseline) | 1.952 | 31.57 |
| loop 1-10-1 | 2.034 | **30.71** |
| loop 121 | 2.053 | 31.34 |

**Measured**: `loop 1-10-1` is 0.86 better in val PPL — the only positive signal here; still near seed-noise scale ("noteworthy, not significant").

## 7. Hidden width H=368

| Arch | Params | Final Loss(EMA) | Best ValPPL |
|---|---|---|---|
| Transformer H368 | 7.46M | 1.877 | 24.47 (seed42); 5-seed 28.41±1.98 |
| Waver H368 | **13.74M** | 1.964 | 29.97 (aborted at epoch 9) |

**Measured**: widening helps Transformer (1.877); Waver H368 inflates to 13.74M (~1.8× Transformer H368) yet val PPL stays ≥30 — **no width-for-capacity payoff**.

## 8. Warmup ablation (Waver, seed42)

| warmup | Final Loss(EMA) | Best ValPPL |
|---|---|---|
| 1977 (baseline) | 1.952 | 31.57 |
| 6000 | 1.954 | **31.07** |
| 6000 + lr5e-4 | 2.001 | 32.28 |

**Measured**: longer warmup raises the initial loss (5.002 vs 4.758) but slightly improves final val PPL; adding lower lr is worse. All within noise.

## 9. Global summarizer dk ablation (line128 / line256)

| dk | Params | Final Loss(EMA) | Best ValPPL |
|---|---|---|---|
| baseline | 7.24M | 1.952 | 31.57 |
| 128 | 7.33M | 2.006 | 31.64 |
| 256 | 7.53M | 1.986 | **31.09** |

**Measured**: within ≤0.55 PPL of baseline — **neutral in this window**.

## 10. 100k-sample comparison (single epoch, no early stop)

| Arch | Params | lr | ValPPL (single reading) |
|---|---|---|---|
| Transformer H368 | 8.94M | 3e-4 | **22.73** |
| Waver v4.2 H268 | 8.91M | 3e-4 | 30.52 |

**Measured**: within one epoch (~89 500 steps) Transformer h368 reaches lower val PPL. **Note: only a single validation reading — not directly comparable to the 10-epoch round3 curves.**

## 11. v6 · Low-precision & training-side speedup (Win11 / RTX 5080 16GB)

| Dir | Content | Steps | Final Loss(EMA) | Note |
|---|---|---|---|---|
| `v6` | fp32 AdamW baseline (supernovel.txt) | 1006 | 7.462 | reference |
| `v6测试8bit` | 8-bit AdamW | 399 | 7.596 | same range as ctrl |
| `v6测试8bit_ctrl` | 8-bit control | 451 | 7.596 | **overlaps 8bit** |
| `v6测试fp8` | FP8 experts | 299 | 7.727 | usable |
| `v6测试moe_ctrl` | MoE control | 299 | 7.713 | reference |

**Measured**: `8bit` and `8bit_ctrl` curves **coincide** (both final 7.596); MON drift is of the same order. FP8 expert quantization works. **These runs have no validation split** — they only support training-side stability claims.
Prior results: 8-bit AdamW saves ~**35.7%** optimizer memory; NVFP4 inference saves **45.6%**; no-grad-checkpoint + train-step CUDA Graph gives ~**3.4×** combined speedup.

## Appendix: overview figures & data tables

| File | Description |
|---|---|
| [`matrix/matrix_loss_all.png`](./matrix/matrix_loss_all.png) | train Loss(EMA) small-multiples of **all 55 round3/100k runs** |
| [`matrix/matrix_valppl.png`](./matrix/matrix_valppl.png) | best ValPPL ranking of **all validation-bearing runs** |
| [`matrix/matrix_seed_variance.png`](./matrix/matrix_seed_variance.png) | 4 multi-seed groups (mean±1σ) |
| [`matrix/matrix.csv`](./matrix/matrix.csv) | one row per run, all metrics |
| [`matrix/summary.json`](./matrix/summary.json) | full per-run summary (incl. val curves) |

> Reproduction: the parse+plot scripts live in the working copy `对照测试/_matrix_gen/` (read-only over the logs, ~3 GB text → 11 MB cache) and are **kept out of the repo** to avoid absolute paths; this directory keeps only the two traceable artifacts — figures and data tables.

## `dashboard/` and `live_dashboard/`

Read-only local dashboards (Python stdlib, SVG). See `dashboard/README.md` (offline log viewer, port 8777) and `live_dashboard/README.md` (live training, port 8778).