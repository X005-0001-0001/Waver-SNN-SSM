# Changelog

本项目所有值得注意的变更都记录在此文件。版本号遵循 `主版本.次版本.修订号`。

> 说明：版本号统一使用纯数字三段式 `主版本.次版本.修订号`（如 `1.0.0`），不使用 `v` 前缀。

---

## 9.1.0 — 2026-10-13

> 定位：**0011 系列 —— 残差连接的 LayerScale 结构化稳定改造**（**只给每层残差加一个可学习标量 γ_i，不改 SSM 内部数学路径**）。诊断已实测确认：0.85B（26 层）的梯度爆炸**从初始化第 1 步即存在**，是**深度连乘**（分层剖面单调，L0 层梯度比顶层高约 6 个数量级，裁剪前总梯度范数达 1e7 量级，loss 全程钉死）；且序列长度由 256 增至 2048 时范数**不升反降**、`delta_v_min` 全程稳定 >1.0 —— 排除了「时间轴前缀和」与「weight_norm 方向退化」两条假设。

### 新增
- **LayerScale（每层残差可学习缩放 γ_i）**：把「残差 = 恒等加法 `out + layer_input`」改为「残差 = `γ_i·wave_out + layer_input`」。γ_i 为**每层独立的可学习标量**（单张量参数 `layer_scales`，形状 `(num_layers, 1)`，进 **no_decay** 组，初值 `LAYER_SCALE_INIT = 1e-3`）。早期 `γ_i·(层内反向增益) ≪ 1`，层间反向传递近似恒等（深度连乘被自动衰减）；模型按需把有贡献的层 γ_i 抬起 —— **无阈值、无钳位、无下界**，纯结构性。
- **三处残差插入点统一覆盖**：训练 forward、并行 prefill、逐步 step（解码）三条路径的残差乘法统一乘 γ_i，保证训练 / 推理严格同构。
- **开关 `LAYER_SCALE_ENABLED`（默认 True）**：置 `False` 时 γ 恒为 1（`layer_scales=None`，forward 不做缩放），与 0010 逐位一致。

### 变更范围
- ultra3：12 个 0011 文件（`210/212/2125/310/312/3125` 的 elite 与 elitelinux）。
- 旗舰版：`0.85B1/3120011elitelinux.py`。
- release 仓：`v9/` 新增 4 个文件（`3120011elite.py`、`31250011elite.py`、`3120011elitelinux.py`、`31250011elitelinux.py`）。

### 边界与注意
- ✅ **语法与格式已校验（实测）**：13 个文件通过 AST 解析；每文件相对其 0010 源均为**最小差异 `+25/-0` 行**（x25 族因改写 `no_decay` 同一行计 `+25/-1`）；三处残差插入点、`layer_scales` 参数、`no_decay` 命中均逐项核对通过；BOM 与换行保持不变。
- ⚠️ **梯度爆炸修复效果「仅结构性论证 / 待端到端实测」**：γ_i 初值 1e-3 使早期层间反向增益乘积从 26 层连乘（≈1.8^26）压到 ≈1.0，这是**由残差缩放公式直接推出的**；但「loss 能否脱离平台 / PPL 能否改善」**尚需端到端训练验证**，本版**不声称**已修复收敛。
- ⚠️ **加载旧 checkpoint 的差异（未验证）**：0011 新增 `layer_scales` 参数，加载 0010 的 `brain.pt` 时该参数取初值 1e-3（非 1），早期前向与前版**不逐位一致**。

---

## 9.0.2 — 2026-10-13

> 定位：**0010 训练可观测性与工程修复**（**加监控、修 bug、省显存；不改 9.0.1 已定的 LR / beta2 / clip 数值**）。9.0.1 修正了学习率与二阶矩，但日志里**没有梯度范数**，无法判断 `GRAD_CLIP_NORM=1.0` 是松是紧；本版补上 `grad_norm` 监控，并顺手修掉三个工程问题。

### 新增
- **`grad_norm` 监控**：训练循环内接住 `clip_grad_()` 的返回值（**裁剪前**总梯度范数），初始化 `last_grad_norm = 0.0`，并在步日志中新增 `GN: {last_grad_norm:.3f}` 字段（与 `LR` 同点打印）。判读基线：长期 <0.5 → 裁剪未生效；长期贴 1.0 → LR 偏高 / 裁剪过紧；偶发 >5.0 → 多为脏 batch spike。

### 修复
- **`fallback_factory` 参数名 bug（潜伏）**：`build_optimizer` 末尾原以关键字 `lr=`/`weight_decay=` 调用回退工厂，而 `train_from_scratch` 传入的 lambda 形参名为 `_lr`/`_wd` ⇒ 回退分支一旦触发（BNB 不可用 / 非 CUDA）会直接 `TypeError`。改用位置参数 `fallback_factory(param_groups, lr, weight_decay)` 消除该隐患。
- **MTP 头 CE 分块（显存）**：`_mtp_loss` 原先对 `(B, T, V)` 一次性物化 fp32 logits 再整体求 CE；现与主头同构，按 `LOSS_CHUNK_SIZE` 沿序列分块累加（`reduction="sum"` 可加 ⇒ 与整体求解**数值等价**），削掉 MTP 头的显存尖峰。
- **RMSNorm `weight` 移出 weight-decay 组**：`no_decay` 原按参数名判定，漏掉了 `layer_norms` / `output_norm` / `mtp_heads.*.norm` 等的 `weight`（`name.split('.')[-1] == 'weight'`），导致 RMSNorm 的尺度被 `weight_decay=0.01` 衰减。改按**模块类型**识别（`isinstance(m, RMSNorm)` 收集 `id(m.weight)` 并排除），不动其它既有 no_decay 成员（`original0` 等仍按名保留；`original0` 实为 2 维 `(out,1)`，**不能**用「按维度分类」判据）。

### 变更范围
- ultra3：12 个 0010 文件（`210/212/2125/310/312/3125` 的 elite 与 elitelinux）。
- 旗舰版：`0.85B1/3120010elitelinux.py`。
- release 仓：`v9/` 的 4 个文件（`3120010elite.py`、`31250010elite.py`、`3120010elitelinux.py`、`31250010elitelinux.py`）。

### 边界与注意
- ✅ **语法与配置已校验（实测）**：13 个文件通过 AST 解析；逐项断点核对（`last_grad_norm` 捕获 / `GN:` 日志字段 / `fallback_factory` 位置参数 / MTP 分块 / `_rms_norm_weight_ids`）全部符合预期；BOM 与换行保持不变。
- ✅ **9.0.1 的 commit 与 tag 已推送（实测）**：`refs/heads/main` 与 `refs/tags/9.0.1` 均指向 `e9bc3dd`；本版补齐 **GitHub Release 页面对象**（9.0.1 与 9.0.2）。
- ⚠️ **收敛效果未实测（未验证）**：`grad_norm` 为新增观测量；MTP 分块与整体求解**理论等价**、RMSNorm 修复**不改前向数值路径**，但均**未做端到端对照实验**。本版**不声称**已打破 8.47 平台 / 改善 PPL。
- ℹ️ **`GRAD_CLIP_NORM` 保持 `1.0`（未动）**：放松 clip（如 3.0）会放大异常 batch 对二阶矩 `v` 的污染，与 9.0.1 的更高 LR + 更低 `beta2` 叠加将过度放大有效步长；预训练场景维持 1.0。

---

## 9.0.1 — 2026-10-13

> 定位：**0010 系列的训练超参数修正**（**仅改优化器配置，不改模型结构 / 数学路径 / 初始化**）。针对 0.85B 旗舰模型在 ~14M token 训练下 loss 长期平台化（step≈3000 起稳定在 8.47，随机基线 `ln(40000)≈10.60`）的现象，把学习率提到**预训练量级**并同步下调 AdamW 二阶矩衰减。

### 变更
- **`LEARNING_RATE`**：`1e-3`（ultra3）/ `3e-5`（旗舰版）→ **`1.0e-4`**（预训练量级；0.85B 建议区间 1e-4 ~ 3e-4）。
- **`ADAM_BETA2`（新增常量）**：`0.999` → **`0.95`**，经 `betas=(0.9, ADAM_BETA2)` 接入 bnb8bit / fused AdamW / torchao8bit / 回退 AdamW **全部优化器分支**。
- **`WARMUP_STEPS_PER_SEGMENT`（新增常量）**：**`50`**；替换原 `warmup_steps = min(2000, int(total_updates * 0.02))`。
- **`COSINE_ETA_MIN`**：`1e-6` → **`1e-5`**（文档值）；实际段末 LR = `eta_min(0.1) × LEARNING_RATE = 1e-5`（调度器 `eta_min` 为相对因子，参数保持 `0.1` 不动）。
- **`GRAD_CLIP_NORM`**：`5.0` → **`1.0`**；**`GRADIENT_ACCUMULATION_STEPS`**：`1` → **`16`**（有效 batch 放大，配合更高 LR 稳定梯度尺度）。
- **`WarmupCosineScheduler.get_lr` warmup 公式**：`factor = (step+1)/(warmup_steps+1e-8)`（从 ~0 爬）→ `factor = eta_min + (1−eta_min)·(step+1)/(warmup_steps+1e-8)`（**从 `eta_min` 起爬**），使段间热重启不归零。

### 变更范围
- ultra3：12 个 0010 文件（`210/212/2125/310/312/3125` 的 elite 与 elitelinux）。
- 旗舰版：`0.85B1/3120010elitelinux.py`（其 `GRAD_CLIP_NORM=1.0` / `GRADIENT_ACCUMULATION_STEPS=16` 已是目标值，未重复改动）。
- release 仓：`v9/` 的 4 个文件（`3120010elite.py`、`31250010elite.py`、`3120010elitelinux.py`、`31250010elitelinux.py`）。

### 边界与注意
- ✅ **语法与配置已校验（实测）**：13 个文件通过 AST 解析；逐项断点核对（`LEARNING_RATE` / `GRAD_CLIP_NORM` / `GRADIENT_ACCUMULATION_STEPS` / `ADAM_BETA2` / `WARMUP_STEPS_PER_SEGMENT` / `COSINE_ETA_MIN` / 调度器 warmup 公式）全部符合预期；BOM 与换行格式保持不变（`2120010elite.py` 与旗舰版文件的「无 BOM」状态一并保留）。
- ⚠️ **收敛效果未实测（未验证）**：本版只声称「学习率与二阶矩配置已按预训练量级修正、语法与配置核对通过」，**不声称**已打破 8.47 平台 / 改善 PPL。平台化归因（学习率过低 / 模型容量饱和 / 数据信息量不足）**仍未经对照实验分离**；`beta2=0.95` 数值为 LLM 主流配置，但「小 batch ⇒ 必须降 beta2」的机制解释**尚待验证**。
- ⚠️ **仅对「从零训练」或「迁移旧 checkpoint」生效**：加载既有 `brain.pt` 会沿用旧优化器状态，本版改动主要体现在后续训练的学习率调度上。

---

## 9.0.0 — 2026-10-12

> 定位：在 009（**含 8.1.1 初始化修复**）基础上派生 **0010 变体** —— 新增**多 token 预测（MTP, Multi-Token Prediction）**：用**共享 `output_norm`** 输出的多个轻量头监督同一隐状态 `norm_last`，主头预测 t+1、MTP 头预测 t+2，以**多时间尺度**目标打破「h 恒定即可最小化 loss」的死通道。**仅追加辅助监督项**，主头结构 / 数学路径不变（`MTP_LOSS_SCALING=0` 或 `MTP_NUM_HEADS≤1` 时与 009 逐位一致）。

### 新增
- **MTP 开关与超参数**（文件头集中定义，独立可关，默认开启）：
  - `MTP_NUM_HEADS`（默认 `2`）：总头数（含主头）。
  - `MTP_PREDICT_DISTANCES`（默认 `[1, 2]`）：主头预测 t+1、MTP 头预测 t+2；长度须等于 `MTP_NUM_HEADS`。
  - `MTP_LOSS_SCALING`（默认 `0.1`）：MTP 辅助 loss 总缩放；`0` 即关闭。
- **`MTPHead`**：`Linear(hidden→hidden, bias=False) → RMSNorm → Linear(hidden→vocab, bias=False)`，**从共享 `norm_last` 出发、不新增独立主 norm**（与主头共用 `output_norm`）。
- **`_mtp_loss`**：对每个 MTP 头按 `head.distance` 平移标签、`ignore_index=-100` 掩码，逐头 `F.cross_entropy(reduction="sum")` 后按**有效 token 数归一**，多头取均值；`seq_len ≤ shift` 的头跳过。
- **训练分支**：主导出后追加 `loss = loss + MTP_LOSS_SCALING * self._mtp_loss(norm_last, labels, seq_len)`。
- **0010 变体**：ultra3 的 12 个 009 文件（6 elite + 6 elitelinux）各派生一个 0010；旗舰版 `0.85B1/312009elitelinux.py` 派生 `3120010elitelinux.py`；本次 release 发布主族 `3120010elite.py` 与 x25 族 `31250010elite.py`，及其对应 linux 变体。

### 修复
- **语义修正（相对原始草案）**：因 `labels[t] = input_ids[t+1]`，草案的 `mtp_labels[:, :seq_len - distance] = labels[:, distance:]` 实际会预测 t+distance+1（**越一位**）；0010 取 `shift = distance − 1`，使 `distance=2` 精确预测 t+2，符合「预测距离 [1,2]」的设计意图。

### 实现要点
- **CUDA Graph 兼容**：`_mtp_loss` **不做任何 device→host 同步**（不用 `int()` / `.item()`），`seq_len` 为 Python int，仅用 `valid.sum().clamp(min=1)`，全程留在图内 ⇒ 训练步 CUDA Graph 可捕获。
- **两族实现一致**：linux 族 MTP 未复用 `_linux_lm_head`（避免 dummy `logit_scale` 与两族代码分叉），改用 eager `F.cross_entropy`。

### 边界与注意
- **关闭即等价（理论可推）**：`MTP_LOSS_SCALING=0`（或 `MTP_NUM_HEADS≤1`）时 MTP 头不实例化、辅助项不计入 ⇒ 前向与 009 逐位一致。
- ⚠️ **效果未实测（未验证）**：本版只声称「MTP 结构已按设计接入、开关可关、训练图可捕获」，**不声称**已改善 PPL / 收敛 / 死通道 —— 其对训练的实际收益**尚未经实验证实**。显存 / 速度的理论增量（每头约 `2·H² + 2·H·V` 参数）亦**未实测**。

---

## 8.1.1 — 2026-10-12

> 定位：009 系列的**初始化修复**（**仅改初始值**，不改数学路径 / 结构 / 优化器）。修复「SSM 从未激活」的初始化死锁：从零训练时 loss 恒等于随机基线、SSM 内部参数无梯度。

### 修复
- **病征（实测）**：从零训练 `312009elite`，三步 loss ≈ 10.63 恒定（≈ `ln 40000`，即随机权重基线）；`wave_raw_std`≈0.0008–0.0027、`amp_std`≈0.0006–0.0023（输入 `I_std`≈1.0 正常，而经 SSM 后输出近常数）；28 层 `delta_mean` 全为 0.0547–0.0549（= `softplus(0.69)×softplus(−2.5)` 的理论值）；`Bn/Dn/An/Mn` 三步逐位不变 ⇒ 梯度未进入 SSM 内部参数。
- **病根（理论可推）**：初始化后 `delta = softplus(delta_raw)·softplus(−2.5) ≈ 0.055`、`_g = 1 − e^{−δν} ≈ 0.03`，`B_disc = B_real·_g·softplus(0)` 被压到 `B_real` 的 ~2%，状态注入信号趋零；形成「δ 小 → _g 小 → B_disc 小 → 对 δ/B 的梯度小」的正反馈锁死，靠梯度无法自行逃出。
- **改法（仅初始化，保留 `GAIN_NORM_ENABLED` 数学不变）**：
  - `ParallelComplexWaveLayer.__init__`：`self.delta_scale` 初值 `−2.5 → −1.0`（softplus 0.075 → 0.313，×4）；`self.B_scale` 初值 `0.0 → 1.0`（softplus 0.693 → 1.313，×1.9）。
  - `_init_weights`：`_small` 移除 `delta_linear` / `B_linear`（其 weight_norm 方向初始化 gain 由 0.1 → 1.0），仅保留 `theta_delta_proj`。

### 变更范围
- ultra3：6 个 009 elite（`210/212/2125/310/312/3125`）+ 6 个 009 elitelinux，共 12 个文件。
- 旗舰版：`0.85B1/312009elitelinux.py`。
- release 仓：`v8/` 的 4 个文件（`312009elite.py`、`3125009elite.py`、`312009elitelinux.py`、`3125009elitelinux.py`）。

### 边界与注意
- ⚠️ **只对「从零初始化」生效**：加载 checkpoint（`brain.pt`）会覆盖新初始值 ⇒ 需从零训练或移走 `brain.pt`，否则修复不生效。
- **对训练收敛的影响尚未实测（未验证）**：本版只声称「修正了初始化量级、解除了 δ/B 的正反馈锁死条件」，**不声称**已修复 PPL 收敛 / 训练效果。

---

## 8.1.0 — 2026-10-12

> 定位：v8 的**Linux 单卡加速变体** —— 在 009 基础上派生 **009elitelinux**：把 Win11 上因 Triton / Inductor 支持受限而无法启用的编译加速（`torch.compile` / Inductor-Triton 融合）与 DataLoader 多进程等**适用于单卡**的优化做成**独立可关**的开关（默认开启）。不改模型结构 / 优化器数学；**全部开关关闭时与对应 009 逐位一致**。

### 新增
- **6 个 Linux 变体**：`210009elitelinux.py`、`212009elitelinux.py`、`2125009elitelinux.py`、`310009elitelinux.py`、`312009elitelinux.py`、`3125009elitelinux.py`（= 对应 009 + Linux 单卡加速开关区）。
- **文件头集中开关区**（每个独立可关；全关 ⇒ 与对应 009 **逐位一致**）：
  - `LINUX_USE_TORCH_COMPILE`（**默认 True**）：用 Inductor + Triton 融合**标准算子** —— `RMSNorm.forward`、FFN / SwiGLU 专家块、LM Head 分块交叉熵。
  - `LINUX_COMPILE_MODE`（默认 `max-autotune-no-cudagraphs`）：显式不用 Inductor 自身的 cudagraph trees，避免与手写训练步 CUDA Graph 叠加冲突。
  - `LINUX_COMPILE_DYNAMIC`（默认 `False`）：静态形状（batch=1、T 固定）利于编译。
  - `LINUX_NUM_WORKERS`（默认 `4`）：DataLoader 多进程 + `persistent_workers`（覆盖文件内 `NUM_WORKERS`）；Windows 曾被迫设 0，Linux 侧稳定。
  - `LINUX_OPTIMIZER`（默认 `'bnb8bit'`）：`'bnb8bit'`（= 009 原样）\| `'fused_adamw'`（`torch.optim.AdamW(fused=True)`）\| `'torchao8bit'`（可选，会改数值）。
  - `LINUX_USE_TF32`（默认 `False`）：可选开 `allow_tf32`，会改数值，默认关。
- **自定义算子隔离**：对 `BlellochScanFn` 扫描核心与矩阵指数闭式实现（x25 族为 `_capturable_expm_skew`，其余为 `_closed_form_expm_2x2` / `_capturable_expm`）加 `torch.compiler.disable`，避免 graph break 破坏其余融合。
- **失败兜底**：`try/except` 包裹编译，编译不可用 / 失败 ⇒ 自动回退 eager 并写日志（与 8-bit 优化器双重兜底同风格）。

### 验证
- **数值等价**：固定输入前向，开关全关时 logits 最大绝对误差 **0.0**（逐位一致）；全开（WSL, sm_120）时 logits 最大绝对误差 **9.537e-07 < 1e-6**。
- **收敛一致性（超小 A/B，同一张 5080）**：Win11 未优化 `312009elite.py` vs WSL 优化 `312009elitelinux.py`（VOCAB≈256 / H=128 / L=2 / T=256，均 534 步）：最终 Loss 1.0182 vs 1.0178、PPL 2.77 vs 2.77，逐步 Loss 曲线走势一致、**无正确性回归**。
- **证据**：`compare/linux/`（A/B 曲线图 + 数据表）。
- **环境**：Linux 侧日志确认 `torch.compile` 命中（RMSNorm / LMHead-CE 首次编译成功，无 graph-break 报错）、`num_workers>0` 生效。

### 安全工作边界
- SSM 核心（自定义 autograd 扫描 + 闭式矩阵指数）被**隔离为 eager**，故编译融合收益只在 norm / FFN / CE 三处标准算子；`torch.compile` 若与手写训练步 CUDA Graph 冲突，默认用 `-no-cudagraphs` 模式并存，仍失败则回退 eager。
- **超小实验只证明「收敛一致 + 无正确性回归」，不能代表真实规模的吞吐收益**；本版本只声称「Linux 单卡加速路径可用 + 收敛一致性已验」，**不声称**具体加速比。
- 单卡场景，**不含** NCCL / DDP / FSDP 等多卡项。

### Release 文件
- `v8/` 新增 `312009elitelinux.py`（主族）与 `3125009elitelinux.py`（x25 族）；其余 4 个 linux 变体仅保留在 ultra3 私有开发目录，不入 release。

---

## 8.0.0 — 2026-10-11

> 定位：v8 的**动力学稳定性分支（注入端结构性修法）** —— 在 007 基础上派生 **009 变体**：把注入项从 **ZOH 连续积分** 改为 **遗忘平衡注入** `B̄ = B̃·(1−|A|)`，使**峰值增益**在**解耦极限（M=0）下恒等** `G_peak ≡ |B̃|`、在**耦合（M≠0）下保持有界**（实测 ≲3.5×），从结构上消除慢极点（ν→0）的直流增益爆炸。不改动模型结构 / 优化器 / 数值路径（`GAIN_NORM_ENABLED=False` 时与 007 逐位一致）。

### 新增
- **峰值增益病灶**：标量复递推 `h_t = A·h_{t−1} + B̄·x_t` 的峰值增益为 `G_peak = |B̄|/(1−|A|)`（单位圆到 A 最近距离的倒数）。旧 ZOH 注入 `B̄ = B·(e^{δγ}−1)/γ = B·(1−A)/γ` 给出 `G_peak = |B|/|γ| · |1−A|/(1−|A|)`，在 `ν→0`（`1−|A|→0`）时**无界** —— 慢极点会把直流增益放大到任意大。实测更正：固定 ν 单独令 `|γ|→0` 增益不变，故无界入口**仅** `ν→0` 一条（此前“双入口”表述已收窄）。
- **遗忘平衡注入开关** `GAIN_NORM_ENABLED`（**默认开启**）：新规则 `B̄ = B̃·(1−|A|)`，其中 `B̃ = B_real·B_scale`（原可学习系数，不动），`|A| = e = exp(−δν)`。于是**解耦（M=0）时 `G_peak ≡ |B̃|`（精确恒等）**；**耦合（M≠0）时** `A` 非正规，`G_peak` 偏离 `|B̃|`、可放大到同量级倍数（实测 K≤8、ν≥1e-4 时 ≲3.5×），但**始终有界**。数值验证：小 ν 通道旧式 `G_peak` 随 ν→0 无界（实测 ν=1e-3→~962、ν=1e-4→~9546），新式保持 O(1)（解耦 ≈1，耦合 ≲2.4）。
- **重参数化而非截断**：任意目标 `B̄` 可由 `B̃ = B̄/(1−|A|)` 精确表示（δ>0、ν>0 严格 ⇒ `0 < 1−|A| ≤ 1`，无除零），**可达集合与 007 完全相同**；实现为纯乘加、无开方无除法，梯度安全（避开 `sqrt(B²)` 在 `B̄=0` 处的 NaN 梯度）。
- **009 变体**：ultra3 的 6 个 007 文件（`210/212/2125/310/312/3125`）各派生一个 009；本次 release 发布主族 `312009elite.py` 与 x25 族 `3125009elite.py`，同时保留 007 供对照（009 = 007 + 遗忘平衡注入）。
- **旗舰版同步**：旗舰版 312007 派生 `312009elite.py`（= 旗舰版 312007 + 遗忘平衡注入；梯度累积 / 裁剪 / LR 全部不动）。

### 破坏性变更与边界
- 开关开启后，加载旧 007 checkpoint 的前向输出**不再逐位一致**（注入公式已改变）；如需与 007 严格对照，置 `GAIN_NORM_ENABLED=False` 即可**逐位回退**。**009 不是 007 的逐位兼容版本。**
- **对训练收敛的影响待验证**：本变体由「峰值增益结构性有界」支撑（已数值验证：**解耦 `G_peak ≡ |B̃|` 精确，耦合有界 ≲3.5×、非严格上界**）；但其对病态 PPL 收敛的长期效果**尚未经实验证实** —— 先前对照实验中把增益压制 ~160×，在 200 步续训窗口内均未观察到 PPL 同步改善。**请勿解读为「已修复 PPL 震荡」。**

### 文档（对照实验矩阵补全）
- `compare/` 由「代表性图集」扩写为**完整对照矩阵**：收录 **60 组实验（53 个 round3 + 2 个 100k + 5 个 v6）**，**每条实验一行**，附条件、可复现指标与**基于原始日志的数据结论**（见 `compare/README.md`，中英双语）。
- 新增图集：`compare/round3/` 扩至 **15 张**（含 h368 多 seed）；新增 `compare/matrix/` **3 张总览图**（全部 53 run 训练曲线小图 / 最优 ValPPL 排序 / 4 组多 seed 稳定性）；新增 `compare/v6/v6_loss_all.png`、`v6_mon_drift.png`。
- 新增机器可读数据：`compare/matrix/matrix.csv`（每 run 一行）与 `compare/matrix/summary.json`。
- 结论口径修正：耦合消融由旧表述「适度耦合最优（强结论）」改为**实测极差仅 0.65 PPL、落在 seed 噪声之上但幅度很小**；补充多 seed 标准差实测（Transformer H256 ±2.89 / Waver H256 ±0.19 / x25 H256 ±0.24）。解析与绘图脚本保留在工作副本 `对照测试/_matrix_gen/`，不入库。

---

## 7.2.0 — 2026-10-11

> 定位：v7 的**动力学稳定性分支（修正 7.1.0 的抓手）** —— 在 007 基础上派生 **008 变体**：**回退 017 的 θ 结构性上界**（回到原生 `θ = exp(theta_raw)`），并新增 **δ 下限** 软约束，直击 0.6B 训练中 δ→0 导致的无遗忘积分器与 amp 自激。不改动模型结构 / 优化器 / 数值路径（`DELTA_MIN_ENABLED=False` 时与 007 逐位一致）。

### 修正（相对 7.1.0）
- **017 的 θ 上界抓错了变量**：旋转本身模 2π，「转过头」不存在；真正的现象是**采样混叠**（`δ·θ > π`）。实测 δ~0.05，混叠起点 `θ ≈ π/δ ≈ 30`，而 π 上限远未触界；且 `π·sigmoid` 在边界附近梯度趋零、反害优化。故 008 **回退原生 `θ = exp(theta_raw)`**，与 007 逐位一致（`THETA_STRUCTURAL_BOUND_ENABLED=False`，开关保留供对照）。

### 新增
- **δ 下限开关** `DELTA_MIN_ENABLED`（**默认开启**）与 `DELTA_MIN`（默认 `0.01`）：病根是 `δ = softplus(delta_raw)·softplus(delta_scale)`，训练可把 `delta_scale` 推向 −∞ 使 δ→0；此时 `A = exp(δ·Ω) → I`，该通道退化为**无遗忘积分器**（记忆无界）→ BPTT 梯度沿时间不衰减而爆掉 → amp 自激。旗舰版 0.6B 实测：`amp_std` 从 0.0001 涨到 1231，`PPL` 长期 2000+ 震荡；而 `|A| = exp(−δν)` 与 θ 无关，故卡 θ 治不了自激，真正的抓手是 δ。
- **加性软下限** `δ ← DELTA_MIN + δ`：保证 `δ ≥ DELTA_MIN` 且**处处可导、无梯度死区**（梯度 1:1 透传）；最坏情形 `|A| ≤ exp(−DELTA_MIN·ν_min) = exp(−1e-4) < 1`，记忆时域有界。`DELTA_MIN_ENABLED=False`（或 `DELTA_MIN=0`）时与 007 逐位一致。
- **008 变体**：ultra3 的 6 个 007 文件（`210/212/2125/310/312/3125`）各派生一个 008；本次 release 发布主族 `312008elite.py` 与 x25 族 `3125008elite.py`，同时保留 `312007elite.py` / `312507elite.py` 供对照（008 = 007 + δ 下限）。
- **旗舰版同步**：`D:\旗舰版\0.6B参数\312008elite.py`（= 旗舰版 312007 + δ 下限），保留 312007 供对照。

---

## 7.1.0 — 2026-10-11

> 定位：v7 的**结构性稳定性分支** —— 在 007 基础上派生 **017 变体**，把相位基频 θ 的 `exp` 参数化改为**结构性上界** `θ = π·sigmoid(theta_raw)`；不改动模型结构 / 优化器 / 数值路径（开关关闭时与 007 逐位一致）。

### 新增
- **θ 结构性上界开关** `THETA_STRUCTURAL_BOUND_ENABLED`（**默认开启**）：物理来源是离散旋转的非混叠（Nyquist）约束 `δ·θ < π` ⇒ 基频 θ 应落在 `(0, π)`。参数化 `θ = π·sigmoid(theta_raw)` **处处可导、无梯度死区、渐近逼近 π 而不真正触界**；替代原生 `θ = exp(theta_raw)`（无上界，θ 可漂移至任意大 → 相位自激）。置 `False` 回退原生 `exp` 参数化，与 007 **逐位一致**。
- **初始化一致**：结构性上界下反解 `theta_raw`，使初始 θ 分布与原生 log-uniform 完全一致（往返误差 ~1e-16）。
- **017 变体**：ultra3 的 6 个 007 文件（`210/212/2125/310/312/3125`）各派生一个 017；本次 release 发布主族 `312017elite.py` 与 x25 族 `3125017elite.py`，同时保留 `312007elite.py` / `312507elite.py` 供对照。

---

## 7.0.3 — 2026-10-11

> 定位：v7 的**生成路径正确性补丁** —— 修复启用 `USE_CUDA_GRAPH_DECODE` 时，菜单 2 对话在 AMP（bf16）下生成若干步后崩溃的问题；不改动模型结构 / 优化器 / 数值路径。

### 修复
- **CUDA Graph 解码 × autocast 缓存冲突**：`generate` 原先让 `prefill` 与每一步解码各自单独进出 `torch.amp.autocast`。抓图前的预热在 autocast 内执行，会把 bf16 权重副本写入 autocast 缓存；随后的抓图直接复用这些缓存张量、未记录 cast 算子；而退出最外层 autocast 会触发 `clear_autocast_cache()` 释放这些副本 —— 图回放便读到已被复用的显存，数值发散产生 NaN/Inf 非法分布，最终 `torch.multinomial` 触发 `TensorCompare.cu:109 Assertion 'input[0] != 0'` 设备端断言、进程硬崩。
- **修复方式**：把整个 `prefill` + 逐 token 解码循环保持在**同一个** autocast 上下文内，不再每步单独进出。autocast 缓存全程存活，抓图烘焙的 bf16 权重始终有效。

### 其他
- **训练步 CUDA Graph 不受影响**：其抓图预热每次迭代都进出 autocast，缓存已在抓图前清空，故训练路径无需改动。
- **验证**：同一 checkpoint（seed=0）下，7.0.2 开启图解码必崩；修复后正常生成无异常；固定 token 序列上「图解码 vs eager」logits 最大绝对误差为 **0.0**（逐位一致）。

---

## 7.0.2 — 2026-10-10

> 定位：v7 的**数据预处理与日志降噪补丁** —— 修复菜单 11 切分后 system 段丢失、对齐切分长度口径，并消除无 triton 环境下的无害告警；不改动模型结构 / 优化器 / 数值路径。

### 修复
- **菜单 11 切分丢失 system 段**：`p4_segment_dataset` 按 `(user, assistant)` 对切分时，会把对话开头的连续 `system` 段丢弃，与训练路径 `_truncate_pairs` 语义不一致。现改为：先摘出开头连续 `system` 段并计入 token 预算，切分后拼接到每条样本前（`{'messages': head + list(pair)}`），保证 system 在多轮切分后仍可见。
- **切分长度口径对齐**：`pair_tokens` 补上真实样本长度的 `+1`（`input_ids = tokens[:-1]`），避免边界样本多留一条，与训练侧长度口径一致。

### 其他
- **日志降噪**：`bitsandbytes` 导入会顺带导入 `torch.utils.flop_counter`，在无 triton 的 CUDA 环境下打一条 `triton not found; flop counting will not work for triton kernels` 的 WARNING。现将其 logger 降至 ERROR（纯降噪，不影响任何功能）。

---

## 7.0.1 — 2026-10-10

> 定位：v7 的**数据管线正确性补丁** —— 修复训练样本中 system 段丢失、以及残缺样本污染生成行为两处问题；不改动模型结构 / 优化器 / 数值路径。

### 修复
- **system 消息丢失**：`_truncate_pairs` 只识别相邻 `(user, assistant)` 对，会把对话开头的连续 `system` 段整段丢弃 —— 于是出现"`render_messages` 支持 system、训练语料里却永远看不到 system"的不一致。现改为：先把开头连续 `system` 段原样摘出并计入 token 预算，再从尾部取 `(user, assistant)` 对，最后合并返回。
- **以 user 结尾样本的护栏**：仅含 user（或 system + user）、没有任何 assistant 回复的残缺样本，其唯一训练目标只剩结尾的 `<|eos|>`，会让模型学到"user 之后立刻结束序列"、污染生成行为。新增 `_has_assistant_target` 护栏：`_render_conversation` / `_tokenize_conversation` 在无 assistant 可训练目标时分别返回 `None` / 空张量，`StreamingJSONLDataset` 迭代时跳过该行。

---

## 7.0.0 — 2026-10-10

> 定位：v6 的训练侧「无损加速 + 降显存」版本 —— 所有新增开关默认启用、失败自动回退、数学逐位一致。

### 新增
- **训练步 CUDA Graph** `USE_TRAIN_STEP_CUDA_GRAPH`：把训练步 fwd + bwd 抓成 CUDA Graph，按序列长度分桶缓存（`TRAIN_GRAPH_BUCKET` / `TRAIN_GRAPH_MAX_GRAPHS`）；`optimizer.step()` / `scheduler.step()` / 梯度裁剪留在图外；仅 CUDA + 非 MoE + batch=1 生效，失败自动回退 eager。
- **8-bit AdamW** `USE_ADAMW8BIT`：主训练优化器改用 bitsandbytes `AdamW8bit`，约省 **36%** 优化器显存；不可用 / 非 CUDA 时自动回退原生优化器。
- **可捕获矩阵指数** `USE_CAPTURABLE_EXPM`：K>2 走「缩放-平方 + Taylor 级数」，纯 matmul / 逐元素、无 LAPACK，可被 CUDA Graph 捕获（**默认关闭**，K>2 默认回退 `torch.linalg.matrix_exp`）。

### 变更
- 矩阵指数 / 抓图相关开关集中到文件头部，便于统一开关。
- 梯度检查点默认开启；loss 缩放与 `zero_grad(set_to_none=False)` 适配训练图。

### 其他
- 精简对照实验纳入 `compare/`；README 重写为 v7 级并新增本 `CHANGELOG.md`。

---

## 6.0.0 — 2026-10-10

### 新增
- **闭式矩阵指数（K=2）** `USE_CLOSED_FORM_EXPM`：`exp(δ·Ω_t)` 走 2×2 复矩阵 Cayley–Hamilton 闭式 + 级数小量保护（`CF_EXPM_MIN_ABS_S2` / `CF_EXPM_SERIES_TERMS`）；与 `torch.linalg.matrix_exp` 数学恒等，**无 LAPACK、可被 CUDA Graph 捕获**。
- **批量分词** `_encode_batch_with_offsets`：一次 `encode_batch` 一批，保持 shuffle 时机与 rng 序列完全一致。
- `_split_reasoning` 同时支持 `think` / `thinking` 双标签。

### 变更
- `USE_CUDA_GRAPH_DECODE` 默认开启（12/12 抓图成功且与 eager 逐位一致）。
- 分词缓存键纳入归一化模式 + `tokenizers` 版本。

### 修复
- 梯度检查点 `preserve_rng_state=False`；BlellochScanFn 去掉多余 clone；保留 `R_cpl` 缓存键（修复抓图区内惰性重算导致的抓图失败）。

---

## 5.0.0 — 2026-10-04

### 新增
- **数据格式层（v5 schema）**：训练数据统一为 `messages` 结构，支持 `reasoning_content`（思维链）与 `system` / `tools` / `tool_calls` 预留字段。
- **模板化渲染** `render_messages` + offsets 掩码：无新字段时与旧格式**逐 token 等价**。
- **流式安全前缀解码**：生成侧按字符边界安全解码，避免多字节字符被截断。
- RL / GRPO 题库与 SFT 输出同步迁移到 v5 schema。

### 修复
- 修掉“吞字 / 吞空格”：`normalize_text` 模式化；新词表走 `ByteLevel`（保空格 / 换行 / 大小写，永不产生 `unk`）。

### 变更
- 配置精简：`SUMMARY_TYPE` 合并为单常量开关；统一补 `id_cache`；移除凑数配置项。
- **不再向后兼容**旧 `conversations` 外壳 / 旧词表元数据 / 旧题库（遇到即报错）。
- **v5 起同时发布主族（`312xxx`）与 x25 族（`3125xx`）** 两个文件。

---

## 4.2.0 — 2026-10-02

### 修复
- 修复 jsonl 数据路径下的显存碎片问题。

---

## 4.1.0 — 2026-09-28

### 新增
- 与经典 Transformer 的**受控对照实验**（同词表 / 同优化器 / 参数量对齐）。
- 层内耦合（Intra-layer Coupling）开 / 关对照。

---

## 4.0.0 — 2026-09-23

### 新增
- **Loop 机制**：权重共享循环层（`USE_LOOP` / `LOOP_RESIDUAL_SCALE` / `LOOP_INDEX_EMBED` / `LOOP_MOE_MODE`）。
- **SSM 分块扫描** `SSM_CHUNK_SIZE`（长序列分块，数学严格等价）。

### 修复
- AMP 采样精度、x25 `step` 路径缩进缺陷、分块扫描形状错误。
- 移除实测无显存收益的 `USE_CHECKPOINT_MATEXP`。

---

## 3.1.0 — 2026-09-17

### 变更（架构重构）
- **统一矩阵指数路径**：`A_t = matrix_exp(δ_t · Ω_t)`，`Ω_t = M + diag(-ν + i·(θ_base + θ_delta_t))`。
- 删除 eig 路径（`_compute_eig` / `USE_EIG_SCAN` / `ScalarScanFn` 等）。

### 新增
- δ 逐通道化，修复 `K>1` 耦合时 `matrix_exp` 不良定义。
- 数据依赖旋转 `theta_delta_proj`（`THETA_DELTA_ENABLED` / `THETA_DELTA_SCALE`）。
- 稳定性由 **Bendixson 定理**结构性保证，不依赖任何阈值。

---

## 2.0.0 — 2026-09-09

### 新增 / 优化
- Blelloch 扫描填充修复：算量 / 显存直接减半。
- Ω 特征分解 → 对角复扫描；分块线性注意力；并行 Prefill。
- CUDA Graph 解码（与 eager 逐位 0 误差）。
- MoE 独立 Copy Stream；MoE BF16 常驻。

### 修复
- Win11 DataLoader spawn 死锁、AMP + `linear_attention` 生成崩溃、COUPLING + MoE 既有 bug。

---

## 1.0.2 — 2026-09-05

### 修复
- `brain.pt` 惰性加载：启动不再 `torch.load`，菜单 1/2/4/7-9 接入惰性加载守卫，聊天不再触发 OOM。

---

## 1.0.1 — 2026-09-05

### 修复
- FastBPE 逐 token 性能问题：新增 `_vocab_size_cache`，`encode()` 改用缓存，消除循环内 `get_vocab_size()` 调用。

---

## 1.0.0 — 2026-09-05

### 首次公开发布
- `ParallelComplexWaveLayer`：复值 SSM 实值化 2×2 块同构（无复数中间量）。
- MoE-FFN、全局摘要器（Mean / LinearAttention / MiniWave）。
- 自定义 Blelloch 并行前缀扫描，O(log T)。
- 训练 / 推理 / 强化学习（GRPO·RL）单文件全流程。
- 永久引用 DOI: [10.5281/zenodo.22309186](https://doi.org/10.5281/zenodo.22309186)