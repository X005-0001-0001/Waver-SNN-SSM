# Changelog

本项目所有值得注意的变更都记录在此文件。版本号遵循 `主版本.次版本.修订号`。

> 说明：版本号统一使用纯数字三段式 `主版本.次版本.修订号`（如 `1.0.0`），不使用 `v` 前缀。

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