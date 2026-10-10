# Changelog

本项目所有值得注意的变更都记录在此文件。版本号遵循 `主版本.次版本.修订号`。

> 说明：版本号统一使用纯数字三段式 `主版本.次版本.修订号`（如 `1.0.0`），不使用 `v` 前缀。

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