# Changelog

本项目所有值得注意的变更都记录在此文件。版本号遵循 `主版本.次版本.修订号`。

> 说明：`v1.x` 为早期补丁版本（保留 `v` 前缀）；`2.0.0` 起改为纯数字版本号。

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

## v1.0.2 — 2026-09-05

### 修复
- `brain.pt` 惰性加载：启动不再 `torch.load`，菜单 1/2/4/7-9 接入惰性加载守卫，聊天不再触发 OOM。

---

## v1.0.1 — 2026-09-05

### 修复
- FastBPE 逐 token 性能问题：新增 `_vocab_size_cache`，`encode()` 改用缓存，消除循环内 `get_vocab_size()` 调用。

---

## v1.0.0 — 2026-09-05

### 首次公开发布
- `ParallelComplexWaveLayer`：复值 SSM 实值化 2×2 块同构（无复数中间量）。
- MoE-FFN、全局摘要器（Mean / LinearAttention / MiniWave）。
- 自定义 Blelloch 并行前缀扫描，O(log T)。
- 训练 / 推理 / 强化学习（GRPO·RL）单文件全流程。
- 永久引用 DOI: [10.5281/zenodo.22309186](https://doi.org/10.5281/zenodo.22309186)