# 对照实验（精简版）

本目录**只保留最有代表性的对照结果与复现脚本**，不收录原始 run 日志、检查点与中间产物。
所有图片均由 `对照测试/` 下的原始训练日志**重新解析生成**（解析器：EMA 平滑 + 最小/最大桶降采样）。

---

## `round3/` —— 第 3 轮受控对照（主力结果）

统一实验条件：**同词表、同优化器、同超参数**，隐藏层 `H=256`、层数 `L=3`、序列长 `T=2048`、
`batch=1`、`lr=1e-3`、数据集 `dialogues_seg.jsonl`（9886 样本/epoch，约 8.9 万步）。
每张图为**三联图**：训练 Loss（EMA，对数轴）｜验证 PPL（按 epoch，对数轴）｜Loss 差值（相对基线）。

| 文件 | 内容 |
|---|---|
| `round3_main.png` | **主对照**：Waver v4.2 / x25 vs 经典 Transformer（h256, seed42）|
| `round3_seeds_transformer.png` | Transformer h256 · 5 seed（42 / 43 / 114514 / 1919810 / 31415926）|
| `round3_seeds_waver.png` | Waver v4.2 h256 · 5 seed |
| `round3_seeds_x25.png` | x25 h256 · 5 seed |
| `round3_lr_transformer.png` | Transformer h256 学习率扫描（5e-4 / 2e-3 / 4e-3）|
| `round3_lr_waver.png` | Waver v4.2 h256 学习率扫描（5e-4 / 1e-3 / 2e-3 / 8e-3）|
| `round3_lr_waver_seeds.png` | Waver v4.2 h256 `lr=2e-3` · 多 seed |
| `round3_coupling.png` | 层内耦合系数消融（0.368 / 1.0 / 2.718 / 无耦合）|
| `round3_state_dim_k4.png` | 状态维度 K=4 消融（K4 h196 vs K2 h256 基准）|
| `round3_loop.png` | Loop 循环层消融（loop 1-10-1 / 121 / 0 基准）|
| `round3_h368.png` | 更宽隐藏层 h368 对照（Transformer vs Waver v4.2）|
| `round3_warmup.png` | Warmup 消融（6000 / 6000+lr5e-4 / 无 warmup）|
| `round3_line.png` | 全局摘要器 dk 消融（line128 / line256 / 基准）|
| `round3_100k.png` | 10 万条数据对照（Transformer h368 vs Waver v4.2 h268）|

**观察（数据事实，非结论包装）**

- 主对照与多 seed 下，Waver / x25 的训练 Loss（EMA）整体**略低于** Transformer 基线，曲线高度接近；
- Waver 的验证 PPL 在 5 个 seed 上**高度稳定**（31.1–31.6），而 Transformer 波动明显更大（22.3–30.0）；
- 学习率扫描：Waver 在 `lr=8e-3` 时出现明显恶化（loss 抬升、提前中断），印证了复值 SSM 对过高学习率的敏感性；
- 耦合消融：完全去掉层内耦合后 loss 抬升约 0.05，耦合系数过大（2.718）同样变差，**适度耦合最优**；
- 其余消融（K4 / loop / h368 / warmup / line / 100k）均为**中性或混合**结果，一并保留以供查证。

---

## `v6/` —— 低精度方案与训练侧加速验证

平台：Windows 11 / RTX 5080 16GB (sm_120) / Python 3.13 / torch + bitsandbytes。
基线：`v6/312006elite.py`（H=256, L=12, T=4096, STATE_DIM=4, K=2, MoE=off），种子统一 42。

| 文件 | 内容 |
|---|---|
| `v6量化与训练侧加速验证报告.md` | 完整结论（8-bit AdamW 稳定性与显存、FP8 / NVFP4 专家量化、训练步 CUDA Graph 收益）|
| `v6_8bit.png` | AdamW8bit vs AdamW：Loss 曲线 + MON 指标相对偏差（对数轴）|
| `v6_memory.png` | 优化器显存 / 专家量化显存对比（双柱状图）|
| `analyze_8bit.py` | 解析训练日志，对比 8-bit 与 fp32 优化器的 MON 指标偏差 |
| `probe_opt_memory.py` | 优化器显存探针 |
| `probe_quant_all.py` / `probe_nvfp4.py` | MoE 量化全谱 / NVFP4 探针 |
| `probe_train_graph.py` / `probe_train_graph_capture.py` | 训练步诊断与抓图收益 |
| `probe_capture_ops.py` / `probe_capture_stage.py` | 抓图阻塞点定位 |

> 结论速览：8-bit AdamW 省约 **35.7%** 优化器显存且无漂移；FP8 / NVFP4 专家量化可用（NVFP4 推理省 **45.6%**）；关梯度检查点 + 训练步 CUDA Graph 合计提速约 **3.4×**。

---

## `dashboard/` —— 对照实验数据看板

只读训练日志、**仅用 Python 标准库**的本地交互看板（SVG 直接绘制，可悬停 / 缩放 / 拖拽 / 开关曲线）。

```bash
python compare/dashboard/server.py 8777
# 浏览器打开 http://127.0.0.1:8777/
```

扫描根目录 = 看板目录的上一级；把含 `training_log.txt` 的 run 目录与它同级即可。详见 `dashboard/README.md`。

## `live_dashboard/` —— 实时训练看板

面向「训练进行中」的增量看板：增量解析 `training_log.txt`、SSE 推送、运行中的 run 带呼吸式光效，
适合直播 / 长时间挂机观察。启动：

```bash
python compare/live_dashboard/server.py 8778
```

详见 `live_dashboard/README.md`。