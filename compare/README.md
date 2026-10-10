# 对照实验（精简版）

本目录**只保留最有代表性的对照结果与复现脚本**，不收录原始 run 日志、检查点与中间产物。

---

## `v4.1/` —— 与经典 Transformer 的受控对照

严格控制变量：同词表、同优化器、同超参数，参数量对齐（Transformer 9.48M vs Waver 9.29M），
数据集为自制中文名著语料（重切分后约 **15.36M tokens**）。

| 文件 | 内容 |
|---|---|
| `loss.png` | 训练损失（线性轴）|
| `loss_log.png` | 训练损失（对数轴）|
| `loss_loglog.png` | 训练损失（双对数轴）|
| `loss_diff.png` | 相对最优曲线的差值 |
| `ppl.png` / `ppl_log.png` | 验证集 PPL（线性 / 对数轴）|
| `activity.png` | 激活值监控 |
| `memory.png` | 显存占用 |
| `lr.png` | 学习率曲线 |
| `compare_viewer.html` | 交互式查看器（本地浏览器打开）|

---

## `dashboard/` —— 对照实验数据看板

只读训练日志、**仅用 Python 标准库**的本地交互看板（SVG 直接绘制，可悬停 / 缩放 / 拖拽 / 开关曲线）。

```bash
python compare/dashboard/server.py 8777
# 浏览器打开 http://127.0.0.1:8777/
```

扫描根目录 = 看板目录的上一级；把含 `training_log.txt` 的 run 目录与它同级即可。详见 `dashboard/README.md`。