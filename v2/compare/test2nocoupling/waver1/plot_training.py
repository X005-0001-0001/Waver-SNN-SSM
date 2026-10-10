# -*- coding: utf-8 -*-
"""从 training_log.txt 解析并绘制训练曲线。

覆盖日志中的全部指标：
  - step 标量：Loss / PPL / Activity / LR / Memory
  - [Layer] 行：I_std / amp_std / wave_raw_mean / wave_raw_std / delta_mean / delta_min / delta_max
  - [MON]   行：cs/Bn/Sn/Dn/Mn/An/nu(min,max)/th(min,max)/betan/Cbn/In

x 轴为跨 epoch 连续的全局步号（每遇到一个 step 行 +1），
因此多 epoch 时曲线连成一条，不会每个 epoch 回到 0 重画。
数值统一支持负号与科学计数法（如 -0.0625、1e-5）。
"""
import re
import sys
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = "training_log.txt"
OUTPUT_DIR = "plots"

# 统一数值：负号 + 小数 + 科学计数法
NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

STEP_RE = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s+Step\s+(\d+)\s+Loss:\s*(' + NUM + r')\s+PPL:\s*(' + NUM + r')\s+'
    r'Activity:\s*(' + NUM + r')\s+LR:\s*(' + NUM + r')\s+Samples:\s*\d+\s+\([\d.]+%\)\s+'
    r'Memory:\s*(' + NUM + r')\s+MB'
)

LAYER_RE = re.compile(
    r'\[Layer\s+(L\d+)\]\s+I_std=(' + NUM + r')\s+amp_std=(' + NUM + r')\s+'
    r'wave_raw_mean=(' + NUM + r')\s+wave_raw_std=(' + NUM + r')'
    r'(?:\s+delta_mean=(' + NUM + r')\s+delta_min=(' + NUM + r')\s+delta_max=(' + NUM + r'))?'
)

MON_RE = re.compile(
    r'\[MON\s+(L\d+)\]\s+Bn=(' + NUM + r')\s+Sn=(' + NUM + r')\s+'
    r'Dn=(' + NUM + r')\s+Mn=(' + NUM + r')\s+An=(' + NUM + r')\s+'
    r'nu=(' + NUM + r')/(' + NUM + r')\s+th=(' + NUM + r')/(' + NUM + r')\s+'
    r'betan=(' + NUM + r')\s+Cbn=(' + NUM + r')\s+In=(' + NUM + r')'
)

MON_NAMES = ['Bn', 'Sn', 'Dn', 'Mn', 'An', 'nu_min', 'nu_max',
             'th_min', 'th_max', 'betan', 'Cbn', 'In']
LAYER_NAMES = ['I_std', 'amp_std', 'wave_raw_mean', 'wave_raw_std', 'delta_mean', 'delta_min', 'delta_max']


def smooth(values, weight=0.9):
    if len(values) < 2:
        return values
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def parse_log(filepath):
    steps = []
    epochs = []
    step_in_epoch = []
    loss = []
    ppl = []
    activity = []
    lr = []
    memory = []

    # layer -> param -> [(global_x, value), ...]
    layer_series = defaultdict(lambda: defaultdict(list))
    mon_series = defaultdict(lambda: defaultdict(list))
    epoch_bounds = []  # [(epoch, global_x_start), ...]

    global_x = 0
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                global_x += 1
                ep = int(m.group(1))
                steps.append(global_x)
                epochs.append(ep)
                step_in_epoch.append(int(m.group(3)))
                loss.append(float(m.group(4)))
                ppl.append(float(m.group(5)))
                activity.append(float(m.group(6)))
                lr.append(float(m.group(7)))
                memory.append(float(m.group(8)))
                if not epoch_bounds or epoch_bounds[-1][0] != ep:
                    epoch_bounds.append((ep, global_x))
                continue

            m = LAYER_RE.search(line)
            if m:
                layer = m.group(1)
                vals = [float(x) for x in m.groups()[1:] if x is not None]  # 兼容旧日志（无 delta 字段）
                for key, v in zip(LAYER_NAMES, vals):
                    layer_series[layer][key].append((global_x, v))
                continue

            m = MON_RE.search(line)
            if m:
                layer = m.group(1)
                vals = [float(x) for x in m.groups()[1:]]
                for key, v in zip(MON_NAMES, vals):
                    mon_series[layer][key].append((global_x, v))
                continue

    return dict(steps=steps, epochs=epochs, loss=loss, ppl=ppl,
                activity=activity, lr=lr, memory=memory,
                layer_series=layer_series, mon_series=mon_series,
                epoch_bounds=epoch_bounds)


def _add_epoch_lines(ax, epoch_bounds):
    for ep, x0 in epoch_bounds[1:]:
        ax.axvline(x0 - 0.5, color='0.6', ls='--', lw=0.8, alpha=0.6)


def plot_single(x, y, title, xlabel, ylabel, filepath, epoch_bounds,
                color='blue', yscale='linear'):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, alpha=0.35, color=color, lw=0.8, label='raw')
    ax.plot(x, smooth(y), color=color, lw=2, label='smoothed (0.9)')
    if yscale == 'log' and min(y) > 0:
        ax.set_yscale('log')
    _add_epoch_lines(ax, epoch_bounds)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)


def plot_group(series, param_names, title, filepath, epoch_bounds, ncols=3):
    if not series:
        return
    layers = sorted(series.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(layers))))
    n = len(param_names)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, param in enumerate(param_names):
        ax = axes[i]
        for li, layer in enumerate(layers):
            pts = series[layer].get(param, [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, alpha=0.25, color=colors[li], lw=0.6)
            ax.plot(xs, smooth(ys), color=colors[li], lw=1.6)
        _add_epoch_lines(ax, epoch_bounds)
        ax.set_title(param, fontsize=11)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    handles = [plt.Line2D([], [], color=colors[li], lw=2, label=layer)
               for li, layer in enumerate(layers)]
    fig.legend(handles=handles, loc='upper right', fontsize=8, ncol=len(layers))
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(filepath, dpi=120)
    plt.close(fig)


def plot_overview(steps, loss, ppl, activity, lr, memory, epoch_bounds, filepath):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    items = [
        (axes[0, 0], loss, 'Loss', 'blue', 'linear'),
        (axes[0, 1], ppl, 'PPL', 'red', 'linear'),
        (axes[1, 0], activity, 'Activity', 'green', 'linear'),
        (axes[1, 1], lr, 'Learning Rate', 'orange', 'log'),
        (axes[2, 0], memory, 'Memory (MB)', 'purple', 'linear'),
    ]
    for ax, data, label, color, yscale in items:
        ax.plot(steps, data, alpha=0.35, color=color, lw=0.8)
        ax.plot(steps, smooth(data), color=color, lw=2, label='smoothed')
        if yscale == 'log' and min(data) > 0:
            ax.set_yscale('log')
        _add_epoch_lines(ax, epoch_bounds)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Step')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[2, 1].axis('off')
    fig.suptitle('Training Overview', fontsize=16)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        log_path = os.path.abspath(sys.argv[1])
    else:
        log_path = os.path.join(script_dir, LOG_FILE)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)

    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found")
        return
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading log: {log_path}")
    d = parse_log(log_path)
    steps = d['steps']
    if not steps:
        print("No training steps found in log (parse failed)")
        return

    epochs = sorted(set(d['epochs']))
    layers = sorted(set(d['layer_series']) | set(d['mon_series']))
    print(f"Steps={len(steps)}  Epochs={epochs}  Layers={layers}")
    print(f"Global x range: {min(steps)} .. {max(steps)}")
    for ep, x0 in d['epoch_bounds']:
        print(f"  Epoch {ep} starts at global step {x0}")
    print(f"Saving plots to: {output_dir}")

    eb = d['epoch_bounds']

    plot_single(steps, d['loss'], 'Training Loss', 'Step', 'Loss',
                os.path.join(output_dir, 'loss.png'), eb, 'blue')
    print("  loss.png")
    plot_single(steps, d['ppl'], 'Perplexity', 'Step', 'PPL',
                os.path.join(output_dir, 'ppl.png'), eb, 'red')
    print("  ppl.png")
    plot_single(steps, d['activity'], 'Activity', 'Step', 'Activity',
                os.path.join(output_dir, 'activity.png'), eb, 'green')
    print("  activity.png")
    plot_single(steps, d['lr'], 'Learning Rate', 'Step', 'LR',
                os.path.join(output_dir, 'lr.png'), eb, 'orange', yscale='log')
    print("  lr.png")
    plot_single(steps, d['memory'], 'Memory (MB)', 'Step', 'MB',
                os.path.join(output_dir, 'memory.png'), eb, 'purple')
    print("  memory.png")

    plot_overview(steps, d['loss'], d['ppl'], d['activity'], d['lr'],
                  d['memory'], eb, os.path.join(output_dir, 'overview.png'))
    print("  overview.png")

    plot_group(d['layer_series'], LAYER_NAMES, 'Layer Statistics (per layer)',
               os.path.join(output_dir, 'layer_stats.png'), eb, ncols=2)
    print("  layer_stats.png")

    plot_group(d['mon_series'], MON_NAMES, 'MON Parameters (per layer)',
               os.path.join(output_dir, 'mon_params.png'), eb, ncols=4)
    print("  mon_params.png")

    print(f"\nDone! {len(os.listdir(output_dir))} plots saved to {output_dir}")


if __name__ == "__main__":
    main()