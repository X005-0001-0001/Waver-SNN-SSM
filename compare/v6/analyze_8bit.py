# -*- coding: utf-8 -*-
"""对照分析：AdamW(基线) vs AdamW8bit —— 提取每步 loss 与 MON 权重范数，
在匹配步数上比较偏差，判定 8-bit 优化器是否发散（类比 Adafactor/Lion 爆炸）。"""
import re
import sys

MON_RE = re.compile(
    r'\[MON L(\d+)\] Bn=([\d.]+) Sn=([\d.]+) Dn=([\d.]+) Mn=([\d.]+) An=([\d.]+) '
    r'nu=([\d.]+)/([\d.]+) th=([\d.]+)/([\d.]+) betan=([\d.]+) Cbn=([\d.]+) In=([\d.]+)')
STEP_RE = re.compile(r'Epoch 1/1 Step (\d+) Loss: ([\d.]+)')
# 关注指标：Bn,Sn,Dn,Mn,An,nu_max,In
KEYS = ['Bn', 'Sn', 'Dn', 'Mn', 'An', 'nu_max', 'In']


def parse(path):
    out = {}
    cur = None
    with open(path, encoding='utf-8') as f:
        for line in f:
            sm = STEP_RE.search(line)
            if sm:
                cur = int(sm.group(1))
                out.setdefault(cur, {'loss': float(sm.group(2)), 'mon': {}})
                continue
            mm = MON_RE.search(line)
            if mm and cur is not None:
                li = int(mm.group(1))
                out[cur]['mon'][li] = [float(mm.group(2)), float(mm.group(3)), float(mm.group(4)),
                                       float(mm.group(5)), float(mm.group(6)), float(mm.group(8)),
                                       float(mm.group(13))]
    return out


def layer_mean(mon, idx):
    vals = [v[idx] for v in mon.values()]
    return sum(vals) / len(vals) if vals else float('nan')


def main(base_path, test_path, label):
    base = parse(base_path)
    test = parse(test_path)
    common = sorted(set(base) & set(test))
    print("基线步数=%d  测试步数=%d  共同步数=%d" % (len(base), len(test), len(common)))
    if not common:
        print("无共同步数，无法比较"); return
    last = common[-1]
    print("\n--- 末步(step %d)对照 ---" % last)
    print("        %10s %10s %10s" % ("基线", label, "相对偏差"))
    print("  loss  %10.4f %10.4f %9.2e" % (base[last]['loss'], test[last]['loss'],
          abs(test[last]['loss'] - base[last]['loss']) / max(abs(base[last]['loss']), 1e-9)))
    worst = 0.0
    worst_k = ''
    for i, k in enumerate(KEYS):
        b = layer_mean(base[last]['mon'], i)
        t = layer_mean(test[last]['mon'], i)
        rel = abs(t - b) / max(abs(b), 1e-9)
        print("  %-5s %10.3f %10.3f %9.2e" % (k, b, t, rel))
        if rel > worst:
            worst, worst_k = rel, k
    # 全程最大相对偏差（逐指标、逐步）
    print("\n--- 全程最大相对偏差（逐指标，跨所有共同步）---")
    for i, k in enumerate(KEYS):
        mx = 0.0
        for s in common:
            b = layer_mean(base[s]['mon'], i)
            t = layer_mean(test[s]['mon'], i)
            if abs(b) > 1e-9:
                mx = max(mx, abs(t - b) / abs(b))
        flag = "  <== 爆炸?" if mx > 1.0 else ("  <== 明显漂移" if mx > 0.1 else "")
        print("  %-5s  max_rel=%.3e%s" % (k, mx, flag))
    # 单调性检查：末步 vs 首步（是否持续增长）
    first = common[0]
    print("\n--- 首末步比值（>3 视为漂移）---")
    for i, k in enumerate(KEYS):
        t0 = layer_mean(test[first]['mon'], i)
        t1 = layer_mean(test[last]['mon'], i)
        r = t1 / max(abs(t0), 1e-9)
        print("  %-5s  %.3f -> %.3f  (x%.2f)" % (k, t0, t1, r))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "测试")