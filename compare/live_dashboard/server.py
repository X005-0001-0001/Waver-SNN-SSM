# -*- coding: utf-8 -*-
"""
对比实验数据看板 · 直播版 —— 后端（仅用 Python 标准库，无需 pip 安装）

与 compare_dashboard/server.py 是同一套后端（API 完全一致，默认端口 8778 避免冲突），
配套的前端 index.html 是**为直播优化**的：大字号、粗曲线、自动聚焦轮换、
持续动画（时钟 / 跑马灯 / 呼吸光效 / 进度条）以防被平台判为挂机。

功能：
  * 扫描本目录（对照测试/）下所有含 training_log.txt 的 run
  * **增量解析**日志（记录字节偏移，只读新增部分）—— 支持 100MB 级日志与实时更新
  * 提取：每步 loss / ppl / activity / lr / rss、epoch 边界、验证 Loss/PPL、
          best_epochN_pplX.pt 给出的验证曲线
  * 从 run 的主脚本提取架构/训练配置；从 checkpoint 统计参数量（带缓存）
  * 提供 JSON API + SSE 事件流；前端 index.html 直接消费

启动：
    python server.py                # 默认 http://127.0.0.1:8778
    python server.py 9000           # 指定端口
    python server.py 9000 0         # 端口 9000，且不自动打开浏览器
"""
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE = os.path.dirname(os.path.abspath(__file__))
# 扫描根目录默认 = 本文件的**上一级**。所以把 compare_dashboard 整个文件夹复制到任何位置，
# 它就会扫描"新位置的上一级"；也可用第三个命令行参数显式指定。
DATA_ROOT = os.path.dirname(BASE)
HERE = BASE

PARAM_CACHE = os.path.join(BASE, "param_cache.json")
VIEWER = os.path.join(BASE, "index.html")

LOG_NAME = "training_log.txt"

# ------------------------------------------------------------------ 正则
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+Step\s+(\d+)\s+Loss:\s*(" + NUM + r")\s+PPL:\s*(" + NUM + r")\s+"
    r"Activity:\s*(" + NUM + r")\s+LR:\s*(" + NUM + r")"
    r"(?:\s+Samples:\s*\d+\s+\([\d.]+%\)\s+Memory:\s*(" + NUM + r")\s+MB)?"
)
VAL_RE = re.compile(r"Epoch\s+(\d+)\s+\S*\s*Loss:\s*(" + NUM + r")\s+PPL:\s*(" + NUM + r")")
BEST_RE = re.compile(r"best_epoch(\d+)_ppl([\d.]+)\.pt$")
# 菜单 6 打印的参数量（权威值，来自 sum(p.numel() for p in model.parameters())）
NPARAM_RE = re.compile(r"可训练参数:\s*(\d+)\s*/\s*总参数:\s*(\d+)")
NPARAM_MARK = "总参数".encode("utf-8")

CFG_KEYS = [
    "VOCAB_SIZE", "HIDDEN_SIZE", "NUM_LAYERS", "STATE_DIM", "MAX_SEQ_LEN",
    "BATCH_SIZE", "EPOCHS", "LEARNING_RATE", "WEIGHT_DECAY", "GRAD_CLIP_NORM",
    "GRADIENT_ACCUMULATION_STEPS", "SUMMARY_TYPE", "SUMMARY_ATTN_DK", "SUMMARY_CHUNK",
    "USE_LOOP", "LOOP_CONFIG", "LOOP_RESIDUAL_SCALE", "LOOP_INDEX_EMBED", "LOOP_MOE_MODE",
    "DISABLE_COUPLING", "COUPLING_IN_B_C", "COUPLING_B_APPLY", "COUPLING_C_APPLY",
    "USE_AMP", "USE_GRADIENT_CHECKPOINT", "LOSS_CHUNK_SIZE", "USE_RESIDUAL",
    "TIE_EMBEDDING_OUTPUT", "USE_CUDA_GRAPH_DECODE", "LOG_INTERVAL", "MOE_ENABLED",
    "EARLY_STOP_PATIENCE", "WINDOW_OVERLAP", "NUM_WORKERS",
    "TRANSFORMER_NUM_HEADS", "TRANSFORMER_FF_EXPAND", "TRANSFORMER_MAX_SEQ_LEN",
    "EMPTY_CACHE_INTERVAL", "A_DISC_CHUNK", "A_DISC_CHUNK_MIN_T", "P4_SEG_MAX_TOKENS",
]


def log(msg):
    try:
        sys.stdout.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        sys.stdout.flush()
    except Exception:
        pass


def safe_float(s, default=float("nan")):
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return v
        return v
    except Exception:
        return default


# ------------------------------------------------------------------ Run
class Run:
    """一个 run 的增量解析状态。"""

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.log_path = os.path.join(path, LOG_NAME)
        self.lock = threading.RLock()
        self.reset()
        self._cfg = None
        self._cfg_key = None
        self._params = None
        self._params_tried = False
        self._rate = deque(maxlen=40)      # (t, n_points) 用于估算速度
        self.error = None

    # -------- 状态重置
    def reset(self):
        self.offset = 0
        self.leftover = b""
        self.x = []          # 横轴：日志中第几个 step 记录（1,2,3,...）
        self.epoch = []
        self.step = []
        self.loss = []
        self.ppl = []
        self.act = []
        self.lr = []
        self.rss = []
        self.val = []        # [(epoch, val_loss, val_ppl)]
        self.ep_sum = {}     # epoch -> [loss_sum, n]
        self.ep_maxstep = {} # epoch -> 该 epoch 见过的最大 step（增量维护，避免每次快照重扫）
        self.nparams_log = None        # 菜单6 打印的"总参数"（权威）
        self.nparams_trainable = None
        self.ver = 0
        self.size = 0
        self.mtime = 0.0
        self.first_t = None
        self.last_t = None
        self.truncated = 0

    # -------- 增量解析
    def poll(self):
        try:
            st = os.stat(self.log_path)
        except OSError:
            return False
        size, mtime = st.st_size, st.st_mtime
        with self.lock:
            if size < self.offset:                  # 日志被截断/重建
                self.reset()
                self.truncated += 1
            self.mtime = mtime
            if size == self.offset:
                self.size = size
                return False
            try:
                with open(self.log_path, "rb") as f:
                    f.seek(self.offset)
                    chunk = f.read()
            except OSError:
                return False
            data = self.leftover + chunk
            nl = data.rfind(b"\n")
            if nl < 0:
                self.leftover = data[-65536:]
                self.offset = self.offset + len(chunk) - len(self.leftover)
                self.size = size
                return False
            block, self.leftover = data[:nl], data[nl + 1:]
            self.offset = self.offset + len(chunk) - len(self.leftover)
            self.size = size
            n0 = len(self.x)
            for line in block.split(b"\n"):
                if b"Loss:" in line:
                    self._parse_line(line)
                elif NPARAM_MARK in line:            # 菜单6 的"可训练参数/总参数"行（权威参数量）
                    m = NPARAM_RE.search(line.decode("utf-8", "replace"))
                    if m:
                        self.nparams_trainable = int(m.group(1))
                        self.nparams_log = int(m.group(2))
            added = len(self.x) - n0
            if added:
                self.ver += 1
                self._rate.append((time.time(), len(self.x)))
                if self.first_t is None:
                    self.first_t = time.time()
                self.last_t = time.time()
            return added > 0

    def _parse_line(self, raw):
        s = raw.decode("utf-8", "replace")
        m = STEP_RE.search(s)
        if m:
            self.x.append(len(self.x) + 1)
            self.epoch.append(int(m.group(1)))
            self.step.append(int(m.group(3)))
            self.loss.append(safe_float(m.group(4)))
            self.ppl.append(safe_float(m.group(5)))
            self.act.append(safe_float(m.group(6)))
            self.lr.append(safe_float(m.group(7)))
            self.rss.append(safe_float(m.group(8)) if m.group(8) is not None else float("nan"))
            e = int(m.group(1))
            a = self.ep_sum.setdefault(e, [0.0, 0])
            a[0] += self.loss[-1]
            a[1] += 1
            st = self.step[-1]
            if st > self.ep_maxstep.get(e, 0):
                self.ep_maxstep[e] = st
            return
        m = VAL_RE.search(s)
        if m:
            self.val.append((int(m.group(1)), safe_float(m.group(2)), safe_float(m.group(3))))

    # -------- 配置 / 参数
    def config(self):
        if self._cfg is not None:
            return self._cfg
        cands = [f for f in os.listdir(self.path)
                 if f.endswith(".py") and ("elite" in f or "transformer" in f)]
        cands = [f for f in cands if not f.startswith("_") and f not in ("plot_training.py",)]
        # 优先 transformer.py / 带 elite 的
        cands.sort(key=lambda f: (0 if "elite" in f else 1, f))
        cfg = {}
        for f in cands:
            p = os.path.join(self.path, f)
            try:
                src = open(p, encoding="utf-8-sig").read()
            except Exception:
                continue
            got = {}
            for k in CFG_KEYS:
                mm = re.search(r"^%s\s*=\s*([^\n#]+)" % k, src, re.M)
                if mm:
                    got[k] = mm.group(1).strip().strip('"').strip("'")[:48]
            if got:
                cfg = got
                cfg["_script"] = f
                break
        self._cfg = cfg
        return cfg

    def params(self):
        """参数量。优先级：
             1) 训练日志里菜单6 打印的"总参数"——权威值，等于 sum(p.numel() for p in model.parameters())
             2) checkpoint 的 state_dict **按张量去重**后求和（近似）
        为什么必须去重：TIE_EMBEDDING_OUTPUT=True 时 output_fc.weight 与 embedding.weight
        共享同一块存储，state_dict 里是两个键指向同一个张量，按键求和会把 embedding 数两遍
        （实测 transformer 会虚高 2,576,000、waver 虚高 1,792,000）。
        返回 (参数量, 来源) —— 来源 'log' 精确 / 'ckpt' 近似（可能含少量 buffer，如 rope.inv_freq）。
        """
        if self.nparams_log:
            return self.nparams_log, "log"
        if self._params is not None or self._params_tried:
            return self._params, "ckpt"
        self._params_tried = True
        cache = load_json(PARAM_CACHE, {})
        cks = [f for f in os.listdir(self.path)
               if f.startswith("best_epoch") and f.endswith(".pt")]
        if not cks:
            cks = [f for f in os.listdir(self.path) if f.endswith(".pt")]
        if not cks:
            return None, None

        def epnum(f):
            m = re.search(r"epoch(\d+)", f)
            return int(m.group(1)) if m else 0
        cks.sort(key=epnum)
        ck = os.path.join(self.path, cks[0])
        try:
            st = os.stat(ck)
            key = "%s|%s|%d" % (self.name, os.path.basename(ck), st.st_size)
        except OSError:
            return None, None
        if key in cache:
            self._params = cache[key]
            return self._params, "ckpt"
        try:
            import torch
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            seen = set()
            n = 0
            for v in sd.values():
                if not torch.is_tensor(v):
                    continue
                k = (v.data_ptr(), v.numel())
                if k in seen:          # 共享存储（权重绑定）只算一次
                    continue
                seen.add(k)
                n += v.numel()
            self._params = int(n)
            cache[key] = int(n)
            save_json(PARAM_CACHE, cache)
            del sd
        except Exception as e:
            log("参数统计失败 %s: %s" % (self.name, str(e)[:80]))
            self._params = None
        return self._params, "ckpt"

    def best_from_filenames(self):
        # 目录 mtime 变化时才重新扫描（避免每次快照都 listdir）
        try:
            st = os.stat(self.path)
            key = (st.st_mtime_ns,)
        except OSError:
            return []
        if getattr(self, "_bf_key", None) == key:
            return self._bf
        out = []
        try:
            with os.scandir(self.path) as it:
                for de in it:
                    m = BEST_RE.search(de.name)
                    if m:
                        out.append((int(m.group(1)), float(m.group(2))))
        except OSError:
            pass
        out.sort()
        self._bf, self._bf_key = out, key
        return out

    def speed(self):
        """近 2 分钟的平均 记录/秒（用 x 长度变化估算）。"""
        with self.lock:
            if len(self._rate) < 2:
                return None
            t1, n1 = self._rate[-1]
            t0, n0 = None, None
            for t, n in self._rate:
                if t1 - t <= 120:
                    t0, n0 = t, n
                    break
            if t0 is None or t1 <= t0:
                return None
            return (n1 - n0) / (t1 - t0)

    def snapshot(self):
        with self.lock:
            n = len(self.x)
            cfg = self.config()
            pnum, psrc = self.params()
            epochs = int(safe_float(cfg.get("EPOCHS"), 0) or 0)
            cur_ep = self.epoch[-1] if n else 0
            cur_step = self.step[-1] if n else 0
            # 每 epoch 的步数：取已完成的 epoch 中最大的（O(1)，增量维护）
            spe = None
            for e in sorted(self.ep_maxstep):
                if e < cur_ep:
                    spe = self.ep_maxstep[e]
                    break
            if spe is None and self.ep_maxstep:
                spe = max(self.ep_maxstep.values())
            spd = self.speed()
            eta = None
            if spe and epochs and cur_ep and spd:
                remain = (epochs - cur_ep) * spe + (spe - cur_step)
                eta = remain / spd if spd > 0 else None
            best_val = None
            if self.val:
                best_val = min(self.val, key=lambda r: r[2])
            bf = self.best_from_filenames()
            bf_best = min(bf, key=lambda r: r[1]) if bf else None
            running = (time.time() - self.mtime) < 120 if self.mtime else False
            return dict(
                name=self.name,
                status="running" if running else "stopped",
                log_size=self.size, mtime=self.mtime,
                points=n, ver=self.ver, truncated=self.truncated,
                epochs_seen=len(self.ep_sum), epochs_total=epochs or None,
                cur_epoch=cur_ep, cur_step=cur_step, steps_per_epoch=spe,
                last_loss=self.loss[-1] if n else None,
                last_ppl=self.ppl[-1] if n else None,
                last_rss=self.rss[-1] if n else None,
                val=self.val,
                best_val_ppl=(best_val[2] if best_val else (bf_best[1] if bf_best else None)),
                best_val_epoch=(best_val[0] if best_val else (bf_best[0] if bf_best else None)),
                best_from_names=bf,
                epoch_train={str(e): (v[0] / v[1] if v[1] else None) for e, v in self.ep_sum.items()},
                speed=spd, eta=eta,
                config=cfg, params=pnum, params_src=psrc,
                params_trainable=self.nparams_trainable,
            )


# ------------------------------------------------------------------ 全局状态
class Hub:
    def __init__(self):
        self.lock = threading.RLock()
        self.runs = {}
        self.version = 0
        self.cv = threading.Condition()
        self.last_scan = 0.0
        self.scan_count = 0

    def discover(self):
        found = {}
        try:
            entries = sorted(os.listdir(DATA_ROOT))
        except OSError:
            entries = []
        for name in entries:
            p = os.path.join(DATA_ROOT, name)
            if not os.path.isdir(p) or name.startswith("."):
                continue
            if name == os.path.basename(BASE):
                continue
            if os.path.exists(os.path.join(p, LOG_NAME)):
                found[name] = p
        with self.lock:
            for name, p in found.items():
                if name not in self.runs:
                    self.runs[name] = Run(name, p)
                    log("发现 run: %s" % name)
            for name in list(self.runs):
                if name not in found:
                    del self.runs[name]
                    log("移除 run: %s" % name)
        return found

    def scan(self):
        self.discover()
        with self.lock:
            runs = list(self.runs.values())
        changed = False
        for r in runs:
            try:
                if r.poll():
                    changed = True
            except Exception:
                r.error = traceback.format_exc()[-300:]
        self.last_scan = time.time()
        self.scan_count += 1
        if changed:
            with self.cv:
                self.version += 1
                self.cv.notify_all()

    def loop(self, interval=2.0):
        while True:
            try:
                self.scan()
            except Exception:
                log("scan 异常: %s" % traceback.format_exc()[-200:])
            time.sleep(interval)


HUB = Hub()


# ------------------------------------------------------------------ 序列生成
def ema(vals, w):
    """指数移动平均。

    起点用**前 k 个点的均值**初始化（k≈等效窗口），而不是 vals[0]：
    否则重平滑（窗口上千步）时，曲线首段会被第一个噪声样本拖住，呈现明显的"起始滞后"。
    """
    n = len(vals)
    if w <= 0 or n < 2:
        return vals
    k = min(n, max(1, int(round(1.0 / max(1e-9, 1.0 - w)))))
    head = [v for v in vals[:k] if v == v]
    last = (sum(head) / len(head)) if head else vals[0]
    out = [0.0] * n
    for i, v in enumerate(vals):
        if v != v:            # nan
            out[i] = last
            continue
        last = w * last + (1.0 - w) * v
        out[i] = last
    return out


def downsample(xs, ys, maxp):
    n = len(xs)
    if n <= maxp or maxp <= 0:
        return xs, ys
    bucket = max(1, math.ceil(n / maxp))
    ox, oy = [], []
    for i in range(0, n, bucket):
        j = min(i + bucket, n)
        seg = ys[i:j]
        if not seg:
            continue
        seg_x = xs[i:j]
        try:
            imin = min(range(len(seg)), key=lambda k: (seg[k] if seg[k] == seg[k] else float("inf")))
            imax = max(range(len(seg)), key=lambda k: (seg[k] if seg[k] == seg[k] else float("-inf")))
        except ValueError:
            continue
        for k in sorted({imin, imax}):
            ox.append(seg_x[k]); oy.append(seg[k])
    return ox, oy


METRIC_FIELD = {"loss": "loss", "ppl": "ppl", "lr": "lr", "act": "act", "rss": "rss"}


def series_for(name, metric, maxp, smooth_w, xaxis):
    r = HUB.runs.get(name)
    if r is None:
        return None
    with r.lock:
        if metric in ("val_ppl", "val_loss"):
            idx = 2 if metric == "val_ppl" else 1
            pts = [(float(e), r.val[i][idx]) for i, (e, _b, _c) in enumerate(r.val)]
            pts = [(x, y) for x, y in pts if y == y]
            pts.sort(key=lambda p: p[0])
            return {"x": [p[0] for p in pts], "y": [p[1] for p in pts], "marks": []}
        f = METRIC_FIELD.get(metric)
        if f is None:
            return None
        ys = list(getattr(r, f))
        if xaxis == "step":
            xs = list(r.step)
        elif xaxis == "epoch":
            xs = list(r.epoch)
        else:
            xs = list(r.x)
        marks = []
        prev = None
        for xi, e in zip(r.x, r.epoch):
            if e != prev:
                marks.append([xi, e])
                prev = e
    ys = ema(ys, smooth_w)
    xs, ys = downsample(xs, ys, maxp)
    return {"x": xs, "y": ys, "marks": marks}


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "compare-dashboard"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, allow_nan=True))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path

        if path in ("/", "/index.html", "/dashboard"):
            if not os.path.exists(VIEWER):
                return self._send(500, "index.html 缺失", "text/plain; charset=utf-8")
            return self._send(200, open(VIEWER, "rb").read(), "text/html; charset=utf-8")

        if path == "/api/meta":
            return self._json(dict(root=DATA_ROOT, scan_interval=2.0,
                                   version=HUB.version, last_scan=HUB.last_scan,
                                   n_runs=len(HUB.runs)))

        if path == "/api/runs":
            with HUB.lock:
                runs = list(HUB.runs.values())
            snaps = []
            for r in runs:
                try:
                    snaps.append(r.snapshot())
                except Exception as e:
                    snaps.append(dict(name=r.name, error=str(e)[:200]))
            snaps.sort(key=lambda s: s.get("name", ""))
            return self._json(dict(version=HUB.version, runs=snaps))

        if path == "/api/series":
            names = []
            for k in ("runs", "run"):
                for v in q.get(k, []):
                    names += [x for x in v.split(",") if x]
            if not names:
                names = sorted(HUB.runs.keys())
            metric = (q.get("metric", ["loss"])[0])
            maxp = int(safe_float(q.get("points", ["1400"])[0], 1400))
            sw = float(safe_float(q.get("smooth", ["0"])[0], 0.0))
            xaxis = q.get("xaxis", ["seq"])[0]
            out = {}
            for n in names:
                s = series_for(n, metric, maxp, sw, xaxis)
                if s:
                    out[n] = s
            return self._json(dict(version=HUB.version, metric=metric,
                                   xaxis=xaxis, smooth=sw, points=maxp, series=out))

        if path == "/api/epochs":
            names = []
            for v in q.get("runs", []):
                names += [x for x in v.split(",") if x]
            if not names:
                names = sorted(HUB.runs.keys())
            out = {}
            for n in names:
                r = HUB.runs.get(n)
                if r is None:
                    continue
                with r.lock:
                    tr = {e: (v[0] / v[1] if v[1] else None) for e, v in r.ep_sum.items()}
                    vl = {e: (b, c) for e, b, c in r.val}
                    bf = dict(r.best_from_filenames())
                eps = sorted(set(tr) | set(vl) | set(bf))
                out[n] = [dict(epoch=e,
                               train_loss=tr.get(e),
                               val_loss=(vl.get(e) or (None, None))[0],
                               val_ppl=(vl.get(e) or (None, None))[1],
                               val_ppl_file=bf.get(e)) for e in eps]
            return self._json(dict(version=HUB.version, epochs=out))

        if path == "/api/run":
            n = q.get("name", [""])[0]
            r = HUB.runs.get(n)
            if r is None:
                return self._json({"error": "no such run"}, 404)
            cfg = r.config()
            with r.lock:
                tail = []
                k = max(0, len(r.x) - int(safe_float(q.get("tail", ["60"])[0], 60)))
                for i in range(k, len(r.x)):
                    tail.append(dict(seq=r.x[i], epoch=r.epoch[i], step=r.step[i],
                                     loss=r.loss[i], ppl=r.ppl[i], act=r.act[i],
                                     lr=r.lr[i], rss=r.rss[i]))
                val = list(r.val)
            return self._json(dict(name=n, config=cfg, params=r.params(),
                                   tail=tail, val=val, error=r.error))

        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                last = -1
                t0 = time.time()
                while True:
                    with HUB.cv:
                        HUB.cv.wait(timeout=5.0)
                        v = HUB.version
                    if v != last or time.time() - t0 > 15:
                        last = v
                        t0 = time.time()
                        with HUB.lock:
                            runs = list(HUB.runs.values())
                        payload = json.dumps(dict(
                            version=v, last_scan=HUB.last_scan,
                            sizes={r.name: r.size for r in runs}))
                        self.wfile.write(("data: %s\n\n" % payload).encode("utf-8"))
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                pass
            return

        return self._send(404, "not found", "text/plain; charset=utf-8")


def load_json(p, d):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return d


def save_json(p, o):
    try:
        json.dump(o, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception:
        pass


def main():
    global DATA_ROOT
    port = 8778
    auto_open = True
    args = [a for a in sys.argv[1:]]
    if args:
        try:
            port = int(args[0])
        except ValueError:
            pass
    if len(args) >= 2:
        auto_open = args[1] not in ("0", "no", "false")
    if len(args) >= 3:
        cand = os.path.abspath(args[2])
        if os.path.isdir(cand):
            DATA_ROOT = cand

    log("看板目录   : %s" % BASE)
    log("扫描根目录 : %s" % DATA_ROOT)
    log("（默认取看板目录的上一级；复制整个 compare_dashboard 文件夹到任意位置即可复用，")
    log("  也可用第三个参数指定：python server.py 8778 1 \"D:\\某处\\对照实验\"）")
    log("首次扫描中（大日志可能需要几秒）...")
    t0 = time.time()
    HUB.discover()
    with HUB.lock:
        n = len(HUB.runs)
    if n == 0:
        log("** 未发现任何 run **（子目录里需要有 training_log.txt）")
        log("如果数据不在 %s，请用第三个参数指定扫描目录。" % DATA_ROOT)
    log("发现 %d 个 run，开始增量解析" % n)
    HUB.scan()
    log("首轮解析完成，用时 %.1fs" % (time.time() - t0))

    t = threading.Thread(target=HUB.loop, kwargs=dict(interval=2.0), daemon=True)
    t.start()

    def _warm_params():
        time.sleep(1.0)
        with HUB.lock:
            runs = list(HUB.runs.values())
        for r in runs:
            try:
                p, src = r.params()
                log("参数量 %-44s %-12s (%s)"
                    % (r.name, "{:,}".format(p) if p else "n/a",
                       "训练日志·菜单6 权威值" if src == "log" else "checkpoint 去重统计(近似)"))
            except Exception:
                pass
        with HUB.cv:
            HUB.version += 1
            HUB.cv.notify_all()
    threading.Thread(target=_warm_params, daemon=True).start()

    url = "http://127.0.0.1:%d/" % port
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    log("看板已启动: %s" % url)
    log("按 Ctrl+C 停止")
    if auto_open:
        def _open():
            time.sleep(0.8)
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("已停止")
    finally:
        try:
            srv.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
