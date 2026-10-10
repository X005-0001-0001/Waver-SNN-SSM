# -*- coding: utf-8 -*-
"""训练侧加速可行性探针：
1) 单个训练步（fwd+bwd+opt）的墙钟时间 vs CUDA kernel 总时间 → 判断是否 launch-bound；
2) kernel 启动次数；
3) 峰值显存（梯度检查点 on/off）。
用于评估「训练步 CUDA Graph 抓图」与「激活 offload」的收益空间。"""
import importlib.util
import os
import sys
import time

import torch

DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)
spec = importlib.util.spec_from_file_location("tmod", os.path.join(DIR, "312006elite.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["tmod"] = m
spec.loader.exec_module(m)

dev = m.DEVICE
tok = m.FastBPE(m.VOCAB_SIZE)
tok.load("tokenizer_incremental.bin")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
model = m.WaveSNN(vocab_size=tok.vocab_size, hidden_size=m.HIDDEN_SIZE,
                  num_layers=m.NUM_LAYERS, pad_token_id=tok.pad_token_id)
m.setup_model_token_ids(model, tok)
model.to(dev).train()

T = m.MAX_SEQ_LEN
ids = torch.randint(1, tok.vocab_size, (1, T), device=dev)
labels = ids.clone()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
amp = torch.amp.autocast('cuda', dtype=torch.bfloat16)


def step_once():
    opt.zero_grad(set_to_none=True)
    with amp:
        loss, _act, _la = model(ids, labels, position_offset=0)
    loss.backward()
    opt.step()
    return loss.item()


def bench(gc, n=15):
    m.USE_GRADIENT_CHECKPOINT = gc
    for _ in range(3):
        step_once()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n):
        step_once()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    peak = torch.cuda.max_memory_allocated() / 1e6
    return dt, peak


print("=== 训练步耗时/显存（H=%d L=%d T=%d B=1 bf16 AMP）===" % (m.HIDDEN_SIZE, m.NUM_LAYERS, T))
for gc in (True, False):
    dt, peak = bench(gc)
    print("grad_checkpoint=%-5s  单步 %.1f ms  峰值显存 %.1f MB" % (gc, dt * 1000, peak))

# kernel 剖析：统计 CUDA kernel 总时间与启动次数
m.USE_GRADIENT_CHECKPOINT = True
from torch.profiler import profile, ProfilerActivity
for _ in range(2):
    step_once()
torch.cuda.synchronize()
N = 5
t0 = time.time()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(N):
        step_once()
torch.cuda.synchronize()
wall = (time.time() - t0) / N
ev = prof.key_averages()
cuda_total_us = sum(e.self_device_time_total for e in ev)
nkernels = sum(e.count for e in ev)
print("=== kernel 剖析（每步）===")
print("墙钟 %.1f ms | CUDA 忙 %.1f ms | 启动次数 ~%d | 空闲占比 %.1f%%" % (
    wall * 1000, cuda_total_us / 1000 / N, nkernels // N,
    max(0.0, (1 - (cuda_total_us / 1e6) / N / wall)) * 100))
top = sorted(ev, key=lambda e: -e.self_device_time_total)[:8]
for e in top:
    print("   %-42s %8.3f ms  x%d" % (e.key[:42], e.self_device_time_total / 1000 / N, e.count // N))