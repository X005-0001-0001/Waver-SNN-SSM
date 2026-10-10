# -*- coding: utf-8 -*-
"""训练步 CUDA Graph 收益测量：
抓图范围 = forward + backward（optimizer.step 留在图外）。
对比 eager(fwd+bwd) 与 graph replay 的墙钟时间。gc on/off 各测一次。

前置条件（本目录 312006elite.py 已按此改造）：
  WaveSNN.forward 内 `avg_activity = (...).item()` 必须改为不触发 host 同步的
  `.detach()`（张量），否则 .item() 的 GPU→CPU 拷贝会让整段抓图
  cudaErrorStreamCaptureInvalidated 失败。optimizer.step() 不进图：
  AdamW 需 capturable=True 才可入图，bnb AdamW8bit 不支持，故留在图外。"""
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
print("MOE_ENABLED=%s" % m.MOE_ENABLED)

T = m.MAX_SEQ_LEN
ids = torch.randint(1, tok.vocab_size, (1, T), device=dev)
labels = ids.clone()
amp = torch.amp.autocast('cuda', dtype=torch.bfloat16)


def run(gc):
    m.USE_GRADIENT_CHECKPOINT = gc
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = m.WaveSNN(vocab_size=tok.vocab_size, hidden_size=m.HIDDEN_SIZE,
                      num_layers=m.NUM_LAYERS, pad_token_id=tok.pad_token_id)
    m.setup_model_token_ids(model, tok)
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = torch.zeros_like(p)

    def fwdbwd():
        with amp:
            loss, _a, _b = model(ids, labels, position_offset=0)
        loss.backward()

    def zero_grads():
        for p in params:
            p.grad.zero_()

    for _ in range(4):
        zero_grads(); fwdbwd()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        zero_grads(); fwdbwd()
    torch.cuda.synchronize()
    eager_ms = (time.time() - t0) / 10 * 1000

    graph_ms, status = float('nan'), "未抓图"
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                zero_grads(); fwdbwd()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fwdbwd()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        graph_ms = (time.time() - t0) / 10 * 1000
        status = "抓图成功"
    except Exception as e:
        status = "抓图失败: %s: %s" % (type(e).__name__, str(e)[:140])

    print("gc=%-5s  eager(fwd+bwd)=%.1f ms  graph=%.1f ms  加速=%.2fx  [%s]" % (
        gc, eager_ms, graph_ms, (eager_ms / graph_ms if graph_ms == graph_ms else 0.0), status))
    del model
    torch.cuda.empty_cache()


for gc in (False, True):
    run(gc)
print("done")