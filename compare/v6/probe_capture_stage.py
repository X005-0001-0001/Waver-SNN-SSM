# -*- coding: utf-8 -*-
"""分阶段定位训练步 CUDA Graph 抓图失败点。
用法: python probe_capture_stage.py <fwd|fwdbwd|full> [gc]
每个阶段独立进程运行，避免失败后污染 CUDA context。"""
import importlib.util
import os
import sys

import torch

DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)
spec = importlib.util.spec_from_file_location("tmod", os.path.join(DIR, "312006elite.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["tmod"] = m
spec.loader.exec_module(m)

stage = sys.argv[1] if len(sys.argv) > 1 else 'fwd'
gc = (sys.argv[2] == '1') if len(sys.argv) > 2 else False
m.USE_GRADIENT_CHECKPOINT = gc

dev = m.DEVICE
tok = m.FastBPE(m.VOCAB_SIZE)
tok.load("tokenizer_incremental.bin")
T = m.MAX_SEQ_LEN
ids = torch.randint(1, tok.vocab_size, (1, T), device=dev)
labels = ids.clone()
amp = torch.amp.autocast('cuda', dtype=torch.bfloat16)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
model = m.WaveSNN(vocab_size=tok.vocab_size, hidden_size=m.HIDDEN_SIZE,
                  num_layers=m.NUM_LAYERS, pad_token_id=tok.pad_token_id)
m.setup_model_token_ids(model, tok)
model.to(dev).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)


def one():
    opt.zero_grad(set_to_none=True)
    with amp:
        loss, _a, _b = model(ids, labels, position_offset=0)
    if stage in ('fwdbwd', 'full'):
        loss.backward()
    if stage == 'full':
        opt.step()
    return loss


s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        one()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        one()
    torch.cuda.synchronize()
    print("STAGE=%s gc=%s => 抓图成功" % (stage, gc))
except Exception as e:
    print("STAGE=%s gc=%s => 抓图失败: %s: %s" % (stage, gc, type(e).__name__, str(e)[:200]))