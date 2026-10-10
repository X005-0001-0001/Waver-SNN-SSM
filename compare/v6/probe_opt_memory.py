# -*- coding: utf-8 -*-
"""AdamW vs AdamW8bit 的 GPU 显存占用对照（同一模型、同样 step）。
报告参数显存、优化器状态显存、峰值显存。"""
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

dev = m.DEVICE
tok = m.FastBPE(m.VOCAB_SIZE)
tok.load("tokenizer_incremental.bin")
T = 512
ids = torch.randint(1, tok.vocab_size, (1, T), device=dev)
labels = ids.clone()
amp = torch.amp.autocast('cuda', dtype=torch.bfloat16)


def build():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    mm = m.WaveSNN(vocab_size=tok.vocab_size, hidden_size=m.HIDDEN_SIZE,
                   num_layers=m.NUM_LAYERS, pad_token_id=tok.pad_token_id)
    m.setup_model_token_ids(mm, tok)
    return mm.to(dev).train()


def measure(kind):
    model = build()
    pbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if kind == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3, weight_decay=0.01)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with amp:
            loss, _a, _b = model(ids, labels, position_offset=0)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    # 稳态显存（优化器状态已分配）
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    print("[%-9s] 参数=%.1f MB  稳态分配=%.1f MB  峰值=%.1f MB" % (
        kind, pbytes / 1e6, alloc, peak))
    del model, opt
    torch.cuda.empty_cache()
    return alloc


print("模型 H=%d L=%d T=%d (MoE=%s)" % (m.HIDDEN_SIZE, m.NUM_LAYERS, T, m.MOE_ENABLED))
a1 = measure('adamw')
a2 = measure('adamw8bit')
print("=> AdamW8bit 相比 AdamW 省显存 %.1f MB (%.1f%%)" % (a1 - a2, (a1 - a2) / a1 * 100))