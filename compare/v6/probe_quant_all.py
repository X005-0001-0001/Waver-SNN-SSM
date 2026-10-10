# -*- coding: utf-8 -*-
"""MoE 专家量化全谱探针：None / 'bf16' / 'nvfp4' 三种推理量化，
比较参数量、显存、logits 相对误差与生成可用性。"""
import importlib.util
import os
import sys
import time

import torch

DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)

spec = importlib.util.spec_from_file_location("nvmod", os.path.join(DIR, "312006elite.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["nvmod"] = m
spec.loader.exec_module(m)

dev = m.DEVICE
tok = m.FastBPE(m.VOCAB_SIZE)
tok.load("tokenizer_incremental.bin")
print("MOE_ENABLED=%s offload(auto)=%s" % (m.MOE_ENABLED, m.decide_moe_offload(m.HIDDEN_SIZE)))

ids = torch.randint(1, tok.vocab_size, (1, 512), device=dev)


def build(quant):
    m.MOE_EXPERT_QUANT = quant
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    mm = m.WaveSNN(vocab_size=tok.vocab_size, hidden_size=m.HIDDEN_SIZE,
                   num_layers=m.NUM_LAYERS, pad_token_id=tok.pad_token_id)
    m.setup_model_token_ids(mm, tok)
    mm.to(dev).eval()
    return mm


def count_params(mm):
    return sum(p.numel() for p in mm.parameters())


def gpu_mem():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
ref = build(None)
mem_ref = gpu_mem()
with torch.no_grad():
    logits_ref = ref(ids, None)
p_ref = count_params(ref)
print("[None   ] params=%.2fM  gpu_alloc=%.1f MB  finite=%s" % (
    p_ref / 1e6, mem_ref, bool(torch.isfinite(logits_ref).all())))
del ref
torch.cuda.empty_cache()

for quant in ('bf16', 'nvfp4'):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mm = build(quant)
    mem = gpu_mem()
    with torch.no_grad():
        lg = mm(ids, None)
    p = count_params(mm)
    peak = torch.cuda.max_memory_allocated() / 1e6
    diff = (lg - logits_ref).abs()
    rel = (diff.norm() / logits_ref.norm()).item()
    try:
        with torch.no_grad():
            out = mm.generate(ids[:, :8], max_new_tokens=16)
        if torch.is_tensor(out):
            gen = "ok tensor%s finite=%s" % (tuple(out.shape), bool(torch.isfinite(out).all()))
        else:
            gen = "ok list len=%d head=%s" % (len(out), list(out[:8]))
    except Exception as e:
        gen = "FAIL %s: %s" % (type(e).__name__, e)
    print("[%-7s] params=%.2fM  gpu_alloc=%.1f MB  peak=%.1f MB  省%.1f%%  "
          "max_abs=%.4f rel_norm=%.4f  gen=%s" % (
              quant, p / 1e6, mem, peak, (mem_ref - mem) / mem_ref * 100,
              diff.max().item(), rel, gen))
    del mm
    torch.cuda.empty_cache()
print("done")