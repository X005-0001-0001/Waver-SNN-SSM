# -*- coding: utf-8 -*-
"""nvfp4 专家量化可用性探针：构建同 seed 的「未量化 MoE」与「nvfp4 MoE」，
比较 logits 误差与显存占用，并验证前向可用（推理路径）。"""
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
ok = tok.load("tokenizer_incremental.bin")
print("tokenizer loaded=%s vocab=%d" % (ok, tok.vocab_size))
print("MOE_ENABLED=%s QUANT=%s offload(auto)=%s" % (
    m.MOE_ENABLED, m.MOE_EXPERT_QUANT, m.decide_moe_offload(m.HIDDEN_SIZE)))


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


ids = torch.randint(1, tok.vocab_size, (1, 512), device=dev)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
ref = build(None)
mem_ref = gpu_mem()
with torch.no_grad():
    logits_ref = ref(ids, None)
p_ref = count_params(ref)
print("[未量化 MoE] params=%.2fM  gpu_alloc=%.1f MB  logits=%s finite=%s" % (
    p_ref / 1e6, mem_ref, tuple(logits_ref.shape), bool(torch.isfinite(logits_ref).all())))
del ref
torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
t1 = time.time()
nv = build('nvfp4')
mem_nv = gpu_mem()
with torch.no_grad():
    logits_nv = nv(ids, None)
p_nv = count_params(nv)
peak_nv = torch.cuda.max_memory_allocated() / 1e6
print("[nvfp4 MoE]   params=%.2fM  gpu_alloc=%.1f MB  peak=%.1f MB  logits=%s finite=%s" % (
    p_nv / 1e6, mem_nv, peak_nv, tuple(logits_nv.shape), bool(torch.isfinite(logits_nv).all())))

# 误差（logits 形状 (B,T,V)）
diff = (logits_nv - logits_ref).abs()
rel = (diff.norm() / logits_ref.norm()).item()
print("[误差] max_abs=%.4f  mean_abs=%.4f  rel_norm=%.4f" % (
    diff.max().item(), diff.mean().item(), rel))
print("[显存] 未量化=%.1f MB  nvfp4=%.1f MB  省=%.1f MB (%.1f%%)" % (
    mem_ref, mem_nv, mem_ref - mem_nv, (mem_ref - mem_nv) / mem_ref * 100))

# 生成可用性（推理路径）
try:
    with torch.no_grad():
        out = nv.generate(ids[:, :8], max_new_tokens=16)
    print("[生成] 输出形状=%s finite=%s" % (tuple(out.shape) if torch.is_tensor(out) else type(out),
                                          bool(torch.isfinite(out).all()) if torch.is_tensor(out) else "n/a"))
except Exception as e:
    print("[生成] 失败: %s: %s" % (type(e).__name__, e))
print("done in %.1fs" % (time.time() - t0))