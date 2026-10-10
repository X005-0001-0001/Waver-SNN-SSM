# -*- coding: utf-8 -*-
"""最小复现：测试单个算子在 CUDA Graph 抓图下是否安全。"""
import sys
import torch


def try_capture(name, fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        print("  %-28s => OK" % name)
    except Exception as e:
        print("  %-28s => FAIL %s: %s" % (name, type(e).__name__, str(e)[:120]))


z = torch.randn(4, 2, 2, dtype=torch.complex64, device='cuda')
r = torch.randn(4, 2, 2, device='cuda')
z2 = torch.randn(2, 2, dtype=torch.complex64, device='cuda')

print("CUDA Graph 抓图算子安全性测试：")
try_capture("complex matmul", lambda: z @ z)
try_capture("torch.linalg.matrix_exp(complex)", lambda: torch.linalg.matrix_exp(z))
try_capture("torch.linalg.matrix_exp(real)", lambda: torch.linalg.matrix_exp(r))
try_capture("torch.linalg.eig(complex)", lambda: torch.linalg.eig(z2))
try_capture("torch.linalg.inv(complex)", lambda: torch.linalg.inv(z2))
try_capture("torch.exp(complex)", lambda: torch.exp(z))
try_capture("torch.cumsum(complex)", lambda: torch.cumsum(z, dim=0))