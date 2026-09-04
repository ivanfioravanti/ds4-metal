#!/usr/bin/env python3
"""Cross-check the kernel test's q2_K / iq2_xxs shadow values against gguf-py.

Run tests/test_qwen4_kernels with QWEN4_DUMP_QUANT=DIR, then:
    python check_quant_dump.py DIR
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/repo/llama.cpp/gguf-py")))
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

d = sys.argv[1]
ok = True
for name, qtype, block_bytes in (("q2_K", GGMLQuantizationType.Q2_K, 84), ("iq2_xxs", GGMLQuantizationType.IQ2_XXS, 66)):
    raw = np.fromfile(f"{d}/{name}.bin", dtype=np.uint8)
    shadow = np.fromfile(f"{d}/{name}.shadow", dtype=np.float64)
    blocks = raw.size // block_bytes
    ref = dequantize(raw.reshape(blocks, block_bytes), qtype).reshape(-1).astype(np.float64)
    diff = np.abs(ref - shadow)
    print(f"{name}: {shadow.size} values, max|d| {diff.max():.3e}, max|ref| {np.abs(ref).max():.3f}")
    ok &= diff.max() < 1e-5 * max(1.0, np.abs(ref).max())
print("ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
