#!/usr/bin/env python3
"""Cross-check the HF oracle fixtures against the mlx-lm qwen4_exp port
(branch add-qwen4-exp), running the same token sequences through mlx-lm in
fp32, including the chunked/decode continuations.  Prints max|d| and top-1
agreement per case; exits non-zero above --tol.

Usage: python oracle_mlx.py --model ~/ds4-gguf/qwen4-mini/hf --fixtures ~/ds4-gguf/qwen4-oracle/hf
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import os

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model


def run(model, tokens: np.ndarray, splits=None) -> np.ndarray:
    if splits is None:
        return np.array(model(mx.array(tokens)[None])[0].astype(mx.float32))
    cache = model.make_cache()
    out, pos = [], 0
    for n in splits:
        out.append(np.array(model(mx.array(tokens[pos:pos + n])[None], cache=cache)[0].astype(mx.float32)))
        pos += n
    return np.concatenate(out, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    model, _ = load_model(Path(args.model), lazy=False)
    model.eval()
    worst = 0.0
    for path in sorted(glob.glob(os.path.join(args.fixtures, "*.npz"))):
        f = np.load(path)
        name = os.path.basename(path)[:-4]
        splits = f["splits"].tolist() if "splits" in f else None
        got = run(model, f["tokens"], splits)
        ref = f["logits"]
        d = np.abs(got - ref).max()
        agree = (got.argmax(-1) == ref.argmax(-1)).mean()
        worst = max(worst, d)
        print(f"{name:16s} L={len(f['tokens']):3d} max|d|={d:.3e} top1 agree={agree:.3f}")
    if worst > args.tol:
        raise SystemExit(f"mlx-lm vs HF mismatch {worst:.3e} > {args.tol}")
    print("ok")


if __name__ == "__main__":
    main()
