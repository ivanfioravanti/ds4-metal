#!/usr/bin/env python3
"""Compare the DS4 qwen4_exp CPU reference (and, with --gpu, the Metal decode
graph) against the HF oracle fixtures.

For every fixture the token ids are fed through
  DS4_QWEN4_FT_TOKENS=<ids> DS4_QWEN4_FT_OUT=<f32 dump> ./ds4 -m GGUF --cpu --first-token-test --raw -p x
and the per-position logits are compared: max|d|, mean|d|, top-1 agreement.
pf_21 also checks the MTP draft logits (DS4_QWEN4_MTP_OUT) against mtp_full.

Usage: python parity_cpu.py --gguf ~/ds4-gguf/Qwen4-Mini-F32.gguf --fixtures ~/ds4-gguf/qwen4-oracle/hf [--tol 2e-3]
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np


def run_ds4(ds4: str, gguf: str, ids, out_path: str, extra_env: dict) -> str:
    env = dict(os.environ)
    env["DS4_QWEN4_FT_TOKENS"] = ",".join(str(int(t)) for t in ids)
    env["DS4_QWEN4_FT_OUT"] = out_path
    env.update(extra_env)
    cmd = [ds4, "-m", gguf] + ([] if extra_env.get("DS4_QWEN4_GPU") else ["--cpu"]) + ["--first-token-test", "--raw", "-p", "x"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"ds4 failed on {cmd}")
    return r.stdout


def compare(name: str, got: np.ndarray, ref: np.ndarray) -> float:
    d = np.abs(got - ref)
    agree = (got.argmax(-1) == ref.argmax(-1)).mean()
    print(f"{name:20s} L={ref.shape[0]:3d} max|d|={d.max():.3e} mean|d|={d.mean():.3e} top1={agree:.3f}")
    return float(d.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--gpu", action="store_true", help="also run the Metal graph (DS4_QWEN4_GPU=1)")
    ap.add_argument("--only", default=None, help="comma-separated fixture names")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    worst = 0.0
    tmp = tempfile.mkdtemp(prefix="qwen4-parity-")
    for path in sorted(glob.glob(os.path.join(args.fixtures, "*.npz"))):
        name = os.path.basename(path)[:-4]
        if only and name not in only:
            continue
        f = np.load(path)
        if "splits" in f:
            continue  # continuations are covered by the single-shot run of the same ids
        ids = f["tokens"]
        V = f["logits"].shape[1]
        out = os.path.join(tmp, name + ".f32")
        extra = {}
        mtp_out = None
        if name == "pf_21" and "mtp_full" in f:
            mtp_out = os.path.join(tmp, name + ".mtp.f32")
            extra["DS4_QWEN4_MTP_OUT"] = mtp_out
        if args.gpu:
            extra["DS4_QWEN4_GPU"] = "1"
        run_ds4(args.ds4, args.gguf, ids, out, extra)
        got = np.fromfile(out, dtype=np.float32).reshape(len(ids), V)
        worst = max(worst, compare(name, got, f["logits"]))
        if mtp_out and os.path.exists(mtp_out):
            mtp = np.fromfile(mtp_out, dtype=np.float32).reshape(len(ids) - 1, V)
            worst = max(worst, compare(name + " mtp(full)", mtp, f["mtp_full"]))
    if worst > args.tol:
        raise SystemExit(f"parity mismatch {worst:.3e} > {args.tol}")
    print("ok")


if __name__ == "__main__":
    main()
