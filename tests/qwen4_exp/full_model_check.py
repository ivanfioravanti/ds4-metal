#!/usr/bin/env python3
"""Full-model accuracy check for Qwen3.8-Flash-Next in DS4.

Teacher-forces a fixed text through the DS4 Metal graph (token by token,
DS4_QWEN4_GPU_ONLY) and, unless --skip-mlx, through the mlx-lm 8-bit model
as an independent reference.  Reports top-1 agreement, |dlogit|, KL and
teacher-forced NLL, plus the MTP draft acceptance rate (greedy-match against
DS4's own next-position argmax and against the true text) for the full-width
variant.  Heavy: loads the full model.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

BUILTIN_TEXT = """The Apollo program was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. It was first conceived in 1960 during the Eisenhower administration as a three-person spacecraft to follow the one-person Project Mercury, which put the first Americans in space. Apollo was later dedicated to President John F. Kennedy's national goal for the 1960s of "landing a man on the Moon and returning him safely to the Earth" in a 1961 address to Congress.

def fibonacci(n):
    \"\"\"Return the n-th Fibonacci number using an iterative loop.\"\"\"
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

if __name__ == "__main__":
    for i in range(10):
        print(i, fibonacci(i))

Water is an inorganic compound with the chemical formula H2O. It is a transparent, tasteless, odorless, and nearly colorless chemical substance, and it is the main constituent of Earth's hydrosphere and the fluids of all known living organisms, in which it acts as a solvent. It is vital for all known forms of life, despite not providing food energy or organic micronutrients.
"""


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def run_ds4(ds4, gguf, ids, out_dir, tag, extra_env):
    env = dict(os.environ)
    env.update({
        "DS4_QWEN4_GPU": "1", "DS4_QWEN4_GPU_ONLY": "1",
        "DS4_QWEN4_FT_TOKENS": ",".join(map(str, ids)),
        "DS4_QWEN4_FT_OUT": os.path.join(out_dir, f"ds4_{tag}.bin"),
        "DS4_QWEN4_MTP_OUT": os.path.join(out_dir, f"mtp_{tag}.bin"),
    })
    env.update(extra_env)
    t0 = time.time()
    r = subprocess.run([ds4, "-m", gguf, "--first-token-test", "--raw", "-p", "x"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit("ds4 failed")
    print(f"  ds4 {tag}: {time.time() - t0:.1f}s")
    n = len(ids)
    logits = np.fromfile(env["DS4_QWEN4_FT_OUT"], dtype=np.float32).reshape(n, -1)
    mtp = np.fromfile(env["DS4_QWEN4_MTP_OUT"], dtype=np.float32).reshape(n - 1, -1)
    return logits, mtp


def mlx_reference(path, ids, cache):
    if os.path.exists(cache):
        return np.load(cache)["logits"].astype(np.float32)
    import mlx.core as mx
    from pathlib import Path
    from mlx_lm.utils import load_model
    model, _ = load_model(Path(path))
    t0 = time.time()
    out = model(mx.array(ids, dtype=mx.int32)[None])
    logits = np.array(out[0].astype(mx.float32), dtype=np.float32)
    print(f"  mlx forward: {time.time() - t0:.1f}s")
    np.savez_compressed(cache, logits=logits.astype(np.float16), ids=np.array(ids, dtype=np.int32))
    return logits


def report(name, got, ref, ids):
    lg, lr = log_softmax(got), log_softmax(ref)
    kl = (np.exp(lr) * (lr - lg)).sum(-1)
    d = np.abs(got.astype(np.float64) - ref)
    top1 = (got.argmax(-1) == ref.argmax(-1)).mean()
    nxt = np.array(ids[1:])
    nll_g = -lg[np.arange(len(nxt)), nxt].mean()
    nll_r = -lr[np.arange(len(nxt)), nxt].mean()
    print(f"{name}: L={len(ids)} top1={top1:.4f} max|d|={d.max():.3e} mean|d|={d.mean():.3e} "
          f"KL(ref||ds4) mean={kl.mean():.3e} max={kl.max():.3e} NLL ds4={nll_g:.4f} ref={nll_r:.4f}")
    return float(kl.mean()), float(top1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--mlx", required=True, help="mlx-lm 8-bit model dir (reference logits)")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer dir (default: --mlx)")
    ap.add_argument("--text", default=None, help="text file (default: built-in passage)")
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--out", required=True, help="directory for dumps / cached reference")
    ap.add_argument("--skip-mlx", action="store_true")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="also run the chunked prefill path")
    ap.add_argument("--max-kl", type=float, default=0.05, help="fail when mean KL(ref||ds4) exceeds this")
    ap.add_argument("--min-top1", type=float, default=0.9, help="fail when top-1 agreement drops below this")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.mlx)
    text = open(args.text).read() if args.text else BUILTIN_TEXT
    ids = tok(text, add_special_tokens=False)["input_ids"][:args.n_tokens]
    print(f"{len(ids)} tokens")

    ds4_logits, mtp_full = run_ds4(args.ds4, args.gguf, ids, args.out, "chunk1", {"DS4_QWEN4_GPU_CHUNK": "1"})
    lg = log_softmax(ds4_logits)
    nxt = np.array(ids[1:])
    print(f"ds4 NLL (teacher-forced) = {-lg[np.arange(len(nxt)), nxt].mean():.4f}, "
          f"greedy hit rate = {(ds4_logits[:-1].argmax(-1) == nxt).mean():.4f}")
    main_next = ds4_logits[1:].argmax(-1)          # argmax at position i+1 (what verify compares to)
    truth = np.array(ids[2:])
    for name, mtp in (("full-width hnorm", mtp_full),):
        d = mtp.argmax(-1)
        acc_greedy = (d[:len(main_next) - 1] == main_next[:-1]).mean() if len(d) > 1 else float("nan")
        acc_truth = (d[:len(truth)] == truth).mean()
        print(f"mtp {name}: greedy-match acceptance={acc_greedy:.4f}, matches text={acc_truth:.4f}")

    if not args.skip_mlx:
        ref = mlx_reference(args.mlx, ids, os.path.join(args.out, f"mlx8_{len(ids)}.npz"))
        kl, top1 = report("ds4 vs mlx-8bit", ds4_logits, ref, ids)
        if kl > args.max_kl or top1 < args.min_top1:
            raise SystemExit(f"accuracy check failed: KL {kl:.3e} (max {args.max_kl}), top1 {top1:.4f} (min {args.min_top1})")

    if args.prefill_chunk > 1:
        env = {"DS4_QWEN4_GPU_CHUNK": str(args.prefill_chunk)}
        pre, _ = run_ds4(args.ds4, args.gguf, ids, args.out, f"chunk{args.prefill_chunk}", env)
        ends = [i for i in range(len(ids)) if (i + 1) % args.prefill_chunk == 0 or i == len(ids) - 1]
        d = np.abs(pre[ends] - ds4_logits[ends])
        top1 = (pre[ends].argmax(-1) == ds4_logits[ends].argmax(-1)).mean()
        print(f"prefill chunk {args.prefill_chunk} vs decode at {len(ends)} chunk ends: "
              f"max|d|={d.max():.3e} mean|d|={d.mean():.3e} top1={top1:.4f}")


if __name__ == "__main__":
    main()
