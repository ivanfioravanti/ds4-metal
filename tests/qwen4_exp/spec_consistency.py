#!/usr/bin/env python3
"""Qwen3.8 MTP speculative-decoding consistency on the mini model.

1. greedy `--mtp` output must equal plain greedy output (drafts are only
   committed when they are the target's argmax);
2. with DS4_QWEN4_SPEC_FORCE_ACCEPT every draft is committed, so the decode
   walks an arbitrary token sequence through the 2-token verify path; every
   token the CLI then picks by argmax must equal the argmax of the sequential
   CPU reference teacher-forced on the same sequence.  That checks the
   recurrent-state, KV and MTP-cache bookkeeping of the accept path.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np


def run(cmd, env):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        raise SystemExit(f"failed: {cmd}")
    return r.stdout, r.stderr


def prompt_ids(ds4, gguf, prompt, env):
    out, _ = run([ds4, "-m", gguf, "--cpu", "--raw", "--dump-tokens", "-p", prompt], env)
    ids = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\s\s", out, re.M)]
    if not ids:
        raise SystemExit("no prompt tokens parsed from --dump-tokens")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--gguf", required=True, help="F32 mini GGUF (with the MTP block)")
    ap.add_argument("--prompt", default="The quick brown fox")
    ap.add_argument("-n", type=int, default=40)
    args = ap.parse_args()
    base_env = dict(os.environ)
    gen = [args.ds4, "-m", args.gguf, "--temp", "0", "--raw", "-n", str(args.n), "-p", args.prompt]

    plain, _ = run(gen, base_env)
    mtp, err = run(gen + ["--mtp", "--mtp-timing"], base_env)
    text = lambda o: o.split("(100.0%)", 1)[-1].strip()
    if text(plain) != text(mtp):
        raise SystemExit(f"greedy mismatch with --mtp:\n{text(plain)}\n{text(mtp)}")
    print("greedy --mtp output matches plain:", re.search(r"mtp: .*", err).group(0) if re.search(r"mtp: .*", err) else "")

    env = dict(base_env, DS4_QWEN4_SPEC_FORCE_ACCEPT="1", DS4_QWEN4_SPEC_TRACE="1")
    _, err = run(gen + ["--mtp"], env)
    ids = prompt_ids(args.ds4, args.gguf, args.prompt, base_env)
    seq = list(ids)
    argmax_pos = []
    for m in re.finditer(r"ds4: spec pos (\d+) token (-?\d+)(?: draft (-?\d+) (accept|reject)| plain)", err):
        pos, tok = int(m.group(1)), int(m.group(2))
        if pos != len(seq):
            raise SystemExit(f"trace position {pos} != sequence length {len(seq)}")
        argmax_pos.append(len(seq))
        seq.append(tok)
        if m.group(3) is not None:
            if m.group(4) != "accept":
                raise SystemExit("forced accept did not accept")
            seq.append(int(m.group(3)))
    n_pairs = sum(1 for _ in re.finditer(r" accept", err))
    if n_pairs < 3:
        raise SystemExit(f"too few verify cycles traced ({n_pairs})")

    tmp = tempfile.mkdtemp(prefix="qwen4-spec-")
    out_path = os.path.join(tmp, "ref.bin")
    ref_env = dict(base_env, DS4_QWEN4_FT_TOKENS=",".join(map(str, seq)), DS4_QWEN4_FT_OUT=out_path)
    run([args.ds4, "-m", args.gguf, "--cpu", "--first-token-test", "--raw", "-p", "x"], ref_env)
    logits = np.fromfile(out_path, dtype=np.float32).reshape(len(seq), -1)
    bad = [(p, seq[p], int(logits[p - 1].argmax())) for p in argmax_pos if int(logits[p - 1].argmax()) != seq[p]]
    print(f"forced-accept walk: {len(seq) - len(ids)} generated tokens, {n_pairs} accepted pairs, "
          f"{len(argmax_pos)} argmax picks checked, {len(bad)} mismatches")
    if bad:
        for p, got, want in bad[:10]:
            print(f"  pos {p}: cli picked {got}, reference argmax {want}")
        raise SystemExit("accept-path bookkeeping mismatch")
    print("ok")


if __name__ == "__main__":
    main()
