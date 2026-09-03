#!/usr/bin/env python3
"""Reference logits for a qwen4_exp checkpoint from HF transformers (torch,
fp32, CPU).  Writes one .npz per case:

  tokens      int32 [L]          input ids
  logits      float32 [L, V]     full-vocab logits at every position
  h_layer<i>  float32 [L, hc*H]  residual stream after decoder layer i (pf_21 only)
  h_final     float32 [L, H]     final hyper-connection mixer output (pf_21 only)
  mtp_full                       MTP draft logits [L-1, V] (pf_21 only): row i predicts
                                 token i+2 from stream i + embedding of token i+1, with the
                                 hidden pre-norm taken over all hc*H dims (vLLM/SGLang) or
                                 per stream (llama.cpp PR #27836)

Chunked-prefill and decode cases are checked against the single-shot run
here, so the fixtures also document what a KV-cache continuation must equal.

Usage: python oracle_hf.py --model ~/ds4-gguf/qwen4-mini/hf --out ~/ds4-gguf/qwen4-oracle/hf
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

EOS = 248044


def make_cases(rng: np.random.Generator, vocab: int) -> dict[str, np.ndarray]:
    def rand(n):
        return rng.integers(0, vocab - 300, size=n, dtype=np.int64)

    cases = {f"pf_{n}": rand(n) for n in (1, 4, 7, 21, 40)}
    eos = rand(21)
    eos[5] = EOS
    eos[12] = EOS
    cases["eos_reset"] = eos
    pat = rand(6)
    cases["repeat"] = np.tile(pat, 4)
    cases["long_sparse"] = rand(64)
    return cases


def load(model_dir: str):
    from transformers import AutoConfig, Qwen4ExpForConditionalGeneration

    cfg = AutoConfig.from_pretrained(model_dir)
    model = Qwen4ExpForConditionalGeneration.from_pretrained(
        model_dir, config=cfg, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    return model


@torch.no_grad()
def run_text(model, ids: np.ndarray, past=None, hidden: bool = False):
    """Returns logits, model output and, when `hidden`, the residual stream
    after every decoder layer (captured with hooks: HF's own hidden_states
    tuple holds the layer inputs)."""
    lm = model.model.language_model
    streams, hooks = [], []
    if hidden:
        for layer in lm.layers:
            hooks.append(layer.register_forward_hook(lambda mod, inp, out: streams.append(out[0].float().numpy())))
    out = lm(input_ids=torch.as_tensor(ids)[None], past_key_values=past, use_cache=True)
    for h in hooks:
        h.remove()
    logits = model.lm_head(out.last_hidden_state)[0]
    return logits.float().numpy(), out, streams


@torch.no_grad()
def mtp_draft(model, model_dir: str, ids: np.ndarray, stream: torch.Tensor) -> np.ndarray:
    """Teacher-forced MTP over positions 0..L-2 of `ids`: input pair (stream[i], ids[i+1])."""
    from safetensors.torch import load_file
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, Qwen4ExpTextRMSNorm)

    lm = model.model.language_model
    tcfg = lm.config
    H, hc = tcfg.hidden_size, tcfg.hc_count
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    mtp = {}
    for fname in sorted(set(idx.values())):
        for k, v in load_file(os.path.join(model_dir, fname)).items():
            if k.startswith("mtp."):
                mtp[k[len("mtp."):]] = v.float()

    cfg = copy.deepcopy(tcfg)
    cfg.num_hidden_layers = 1
    cfg.layer_types = ["qwen_sparse_attention"]
    cfg.ple_layer_ids = []
    layer = Qwen4ExpTextDecoderLayer(cfg, 0).eval()
    layer.load_state_dict({k[len("layers.0."):]: v for k, v in mtp.items() if k.startswith("layers.0.")})
    mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False).eval()
    mixer.load_state_dict({k[len("hyper_connection_mixer."):]: v for k, v in mtp.items()
                           if k.startswith("hyper_connection_mixer.")})
    enorm = Qwen4ExpTextRMSNorm(H, eps=cfg.rms_norm_eps)
    enorm.weight.data = mtp["pre_fc_norm_embedding.weight"]
    hnorm = Qwen4ExpTextRMSNorm(hc * H, eps=cfg.rms_norm_eps)
    hnorm.weight.data = mtp["pre_fc_norm_hidden.weight"]

    L = len(ids) - 1
    e = lm.embed_tokens(torch.as_tensor(ids[1:])[None])                 # [1, L, H]
    e = torch.nn.functional.linear(enorm(e), mtp["fc_embedding.weight"])
    h = hnorm(stream[:L][None]).view(1, L, hc, H)
    h = torch.nn.functional.linear(h, mtp["fc_hidden.weight"])
    x = (h + e[:, :, None, :]).reshape(1, L, hc * H)

    pos = torch.arange(1, L + 1)[None]
    cos, sin = lm.rotary_emb(x, pos[None].expand(3, 1, L))
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool))[None, None]
    y = layer(x, position_embeddings=(cos, sin), attention_mask=mask, conv_mask=None,
              past_key_values=None, ple_input_ids=None)
    return model.lm_head(mixer(y))[0].float().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    model = load(args.model)
    vocab = model.config.text_config.vocab_size
    cases = make_cases(np.random.default_rng(args.seed), vocab)

    full = {}
    for name, ids in cases.items():
        hidden = name == "pf_21"
        logits, out, streams = run_text(model, ids, hidden=hidden)
        full[name] = logits
        extra = {}
        if hidden:
            for i, h in enumerate(streams):
                extra[f"h_layer{i}"] = h
            extra["h_final"] = out.last_hidden_state[0].float().numpy()
            pre_mixer = torch.as_tensor(streams[-1])
            extra["mtp_full"] = mtp_draft(model, args.model, ids, pre_mixer)
        np.savez(os.path.join(args.out, f"{name}.npz"), tokens=ids.astype(np.int32), logits=logits, **extra)
        top = logits.argmax(-1)
        print(f"{name:12s} L={len(ids):3d} top1[:8]={top[:8].tolist()} |logit|max={np.abs(logits).max():.3f}")

    # chunked prefill and decode continuations must reproduce the single-shot logits
    checks = {"chunk_21_7": ("pf_21", [7, 14]), "chunk_40_33": ("pf_40", [33, 7]),
              "chunk_40_39": ("pf_40", [39, 1]), "decode_21_8": ("pf_40", [21] + [1] * 8),
              "sparse_64_split": ("long_sparse", [17, 30, 17])}
    worst = 0.0
    for name, (base, splits) in checks.items():
        ids = cases[base]
        pos, past, pieces = 0, None, []
        for n in splits:
            lg, out, _ = run_text(model, ids[pos:pos + n], past=past)
            past = out.past_key_values
            pieces.append(lg)
            pos += n
        chunked = np.concatenate(pieces, 0)
        d = np.abs(chunked - full[base][:pos]).max()
        worst = max(worst, d)
        np.savez(os.path.join(args.out, f"{name}.npz"), tokens=ids[:pos].astype(np.int32), logits=chunked,
                 splits=np.array(splits, dtype=np.int32))
        print(f"{name:16s} splits={splits} max|d| vs single-shot = {d:.2e}")
    if worst > 1e-3:
        raise SystemExit(f"HF cache continuation mismatch {worst:.3e}")
    print("ok")


if __name__ == "__main__":
    main()
