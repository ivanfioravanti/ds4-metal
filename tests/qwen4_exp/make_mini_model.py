#!/usr/bin/env python3
"""Write a small random Qwen3.8-Flash-Next (qwen4_exp) checkpoint in the
official HF layout: model.language_model.* / lm_head / mtp.* tensors, the
n-gram table split into shard_J parts, int64 hash constants, and the real
tokenizer copied from a Qwen3.8-Flash-Next snapshot (so vocab_size and the
PLE hash multipliers match the release model).

The mini keeps every structural feature of the release model (hc streams,
PLE layer at 1-based id 2, GDN with 3 value heads per key head, QSA indexer,
top-10 routing, MTP block) at toy widths, so the HF/mlx oracles, the GGUF
converter and the DS4 engine can all be checked on the same file.

Usage:
  python make_mini_model.py --out ~/ds4-gguf/qwen4-mini/hf \
      --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/<sha>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil

import numpy as np
from safetensors.numpy import save_file

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PLE_LAYER_PRIME = 10007

MINI = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 248044,
    "dtype": "float32",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "hc_count": 4,
    "hc_lowrank": 8,
    "head_dim": 32,
    "heads_per_ngram": 8,
    "hidden_act": "silu",
    "hidden_size": 64,
    "indexer_budget": 8,
    "indexer_compress_ratio": 4,
    "indexer_head_dim": 32,
    "indexer_kv_heads": 1,
    "indexer_n_heads": 4,
    "initializer_range": 0.02,
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 6,
    "linear_value_head_dim": 32,
    "make_ngram_vocab_size_divisible_by": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "model_type": "qwen4_exp_text",
    "moe_intermediate_size": 32,
    "mtp": {
        "hybrid": True,
        "layer_types": ["full_attention"],
        "mtp_use_hidden_state_from_layer": None,
        "num_hidden_layers": 1,
        "rope_theta": 10000000,
    },
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "ngram_size": 3,
    "ngram_vocab_size_base": 97,
    "num_attention_heads": 4,
    "num_experts": 32,
    "num_experts_per_tok": 10,
    "num_hidden_layers": 8,
    "num_key_value_heads": 2,
    "output_gate_type": "sigmoid",
    "output_router_logits": False,
    "pad_token_id": None,
    "partial_rotary_factor": 0.25,
    "ple_conv_kernel_size": 4,
    "ple_embed_dim": 64,
    "ple_layer_ids": [2],
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
    "router_aux_loss_coef": 0.001,
    "seed": 1234,
    "shared_expert_intermediate_size": 32,
    "split_ngram_parts": 4,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 248320,
}

VISION = {
    "deepstack_visual_indexes": [],
    "depth": 1,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 16,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 32,
    "model_type": "qwen4_exp",
    "num_heads": 2,
    "num_position_embeddings": 16,
    "out_hidden_size": 64,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "generation_config.json",
]


def splitmix64(v: int) -> int:
    v = (v + SPLITMIX_GAMMA) & MASK64
    v = ((v ^ (v >> 30)) * SPLITMIX_M1) & MASK64
    v = ((v ^ (v >> 27)) * SPLITMIX_M2) & MASK64
    return (v ^ (v >> 31)) & MASK64


def layer_multipliers(vocab: int, ngram: int, ple_index: int, seed: int) -> list[int]:
    half_bound = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    base = seed + PLE_LAYER_PRIME * ple_index
    return [2 * (splitmix64((base + SPLITMIX_GAMMA * (j + 1)) & MASK64) % half_bound) + 1
            for j in range(ngram)]


def is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


class Init:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def linear(self, out_dim: int, in_dim: int) -> np.ndarray:
        return (self.rng.standard_normal((out_dim, in_dim)) / math.sqrt(in_dim)).astype(np.float32)

    def normal(self, shape, std: float) -> np.ndarray:
        return (self.rng.standard_normal(shape) * std).astype(np.float32)

    def uniform(self, shape, lo: float, hi: float) -> np.ndarray:
        return self.rng.uniform(lo, hi, shape).astype(np.float32)

    def zero_centred_gamma(self, n: int) -> np.ndarray:
        return self.uniform((n,), -0.2, 0.2)


def hc_tensors(init: Init, prefix: str, hc: int, hc_dim: int, rank: int, inject: bool) -> dict:
    t = {
        f"{prefix}.hc_norm.weight": init.zero_centred_gamma(hc_dim),
        f"{prefix}.input_mix_weight_down.weight": init.linear(rank, hc_dim),
        f"{prefix}.input_mix_weight_up.weight": init.linear(hc_dim, rank),
    }
    if inject:
        t[f"{prefix}.block_inject_weight.weight"] = init.linear(hc, hc_dim)
    return t


def attn_tensors(init: Init, p: str, c: dict) -> dict:
    h, dh = c["hidden_size"], c["head_dim"]
    nh, nkv = c["num_attention_heads"], c["num_key_value_heads"]
    di, ni = c["indexer_head_dim"], c["indexer_n_heads"]
    return {
        f"{p}.self_attn.q_proj.weight": init.linear(nh * dh * 2, h),
        f"{p}.self_attn.k_proj.weight": init.linear(nkv * dh, h),
        f"{p}.self_attn.v_proj.weight": init.linear(nkv * dh, h),
        f"{p}.self_attn.o_proj.weight": init.linear(h, nh * dh),
        f"{p}.self_attn.q_norm.weight": init.zero_centred_gamma(dh),
        f"{p}.self_attn.k_norm.weight": init.zero_centred_gamma(dh),
        f"{p}.self_attn.indexer.index_qk_proj.weight": init.linear((ni + 1) * di, h),
        f"{p}.self_attn.indexer.q_layernorm.weight": init.zero_centred_gamma(di),
        f"{p}.self_attn.indexer.k_layernorm.weight": init.zero_centred_gamma(di),
    }


def gdn_tensors(init: Init, p: str, c: dict) -> dict:
    h = c["hidden_size"]
    hk, hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    conv_dim = 2 * hk * dk + hv * dv
    k = c["linear_conv_kernel_dim"]
    return {
        f"{p}.linear_attn.A_log": np.log(init.uniform((hv,), 1.0, 16.0)).astype(np.float32),
        f"{p}.linear_attn.dt_bias": init.uniform((hv,), 0.2, 1.5),
        f"{p}.linear_attn.conv1d.weight": init.uniform((conv_dim, 1, k), -0.5, 0.5),
        f"{p}.linear_attn.in_proj_a.weight": init.linear(hv, h),
        f"{p}.linear_attn.in_proj_b.weight": init.linear(hv, h),
        f"{p}.linear_attn.in_proj_qkv.weight": init.linear(conv_dim, h),
        f"{p}.linear_attn.in_proj_z.weight": init.linear(hv * dv, h),
        f"{p}.linear_attn.norm.weight": (1.0 + init.uniform((dv,), -0.1, 0.1)).astype(np.float32),
        f"{p}.linear_attn.out_proj.weight": init.linear(h, hv * dv),
    }


def moe_tensors(init: Init, p: str, c: dict) -> dict:
    h, e, f, fs = c["hidden_size"], c["num_experts"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
    return {
        f"{p}.mlp.gate.weight": init.linear(e, h),
        f"{p}.mlp.experts.gate_up_proj": (init.rng.standard_normal((e, 2 * f, h)) / math.sqrt(h)).astype(np.float32),
        f"{p}.mlp.experts.down_proj": (init.rng.standard_normal((e, h, f)) / math.sqrt(f)).astype(np.float32),
        f"{p}.mlp.shared_expert.gate_proj.weight": init.linear(fs, h),
        f"{p}.mlp.shared_expert.up_proj.weight": init.linear(fs, h),
        f"{p}.mlp.shared_expert.down_proj.weight": init.linear(h, fs),
        f"{p}.mlp.shared_expert_gate.weight": init.linear(1, h),
    }


def ple_tensors(init: Init, p: str, c: dict, ple_index: int) -> tuple[dict, dict]:
    h, hc = c["hidden_size"], c["hc_count"]
    hc_dim = h * hc
    pe = c["ple_embed_dim"]
    n_heads = (c["ngram_size"] - 1) * c["heads_per_ngram"]
    head_dim = pe // n_heads
    sizes, offsets, total = [], [], 0
    for hd in range(n_heads):
        s = nth_prime_after(c["ngram_vocab_size_base"] - 1, ple_index * n_heads + hd + 1)
        sizes.append(s)
        offsets.append(total)
        total += s
    div = c["make_ngram_vocab_size_divisible_by"]
    padded = (total + div - 1) // div * div
    parts = c["split_ngram_parts"]
    if padded % parts != 0:
        raise SystemExit(f"padded n-gram rows {padded} not divisible by split_ngram_parts {parts}")
    rows = padded // parts
    t = {
        f"{p}.ple.key_proj.weight": init.linear(hc_dim, pe),
        f"{p}.ple.value_proj.weight": init.linear(h, pe),
        f"{p}.ple.norm_key.weight": init.zero_centred_gamma(hc_dim),
        f"{p}.ple.norm_query.weight": init.zero_centred_gamma(hc_dim),
        f"{p}.ple.norm_conv.weight": init.zero_centred_gamma(hc_dim),
        f"{p}.ple.conv1d.weight": init.uniform((hc_dim, 1, c["ple_conv_kernel_size"]), -0.4, 0.4),
        f"{p}.ple.ple_embedding.layer_multipliers": np.array(
            layer_multipliers(c["vocab_size"], c["ngram_size"], ple_index, c["seed"]), dtype=np.int64),
        f"{p}.ple.ple_embedding.ngram_heads_offsets": np.array(offsets, dtype=np.int64),
        f"{p}.ple.ple_embedding.ngram_heads_vocab_sizes": np.array(sizes, dtype=np.int64),
    }
    for j in range(parts):
        t[f"{p}.ple.ple_embedding.ngram_embedding.shard_{j}.weight"] = init.normal((rows, head_dim), 0.5)
    meta = {"sizes": sizes, "offsets": offsets, "total": total, "padded": padded, "head_dim": head_dim}
    return t, meta


def build(c: dict, seed: int) -> tuple[dict, dict]:
    init = Init(seed)
    h, hc, rank, v = c["hidden_size"], c["hc_count"], c["hc_lowrank"], c["vocab_size"]
    hc_dim = h * hc
    lm = "model.language_model"
    t = {
        f"{lm}.embed_tokens.weight": init.normal((v, h), 0.5),
        "lm_head.weight": init.linear(v, h),
    }
    t.update(hc_tensors(init, f"{lm}.hyper_connection_mixer", hc, hc_dim, rank, inject=False))
    meta = {}
    for il, kind in enumerate(c["layer_types"]):
        p = f"{lm}.layers.{il}"
        t.update(hc_tensors(init, f"{p}.attn_hyper_connection", hc, hc_dim, rank, inject=True))
        t.update(hc_tensors(init, f"{p}.mlp_hyper_connection", hc, hc_dim, rank, inject=True))
        if kind == "linear_attention":
            t.update(gdn_tensors(init, p, c))
        else:
            t.update(attn_tensors(init, p, c))
        t.update(moe_tensors(init, p, c))
        if il + 1 in c["ple_layer_ids"]:
            pt, meta = ple_tensors(init, p, c, c["ple_layer_ids"].index(il + 1))
            t.update(pt)
    # MTP block: one full-attention layer with its own hc mixes and mixer
    t.update({
        "mtp.pre_fc_norm_embedding.weight": init.zero_centred_gamma(h),
        "mtp.pre_fc_norm_hidden.weight": init.zero_centred_gamma(hc_dim),
        "mtp.fc_embedding.weight": init.linear(h, h),
        "mtp.fc_hidden.weight": init.linear(h, h),
    })
    t.update(hc_tensors(init, "mtp.hyper_connection_mixer", hc, hc_dim, rank, inject=False))
    for il in range(c["mtp_num_hidden_layers"]):
        p = f"mtp.layers.{il}"
        t.update(hc_tensors(init, f"{p}.attn_hyper_connection", hc, hc_dim, rank, inject=True))
        t.update(hc_tensors(init, f"{p}.mlp_hyper_connection", hc, hc_dim, rank, inject=True))
        t.update(attn_tensors(init, p, c))
        t.update(moe_tensors(init, p, c))
    return t, meta


def write_shards(out: str, tensors: dict, n_files: int) -> None:
    names = list(tensors)
    per = (len(names) + n_files - 1) // n_files
    weight_map, total = {}, 0
    for i in range(n_files):
        fname = f"model-{i + 1:05d}-of-{n_files:05d}.safetensors"
        chunk = {k: tensors[k] for k in names[i * per:(i + 1) * per]}
        if not chunk:
            continue
        save_file(chunk, os.path.join(out, fname), metadata={"format": "pt"})
        for k, a in chunk.items():
            weight_map[k] = fname
            total += a.nbytes
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="Qwen3.8-Flash-Next snapshot dir")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--files", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    c = dict(MINI)
    tensors, meta = build(c, args.seed)
    write_shards(args.out, tensors, args.files)

    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 248056,
        "language_model_only": False,
        "model_type": "qwen4_exp",
        "text_config": c,
        "tie_word_embeddings": False,
        "transformers_version": "5.8.0.dev0",
        "video_token_id": 248057,
        "vision_config": VISION,
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    for name in TOKENIZER_FILES:
        shutil.copyfile(os.path.join(args.tokenizer, name), os.path.join(args.out, name))
    with open(os.path.join(args.out, "mini_meta.json"), "w") as f:
        json.dump({"seed": args.seed, "ple": meta, "n_tensors": len(tensors),
                   "n_params": int(sum(a.size for a in tensors.values()))}, f, indent=2)
    print(f"wrote {len(tensors)} tensors, {sum(a.size for a in tensors.values()) / 1e6:.1f}M params to {args.out}")
    print("ple", meta)


if __name__ == "__main__":
    main()
