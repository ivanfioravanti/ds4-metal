#!/usr/bin/env python3
"""Convert the three matching V4.1 DSpark shards to a separate support GGUF.

Download shards 44--46, config.json, inference/config.json and the original
model.safetensors.index.json from the target GGUF's source revision first.
The target embedding and output head are shared at runtime, not duplicated.
"""
import argparse
import json
import os
import re
import threading
import sys

from deepseek41_quantize import write_gguf, scale_name
from glm53_quantize import (SourceDB, TensorPlan, load_index,
    load_safetensors_header, QTYPE_F32, QTYPE_F16, QTYPE_Q8_0, QTYPE_Q4_K,
    align, qtype_nbytes, kv_string, kv_u32, kv_u32_array, print_plan)
from deepseek41_metadata import GGUF_ALIGNMENT


class DraftSource(SourceDB):
    def __init__(self, directory):
        self.hf_dir = directory
        _, original = load_index(os.path.join(directory, 'model.safetensors.index.json'))
        self.weight_map = {k: v for k, v in original.items() if k.startswith('mtp.')}
        if not self.weight_map:
            raise ValueError('checkpoint has no DSpark tensors')
        self.tensors, self._fds = {}, {}
        self._fd_lock = threading.Lock()
        for shard in sorted(set(self.weight_map.values())):
            for name, info in load_safetensors_header(os.path.join(directory, shard)).items():
                if not name.startswith('mtp.'):
                    continue
                if self.weight_map.get(name) != shard or name in self.tensors:
                    raise ValueError(f'inconsistent source index: {name}')
                self.tensors[name] = dict(info, shard=shard)
        if set(self.tensors) != set(self.weight_map):
            raise ValueError('incomplete DSpark source shards')


def build_plan(db, config):
    expected = dict(dim=5120, n_layers=40, n_mtp_layers=3, dspark_block_size=5,
                    dspark_noise_token_id=128799, dspark_markov_rank=256,
                    dspark_n_routed_experts=128, dspark_n_activated_experts=3,
                    dspark_target_layer_ids=[37, 38, 39])
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('unsupported V4.1 DSpark configuration')
    d, f, h, hd = config['dim'], config['moe_inter_dim'], config['n_heads'], config['head_dim']
    qr, groups, rank = config['q_lora_rank'], config['o_groups'], config['o_lora_rank']
    hc, ne, vocab = config['hc_mult'], config['dspark_n_routed_experts'], config['vocab_size']
    plan, used = [], set()

    def claim(source, shape):
        info = db.info(source)
        if info['shape'] != list(shape):
            raise ValueError(f'{source}: shape {info["shape"]}, expected {shape}')
        used.add(source)
        if info['dtype'] in ('I8', 'F8_E4M3'):
            scale = db.info(scale_name(source))
            expected_scale = [shape[0], shape[1] // 16] if info['dtype'] == 'I8' else [(x + 31) // 32 for x in shape]
            if scale['dtype'] != 'F8_E8M0' or scale['shape'] != expected_scale:
                raise ValueError(f'{source}: invalid native quantization scales')
            used.add(scale_name(source))

    def regular(name, source, shape, qt, role='draft'):
        claim(source, shape)
        plan.append(TensorPlan(name, tuple(reversed(shape)), qt, role, source=source))

    for stage in range(3):
        p = f'mtp.{stage}'
        for site in ('attn', 'ffn'):
            for part, shape, qt in [('fn', (hc * (hc + 2), hc * d), QTYPE_F16),
                                     ('base', (hc * (hc + 2),), QTYPE_F32),
                                     ('scale', (3,), QTYPE_F32)]:
                regular(f'{p}.hc_{site}_{part}.weight', f'{p}.hc_{site}_{part}', shape, qt)
            regular(f'{p}.{site}_norm.weight', f'{p}.{site}_norm.weight', (d,), QTYPE_F32)
        for dst, src, shape, qt in [
            ('attn_sinks', 'attn_sink', (h,), QTYPE_F32),
            ('attn_q_a', 'wq_a.weight', (qr, d), QTYPE_Q8_0),
            ('attn_q_b', 'wq_b.weight', (h * hd, qr), QTYPE_Q8_0),
            ('attn_q_a_norm', 'q_norm.weight', (qr,), QTYPE_F32),
            ('attn_kv', 'wkv.weight', (hd, d), QTYPE_Q8_0),
            ('attn_kv_a_norm', 'kv_norm.weight', (hd,), QTYPE_F32),
            ('attn_output_a', 'wo_a.weight', (groups * rank, h * hd // groups), QTYPE_Q8_0),
            ('attn_output_b', 'wo_b.weight', (d, groups * rank), QTYPE_Q8_0)]:
            regular(f'{p}.{dst}.weight', f'{p}.attn.{src}', shape, qt)
        regular(f'{p}.ffn_gate_inp.weight', f'{p}.ffn.gate.weight', (ne, d), QTYPE_F32)
        for src, dst in [('bias', 'exp_probs_b.bias'), ('bias_vl', 'exp_probs_b_vl.bias')]:
            regular(f'{p}.{dst}', f'{p}.ffn.gate.{src}', (ne,), QTYPE_F32)
        for part, src, shape in [('gate', 'w1', (f, d)), ('up', 'w3', (f, d)), ('down', 'w2', (d, f))]:
            regular(f'{p}.ffn_{part}_shexp.weight', f'{p}.ffn.shared_experts.{src}.weight', shape, QTYPE_Q8_0)
            pattern = f'{p}.ffn.experts.{{expert}}.{src}.weight'
            for expert in range(ne):
                name = pattern.format(expert=expert)
                claim(name, (shape[0], shape[1] // 2))
                if db.info(name)['dtype'] != 'I8':
                    raise ValueError(f'{name}: expected packed FP4')
            plan.append(TensorPlan(f'{p}.ffn_{part}_exps.weight', (*reversed(shape), ne),
                QTYPE_Q4_K, 'experts', source=pattern, expert_layer=stage, expert_part=part, expert_count=ne))
    regular('mtp.0.main_proj.weight', 'mtp.0.main_proj.weight', (d, 3 * d), QTYPE_Q8_0)
    regular('mtp.0.main_norm.weight', 'mtp.0.main_norm.weight', (d,), QTYPE_F32)
    regular('mtp.2.norm.weight', 'mtp.2.norm.weight', (d,), QTYPE_F32)
    regular('mtp.2.markov_head.markov_w1.weight', 'mtp.2.markov_head.embed.weight', (vocab, 256), QTYPE_F16)
    regular('mtp.2.markov_head.markov_w2.weight', 'mtp.2.markov_head.head.weight', (vocab, 256), QTYPE_F32)
    regular('mtp.2.confidence_head.proj.weight', 'mtp.2.confidence_head.proj.weight', (1, d + 256), QTYPE_F32)
    if used != set(db.tensors):
        raise ValueError(f'unclaimed draft tensors: {sorted(set(db.tensors) - used)[:10]}')
    offset = 0
    for item in plan:
        item.offset, item.nbytes = offset, qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    parser.add_argument('--quants-library', default=os.path.join(os.path.dirname(__file__), f'libds4quants.{suffix}'))
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_revision) or args.threads < 1:
        parser.error('use a full source revision and a positive thread count')
    args.imatrix = None
    with open(os.path.join(args.hf, 'config.json')) as fp:
        if json.load(fp).get('model_type') != 'deepseek_v41':
            parser.error('expected a DeepSeek V4.1 checkpoint')
    with open(os.path.join(args.hf, 'inference/config.json')) as fp:
        config = json.load(fp)
    db = DraftSource(args.hf)
    plan = build_plan(db, config)
    records = [kv_string('general.architecture', 'deepseek41-dspark'),
        kv_string('general.name', 'DeepSeek V4.1 Flash DSpark'),
        kv_string('general.source.url', 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash'),
        kv_string('general.source.revision', args.source_revision),
        kv_string('deepseek4.checkpoint_variant', 'v4.1-flash'),
        kv_u32('general.alignment', GGUF_ALIGNMENT),
        kv_u32('deepseek4.dspark.block_size', 5),
        kv_u32('deepseek4.dspark.markov_rank', 256),
        kv_u32('deepseek4.dspark.noise_token_id', 128799),
        kv_u32_array('deepseek4.dspark.target_layer_ids', [37, 38, 39])]
    try:
        if args.dry_run:
            print_plan(plan, records, [], GGUF_ALIGNMENT)
        else:
            write_gguf(args, plan, records, db)
    finally:
        db.close()


if __name__ == '__main__':
    main()
