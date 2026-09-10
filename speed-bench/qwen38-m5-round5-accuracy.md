# Qwen3.8-Flash-Next on M5 Max: round 5 accuracy follow-up (routed tiles)

Base: the round-5 branch (`m5-round5-nax`, PR #10), same machine, pack
(`qwen38-q4k`: Q4_K routed gate/up, MXFP4 routed down) and harnesses as the
earlier rounds.  Question: can the tensor-op routed tiles keep most of their
prefill win while becoming mathematically more accurate?

## Tile-level instrument: an exact double reference

`test_moe_mm_tiles_exact` now keeps the double shadows of the quantized
fixtures and computes an exact reference in double precision (double products
of the dequantized weights and the float activations, double accumulation),
then prints each path's max/mean |error| against it.  The simdgroup default
and the half-operand tensor tiles (`=1`/`=2`) sit at the same error
(6.779e-06 mid / 3.55e-06 down mean) — the cooperative matmul's accumulation
order contributes almost nothing to error; both paths are operand-rounding
bound (x and the dequantized weights rounded to half, 2^-11 relative).

## What was tried

| variant | level | tile error (mid / down, mean) | routed microbench (dense / sparse) | verdict |
| --- | --- | ---: | ---: | --- |
| full float operands (x gathered as float, no conversion pass) | 3/4 | 4.998e-06 / 2.444e-06 (−26% / −31%) | 15.8 / 18.8 ms | rejected: the MPP fast path has no useful fp32 tgmem-left kernel (`run_tg_f32_tg_f16_*` exists but runs ~2.4x slower); the prefill win disappears |
| compensated: stage the half rounding residual r = half(x − half(x)) beside the half operand and run the tensor op twice per K step on the same staged weights | 5 (64-tok) / 6 (32-tok tail) | 4.998e-06 / 2.444e-06 (−26% / −31%) | 12.9 / 15.0 ms | adopted as opt-in level 5/6: the fp16 tensor-op rate is kept, so the real-model prefill win survives |

The compensated mid tile gathers the float `x` directly (the pre-round
conversion pass disappears), writes the half copy of `mid` and its residual
for the down tiles, and keeps threadgroup memory at 16 KB (64 tokens) /
12 KB (32 tokens) — no residency change.  The compensated down tile reads the
half copy plus the residual.  Compensated levels fall back to the
uncompensated tiles if the matching mid launch did not just produce the
residual (stand-alone down dispatch).

## Measurements (M5 Max, chunk-interleaved, 8192-token chunks)

| comparison | control | candidate | delta |
| --- | ---: | ---: | ---: |
| `=2` vs default, 8K->40K, same session | 1077.6 t/s | 1348.5 t/s | +25.1% |
| `=5` vs default, 8K->40K, same protocol | 840.5 t/s | 1025.5 t/s | **+22.0%** |
| `=5` vs `=2`, 8K->40K, direct | 1369.0 t/s | 1076.4 t/s | −21.4% |
| decode (MTP, 8K prefix), `=5` vs default | 50.46 t/s | 50.43 t/s | −0.05%, byte-exact |

(An earlier `=5` pair that showed +28/+33% was measured with a stale bench
binary whose dispatch predated the compensated levels — its drift line was
bit-identical to `=2`'s, which is how the stale measurement was caught.  The
rebuilt binary shows a distinct drift signature: max 1.065 / mean 0.138 at the
40960 boundary, top-1 flip on a 0.852 margin.)

## Model-level quality (99-case BF16 fixture, same session and build)

| path | target NLL | top-1 vs BF16 | first-token | logprob MAE |
| --- | ---: | ---: | ---: | ---: |
| default (simdgroup) | 0.20505 | 96.34% | 86/99 | 0.046788 |
| `=2` (half tensor) | 0.20503 | 96.25% | 86/99 | 0.046591 |
| `=5` (compensated) | **0.20430** | 96.21% | 86/99 | 0.046653 |

The default and `=2` rows reproduce the round-5 report's published values
exactly.  The compensated path posts the best NLL of all paths (−0.00075 vs
the simdgroup default) and improves logprob MAE over the default, at the cost
of 3 top-1 tokens of BF16 agreement (96.21% vs 96.34%); first-token matches
are unchanged.  All paths remain two orders of magnitude inside the Q4_K->Q8
quantization spread.

## Verdict

Mathematical accuracy and the tensor-op prefill win trade off directly on
this hardware: the fast path is fp16-only, so accuracy can only be bought
with extra work (compensation: 2x tensor-op) or lost rate (fp32 operands:
~2.4x).  `DS4_QWEN4_MOE_MM_NAX=5` keeps a good boost (+22% prefill,
decode untouched) while making the routed tiles the most numerically faithful
path measured.  Levels 3/4 are kept for the record but rejected.  Everything
remains opt-in; the default path is untouched and byte-exact.
