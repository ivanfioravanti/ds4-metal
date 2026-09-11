# Qwen3.8-Flash-Next on M5 Max: round 5 (routed tiles on the tensor ops)

Base: upstream `qwen3.8-flash-next` at 6c1e836 (rounds 3 and 4 merged, PR #6).  Same machine,
pack (`qwen38-q4k`: Q4_K routed gate/up, MXFP4 routed down) and harnesses as the
earlier rounds.

Round 4 left the routed expert tiles MAC-bound: about 80% of a tile's time is the
fp16 `simdgroup_matrix` loop, and every exact lever below that loop is used up.
This round moves the two routed tile kernels onto the Metal 4 tensor ops (the M5
neural accelerators, `matmul2d` from MetalPerformancePrimitives), the same API the
dense Q8 GEMMs already use on M5.

**This is not bit-exact.**  The staged operands are unchanged (the same Qwen
dequantizers, activations rounded to half, exactly as the simdgroup tiles stage
them), but the cooperative matmul accumulates in its own order.  So the path is
opt-in (`DS4_QWEN4_MOE_MM_NAX=1` for 32-token tiles, `=2` for 64-token tiles),
off by default, and this report carries the drift and quality numbers that a
default decision needs.

## Kernels

`kernel_qwen4_moe_mm_mid_nax[64]` and `kernel_qwen4_moe_mm_down_nax[64]`
(`metal/qwen4.metal`, under `DS4_METAL_HAS_TENSOR`): 64 expert rows x 32 or 64
tokens per threadgroup, K in 32-wide steps, 128 threads.  Per K step the threads
stage the dequantized weight block (two 16-wide halves per row) and the token
block (8 floats per item, rounded to half) into threadgroup memory, then run
`matmul2d<...(NR1, 64, 32, false, true, false, multiply_accumulate),
execution_simdgroups<4>>` with the token block as the left operand and the weight
block as the transposed right operand.  Gate and up share one element layout, so
the mid epilogue applies `silu(gate) * up` on the cooperative tensors and stores
one float C tile; the scatter to the `mid`/`part` rows is the simdgroup kernels'.
Threadgroup memory: mid 10 KB (32 tokens) / 16 KB (64), down 8 KB / 16 KB.

Tried and dropped: double-buffered staging (stage step k+1 while the tensor op
consumes step k) and the descriptor's relaxed-precision accumulation.  Neither
moved the microbench (dense 9.2-9.5 ms vs 9.0-9.1 ms single-buffered; sparse 11.4
vs 10.6 ms), so the tiles are staging-bound on the tensor path, not MAC-bound.
The register-resident SiLU epilogue (one C tile instead of two, 16 KB -> 10 KB)
gained about 1.5%.

## Microbench (`QWEN4_BENCH=1 QWEN4_BENCH_ONLY=...`, mid + down, 3 runs each)

| case | simdgroup tiles | tensor tiles, 32 tokens | tensor tiles, 64 tokens |
| --- | ---: | ---: | ---: |
| `moe mm q4k/mxfp4 32` (32 experts, 2048 tokens x 10 slots) | 16.07-16.12 ms | 8.87-8.97 ms | 7.50-7.62 ms |
| `lo 256` (256 experts, sparse lists) | 17.99-18.13 ms | 10.63-10.79 ms | 11.37-11.43 ms |

## Prefill (chunk-interleaved harness, 8192-token chunks, `--tolerate-drift`)

| context | control (simdgroup) | tensor tiles | delta |
| --- | ---: | ---: | ---: |
| 8K -> 40K, 32-token tiles | 1148.0 tok/s | 1359.2 tok/s | +18.4% |
| 8K -> 40K, 64-token tiles vs 32-token tiles | 1376.0 tok/s (32) | 1422.2 tok/s (64) | +3.4% |
| 96K -> 128K, 64-token tiles (first repeat of two runs) | 1010.2 / 1023.2 tok/s | 1209.5 / 1234.6 tok/s | +19.7% / +20.7% |
| 96K -> 128K, after the staging follow-up (one unthrottled repeat each) | 1114.6 / 1129.5 tok/s | 1452.0 / 1464.7 tok/s | +30.3% / +29.7% |

The 64-token and 32-token tensor tiles produce bit-identical logits (each output
element accumulates over the same K order); the 64-token tiles are the opt-in
level 2.  At 128K each repeat first prefills a 96K prefix, and in both runs
(one after 15 minutes of continuous GPU load, one after a three-minute
cool-down) the second repeat throttled: both variants fell together and the
interleaved pairs stayed at +16 to +28%.  The table shows the first, unthrottled
repeat of each run; the two-repeat aggregates are +17.2% and +23.3%.

## Drift (full-vocabulary logits at the last position, candidate vs control, fresh sessions)

| prefix | max abs | mean abs | top-1 | reference margin |
| --- | ---: | ---: | --- | ---: |
| 2048 | 0.873 | 0.144 | agree (303) | 0.195 |
| 32768 | 0.653 | 0.094 | agree (85952) | 3.09 |
| 40960 (mixed histories, interleaved run) | 0.775 | 0.110 | flip on a 0.233 near-tie | 0.233 |
| 131072 (mixed histories, interleaved run) | 1.375 | 0.150 | agree (296) | 0.992 |

The perturbation does not grow with context: it is the same size at 2K, 32K and 128K.
Per tile, `test_moe_mm_tiles_exact` bounds the difference against the simdgroup
tiles at 7.7e-7 (mid, scale 1.3) and 1.5e-5 (down, scale 0.12) on a
production-shaped fixture (bound 2e-3 x scale in the test).

## Quality gate (BF16-reference continuation fixture, 99 aligned cases, 2376 target tokens)

`speed-bench/qwen38_q2down_compare.py`, production prefill of each prompt (the tile
path) and teacher-forced targets, same tokenizer for both runs:

| metric | simdgroup tiles | tensor tiles | delta |
| --- | ---: | ---: | ---: |
| target NLL | 0.20505 | 0.20503 | -0.00002 |
| top-1 agreement with BF16 | 96.34% | 96.25% | -0.08 pt (2 tokens) |
| first-token matches | 86 / 99 | 86 / 99 | 0 (one case each way) |
| logprob MAE | 0.04679 | 0.04659 | -0.00020 |

NLL moved in either direction on 51 vs 48 cases.  For scale, the Q4_K -> Q8 pack
spread on the same fixture is 0.015 NLL and 0.022 MAE (`qwen38-iq2-quality.md`);
the tensor-tile drift is two orders of magnitude below it.

## Follow-up: what the tensor tiles are bound by, and two staging fixes

Skip-variants of the tensor-tile source (`DS4_METAL_QWEN4_SOURCE` override, 64-token
tiles, `moe mm q4k/mxfp4 32` dense / `lo 256` sparse):

| variant | dense | sparse | attribution |
| --- | ---: | ---: | --- |
| full | 7.40 ms | 11.17 ms | |
| no tensor op | 4.33 ms | 7.12 ms | tensor op 41% / 36% (about 66 TFLOPS) |
| no weight dequant | 4.84 ms | 6.84 ms | weight staging 35% / 39% |
| no activation staging | 6.05 ms | 9.07 ms | activation staging 18% / 19% |
| no epilogue scatter | 7.23 ms | 10.76 ms | epilogue 2-4% |

The parts add up to the whole: staging and the tensor op barely overlap inside a
threadgroup, so the staging is the lever.  Two changes, both bit-identical (the
test prints an FNV hash of the tensor-tile outputs; it did not move):

- **Pre-rounded half activation operand.**  `kernel_qwen4_rows_f32_to_f16` rounds
  the operand once per call into a scratch tensor; each K step then gathers one
  16-byte word per item instead of two float4 gathers and eight conversions per
  row block.  7.40 -> 7.13 ms dense, 11.17 -> 10.5 ms sparse (conversion pass
  included).  8K-40K prefill: +28.7% over the simdgroup tiles (was +22%).
- **Register prefetch of the next K step's weight words.**  Rewriting the dequant
  arithmetic did nothing (vector nibble unpack, MXFP4 bit decode: within noise or
  worse); the cost is the scattered device loads issued in the step that consumes
  them.  `qwen4_load_raw16` fetches step k+1's raw words (Q4_K header + nibbles,
  or the 17 MXFP4 bytes) into registers during step k; `qwen4_dequant_raw16` runs
  the unchanged float expressions on them.  7.13 -> 6.70 ms dense, 10.5 -> 9.7 ms
  sparse.  8K-40K prefill: **1193 -> 1556 tok/s, +30.4%** over the simdgroup tiles.
- **32-token tail tile.**  An expert whose count leaves a remainder of at most 32
  tokens paid a half-empty 64-token tile (full dequant, half the tensor work).  The
  simdgroup kernels' tail mechanism (`qwen4_moe_tail_base` = 64) now applies to the
  tensor tiles: the 64-token kernel keeps the full tiles, the 32-token kernel takes
  the remainder.  Sparse microbench 9.5 -> 8.1 ms, dense unchanged; 8K-40K prefill
  +1.0% with the tensor tiles on both sides (three of four interleaved pairs).
- **Half copy of `mid` written by the mid tiles.**  The conversion pass feeding the
  down tiles read and wrote `mid` again (315 MB per layer), 3.4% of the chunk in the
  timeline.  The mid epilogue now writes the half copy beside the float `mid`, into
  a second scratch the down dispatch uses when the mid tiles just produced it (it
  converts `mid` itself otherwise); the x conversion does 16 values per thread.
  6.64-6.68 -> 6.52-6.60 ms dense; 8K-40K prefill **1186 -> 1572 tok/s, +32.6%**
  over the simdgroup tiles.

Rejected with numbers (all exact where applicable):

| candidate | result |
| --- | --- |
| vectorized threadgroup stores in the tensor-tile staging | bit-identical, 7.37-7.42 vs 7.36-7.47 ms |
| Q4_K vector nibble unpack / MXFP4 bit decode / vectorized table path | bit-identical, 7.06-7.15 / 7.29-7.37 / 7.10-7.15 ms vs 7.11-7.14 |
| attention K/V gather pipelined one tile ahead (`kernel_qwen4_attn_mm`) | exact, +0.24% at 8K-40K (noise); the per-token threadgroups already hide the gather latency |
| GDN scan next-token operand prefetch (`kernel_qwen4_gdn_scan_r4`) | exact, scan bench 1120 vs 1112-1122 us; the scan is bound by its reduction chain |
| register prefetch of the next k step's Q8 words in the dense tensor-op GEMM (`kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128`, default path) | exact, -1.6% at 8K-24K (3.3 -> 3.6 ms per dispatch) |
| dense tensor-op GEMM with the activation rows pre-rounded to half (`kernel_mul_mm_q8_0_f16_nax_direct_rhs`) | not exact (the unit multiplies the fp32 rows at full precision; first logit 4.046 -> 4.153), kernel 3.3 -> 3.06 ms but -1% overall with the conversion pass |
| tensor tiles with 64-wide K steps (level 3, 24 KB of threadgroup memory) | bit-identical to K=32 (max |d| 0, same hash), 6.51 -> 7.37 ms dense, 7.99 -> 8.78 ms sparse: occupancy |
| `kernel_swiglu_flat_f32` with eight elements per thread (20k -> 2.5k threadgroups) | exact, -0.7% at 8K-24K: the kernel's timeline share was encoder wait, not work |
| upstream's blocked conv (`kernel_qwen4_conv_halo` + `_blocked`, scoped to M3 Ultra) enabled on M5 | exact, 3.4 -> 1.2 ms per dispatch but +0.4% over three repeats (noise); the whole prefill-reuse path +0.8% |
| register prefetch of the weight words in the simdgroup routed tiles (`kernel_qwen4_moe_mm_mid/down_nt8`, default path) | 16.0 -> 16.7 ms dense, 17.8 -> 18.4 ms sparse; the simdgroup tiles are MAC-bound, the extra registers cost more than the hidden latency |

Where a chunk goes (8192 tokens at prefix 0, encoder timeline): with the first
tensor tiles, routed mid 21.7%, dense Q8 tensor-op GEMMs 19.3%, routed down 11.7%,
attention 11.5%, GDN scan 6.4%, HC low-rank GEMMs 5.3%; after the staging fixes the
chunk is 6.5% shorter and the order is dense Q8 GEMMs 20.7%, routed mid 15.9%,
attention 12.3%, routed down 9.2%, GDN scan 6.8%, HC GEMMs 5.7%.  Over the thirteen
chunks to 104K the attention share rises to 12.9%.  The attention kernel gathers about 2048
selected keys per token; an exact prefetch does not help it, so its remaining lever
is a multi-token restructure sharing the selected blocks across a query tile, which
changes the online-softmax order (drift class) and is left for a decision.

## Can the tensor tiles be bit-exact?

No, on this hardware.  The reference accumulates each output with
`simdgroup_multiply_accumulate` (8-term dot products added to the fp32
accumulator in ascending K order, one accumulator per element).  On the tensor
ops: a static K must be a multiple of 16 (`K must be dynamic or a multiple of
16`); K=16 runs give outputs bit-identical to K=32 runs (the unit reduces in
16-wide chunks internally), 7.5 ms dense; a dynamic-K partition of 8-term runs in
the same order changes the outputs (different hashes) but still does not match
the simdgroup tiles (max |d| 6.6e-7 for mid, 1.5e-5 for down) and runs at the
simdgroup speed (15 ms).  The unit's internal rounding of a dot product is not
the simdgroup unit's and is not controllable, so a bit-exact tensor path is not
available; the tensor tiles stay opt-in.

## Verification

- `tests/test_qwen4_kernels`: all passes, including the bounded tensor-tile pass
  for both tile widths (skips when the tensor API is unavailable).
- Default path untouched: the simdgroup kernels and their dispatch are unchanged;
  the opt-in branch is taken only with `DS4_QWEN4_MOE_MM_NAX` set and the tensor
  API available.  Checked across binaries: the 99 fixture cases scored by the
  pre-round build and by this branch with the variable unset give byte-identical
  full-vocabulary logits (99/99 files) and the same metrics.
- The follow-up staging changes keep the tensor-tile outputs bit-identical to the
  version the quality gate scored (the test's FNV hashes of the mid and down tiles
  did not move across them, and the A/B drift lines are identical), so the gate
  numbers above stand for the final kernels.
