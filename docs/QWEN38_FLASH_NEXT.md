# Qwen3.8 Flash Next

[Back to README](../README.md)

Qwen3.8-Flash-Next (`qwen4exp` in GGUF terms) runs on its own Metal graph:
36 gated delta-net layers and 12 gated GQA layers with the QSA block-sparse
indexer, four-stream hyper-connections, the hashed per-layer n-gram table,
512-expert MoE with a shared expert, and the built-in MTP block.

## Download and run

The [DS4 Q2 release](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2)
contains a **41.73 GiB** combined main/MTP GGUF. It uses IQ2_XXS gate/up
experts and Q2_K down projections, with weight rows padded from 640 to 768
columns. The required external Q4_1 PLE sidecar is reused from the Q4 repo;
together the files use about **76.81 GB (71.53 GiB)** on disk. On a 64 GB Mac,
64K is the practical daily context (about 44.8 GiB planned, stable alongside a
normal desktop load); 128K fits at about 47.0 GiB but only on a clean host —
once the OS is several GB into swap, prefill can fail inside the Metal command
buffer with `Insufficient Memory`. The Qwen path defaults to an 8192-token
prefill chunk (`DS4_QWEN4_PREFILL_CHUNK` overrides it; `--help` still lists the
generic 4096 default). Prefill throughput barely moves with the chunk size, so
1,024 is a fine low-memory choice:

```sh
./download_model.sh qwen38-q2
./ds4 --ple gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --ctx 65536 --prefill-chunk 1024
./ds4-server --ple gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --ctx 65536 --prefill-chunk 1024 --mtp --mtp-exact-sampling
./ds4-agent --ple gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --ctx 65536 --prefill-chunk 1024 --mtp
```

The downloader links `ds4flash.gguf` to the combined model. Both ordinary and
MTP decode use this same GGUF and PLE sidecar; omit `--mtp` for ordinary decode.
One physical 64 GB data point, an M1 Max (32-core GPU, 64 GB) at `18ca8ec`
with the same files and `--temp 0 --nothink --prefill-chunk 1024`:

| ctx | decode | decode `--mtp` | MTP acceptance | prefill (3K prompt) |
|---:|---:|---:|---:|---:|
| 65,536 | 20.9 t/s | 27.6 t/s | 96.7% | 242 t/s |
| 131,072 | 20.2 t/s | 26.8 t/s * | 96.7% | 200–217 t/s |
| 262,144 | 19.7 t/s | – | – | – |

\* 128K decode was run-to-run unstable while host swap grew (26.8 down to
3.0 t/s over five identical runs); `DS4_QWEN4_PLE_PREFETCH_FULL=1` was steady
but slower at 21.9 t/s. Planned memory on that host: 44.79 GiB at 64K (stable
with a browser and a 1.6 GB VM alive), 47.00 GiB at 128K (works only with host
swap at 0), 51.42 GiB at 256K — not a daily setting. The
[Q2 comparison](../speed-bench/qwen38-q2down.md) was measured on an M3 Ultra
with 512 GiB. The larger `./download_model.sh qwen38-q4k` target remains
available for higher precision.


### PLE sidecar residency

The sidecar is prefetched whole when installed RAM covers the resident model
plus the sidecar plus a 16 GiB margin, and demand-paged otherwise; the startup
line reports which path was taken. Full prefetch makes the first cold-cache
prefill faster; demand paging keeps tight hosts lean. Two overrides exist:

* `DS4_QWEN4_PLE_PREFETCH_FULL=1` forces the full prefetch, `=0` forces
  demand paging. On a host already several GB into swap at 128K, forcing the
  prefetch traded speed for steadiness in the measurements above.
* `DS4_QWEN4_PLE_EVICT_TOKENS=N` (default off) bounds the demand-paged
  working set, which otherwise grows with tokens processed as touched sidecar
  pages accumulate: every N tokens the pages are dropped, so recurring n-grams
  re-fault. N=256 held residency near 130 MiB for roughly 3% decode cost on
  an M5 Max; N=1024 near 480 MiB for a noise-level cost.

Neither knob changes model output.

Use `ds4-agent` for native terminal and web tools. Its `bash` tool executes
commands; `google_search` and `visit_page` use a visible Chrome browser.
Starting that browser requires approval in interactive mode. A
`--non-interactive` run cannot approve browser startup; HTTP fetching through
`bash`/`curl` does not require Chrome. With `ds4-server`, the API client must
provide tool definitions, execute returned calls and send back their results.
See the [terminal and web smoke tests](../speed-bench/qwen38-real-tools.md).

Vision needs llama.cpp's mmproj encoder, which is a separate download:

```sh
./download_model.sh qwen38-vision
```

## Build your own GGUF

The current Q2 release adds [padded Q2_K down projections](../gguf-tools/README.md#experimental-padded-q2_k-down-projections)
to the calibrated IQ2_XXS gate/up build. It reduces the main file from 46.89
to 41.73 GiB and requires padded-down runtime support (commit `5bd8796` or
later on the `qwen3.8-flash-next` branch).

For the previous calibrated IQ2_XXS gate/up build with MXFP4 down projections, embedded
MTP and external PLE, use the [two-stage BF16 build](../gguf-tools/README.md#qwen38-iq2_xxs-experiment).
Its main GGUF is 50.34 GB; the Q4_1 PLE sidecar is required separately.
Context buffers, runtime allocations and resident PLE pages also consume
memory. File size alone does not establish whether it fits a 64 GB Mac.
See the [quality and memory measurements](../speed-bench/qwen38-iq2-quality.md)
for the comparison with Q4_K, Q4_0 and Q8.

For the padded Q2_K-down build, the 640 live activation values use a
768-value physical weight row; padding is skipped without changing the
activation layout. On M3 Ultra, IQ2_XXS and Q2_K expert decode specializes
the quantization and logical width and uses one row per SIMD group, with
eight SIMD groups per threadgroup for IQ2_XXS and sixteen for Q2_K.
Set `DS4_QWEN4_MOE_MV_SPECIALIZE=0` to
restore generic decode. `DS4_QWEN4_MOE_MV_NR` (1, 2 or 4) and
`DS4_QWEN4_MOE_MV_NSG` (1 through 16) allow explicit geometry comparisons.

For these low-bit expert formats on M3 Ultra, prefill uses 8-token tiles
through 512-token batches, 16-token tiles through 1024, and 32-token tiles
for larger batches. This avoids unused matrix products when few tokens
route to each expert. A final partial expert tile uses eight or sixteen
tokens when that is sufficient, with disjoint launches for the full tiles
and the remainder. The K accumulation order is unchanged.
`DS4_QWEN4_MOE_MID_NT=4 DS4_QWEN4_MOE_DOWN_NT=4` restores the original
prefill tile width; each override accepts 1, 2 or 4 groups of eight tokens.
Also set `DS4_QWEN4_MOE_TAILS=0` to disable the smaller remainder tiles.
Other devices and quantizations retain their previous default geometry.
See [the padded Q2 speed measurements](../speed-bench/qwen38-q2-speed.md)
and [the second optimization round](../speed-bench/qwen38-q2-round2.md)
for end-to-end timings and numerical checks.

Ordinary single-token decode on M3 Ultra also combines the residual update
with the following F16 hyper-connection normalization, and dispatches the two Q8 GDN
input projections together. Both retain the original FP32 reduction order.
Set `DS4_QWEN4_DECODE_FUSIONS=0` to restore the separate operations;
MTP sessions retain the separate operations, as do other devices by default. See the
[ordinary-decode fusion measurements](../speed-bench/qwen38-q2-decode-fusions.md)
for the measured gain, exact-logit checks, and rejected experiments.

On Apple M5, routed-expert prefill GEMMs specialize the bound quantization
by default, single-token Q4_K gate/up decode uses one row per SIMD group with
four groups per threadgroup, and MXFP4 routed-down decode rows specialize the
quantization with sixteen SIMD groups per threadgroup. The same overrides
apply (`DS4_QWEN4_MOE_MM_SPECIALIZE`, `DS4_QWEN4_Q4K_MID_NR`/`_NSG`,
`DS4_QWEN4_MOE_MV_SPECIALIZE`/`_NSG`); the M3 Ultra ordinary-decode fusions
stay off on M5, where they measured slower. See
[the M5 Max round-one report](../speed-bench/qwen38-m5-round1.md) for the
single-engine A/B measurements, exact-logit checks and the sweep through 128K.

The second M5 round adds, still bit-exact: narrow single-token F16/F32
matvecs (hyper-connection low-rank down, router) launched one row per SIMD
group (`DS4_METAL_PLAIN_MV_NR0`), 1024-thread GDN front threadgroups
(`DS4_QWEN4_GDN_FRONT_THREADS`), expert-major prefill tile order
(`DS4_QWEN4_MOE_MM_ORDER`), a decode indexer scorer with staged queries and
vector key loads, a one-thread-per-dim split-attention merge and an exact
tile-max prefiltered top-k for long contexts (`DS4_QWEN4_IDX_SCORE_VEC`,
`DS4_QWEN4_ATTN_MERGE_WIDE`, `DS4_QWEN4_IDX_PREFILTER`), and, on every
device, an MTP rejection rollback that swaps the live and snapshot state
buffers instead of copying them (`DS4_QWEN4_MTP_SWAP_RESTORE=0` restores the
copy). See [the round-two report](../speed-bench/qwen38-m5-round2.md).

Round 3 on M5 Max (`speed-bench/qwen38-m5-round3.md`) turned the round-2 fast-math lesson into a method: dump the runtime library to AIR, replicate each hot loop's *compiled* op order with `#pragma clang fp reassociate(off)` / `contract(off)`, then find the backend's fusion form by a fixture sweep (no fma for the single-row mixer, fma chains for the paired mixer and the Q8 matvec, a mixed form for Q4_K, fma inner sums for MXFP4). With that, register-prefetch rewrites became exact: the F16 hyper-connection gate/mix (+2.05% plain decode; `DS4_QWEN4_HC_MIX_PREFETCH`), its paired MTP-verify kernel (+1.1 to +1.3% MTP), and the MXFP4 routed down rows (+0.4%; `DS4_QWEN4_MOE_DOWN_PREFETCH`). The one-row Q4_K gate/up geometry now also covers the two-row MTP passes (+1.0 to +2.8% MTP; `DS4_QWEN4_Q4K_MID_NR`/`_NSG`). Rejected with numbers in the report: Q8 dense and Q4_K mid prefetches, row-block-major tensor-op GEMM order, and a prefill-row prefilter. Tooling: `qwen38_decode_variant_bench --mtp`, `qwen38_mtp_compare.py --baseline-env`, `DS4_METAL_MV_EXT_NSG`. Opt-in, off by default: the MTP draft head can score a vocabulary subset (`DS4_QWEN4_MTP_DRAFT_ROWS`, `DS4_QWEN4_MTP_DRAFT_VOCAB`; MTPLX's FR-Spec idea), +1.8 to +2.8% MTP with identical outputs but acceptance counters that can move. See [the round-three report](../speed-bench/qwen38-m5-round3.md).

Round 4 on M5 Max focused on prefill (`speed-bench/qwen38-m5-round4.md`). A production-shaped microbench with skip-variants of the routed tile kernels shows the fp16 simdgroup matmul loop at about 80% of the tile time, so load-side rewrites (coalesced epilogue, double-buffered staging, 32-wide K steps, 64-row tiles, half activation operands) moved little or lost; the tile geometry did pay: 64-token gate/up tiles beside the existing down NT=8 are the M5 default for the Q4_K / MXFP4 pack at batches of 4096 tokens or more (`DS4_QWEN4_MOE_MID_NT` / `DS4_QWEN4_MOE_DOWN_NT` override), +2.6% at 8192-token chunks and +2.9% at 4096-token chunks with logits exact. A register-query attention variant, 64-wide K steps for the tensor-op GEMMs and an eight-row GDN scan were exact but not faster. The remaining large prefill lever, routed experts on Metal 4 tensor ops, is a drift-class change left for a decision.

Round 5 on M5 Max took that lever, opt-in (`speed-bench/qwen38-m5-round5.md`). The routed Q4_K gate/up and MXFP4 down tiles run on the Metal 4 tensor ops (`kernel_qwen4_moe_mm_mid_nax[64]` / `_down_nax[64]`): the same dequantized halves are staged, only the cooperative matmul's accumulation order differs, so logits drift instead of matching bit for bit. `DS4_QWEN4_MOE_MM_NAX=1` selects 32-token tiles, `=2` 64-token tiles (bit-identical to each other), off by default. Measured with the chunk-interleaved harness in its new `--tolerate-drift` mode: +18.4% at 8K-40K with 32-token tiles, +3.4% more with 64-token tiles, +19.7 to +20.7% at 96K-128K (unthrottled repeats; +17 to +23% aggregates); the routed tile microbench halves (16.1 ms to 7.5 ms). The drift is a fixed-size perturbation (mean |delta logit| 0.09-0.15 on the last row at 2K, 32K and 128K, top-1 agreeing) and the BF16-reference fixture (99 cases) moves within noise: target NLL 0.20505 to 0.20503, logprob MAE 0.0468 to 0.0466, top-1 agreement 96.34% to 96.25%, first-token matches 86/99 both ways, two orders of magnitude under the Q4_K to Q8 spread. Double-buffered staging and relaxed-precision accumulation did not pay; a register-resident SiLU epilogue did. Skip-variants then put the tile at 41% tensor op, 35% weight staging, 18% activation staging, nearly serial, and three bit-identical staging changes followed (output hashes unchanged): a pre-rounded half activation operand (one conversion pass per call), a register prefetch of the next K step's raw weight words, a 32-token tail tile for the 64-token kernel, and the half copy of `mid` written by the mid tiles instead of a conversion pass; together the tensor path reaches **+32.6% at 8K-40K** (1186 -> 1572 tok/s) and +30% at 96K-128K over the simdgroup tiles. Exact prefetches of the attention K/V gather and of the GDN scan operands were tried and rejected (noise). `test_qwen4_kernels` bounds both tile widths against the simdgroup tiles and prints their output hashes.

An accuracy follow-up (`speed-bench/qwen38-m5-round5-accuracy.md`) added operand-precision
levels to the opt-in tensor tiles: `DS4_QWEN4_MOE_MM_NAX=3/4` stage the activation operand at
full float precision, and `=5/6` compensated tiles stage the half rounding residual of the
operand beside it and run the tensor op twice, so the activation operand enters at ~2^-22
relative instead of 2^-11. Both reach the same tile accuracy — mean |error| versus a double
reference drops 26% (mid) and 31% (down) against the simdgroup tiles — but the fp32 path gives
up the whole prefill win (the unit's fp32 tiles run ~2.4x slower), while the compensated path
keeps most of it: +22.0% at 8K-40K versus +25 to +33% for `=2`, with the best fixture NLL of
all paths (0.20430 versus 0.20505 simdgroup and 0.20503 `=2`) on the 99-case BF16 fixture.

Level 2 (the uncompensated 64-token tiles) is the promoted default on devices with the
Metal tensor API (`DS4_QWEN4_MOE_MM_NAX` unset selects it): +48% at 8K-40K and +51% at
96K-128K prefill over the simdgroup tiles, with the same operand rounding and a fixture NLL
within noise of them. The compensated level `=5` keeps the best absolute NLL and stays one
environment variable away; `=0` restores the simdgroup tiles, and decode is unaffected at
every level (the decode MoE path never dispatches these tiles).

The Q2 pack (IQ2XXS gate/up, Q2_K down) runs the same level-2 default: +29% at
8K-40K and +36% at 96K-128K prefill (microbench tile time 19.33 -> 7.38 ms), again
with simdgroup-identical operand rounding and accumulation-order-only drift (NLL
+0.0007, one near-tie top-1 flip per 1000 tokens, 100% top-5 containment).
`DS4_QWEN4_MOE_MM_NAX=0` still restores the simdgroup tiles for either pack.

The older recipes below keep PLE inside the main GGUF, so their file sizes
are not directly comparable with the external-PLE builds.

Build the
GGUF from the Hugging Face checkpoint with the converter in `gguf-tools/`;
the stock `ggml-org` Q8_0 GGUF also loads once its two parts are merged with
`llama-gguf-split --merge`:

```sh
python gguf-tools/qwen4_exp_convert.py --src /path/to/Qwen3.8-Flash-Next \
  --out gguf/Qwen3.8-Flash-Next-Q8.gguf --outtype q8_0            # about 192 GB
python gguf-tools/qwen4_exp_convert.py --src /path/to/Qwen3.8-Flash-Next \
  --out gguf/Qwen3.8-Flash-Next-MXFP4.gguf --outtype q8_0 --experts mxfp4  # about 126 GB
llama-quantize --allow-requantize --tensor-type hc_=f16 --tensor-type ffn_gate_exps=Q4_K \
  --tensor-type ffn_up_exps=Q4_K --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-Q4K.gguf Q8_0  # about 124 GB
llama-quantize --allow-requantize --tensor-type hc_=f16 --tensor-type ffn_gate_exps=Q2_K \
  --tensor-type ffn_up_exps=Q2_K --tensor-type ffn_down_exps=MXFP4 \
  --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-Q2K.gguf Q8_0   # about 83 GB
llama-quantize --imatrix imatrix.gguf --allow-requantize --tensor-type hc_=f16 \
  --tensor-type blk.48.ffn_gate_exps=MXFP4 --tensor-type blk.48.ffn_up_exps=MXFP4 \
  --tensor-type ffn_gate_exps=IQ2_XXS --tensor-type ffn_up_exps=IQ2_XXS \
  --tensor-type ffn_down_exps=MXFP4 --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-IQ2.gguf Q8_0   # about 77 GB
```

The Q8 file keeps every weight at 8 bits except the QSA indexer projections,
which stay at the released BF16. The MXFP4 file keeps the routed experts in
native MXFP4 blocks, the Q4K file requantizes the expert gate/up projections
to Q4_K, and the two-bit files take them to Q2_K or, with an importance
matrix from `llama-imatrix`, IQ2_XXS; the 640-wide expert down projections
are too narrow for 256-value blocks and go to MXFP4. The first matching
`--tensor-type` wins, and the MTP block's experts (`blk.48`) stay MXFP4
because an importance matrix collected with llama.cpp never sees them. Keep
the `hc_=f16` override: the Q8_0 base type would otherwise requantize the
hyper-connection mixers, which slows prefill. Q2_K and IQ2_XXS trade size
against quantization error; compare measured quality for the specific recipe.
These inline-PLE builds need additional memory beyond the file size for
context and runtime buffers.

```sh
./ds4 -m gguf/Qwen3.8-Flash-Next-Q8.gguf --ctx 32768
./ds4 -m gguf/Qwen3.8-Flash-Next-Q8.gguf --mtp --temp 0
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q8.gguf --ctx 65536 --kv-disk-dir ~/.ds4/server-kv
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q8.gguf --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
./ds4-agent -m gguf/Qwen3.8-Flash-Next-MXFP4.gguf
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q4K.gguf --mtp --mtp-exact-sampling --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
```

The MTP block is inside the same GGUF; `--mtp` enables it and speeds up
greedy decoding. At non-zero temperature it keeps target-matching greedy
drafts, like the GLM path, which skews sampled output toward the greedy
choice; `--mtp-exact-sampling` preserves the ordinary sampling distribution
at a smaller speedup. Prefill runs in 8192-token chunks
(`DS4_QWEN4_PREFILL_CHUNK` overrides it; the transient buffers scale with the
chunk size). For A/B checks, `DS4_QWEN4_NO_FUSE=1` selects the unfused decode
kernels, `DS4_QWEN4_NO_IDX_SELECT=1` the argsort top-k and
`DS4_QWEN4_NO_ATTN_MM=1` the per-token attention kernel for prefill batches;
`DS4_QWEN4_TIMING=1` prints stage timings.

MTP decode batches the accepted token's predictor history update with its next
draft. Two-token hyper-connection mixers share weights, the 128-wide GDN scan
processes four value rows per SIMD group, and Q4_K expert gate/up projections
share input loads. These paths are enabled by default. For individual A/B
checks, set `DS4_QWEN4_NO_MTP_BATCH=1`, `DS4_QWEN4_NO_HC_PAIR=1`,
`DS4_QWEN4_NO_GDN_R4=1` (decode only), or `DS4_QWEN4_NO_Q4K_MID=1`.
When the predictor is asked only for its next token, it selects that token
on the GPU and reads back one index. Full predictor logits remain available
to callers that request them. `DS4_QWEN4_MTP_GPU_ARGMAX=0` restores the
full-vocabulary readback and CPU selection for comparisons.
See the [predictor token-selection measurements](../speed-bench/qwen38-q2-mtp-argmax.md)
for numerical checks and timings.
At the 2560-wide, rank-320 shape on M3 Ultra, paired mixers use sixteen
SIMD groups per threadgroup to share activated inputs across more rows.
`DS4_QWEN4_HC_PAIR_NSG=4` restores the previous grouping; values from 1 to
16 are supported. This override applies only to the two-token paired mixer.
See the [Q2 MTP measurements](../speed-bench/qwen38-q2-mtp.md) for the
measured gain and parity checks.
The GDN verification layout uses eight SIMD groups on M3 Ultra;
`DS4_QWEN4_GDN_NSG` overrides the decode group count from 1 to 8.
GDN prefill uses four SIMD groups on M3 Ultra for batches of at least 8192
tokens with 16 key heads, 48 value heads and head width 128.
`DS4_QWEN4_Q4K_MID_NR` selects 1 or 2 rows per SIMD group for Q4_K gate/up.
One- and two-token decode on M3 Ultra default to one row and eight SIMD groups;
other batches and devices retain two rows and two SIMD groups.
`DS4_QWEN4_Q4K_MID_NSG` overrides the group count from 1 to 8. Setting
`DS4_QWEN4_Q4K_MID_NR=2` restores the former layout without an NSG override.
See [the MTP decode benchmark](../speed-bench/qwen38-mtp-decode.md) for measurements
and reproduction commands. The [earlier 262K comparison](../speed-bench/qwen38-262k-compare.md)
found decode drift in the previous Q4_K optimization. The current gate/up kernel
preserves the original accumulation order. Before the main rebase, it matched
all 896 recorded FP32 decode logit vectors through 262K; see the
[non-MTP benchmark](../speed-bench/qwen38-nonmtp-round3.md) for those measurements.
The [rebase validation](../speed-bench/qwen38-main-rebase.md) matched 256 full
vectors through 8K and preserved the tested MTP outputs; it did not rerun 262K. The
[previous round](../speed-bench/qwen38-nonmtp-decode.md) records the numerical correction.

The trunk submits commands after two layers to overlap GPU execution with host
encoding; `DS4_QWEN4_FLUSH_LAYER=0` restores one submission. On M3 Ultra, MoE
prefill launches use tile caps of 8, 16, and 32 at batch sizes below 4096, from
4096, and from 8192 tokens. `DS4_QWEN4_MOE_MID_TILES` and
`DS4_QWEN4_MOE_DOWN_TILES` override these caps from 1 to 32.
Hyper-connection normalization reuses each stream's RMS for batches of at least
8192 tokens on M3 Ultra with embedding size 2560, four streams and four injection
outputs. It preserves the original chunk reductions. `DS4_QWEN4_HC_NORM_REUSE=0`
disables reuse; `=1` enables it for batches larger than two tokens.
Separate scratch for each chunk removes redundant overwrite barriers without
changing the reduction order. See [the latest MTP optimization report](../speed-bench/qwen38-mtp-round3.md)
for full-vocabulary numerical checks and prefill/decode measurements through 262K.

`--batched-session N` keeps N sessions resident; their decode steps run one
after another rather than as one grouped batch, so it buys concurrency, not
throughput. Thinking is on by default with the model's `xhigh` reasoning
instruction; `reasoning_effort` `low`, `medium` or `xhigh` selects the model
card's levels (`chat_template_kwargs` with `enable_thinking` and
`reasoning_effort` is accepted too), the server's `qwen3.8-flash-next-chat`,
`qwen3.8-flash-next-nothink` and `qwen3.8-flash-next-no-think` aliases (a
`qwen/` prefix also works) disable thinking and
`qwen3.8-flash-next-reasoner` forces it. Tool
calls use the model's native `<tool_call><function=...><parameter=...>` format
in both the server and the agent. Disk KV checkpoints and live prefix reuse
work as for the other models; the recurrent state travels with the checkpoint.
Rewinding to an arbitrary earlier position resets that state and replays the
kept token prefix on the next evaluation. A shorter prompt is prefilled again.

The model's native window is 262144 tokens. For longer prompts set
`DS4_QWEN4_YARN_FACTOR=4`, which
applies the static YaRN scaling from the model card, up to about 1M tokens;
use a factor of 2 for 512k. Static YaRN costs a little accuracy on short
prompts, so leave it off otherwise.

Images go through the model's Qwen3-VL vision tower. Download llama.cpp's
mmproj file with `./download_model.sh qwen38-vision` (it comes from
`ggml-org/Qwen3.8-Flash-Next-GGUF`; alternatively run
`convert_hf_to_gguf.py --mmproj --outtype f16` on the checkpoint) with
`--vision` and send `image_url` parts as usual. Each image is resized to
multiples of 32 pixels within 64 to 1024 tokens (`DS4_QWEN4_IMAGE_MAX_TOKENS`
raises the cap), encoded on the GPU, and takes the model's 3D rope positions;
live KV reuse keys on the image fingerprints. `make test-qwen4-vision`
compares the tower with the Hugging Face implementation using the same GGUF
weights, dequantized to float32. It separately reports GGUF-versus-original
checkpoint quality and Metal-versus-original agreement. Both comparisons use
the unchanged minimum per-token cosine threshold of 0.99; implementation
parity gates the default exit status. Add `--require-quality` to also fail on
GGUF-versus-original quality loss. A passing implementation check alone does
not mean the quantized encoder matches the original checkpoint.

For a small suite including OCR, diagrams, a photograph, and resizing:

```sh
make tests/test_qwen4_vision
uv run --with numpy --with torch --with torchvision --with pillow \
  --with safetensors --with transformers --with gguf tests/qwen4_vision_ref.py \
  --snapshot /path/to/HF-checkpoint \
  --mmproj gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image tests/vision-fixtures/qwen38/orbit.png \
  --image tests/vision-fixtures/qwen38/maple.png \
  --image tests/vision-fixtures/glm53/diagram.png \
  --image tests/vision-fixtures/glm53/text.png \
  --image tests/vision-fixtures/glm53/earth.jpg \
  --image tests/vision-fixtures/glm53/screenshot.png \
  --json-report /tmp/qwen38-vision-parity.json
```

Transformers must include `qwen4_exp`; the reference runs on CPU. Repeat
`--image` to add cases. The original single-image `--out` embedding dump is
still supported. Metric checks without model weights run with
`uv run --with numpy python -m unittest discover -s tests -p test_qwen4_vision_ref.py`.
JPEG decoder agreement with Pillow, including progressive scans, subsampling,
and image edges, is checked separately with
`uv run --with numpy --with pillow python -m unittest discover -s tests -p test_jpeg_decode.py`.

To check CLI `/read` image turns and text follow-ups with ordinary and MTP
decode, use two different PNG or JPEG images with the model-backed regression:

```sh
make ds4
uv run tests/test_qwen4_cli_vision.py --model /path/to/main-with-mtp.gguf \
  --ple /path/to/ple.gguf --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image /path/to/first.png --image /path/to/second.png
```

The test checks that all four turns complete in each mode, including errors
that the interactive CLI can report without a nonzero process exit. It saves
the responses and diagnostics for inspection; it does not grade image content.

Metal only for now. The Metal graph accepts Q8_0, Q4_0, F16, BF16 and F32
dense weights, Q8_0/MXFP4/Q4_0/Q4_K/Q2_K/IQ2_XXS experts, F16/F32/Q8_0
hyper-connection mixers and a Q8_0/Q4_0/MXFP4/F16/F32 n-gram table.
Multi-node tensor parallelism is not implemented yet.

## Qwen3.8 Flash Next (campaign branch merge)

This branch carries the qwen4-exp engine adoption merge. The native fast-pack
qwen4 engine developed during the campaign is superseded; its history, notes
and performance handoff live in [QWEN38_FLASH_NEXT.md](../QWEN38_FLASH_NEXT.md)
and [QWEN38_PERF_HANDOFF.md](../QWEN38_PERF_HANDOFF.md), and its source is kept
in-tree (`ds4_qwen4.c`/`ds4_qwen4.h`, unreferenced by the build).

The published Q4_K-imatrix build combines the main and MTP weights and uses
the external PLE sidecar, as shown in [Download and run](#download-and-run).
To select it explicitly instead of using the default symlink:

```
./ds4 -m gguf/Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf \
      --ple gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --metal --mtp
```

Models still in the old fast-pack format can be converted with
`gguf-tools/qwen4_pack_to_qwen4exp.py` and then run the same way.
