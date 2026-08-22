# Decode campaign — handoff (session of Aug 21, M3 Ultra, round 2)

Goal: push greedy decode past **45 tokens/s** on the MXFP4 `ds4flash.gguf`
model, bit-exact. **Reached: 45.5 t/s interleaved (45.47/45.55), all
transcripts md5 `db0c504c…`, `make test` 44/44, harness bit-identical.**
Round-1 state was 43.2–44.0 t/s; round 2 added +2.7% (attention parity) and
+1.75% (greedy chain decode), both validated per the campaign protocol.

## Round-2 changes (working tree; commit state at bottom)

1. **Raw-layer gathered attention** — `n_comp == 0` decode layers (ratio 0/128)
   now use the gathered path (fused staging + packed32 reduce + fused inverse
   RoPE) instead of the five-dispatch raw path. Found via commit-only stage
   counters: the raw path cost ~65 µs vs ~32 µs per layer, invisible in the
   averaged ledger ("attention core 44.3 µs flat"). ~0.7 ms/token at short
   context. Rollback envs: `DS4_METAL_DISABLE_DECODE_RAW_GATHERED_ATTN`,
   `DS4_METAL_DISABLE_DECODE_RAW_PACKED32`. Also fixed the staging wrapper's
   `n_comp == 0` guard in `ds4_gpu_encode_flash_kv_stage_f16`.
2. **Greedy chain decode** — `metal_graph_greedy_chain` (ds4.c, engaged from
   `generate_metal_graph_raw_swa` only): GPU argmax writes the next token id
   into a device ring; the next token's embedding gathers it from the ring
   (the by-value embed and the batched embed use the same get_rows/repeat
   kernels — bit-identical). Encode runs two tokens ahead of the host's
   confirm cursor; the host only lags (one MTLSharedEvent wait per token) to
   print and stop-check. Removes the per-token `waitUntilCompleted` + 517 KiB
   logits readback + CPU argmax boundary (~0.5 ms/token). Kill switch:
   `DS4_DISABLE_GREEDY_CHAIN=1`. Diagnostics: `DS4_GREEDY_CHAIN_DEBUG`,
   `DS4_GREEDY_CHAIN_DUMP_IDS`, `DS4_GREEDY_CHAIN_VERIFY`.
   **Hash-layer gotcha that cost an hour**: the first `DS4_N_HASH_LAYER`
   layers route experts by token id (`ffn_gate_tid2eid`); the select kernel
   already supports a device-resident token (`use_token_buffer` in
   `kernel_dsv4_router_finalize_one`), plumbed via
   `ds4_gpu_router_select_tensor_devtoken` + `g->chain_token_view`. The
   host-side `metal_graph_decode_set_hash_selected_override` is skipped in
   chain mode — the resident fixed-route MoE never consumes it (the override/
   readback blocks live under `use_selected_slots`, all gated on
   `g_ssd_streaming_mode`). Feeding token=0 instead was the one divergence:
   deterministic drift from layer 0, tipping an argmax 33 tokens in.
   Symptom signature if it regresses: transcripts match for N tokens then
   diverge deterministically.

## Verified numbers (interleaved CLI, `-c 8192 -n 128 --temp 0`, lighthouse prompt)

| variant | t/s | md5 |
|---|---|---|
| round-1 baseline | 43.40–43.52 | `db0c504c…` |
| + attention parity only | 44.59–44.67 | `db0c504c…` |
| + greedy chain (current) | **45.47–45.55** | `db0c504c…` |

Long-context (2K prompt): 44.12 chained vs 43.40 classic. 1024-token run and
an early-stop prompt: md5-identical. Harness (prefix 2048): 529 rows /
68,389,120 logits / 528 ids bit-identical, 43.18 vs 43.10 t/s (only layers
0–1 qualify at long prefix). SSD streaming decode smoke: OK (chain declines,
classic path taken).

## Remaining headroom (re-estimated)

GPU busy ≈ 21.85 ms/token now; wall ≈ 21.96 ms. The boundary is ~0.1–0.15 ms
(argmax→embed dependency + drain). Still open, expectations lowered by the
round-1 evidence: HC epilogue tail fusions (~0.3 ms), KV-staging direct read
(~0.25 ms; a prior attempt diverged — the addressing facts are in README's
stage-counter section and the code comments at cpy.metal:147), MXFP4
concurrent shared-expert stream (machinery exists hard-gated to IQ2,
ds4_metal.m:9286; medium confidence, and the "more parallelism throttles this
GPU" lesson applies). Each could add ~0.2–0.6 t/s; none is needed for 45.

## Watch item

The balanced harness failed twice at `step=0 variant=control` ("metal decode
failed") with an early round-2 binary, then passed 9+ consecutive runs with
semantically identical code. Unexplained; both failures immediately followed
sustained benching (the documented thermal-transient window). If it recurs,
reproduce with `--warmup 1 --tokens 2` and capture full stderr.

## Tools (unchanged from round 1)

- Commit-only stage profiler: `DS4_METAL_DECODE_STAGE_PROFILE=1
  DS4_METAL_STAGE_COUNTERS=1 ./ds4 …` (trustworthy; end-and-wait inflates).
- Balanced A/B harness: `make metal-decode-schedule-bench && ./speed-bench/
  metal_decode_schedule_bench -m ds4flash.gguf --candidate-env NAME
  --include-selection --tokens 512` (aborts unless bit-identical).
- Transcript md5 oracle: `./ds4 -m ds4flash.gguf -p "Write a short story
  about a lighthouse keeper." -c 8192 -n 128 --temp 0 | md5` → `db0c504c…`.
- Tensor dump bisect: `DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/x
  DS4_METAL_GRAPH_DUMP_NAME=<stage> DS4_METAL_GRAPH_DUMP_LAYER=N
  DS4_METAL_GRAPH_DUMP_POS=P ./ds4 …` (synchronizes and dumps; comparing
  classic vs chain dumps located the hash-router divergence in one pass).
- One `ds4` instance at a time; idle ~60 s after heavy runs (transient
  2–10× slowdown recovers by itself).

## Round-1 closed avenues still stand

Eight bit-exact kernel variants (all ≤0.03%), DSpark strict/non-strict
(43.07 / peaks 40.2), three microbatch increments, split-schedule sweep
(default 4 optimal), readback-bubble premise (disproved). Details in
speed-bench/README.md. The round-1 verdict "bit-exact and >45 t/s are
jointly infeasible" was wrong: the ledger's averaged `attn_inv_rope` line
hid the 65/32 µs parity alternation, and the boundary's "wake/launch
latency" was recoverable by keeping the token id on-device.

## Round-3 addendum (Aug 22): session chain + headroom list exhausted

- **Session chain decode**: `ds4_session_eval_chain_greedy` (ds4.c) reuses
  `metal_graph_greedy_chain` for session API callers; `ds4-eval` decodes in
  bursts capped to stay out of the think-close controller window. Bit-exact
  (traces identical), eval decode **44.5 → 45.2 t/s**. Kill switch
  `DS4_DISABLE_GREEDY_CHAIN=1` covers both CLI and session paths.
- **MoE down-sum6+HC4 tail fusion**: landed bit-exact, speed-neutral,
  default on (rollback `DS4_METAL_DISABLE_DECODE_MOE_HC_FUSION`).
- **KV-staging direct read**: implemented bit-exact (wrap-safe) but −7%
  (per-head F32 re-read/re-convert amplification); landed gated OFF
  (opt-in `DS4_METAL_ENABLE_DECODE_RAW_DIRECT_KV`).
- The round-2 "remaining headroom" list is now closed out: both quantified
  items measured (neutral / negative), the concurrent shared-expert stream
  stays contraindicated by the throttling evidence. Wall−GPU gap is ~0.1
  ms/token; the next real gain must come from the MoE or attention core
  itself.

## Round-4 addendum (Aug 22, branch perf/decode-50tps): 50 t/s campaign

Executing speed-bench/ROADMAP-50TPS.md. Ledger baseline re-verified on the
chain path (all md5 `db0c504c…`, `make test` 44/44):

- **Item 0 (head attribution tooling)**: `logits` output-stage boundary +
  per-token stage-counter reporting inside the greedy chain (report at each
  token's confirm wait, encode-ahead samples compacted; chain now engages
  under commit-only counters). Head = **0.90 ms/token**: Q8_0 logits matvec
  0.769 (at the wall), GPU argmax 0.086 (full bitonic sort for top-1 — a
  dedicated reduce is ~10× cheaper; A5's ceiling), HC collapse 0.045.
  Ledger fully closes: 6.86 MoE + 5.17 q_path + 4.79 attn_output + 1.92
  router + 1.37 attn core + 1.69 HC pre×2 + 0.90 head = 22.7 ms counter
  total (wall 21.96).
- **P2 (prefill stage counters)**: commit-only counters ported to prefill —
  `lNN:stage` labels, no per-layer drain under counters, per-chunk
  reset/report in both prefill schedules. Settled: routed MoE at n=3092
  runs 20.6 TFLOPS effective (compute-bound estimate confirmed); per-layer
  cost ≈ 8.5 ms fixed + 36 µs/token marginal; small-N fixed map at n=6:
  routed_moe 1.82 (≈3× its byte floor), hc_pre ×2 1.70, output_proj 0.82
  ms/layer. Also found: the one-shot CLI passes its progress callback as
  `display_progress` unconditionally, so the 43-drain `callback_split`
  schedule hits every ≥32-token one-shot prefill regardless of TTY (P1's
  exact target).
- **A1 (HC producer tail)**: MEASURED NEGATIVE, reverted. Parallel
  4-lane Sinkhorn comb + shrunk completion fences: (a) not bit-exact —
  the serial comb's four unrolled rows compile to row-position-dependent
  reduction trees; per-lane rewrite lands rows 0/3 one ulp off
  (dump-bisected: mixes and collapse bitwise identical, comb differs);
  (b) speed-neutral anyway: 45.90/45.87 vs 45.92/45.91 t/s interleaved.
  The stage's ~19.6 µs/call is dispatch floor + phase streams, not the
  tail. Do not retry the tail; any real hc_pre win must attack the dispatch
  boundary itself (fusion) or the phase-1/2 streams.
- **A3 (defer Q head norm+RoPE into the packed32 attention consumer)**:
  BLOCKED BY FAST-MATH, reverted. What was proven along the way:
  (a) the plumbing is bit-exact — skipping the standalone norm+RoPE at the
  q_b site and re-running it at the attention site (or the deferred
  prologue's norm tree emulation: 4 simdgroups + zero-padded 32-slot plane
  + 32-lane simd_sum) reproduces q bitwise on all 64 heads;
  (b) under DS4_METAL_MATH_SAFE=1 the fully fused kernel matches the
  rollback transcript exactly — the deferral logic is correct;
  (c) under the production fast-math library the YaRN rope tail
  (theta blend `interp*(1-mix)+extrap*mix`, mscale, rotation mul-adds)
  contracts differently inside the 64-TG attention kernel than inside the
  64-head standalone kernel — 1-ulp drifts on compressed layers, and every
  source-level pin (strict/fma/lerp forms, noinline shared helper) either
  misses by ~26 elements or cascades the whole kernel's contraction,
  including the norm dots. Reproducing contraction-sensitive arithmetic
  across kernel contexts is not source-controllable here. Rule of thumb for
  future fusions: only contraction-free code (explicit reduction trees,
  builtins, single muls) can move between kernels bit-exactly.
  Timing signal from the (divergent) fused build was also not promising
  (43.77 vs 45.5 t/s single-run): the in-place prologue + 3 extra barriers
  may not even beat the removed dispatch.
- **A4 (router_project_select port to pre-M5/MXFP4)**: CLOSED BY ANALYSIS,
  no code. The fused select computes no shared-expert work; on MXFP4 decode
  the FFN block is 5 dispatches either way, and the port would drop the
  shared expert's free-ride overlap inside the router matvec AND disqualify
  the landed sum6+HC4 tail fusion (`fuse_moe_down_hc` gates on
  `router_shared_done != 0`). The M5 win requires the IQ2-only parallel-FFN
  concurrent encoder, which MXFP4 does not have. Dispatch-neutral by direct
  reading; not measured.
- **P1 (prefill progress drains)**: LANDED. The callback_split schedule
  (display_progress != NULL && n_tokens >= 32) forced 43 per-layer drains
  per chunk just for the progress bar — and the one-shot CLI passes its
  progress callback as display_progress unconditionally, so even piped
  one-shot prefills paid them. Now: commit-only per-layer flush +
  completion-handler progress (`ds4_gpu_flush_commands_progress`) for
  n_tokens <= 2048; >2048 chunks keep draining (transient-memory bound).
  Zero arithmetic change (md5 ed17c76a… long-prompt identical, db0c504c…
  holds, make test OK, SSD streaming smoke OK). +2.7% prefill at ~250
  tokens, +1.8% at 637, control unchanged at 3092. Rollback
  `DS4_METAL_DISABLE_PREFILL_FLUSH_PROGRESS=1`.
- **A2 (MXFP4 dequant constant-space LUT)**: MEASURED NEGATIVE, gated OFF.
  Bit-identical (harness 529 rows/68.4M logits/528 ids exact) but −2.7%
  decode (42.41 vs 43.59 t/s): divergent constant-cache gathers lose to the
  threadgroup-staged LUT.  Other prescribed variants are structurally
  blocked (17-byte block stride ⇒ unaligned loads; register select chains
  raise ops/byte).  Opt-in kept: `DS4_METAL_ENABLE_MXFP4_CONST_LUT`.
  The routed-MoE dequant stays issue-rate-bound with no safe lever found —
  A2's 0.5–1.5 ms is not in this loop.
- **P5 (chunk sweep)**: DONE, no change. `metal_prefill_variant_bench` grew
  `--prefill-chunk`. At 8192-token prefix (balanced, 8 runs): 615.9 t/s at
  chunk 2048, 640.4 at 4096, 645.0 at 8192 — 4096 is near-optimal (8192's
  +0.7% not worth 2x transients). At 32768 the blocks skew with thermal
  soak; cross-config still shows no big lever. Keep 4096.
- **P3 (PSO pre-warm)**: CLOSED BY MEASUREMENT, no warmup-list change. New
  env-gated diagnostic `DS4_METAL_LOG_PIPELINE_CREATES` (logs every lazy
  PSO creation with its cost) shows a prefill lazily compiles 108 pipelines
  for **~4.2 ms total**, once per process — harness/bench warmups already
  absorb it, and a hardcoded pre-warm list would silently drift from the
  encode path for a gain far below first-turn noise. Diagnostic kept;
  nothing to warm.
- **P6 (long-context prefill decay)**: PROFILED with the P2 counters. A
  ~59k-token prompt (15 chunks of 4096) grows per-chunk GPU busy
  6.81 → 10.51 s monotonically. The decay is NOT the attention core
  (1.29x): it is the indexer machinery, which is O(n_comp) per chunk —
  `score` 95→1782 ms/chunk (18.7x), `compressor` 0→1700, `indexer_setup`
  102→1142, `topk` 20→275 (chunk-0 MoE-inflated ~25% by first-touch page
  faults; steady-state compute stages are flat). Target for a follow-on:
  the indexer score pass (`ds4_gpu_indexer_scores_batch_tensor` →
  kernel_dsv4_indexer_scores_tiled_*) at long prefixes — its GEMM shape
  versus n_comp, and the compressor's per-chunk work at long context.
