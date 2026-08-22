# Roadmap to 50 t/s decode + faster prefill — execution plan for a dedicated branch

Scope decisions already made: decode follows **Track A (bit-exact kernel work,
current model)**; CPU-parallel and prefill were analyzed and are folded in
below. Execute on a dedicated branch in a fresh session. Campaign protocol
applies to every item: commit-only stage counters → md5 oracle
(`db0c504c…` n=128, `aabbc1d0…` n=512) → `metal_decode_schedule_bench
--candidate-env … --include-selection --tokens 512` (aborts unless
bit-identical) → `make test` 44/44 → interleaved t/s A/B; every variant lands
behind a `DS4_METAL_DISABLE_*` rollback env; one ds4 instance at a time,
`sleep 60` after heavy runs.

## Part 1 — Decode to 50 t/s (Track A, approved)

Budget: 21.85 ms GPU-busy/token → need −2.2 ms. Ledger: routed_moe 6.89 ms
(issue-rate-bound MXFP4 dequant, 550–577 GB/s effective), q_path 5.21 (q_b at
the wall; norm/rope 21 µs/layer latency), attn_output 4.79 (at the wall),
router 1.95, HC pre ×2 1.61 (~90% latency), attn core 1.42, head ~0.75–0.9,
wall−GPU gap 0.1–0.15.

0. **Pre-step: measure the output head explicitly** (~0.8 ms is unattributed
   in the ledger; chain flush doesn't record to `g_stage_cabs`). Boundary
   around the head or `DS4_METAL_GPU_BUSY_PROFILE`; settles whether
   logits+argmax fusion (item D5) is worth it.
1. **A1 — HC producer restructure** (recover 0.7–1.0 ms). Target:
   `kernel_dsv4_hc_rms_norm_mix_f16_cluster2_pre_norm` (metal/dsv4_hc.metal:1551):
   6 TGs, serialized phases, single-lane 20-iteration Sinkhorn
   (`n_hc_sinkhorn_iter = 20`, ds4.c:686), device-scope atomic completion
   (dsv4_hc.metal:1749–1770). Parallelize Sinkhorn across lanes (fixed
   iteration count, deterministic), shrink the completion protocol. Must
   reproduce both reduction trees verbatim (dsv4_hc.metal:1665–1668).
   Untried by rounds 1–3; biggest medium-confidence item.
2. **A3 — defer per-head Q RMS-norm + RoPE into the attention consumer**
   (0.3–0.5 ms). Mirror of the landed inverse-RoPE deferral (ds4.c:21853);
   each attention TG recomputes its head's 512-wide norm verbatim on raw q_b
   output; removes one dispatch per layer. Adjust Qnorm/Qcur debug dumps.
3. **A4 — port `router_project_select` fusion to pre-M5/MXFP4** (0.1–0.25 ms).
   M5-only kernel (dsv4_misc.metal:5025) folds top-6 select into the router
   matvec's last TG via atomic completion; deterministic bitonic ⇒
   bit-exact-safe. Currently gated on M5 device + IQ2-only
   `parallel_full_ffn_eligible` (ds4.c:23787/21941).
4. **A2 — MXFP4 dequant inner-loop cost cut** (0.5–1.5 ms; timebox; the only
   lever that alone reaches 2.2 ms). metal/moe.metal:6561–6600: 8 scalar uchar
   loads + 16 threadgroup LUT gathers per 17 B block ≈ 81 mixed ops/8 B.
   Vectorize loads (uint2), move the 16-entry LUT to constant space or
   register select chains. Loads/LUT values unchanged, accumulation order and
   simd_sum trees untouched ⇒ bit-exact-safe. Excluded: the half-LUT
   (moe.metal:32) — changes rounding. Eight prior variants attacked grid
   shape, not ops/byte; confidence low-medium.
5. **A5 — logits matvec + argmax fusion** (~0.01–0.02 ms). Blockwise top-1 in
   the logits epilogue + tiny final reduce; tie-break must match
   `sample_argmax`. Only if pre-step 0 shows the head cost matters.
6. **A6 — whole-token ICB replay** (0.1–0.15 ms). No MTLIndirectCommandBuffer
   machinery exists (decode-island capture is CUDA-only, stubs ds4.c:130–134).
   Largest effort; do last, only if the wall−GPU gap still matters.

Expected: realistic +1.2–1.9 ms ⇒ ~48–48.7 t/s; optimistic ⇒ ~50.6 t/s.
**Decision point: if 50 remains firm after A1+A3+A4 land, the honest route is
the requant track (attention projections + output head Q8_0→MXFP4 ≈ −4.9 ms ⇒
~55 t/s) — a model change, quality-gated by ds4-eval parity, not md5.**

## Part 2 — Prefill program

Findings: large-prefill is **compute-bound** (~19 TFLOPS effective ≈ 70% of
the M3 Ultra FP16 MMA peak; 2048-token prefix = 663 t/s vs a 240 ms/pass byte
floor ⇒ 12.9× above it), so big-N wins are ≤1.5× and only via GEMM kernels.
The short/medium-prompt regime has real structural waste:

1. **P1 — kill the 43 per-layer GPU drains in TTY chat prefill** (biggest
   chat win, bit-exact by construction — zero arithmetic change).
   `callback_split` (ds4.c:34918) forces the split schedule — per-layer
   `end_commands` = commit + `waitUntilCompleted` (43 full drains/chunk) —
   solely so the progress bar updates. Replace with commit-only progress:
   command-buffer completion handlers (the `DS4_METAL_STAGE_COUNTERS` trick)
   or progress reported post-commit without waiting. Immediate A/B: chat with
   stderr TTY vs redirected.
2. **P2 — prefill stage attribution tooling.** No commit-only counters exist
   for prefill (decode-only today). Port the mechanism; settle the
   compute-bound estimate and find the per-layer fixed costs. Foundation for
   everything below.
3. **P3 — PSO pre-warm.** Lazy `newComputePipelineStateWithFunction` for every
   batch kernel lands inside the first timed prefill; extend
   `ds4_session_gpu_warmup` (ds4.c:61344) to cover the batch pipelines.
4. **P4 — small-N grouped MoE GEMM (6–31 tokens).** Below N=32
   (ds4_metal.m:41633) expert rows re-read per token; a small-N grouped kernel
   streams each distinct expert once. Wins the 6–31-token prompt regime.
5. **P5 — chunk-size sweep on Metal** (`DS4_METAL_PREFILL_CHUNK`
   2048/4096/8192 at long prompts; never swept — 4096 was chosen to bound
   transient buffers). Compute-bound at large N ⇒ modest expectations; cheap
   to measure with `metal_prefill_variant_bench`.
6. **P6 — long-context prefill decay** (596.8 t/s @2k ctx → 410.9 @59k for
   2048-token chunks; attention-KV growth). Profile first (P2), then look at
   the indexed-attention staging path at long prefixes.

Harnesses: `make metal-prefill-variant-bench` (ABBA, bit-identical abort),
`ds4-bench` frontier increments (CSV in speed-bench/), CLI
`prefill: X t/s` line.

## Part 3 — CPU-parallel: honest verdict (analyzed, mostly closed)

Memory model permits it (weights mmap'd Shared, all scratch/KV Shared,
zero-copy CPU access), and sync primitives exist (MTLSharedEvent + the TP
flag-spin rendezvous). But:

- **Mid-layer offload is dead by sync math**: measured MTLSharedEvent wake
  latency is 100s of µs (ds4_metal.m:9625-9629: flag-spin wakes "hundreds of
  microseconds earlier" than event polling); offloadable stages are 10–80 µs
  and serially dependent. 1–2 orders of magnitude short.
- **Bandwidth is shared**: GPU matvecs already stream at 590–650 GB/s of the
  ~800 GB/s fabric; CPU compute would subdivide the same DRAM. The repeated
  measured lesson: more parallel traffic throttles this GPU (KV direct-read
  −7%; concurrent shared-expert contraindicated).
- **No CPU GEMM path exists** (NEON matvec only; no Accelerate/AMX/SME), so
  prefill batch work can't move to CPU either.
- Already-landed CPU help: streaming page-in on 16 pthreads overlapped with
  GPU chunks; the CPU router exists for SSD-streaming (early expert ids), not
  speed; selected-id overlap kicks async expert loads — the one working
  overlap, precisely because it doesn't touch GPU bandwidth.

Two narrow openings remain, both cheap probes, neither blocking:
- **C1 — flag-spin sync microbenchmark**: measure CPU↔GPU rendezvous latency
  of the TP-style Shared-memory flag spin (not MTLSharedEvent) at decode
  cadence. If it lands <5 µs round-trip, revisit offloading the latency-bound
  HC-pre producer to CPU *overlapped with the GPU's bandwidth-bound
  stages of the same layer* (the one place the GPU leaves bandwidth idle).
  If ≥10 µs, close the CPU question permanently.
- **C2 — streaming-mode only**: if the bigger PRO model is ever run via SSD
  streaming, FreeToken-style CPU–GPU co-execution (bandwidth-adaptive expert
  split) applies there. Resident decode: not applicable.

## Part 4 — FreeToken ideas: what transfers

[FreeToken](https://github.com/FlashML-org/FreeToken) is an edge MoE engine
for *discrete* GPU + host-RAM configs. Idea-by-idea against this repo:

- **Bandwidth-adaptive CPU–GPU co-execution (q\* policy)**: targets weights
  spilling to host RAM. On M3 Ultra the model is fully resident in unified
  memory and the GPU is at 590–650 GB/s of the shared fabric — co-execution
  subdivides the same bandwidth. Only relevant to SSD-streaming mode (C2).
- **Full-layer double-buffered prefill streaming**: the repo's SSD-streaming
  prefill already overlaps page-in; double-buffered *full-layer* streaming
  could cut streaming-prefill stalls. Streaming-mode item, pairs with C2.
- **Global LRU expert caching**: exists (`--ssd-streaming-cache-experts`).
- **Graph-compatible execution**: CUDA graphs — exists on CUDA here
  (decode-island capture); Metal equivalent is A6 (ICB replay).
- **Semantic-aware KV caching** (reuse across agentic context edits):
  orthogonal *feature* — would cut real-world agentic latency more than any
  kernel work; the repo's session checkpoint/prefix-reuse machinery
  (`ds4_session_rewrite_from_common`, prefix resume) is the foundation.
  Optional product item, not a t/s lever.
- **FTW fast weight format**: a layout optimized for streaming reads; the
  in-repo analog is the GGUF mixed-quant tooling — the requant decision point
  in Part 1.

## Execution logistics

- Branch: `perf/decode-50tps` (or similar); land each item as its own commit
  with its rollback env and A/B numbers in the message.
- Order: Part-1 item 0 → P2 (attribution tooling benefits everything) → A1 →
  A3 → A4 → P1 → A2 (timeboxed) → P3–P5 → A5/A6 if still justified → C1
  probe anytime.
- Record every outcome (positive or negative) in speed-bench/README.md +
  the campaign handoff, per convention.
- Success criteria: decode ≥ 48.5 t/s bit-exact (stretch 50); chat prefill
  without the TTY-drain penalty (P1 target: parity with redirected-stderr
  numbers); no regressions in `make test`, md5 oracles, harness bit-identity,
  or ds4-eval traces.

## Closed avenues — do not redo (measured, rounds 1–3)

More parallelism/concurrent streams; MTP/DSpark speculation; kernel grid/tile
micro-variants (8 tried, ≤0.03%); split-schedule sweep; Wa+Wb fusion;
pair+down single dispatch; expert prefetch/affinity; KV direct read (−7%);
mid-layer CPU offload (sync math); HC tail fusion (landed, neutral);
eight-row prefill staging (negative).

Round 4 (this branch, measured): **A1 HC producer tail restructure** —
parallel-lane Sinkhorn comb is not bit-exact (the serial 4-row-unrolled
source compiles to row-position-dependent reduction trees; rows 0/3 land one
ulp off) AND speed-neutral (45.90 vs 45.92 t/s); the stage cost is the
dispatch floor + phase streams, not the completion tail. **A3 Q norm/RoPE
deferral into the attention consumer** — plumbing and norm-tree emulation
proven bit-exact (math-safe transcripts match exactly), but the YaRN rope
tail's mul-add blends contract differently in the attention kernel context
under the fast-math library; not landable bit-exactly. **A4
router_project_select port to pre-M5/MXFP4** — closed by analysis, no code:
dispatch-neutral (5 FFN dispatches either way), the shared expert loses its
free-ride overlap slot in the router matvec, and the landed sum6+HC4 tail
fusion disqualifies itself (gates on router_shared_done != 0). The M5 win
is inseparable from the IQ2-only parallel-FFN concurrent encoder.
**A2 MXFP4 dequant constant-LUT** — measured negative (bit-exact but −2.7%:
constant-cache divergent gathers lose to threadgroup staging); the uint2
and register-select variants are structurally blocked (17-byte stride ⇒
unaligned loads; select chains raise ops/byte). Opt-in kept as
`DS4_METAL_ENABLE_MXFP4_CONST_LUT`. Details in speed-bench/README.md.

**Decode status after A1–A4 + item 0:** no bit-exact headroom landed in
this round; decode stays ~45.5 t/s. The Part-1 decision point is reached:
further decode gains require the requant track (attention projections +
output head Q8_0→MXFP4, quality-gated by ds4-eval parity, not md5).
