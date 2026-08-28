## Benchmarking

Session handoff (Aug 21 decode campaign): see
`speed-bench/DECODE-CAMPAIGN-HANDOFF.md` for the cold-restart kit — verified
baselines, the closed-avenue list, tool usage, and the decision required
before resuming work toward >45 t/s.

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### DSpark speculation on M3 Ultra (measured, not yet profitable)

DSpark non-strict decode was measured end to end on M3 Ultra with the MXFP4
model and the 0731 support GGUF.  Three systemic costs keep it below plain
decode (~43.7 t/s) today; scheduler tuning alone cannot fix them:

1. The verify pass runs the speculative suffix through the generic batch
   prefill kernels (`metal_graph_encode_layer_batch`): ~46-60 ms per verify
   vs the ~29-33 ms memory floor for two rows (the extra routed-expert reads
   are inherent).  A commit-only stage decomposition showed a uniform
   1.5-2.5x per-stage excess (routed MoE 14.3 ms vs 5.9 decode, HC pre
   7.2 vs 1.7, output projection 7.1 vs 4.8, attention 4.4 vs 2.1).
   The fix is the N<=6 microbatch verifier on the decode-grade kernels that
   the verifier's own header comment calls out as "not yet" written.
   Caution for that build: three measured increments (per-row MoE, per-row
   HC pre, and a strict-oracle-verified dual-row HC-pre kernel) each
   recovered nothing, and the batch MoE already runs at its distinct-expert
   floor.  The honest reading is that the ~29-33 ms "perfect sharing"
   verify floor is not deliverable on this GPU; the N=2 verify near 50 ms
   is close to its real floor, so speculation is unlikely to beat the
   43.7 t/s plain decode on M3 Ultra with this model.
2. The draft propose chain costs ~3-8 ms/cycle and its confidence gate
   (`sigmoid(confidence0) >= threshold`, Metal default 0.6) declines on
   45-75% of cycles; each such cycle still pays the propose before falling
   back to one plain decode.  Plain decode inside a DSpark session measures
   a normal ~23.3 ms (the hidden-state capture is not a decode tax).
3. Per-cycle bookkeeping (checkpoint, snapshot, commit, propose-fail
   waste) accounts for a further ~4-5 ms/cycle beyond the measured
   propose+verify+decode components.

Scheduler and confidence knobs were swept: `DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP=0`
(retry the draft every cycle) raised proposals from 91/179 to 125/147 cycles
and accepted drafts to 101 (80.8% accept); the best combination measured
39.5 t/s at `--dspark-confidence 0.75` on a code prompt, still below the
43.7 t/s plain-decode equilibrium.

A genuine dual-row HC-pre producer kernel (one F16 mix-weight fetch serving
both rows, each row keeping the one-row kernel's exact reduction trees) was
also built for the N=2 verify.  Correctness was proven with the strict-mode
oracle — `--dspark-strict` output matched plain decode byte-for-byte — after
fixing a buffer-index mismatch that silently misbound five kernel arguments
(the strict oracle is the right validation tool for any verify-kernel work).
Measured honestly, per-verify cost was unchanged (~52 ms, n=4, vs ~50 ms
rollback over 57 verifies) and the diverged non-strict token stream made the
draft decline on 99% of cycles: the serialized hc_pre share had been
overestimated by the stage decomposition, and the win is not there.

Two dispatch-routing microbatch increments were also built, validated, and
measured negative before reverting: per-row routed MoE through the
single-token static-trip kernels (bit-identical output, verify_layer
unchanged at 1225 vs 1217 ms) and per-row HC pre through the decode fused
producer kernel (valid non-strict output, 39.3 vs 40.2 t/s).  Both show the
verify excess lives in the batch kernels' execution, not in dispatch
routing; the microbatch build must write genuine small-N batched kernels.  Strict mode (`--dspark-strict`)
measures 43.07 t/s, i.e. no gain, as accepted blocks are re-run through
one-token decode to stay byte-identical.

Round-5 addendum (Aug 22, branch perf/decode-50tps, MXFP4): propose-path
work, validated by the dspark fixture + verify-depth invariant
(worst_argmax_gap 0.000) + scheduler stats.

- **Logits re-verified OK before touching anything**: 128 speculative
  tokens teacher-forced through plain decode show zero argmax gap; all five
  fixture prompts produce byte-identical greedy output.  The verifier is
  not the problem; the cost structure is.
- **LANDED: chained Metal markov+argmax fusion.**  The propose loop used to
  read back every draft row's 517 KiB logits row and run the full-vocab
  markov bias + argmax on CPU (~175 MB of w2 traffic + 165 M MACs per
  round, prop_markov ~1.33 ms).  Now a fused two-dispatch kernel pair
  (q8_0 dot vs the dequantized w1 row + lexicographic top-1, CUDA-numerics
  precedent, ties keep the lowest index) scores all five draft rows in ONE
  command buffer, each row's kernel reading the previous row's token from a
  GPU-resident key ring (`g->dspark_markov_ring`), ordered by dispatch
  boundaries.  prop_markov 1.33 -> 0.59-0.69 ms/round (python_reverse
  fixture 6.65 -> 3.23 ms over 5 rounds).  A per-row variant (one
  commit+wait per draft row) measured WORSE than CPU (1.55 ms/round) — five
  drained round trips eat the kernel win; the chained form is the one that
  lands.  CPU fallback intact, rollback `DS4_DSPARK_NO_GPU_MARKOV=1`.
  Outputs identical on all fixture cases; verify-depth invariant holds.
- **MEASURED NEGATIVE, reverted**: routing the block_size-row base-head
  matmul through `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor` (the
  multi-session mul_mv batch path) re-reads the 562 MB Q8_0 head per row —
  prop_logits 2.0 -> 4.1 ms/round.  The generic small-N matmul (mv_ext)
  stays; the head's ~280 GB/s vs the single-row decode head's ~730 GB/s is
  the same small-N kernel-efficiency disease as verify, not a dispatch
  routing problem.
- Updated per-round cost structure (python_reverse, default confidence):
  propose ~10 ms = chain 5.0 (support-model 3-stage forward over 6 rows,
  batch-kernel small-N disease in metal_graph_encode_layer_ffn_batch) +
  logits 2.0 (562 MB head at ~280 GB/s) + cache ~1.6 (one-time per
  generation stage-0 seed, amortizes) + markov 0.65 (was 1.33) + setup 0.4.
  Verify ~52-74 ms for <=6 rows (unchanged).  All three remaining propose
  costs and the entire verify excess are the same root cause: no
  decode-grade small-N (N<=8) kernels.  That is the only remaining lever
  and it is the P4-scoped grouped-kernel program (bit-exact conditions in
  the round-4 handoff), which the three failed verify increments above did
  not try (they were dispatch routing, not new kernels).
- One-time cost to know about: the first propose round of a process pays
  ~45 ms of markov-head first-GPU-touch (PSO create + 70 MB of w1/w2 page
  faults).  Amortizes over any real session; visible in 32-token fixtures.

Round-6 addendum (Aug 22, branch perf/decode-50tps, MXFP4): the small-N
kernel program.  Verify pass measured truthfully first (commit-only stage
counters inside the verify loop — the decode eval used to wipe them with
its per-token reset; a reset/report bracket in the verify driver fixes
that):  verify wall ~74 ms/pass for <=6 rows, GPU-bound with a UNIFORM
~2.9x per-stage excess over the ~30 ms bandwidth floor (routed_moe 22.8,
hc_pre 11.5, output_proj 10.1, indexer+compressor 13.7, q_path 7.1,
shared 10.8, router 4.3 ms/verify).  Expert footprint: ~12.8 distinct
experts/layer (2.1x decode's 6) — the per-pair dispatch already dedups
through L2.

- **MEASURED NEGATIVE, gated OFF: expert-major grouped routed MoE.**
  New grouped kernels (one threadgroup per distinct expert, sharing the
  weight block read and LUT gathers across its (token,slot) pairs,
  bit-exact per pair vs the reference matvec) went SLOWER: verify
  74.0 -> 91.9 ms/pass.  At ~2.3 pairs/expert average overlap the issue
  sharing (~2.9x at npair>=4) is eaten by the pair-loop/setup/register
  overheads and the sentinel-padded grid — L2 was already doing the
  dedup.  Opt-in kept as DS4_DSPARK_ENABLE_VERIFY_GROUPED_MOE for
  re-measurement at other overlaps.  This closes the MoE angle at this
  overlap factor: the recoverable time is NOT in the routed MoE.
- **LANDED: nrow small-N Q8 matvec for the verify path** (verify
  74.3 -> 68.9 ms/pass, -7.3%, interleaved A/B x3).  The batch small-N
  matmuls (shared gate/up/down, q_a/q_b, kv, router, attention out-B)
  ran through mv_ext at 27-280 GB/s where the single-row decode matvec
  gets ~730 GB/s.  New kernel_mul_mv_q8_0_nrow_f32_n{2..6} keeps the
  single-row kernel's exact per-element math (lane map, ascending block
  stride, sequential per-lane sumq*scale, same two-stage simd_sum
  epilogue tree per row) with one accumulator per token — bit-exact with
  mul_mv per token, one weight read for all rows.  Routed in
  ds4_gpu_matmul_q8_0_legacy_tensor when g_dspark_nrow_active is set
  (verify layer loop only), n_tok 2..6; rollback
  DS4_METAL_DISABLE_Q8_NROW_MATMUL.  Stage deltas per verify:
  shared_down 6.4 -> 1.2, q_path 7.1 -> 4.3, shared_gate_up 4.4 -> 2.6,
  attention out-B included.  Validation: fixture output_match=1 on all
  five cases, verify-depth worst_argmax_gap 0.000, md5 oracle db0c504c…
  holds (prefill untouched by construction), make test 44/44.
- **LANDED: attention out-A grouped nrow** (output_proj 9.0 -> 6.2
  ms/verify): kernel_dsv4_attn_out_low_q8_0_nrow_f32_n{2..6} runs all
  eight head-group projections in one dispatch with the same nrow
  structure.  Combined V2a+V2b: verify 74.8 -> 67.0 ms/pass (-10.4%,
  interleaved A/B x3).
- **LANDED: nrow for the propose side** (propose 9.4 -> 8.2 ms/round):
  the block_size-row base head (562 MB, 280 -> ~730 GB/s) and the
  support chain's Q8 projections set g_dspark_nrow_active around their
  calls.  prop_logits 2.0 -> 1.2, prop_chain 5.0 -> 4.3 ms/round.
  Guard: nrow only routes out_dim >= 512 — tiny extents (hc projections
  out=24, router out=256) starve the no-k-split grid and stay on mv_ext
  (an unguarded nrow made the chain 3x slower; measured, fixed).
- End-to-end so far (python_reverse fixture, code): dspark 48.9 -> 52.9
  t/s vs 47.0 baseline (+12%).  Fixture hello/math/python net-positive;
  redis/medium-yield and story still scheduler-limited.  c_add fixture
  guard still trips at 32 tokens (pre-existing scheduler conservatism;
  net_saved improved -3.8 -> +10.2).
- End-to-end at 128 tokens (temp 0): medium-yield code 40.85 -> 42.34
  t/s vs 45.74 baseline (-7.4%, was -10.4%); story 39.23 -> 40.59 vs
  45.38 (-10.5%, was -13.8%).  The remaining negatives are yield-side
  (0.95-0.27 accepted/cycle), not cost-side: 24 verifies at ~48.7 ms
  (n~4 rows) confirm the nrow verify scales with rows as expected.
- Remaining verify map after nrow (target in order): routed_moe 22.5
  (floor ~13, no lever found), hc_pre 11.3, output_proj 9.0 (the
  attention out-A grouped projection is next), indexer_setup+compressor
  13.7, router 4.3 (out=256 grid too small — needs k-split nrow),
  attention 2.9.  The grouped-MoE negative + this map say the rest of
  the floor is won stage-by-stage with nrow-family kernels, not by
  restructuring dispatch.
- Next candidates: k-split nrow for out_dim < 512 (router 4.3 -> ~0.5);
  hc_pre 11.3 (F16 HC projections + sinkhorn at n=6); indexer/compressor
  13.7; attention 2.9; then the support-model MoE per-pair MXFP4 (same
  blocked angle as the target's routed_moe).
- **ds4-eval wiring fix (Aug 23)**: eval never speculated at all — the
  decode loop used only sample+eval, and loading the support model
  disabled chain bursts (chain_greedy_supported rejects support_kind !=
  NONE), so --dspark eval was the slowest decode configuration.  Added
  --dspark/--dspark-confidence/--dspark-strict to ds4_eval.c and a
  speculative-burst branch in the decode loop (same gating as the CLI:
  temp <= 0, ds4_engine_mtp_draft_tokens > 1; think-close window guard
  identical to chain bursts; accepted tokens go through
  eval_chain_on_token so the emitted stream is byte-identical).
  Measured on GPQA Diamond case 1, nothink, 1024-token budget, M3 Ultra:
  plain 6.1s/278 tok, dspark 6.0s/277 tok, identical text, PASS both
  ways.  DS4_DSPARK_STATS: cycles=114 avg_accept=1.44 accept_rate=85%
  no_draft=55 scheduler_skips=45 net_saved=+192 ms — break-even is
  ~1.4 accepted/cycle here (propose 4.2 + verify 26.1 vs 21.9 ms/token
  baseline), so science/LaTeX content sits at parity; the 52.9 t/s
  fixture number needs high-yield code.  Confidence sweep on the same
  case: 0.4 -> net_saved -266 ms (cycles 141, avg 0.97), 0.8 -> -163 ms
  (cycles 169, avg 0.65); the 0.6 default is already optimal for this
  content.

Round-7 addendum (Aug 23, branch perf/decode-50tps, MXFP4): the small-out
end of the nrow program plus the first scheduler change.  Validation per
item: fixture output_match=1 on all five cases (the c_add accepted_draft
guard trip at 32 tokens is pre-existing and byte-identical before/after),
verify-depth worst_argmax_gap 0.000, md5 oracle db0c504c…, make test 44/44,
interleaved A/B x3 for wall time, commit-only counters for stage maps.

- **LANDED: small-out nrow matvecs (Q8 + F16).**  The verify router is F16
  (4096->256), not Q8 as the round-6 guard comment implied — the Q8
  small-out variant only fires for the support model's Q8 router.  New
  kernel_mul_mv_f16_nrow_f32_n{2..6} (and the Q8 _so_ sibling) widen the
  nrow structure to NSG=8 with one output row per threadgroup: the router
  dispatches 256 threadgroups of 256 threads where mv_ext had (32,2).
  Routed when g_dspark_nrow_active && 64<=out<512 && in>=2048 (round-6
  nrow keeps out>=512).  NOT bit-exact with mul_mv/mv_ext (different
  per-lane block subsets and reduction tree); the verify tolerates the
  order difference by design, as it already did for mv_ext.  Interleaved
  A/B x3 (32-token fixture prompt): verify 329.3 -> 318.8 ms per 5 passes
  (-2.1 ms/verify), e2e 53.7 -> 54.7 t/s.  Rollbacks
  DS4_METAL_DISABLE_F16_NROW_SMALLOUT / DS4_METAL_DISABLE_Q8_NROW_SMALLOUT.
  Measurement note: the router stage's cost was half select, not all
  matmul — a temporary router_mm boundary showed 2.6 ms matmul / 1.7 ms
  select per verify (counter mode).
- **LANDED: fused batch router select** (kernel_dsv4_router_select_batch_
  fused): one 256-thread threadgroup per token fuses the sqrt(softplus)
  transform (device store + volatile reload, same rounding boundary as the
  standalone dispatches), the bias add, and the 256-wide bitonic top-6 from
  kernel_dsv4_router_finalize_one_simd — replacing softplus + sqrt +
  bias-add + two argsort merge dispatches per bias layer.  The weight
  finalization stays in kernel_dsv4_router_weights_batch (its simd_sum /
  p/denom/scale rounding sequence is deliberately not reproduced in the
  fused kernel — the comment there about 43-layer amplification applies).
  Engages only under g_dspark_nrow_active, 2..6 tokens, non-hash layers.
  Tie order inside the bitonic network can differ from the argsort merge
  it replaces; router scores are distinct in practice, and the fixture +
  verify-depth gates held.  A/B x3: verify 318.8 -> 315.3 per 5 passes
  (-0.7 ms/verify), e2e 54.7 -> 55.0 t/s.  Rollback
  DS4_METAL_DISABLE_ROUTER_SELECT_BATCH_FUSION.
- **LANDED: nrow for the HC projections** (gate widened to out>=24): the
  verify HC projections (F16 16384->24, hc_attn_fn/hc_ffn_fn, twice per
  layer) ran mv_ext on a (3,2) grid of 64-thread threadgroups — 120 us per
  call.  The nrow kernel gives each of the 24 output rows a 256-thread
  threadgroup: hc_proj 10.3 -> 2.4 ms/verify (counter mode), production
  verify -8.6 ms/verify (324.1 -> 281.3 per 5 passes, interleaved A/B x3),
  prop_chain -3 ms/round (the support chain's HC projections ride too).
  e2e fixture 54.1 -> 58.6 t/s.  Same rollback env as the router nrow.
- **LANDED: scheduler no-draft streak escalation.**  Low-yield content
  (story: avg_accept 0.31) loses ~12% to doomed propose+verify cycles.
  Static longer skip knobs recover most of it (story 40.6 -> 43.7-43.9)
  but collapse the code yield (avg_accept 1.93 -> 0.65, 47.1 -> 44.9 t/s):
  the no_draft pause length must depend on regime persistence, not be
  global.  Now consecutive no-draft cycles escalate the pause 3 -> 6 ->
  12 -> 24 -> 32 (cap DS4_DSPARK_SCHEDULER_NO_DRAFT_STREAK_CAP, default 32;
  a cap at/below the base pause disables), and a live draft (accepted or
  missed) decays the streak by one instead of resetting it — transitional
  declines between productive drafts stay cheap, persistent loser regimes
  clamp.  Interleaved A/B x2 at 128 tokens: story 39.9 -> 42.1 t/s (code
  neutral at 47.0, avg_accept and skips byte-identical), GPQA Diamond
  case 1 identical text/stats PASS both ways.  Residual story gap vs plain
  decode (~46.2) is structural: with drafting fully disabled
  (--dspark-confidence 1.0) the session still pays ~1.8 ms/token of
  propose-attempt + bookkeeping, i.e. the story ceiling is ~43.5-44.
- **Scoped, not built: compressor/indexer small-N fusion** (compressor 7.9
  + indexer_setup 5.6 ms/verify counter mode).  The n=6 verify runs the
  per-token compressor loop: ~30 tiny dispatches per ratio-4 layer (6 APE
  stores, 2 emit chains of pool+norm+rope+shift, 5x2 prefix-capture state
  copies), dispatch-latency-bound.  Fusing them into one kernel per layer
  is semantically possible only whole: the per-token prefix snapshots must
  include the interleaved emit/shift effects, so the fused kernel would
  have to reproduce the softmax pool, the rms_norm reduction tree, and the
  YaRN rope tail inside one new kernel context — the A3 fast-math
  contraction trap applies to the last two, with only the empirical gates
  to catch it.  Deferred; the safe sub-fusions (stores, captures) are
  blocked by the same interleaving.
- End-to-end state (fixture-scale, temp 0): python_reverse dspark 57.9-58.6
  t/s vs plain 46.7-47.6; all five fixture prompts net-positive, three of
  them with dspark above plain decode outright (hello 53.7/40.8, math
  49.1/40.3, python_reverse 57.9/46.7; redis 52.1/46.5).  128-token:
  medium code 47.8-47.9 vs 46.4 plain (+3%, was -7.4% in round 6); story
  42.1 vs 46.2 (-9%, was -10.5%; the residual is the machinery floor tax
  plus yield reality, not verify cost).  GPQA parity with net_saved +479
  ms (was +192 at the eval-wiring note).  Verify pass 66.6 -> 56.3 ms
  production (-15.5%).
- Updated verify map (counter mode, after this round): routed_moe 22.8
  (floor-bound, no lever), compressor 7.9 + indexer_setup 5.6 (per-token
  loops, see above), output_proj 6.1, attention 3.3, q_b 3.1 (at the wall),
  router 2.9 -> fused split, hc_proj 2.4 + hc_pre 1.2 (was 11.5), kv_path
  2.7.  shared_down misreads ~13 under counters since the fused select
  landed (a commit/scheduling artifact — production wall disagrees);
  treat that line as ~1.2.

Round-8 addendum (Aug 23, branch perf/decode-50tps, MXFP4): the
compressor/indexer emit-chain fusion — item 3 landed as dispatch routing,
reusing the decode-proven bit-exact kernels.  Validation: fixture
output_match=1 on all five cases (c_add guard trip pre-existing),
verify-depth worst_argmax_gap 0.000, md5 oracle db0c504c…, make test
44/44, interleaved A/B x3 for wall time.

- **LANDED: verify emit chain routed to the decode fused kernels.**  The
  round-7 scoping read this fusion as a new-kernel build with A3-class
  numerics risk; the full map showed the bit-exact kernels already exist
  on the decode path:
  kernel_dsv4_compressor_exact_pool_ratio4_decode_ggml (one dispatch,
  reproduces the GGML pool's soft_max/mul/sum_rows topologies reading
  the state rows directly — the same logical content the verify path's
  concat pack builds) and kernel_dsv4_comp_row_finalize_f32 (norm + rope
  + fp8 round-trip + F16 commit, indexer norm + rope + QAT, and both
  ratio-4 state shifts in one 22-threadgroup dispatch).  Verify never
  routed to them: decode_one_token=false forced the 7-dispatch GGML
  pool plus separate norm/rope/shift/quantize/commit per emit.  The
  finalize kernel grew a mode field (0 = both compressors, decode
  default; 1 = attention only; 2 = indexer only — the verify loops emit
  the two compressors from separate per-token loops, so the combined
  kernel would touch indexer state mid-window).  The verify loops engage
  the routing per emit under g_dspark_nrow_active with decode's safety
  gates (F16 cache, F32 norms, kv_rope_fp8 availability, pre-M5/M5
  feature gate); each emit goes from 12 dispatches per compressor to 2
  (exact pool + mode finalize), and the separate quantize/commit/QAT
  calls are skipped.  Rollback
  DS4_METAL_DISABLE_DSPARK_COMP_FINALIZE_FUSE.
- A/B x3 (python_reverse, 32 tokens, interleaved, 20 s pauses): verify
  283.4 -> 271.7 ms per 5 passes (-2.4 ms/verify, -4.2%); e2e fixture
  58.6 -> 58.9 t/s.  Counter-mode stage map: compressor 7.9 -> 5.0,
  indexer_setup 5.6 -> 2.7 ms/verify.
- What remains in those stages (~7.7 ms/verify counter-mode) is the 12
  per-token stores, the ~16-20 prefix-capture blits, and the residual
  pool/finalize dispatches.  Stores cannot hoist past captures (each
  capture must observe the canonical per-token-boundary state) and a
  capture cannot share a dispatch with the store it depends on (Metal
  has no cross-threadgroup ordering within a dispatch), so the residual
  is structural short of the whole-loop fusion that reproduces the
  intermediate states inside one kernel — still deferred, now worth at
  most ~2-3 ms/verify.
- Updated verify map (counter mode): routed_moe 22.8 (floor-bound),
  output_proj 6.1, compressor 5.0, q_path 4.2, hc_pre 3.4, attention
  2.9, indexer_setup 2.7, router 2.7 (fused split), hc_proj 2.4,
  kv_path 1.2.  Production verify pass 56.3 -> ~54.3 ms.

Round-9 addendum (Aug 23, branch perf/decode-50tps, MXFP4): handoff item 1
(session bookkeeping trim) executed profiling-first; the premise measured
false, so no code landed — the working tree is net-unchanged and the full
ritual re-ran green at close.  Item 3 (ds4-eval re-measurement) landed
positive.  All wall numbers interleaved x3, 128-token lighthouse prompt,
20 s thermal pauses, one ds4 process at a time.

- **Corrected story machinery map (the handoff's ~1.8 ms/token tax is really
  ~1.15, and it is not bookkeeping).**  Plain chain 45.92 t/s; classic
  (DS4_DISABLE_GREEDY_CHAIN=1) 44.62; --dspark-confidence 1.0 (drafting
  dead, streak escalation active: 112 skips / 7 proposes / 119 cycles)
  43.59; the same probe with the per-token capture dispatch skipped via a
  temporary profiling env 43.86.  Decomposition per token: **0.62 chain
  loss** (support_kind != NONE rejects ds4_session_chain_greedy_supported,
  so every DSpark token runs the classic wait/readback/argmax loop) +
  **0.24 propose machinery** (dominated by the ~20 ms first-propose
  markov-head first-touch amortized over 119 cycles; 6 further explores)
  + **~0.05 capture+ring** + **<=0.29 bounded residual**.  The capture is
  ~free: target-layer count is 3 (layers 40,41,42), one weighted-sum
  dispatch each, target_ms unchanged 24.4 vs 24.5 ms/token and GPU
  stage-busy identical (2798 vs 2796 ms over 127 positions, commit-only
  counters).  The residual sits outside every DS4_DSPARK_STATS timer with
  identical GPU streams — submission-side or thermal, not attributed, not
  actionable at this size.
- **Why nothing was cut:** the capture feeds metal_graph_dspark_ring_maintain,
  which appends one support-KV row per skipped token so the first real
  propose after a streak finds the cache ending at pos — gating either
  deadlocks the streak-recovery propose (capture_ok / cache_ends_at fail,
  no-draft escalates again), and the 7 explores are the scheduler's regime
  probes (load-bearing conservatism).  The one real lever, restoring chain
  execution between bursts, is scoped with measured ROI: story-only
  ~+0.58 ms/token on ~93% of tokens (42.1 -> ~43.2, still -6.5% vs plain
  46.2); code gains ~0 (every cycle bursts, the chain never gets ahead);
  GPQA +~0.06.  Not built: emit-path surgery plus ahead-encode/verify KV
  interaction risk for a story-only partial recovery.
- **GPQA Diamond case 1 re-measured (handoff item 3): now a wall-clock
  win.**  Identical grading both ways (PASS, answer B): dspark 5.5 s /
  277 tok vs plain 6.1 s / 278 tok — 0.6 s faster end to end (round 7 was
  parity 6.0 vs 6.1 with net_saved +479 ms).  With stats:
  cycles=99 avg_accept=1.798 accept_rate 87.7% scheduler_skips=30
  net_saved=+809.7 ms; propose 446.9 ms total (prop_chain 256.7,
  prop_logits 71.8, prop_cache 51.9, prop_markov 36.0); verify_layer
  2743.6 ms over 58 fused-head verifies ~ 47.3 ms average (row-count
  weighted; the 54.3 fixture number is for ~5-row blocks).
- **Re-ranked open work for round 10:** (1) the propose chain — 4.2-4.5
  ms/round is ~8% of the code cycle and it is dispatch-bound small-model
  work, the same disease the verify's nrow program treated once
  (prop_chain 5.0 -> 4.3); a dispatch/stage map of
  metal_graph_eval_dspark_stage_chain is the entry point.  (2) Chain
  restore between bursts (story-only, ROI above).  (3) Whole-loop
  compressor fusion (~2-3 ms/verify ceiling, A3-class risk, unchanged).
- Ritual at round-9 close: md5 oracle db0c504c…, verify-depth
  worst_argmax_gap 0.000, acceptance output_match=1 on all five (c_add
  guard trip pre-existing, accepted_draft 6 class), make test 44/44.

Round-10 addendum (Aug 26, branch perf/decode-50tps, M3 Ultra): items 1
and 4 of the round-9 ranking executed; item 1 closed NEGATIVE by
measurement, item 4 produced the corrected GPQA band.  Two tooling
landings (uncommitted at write time): the propose-chain stage-counter map
(bracket at the propose driver + `ds%u:%s` labels in
metal_graph_dspark_stage_profile_boundary; env pair
DS4_METAL_STAGE_COUNTERS=1 DS4_DSPARK_STAGE_PROFILE=1) and a ds4-eval
`--source SUBSTR` case filter (the 92 embedded cases are interleaved by
source; `--questions N` cannot isolate one).

- **Propose chain map (the round-9 entry point, now measured):** per
  round, counter mode, python_reverse 32 tok — ffn **1.72 ms (48%)**,
  attn_output_hc 0.48, kv_path 0.48, q_path 0.42, attention 0.18,
  hidden 0.15, attn_hc_pre 0.09, attn_norm 0.04 (total 3.57 GPU-busy vs
  3.75 wall).  The chain is **kernel-bound, not dispatch-bound**: the
  round-9 "dispatch-bound small-model work" premise is false.  The FFN's
  ~12 dispatches average ~130 µs of real kernel time (support routed MoE
  IQ2_XXS/Q2_K over 5x6 pairs, shared expert, router); the removable
  dispatches (blits, ropes, small norms) are µs-scale.
- **MEASURED NEGATIVE, reverted: propose dispatch-fusion bundle.**  (a)
  kv target+draft rope merge (2 dispatches -> 1, bit-identical per-row
  math); (b) q head_rms_norm+rope_tail -> the decode-production fused
  kernel; (c) stage-input copy fold — the FFN epilogue (grown a
  next_hc_override param) writes draft rows straight into
  stage_input_hc rows 1..block, eliminating the 2 inter-stage blit
  encoder churns.  Outputs md5-identical to the rollback path and plain
  decode on every gate, but per-round GPU busy and prop_chain were
  UNCHANGED (3.62 vs 3.65 ms/round, counter mode, n normalized by chain
  rounds).  The removed dispatches were too small to matter; reverted
  per the no-noise-landings rule.
- **MEASURED NEGATIVE, reverted: f16-nrow at out_dim<24** (the support
  head 16384->4 missed the small-out gate).  Lowering the gate to >=4
  reroutes hc_head_fn off mv_ext; the reduction-tree difference shifts
  conf0 enough that the fixture's scheduler-stats class broke
  (python_reverse accept 95.65% -> 57.69%, c_add 6 -> 3).  Gate restored
  to out_dim >= 24; the comment in ds4_gpu_matmul_f16_tensor documents
  it.  Draft-side "may shift at ties" has a practical limit: the
  confidence gate sits near the threshold, so ulp shifts flip gate
  decisions, not just tie-breaks.
- **Scheduler time-basin hazard (new measurement rule).**  The DSpark
  scheduler accumulates wall-clock extra-ms and decides on
  slow_accept/measured_unprofitable ratios (ds4_session_dspark_scheduler
  accounting, ~ds4.c:50490).  Any propose-timing perturbation — even an
  output-identical fusion, counter envs, or thermal state — flips the
  whole run into a different schedule basin (observed same binary+env:
  avg_accept 4.400/95.65% vs 1.500/57.69%, fixture 58.9 vs 45.9 t/s,
  outputs byte-identical both ways).  Consequences: (1) single-run
  fixture t/s is not a build comparator — use the stats-class +
  counter-map arbiters, or interleave x3 and take medians; (2) the
  "scheduler stats byte-identical class" ritual requirement is only
  achievable by timing-identical builds; (3) a scheduler-freeze env would
  make fixture A/Bs deterministic (candidate tooling item).
- **GPQA Diamond 25-case sweep (round-9 item 4): the wall win does NOT
  generalize.**  `--source "GPQA Diamond"`, nothink, 1024-token budget,
  temp 0, plain vs --dspark, interleaved x3, medians: aggregate **220.5 s
  plain vs 227.6 s dspark = -3.2% (dspark slower)**, 10 wins / 13 losses
  / 2 ties (>20 ms thresholds).  Case 1 (+0.63 s) was the favorable end:
  the biggest win +4.7 s is a budget-hit case whose diverged stream
  finished 237 tokens early; losses concentrate in low-yield cases
  (machinery tax) and diverged-longer streams.  **17/25 dspark token
  streams differ from plain** (non-strict verify-kernel numerics; 3
  answer flips, here all F->P, passes 16->19 — treat as divergence, not
  quality gain).  Aggregate stats: cycles=6976 avg_accept 0.468
  accept_rate 82.2%, no_draft 5601, verify_layer 55.4 s, propose 11.0 s
  (prop_chain 7.0 over 3972 attempts ~= 1.76 ms/attempt — the fixture's
  4.3 counts full chains only; failed/short attempts average in).
  Verdict: science content sits at parity-to-negative wall at verify 54
  ms; DSpark's real win band is high-yield code-class only.  The
  ds4-eval `--source` flag landed to make this sweep repeatable.
- Ritual at round-10 close: md5 oracle db0c504c…, acceptance
  output_match=1 all five (python_reverse 59.07 t/s / 95.65% / c_add
  accepted_draft 6 class — canonical), make test 44/44.
- **Re-ranked open work after round 10:** (1) story chain restore
  (unchanged ROI, ~+1.1 t/s story-class, emit-path + KV-rewind risk);
  (2) whole-loop compressor fusion (~2-3 ms/verify, A3-class risk);
  (3) scheduler-freeze env for deterministic fixture A/Bs (tooling,
  small); (4) support-model FFN kernel slivers (shared gate+up dual
  nrow, ~30-50 µs/stage — the only untried dispatch-level item, low
  ceiling).  The propose chain and the GPQA band are closed.  Without
  the requant track, the bit-exact DSpark program is at its local
  optimum.

### Q4_K attention+head requant — the parked track, executed (Aug 26)

The roadmap's Part-1 decision point ("further decode gains require the
requant track, quality-gated by ds4-eval parity, not md5"), taken without
the HF safetensors on disk.  Two landed pieces:

- **`gguf-tools/gguf-requantize-dense`** (new tool, the name .gitignore
  expected): GGUF→GGUF re-encoder for dense families.  Dequantizes only the
  requested tensors (sources f32/f16/bf16/q8_0), re-encodes via
  ds4q_quantize_chunk, copies every other tensor and the whole KV blob
  verbatim (--verify checksums the copied tensors).  The reference is the
  input GGUF's values (double quantization: q8_0→q4_K adds only q8_0's
  near-lossless stage on top of q4_K's own error — acceptable for
  quality-gated experiments).  Gotchas encoded: GGUF v3 value-type enum
  must match the quantizer's (UINT32=4/FLOAT32=6/BOOL=7), ds4q_type_name
  returns NULL at enum holes, the type string is `q4_K`.
- **Variant GGUF**: `gguf/DeepSeek-V4-Flash-MXFP4Experts-F16HC-
  F16Compressor-F16Indexer-Q4KAttn-Q8Shared-Q4KOut-chat-v2-mxfp4-0731.gguf`
  (153.4 GB): 216 tensors changed = 43 layers × 5 attention projections
  (attn_q_a/q_b/kv/output_a/output_b) + output.weight; indexer/compressor/
  HC/norm/shared/embd untouched (the indexer's own attn_q_b suffix-matches
  the projection pattern and is explicitly excluded).

**Speed** (interleaved ×3, lighthouse 128 tok): generation 45.72 →
**51.47 t/s median (+12.6%)**, ≈ −2.4 ms/token of the estimated −3.5–4
(q4_K dense dequant is not free); prefill −3%.

**Quality gate (the agreed "intelligence" check, full ds4-eval):**
- Screen (92 cases, --nothink --tokens 1024): Q4_K 61/92 vs baseline 58/92;
  31 answer flips splitting 7 F→P / 4 P→F — noise.
- Canonical (92 cases, thinking mode, 16k budget): Q4_K **85/92 vs baseline
  82/92**.  AIME2025 22/25 → 24/25, GPQA Diamond 20/24 → 24/24, SuperGPQA
  22→21, COMPSEC 17/17 → 15/17 (the one soft spot: two CVE-localization
  regressions, precision-sensitive line recall).  12 answer flips, 6 F→P vs
  3 P→F.  Wall 6549 s → 5770 s (−12%) at 272k → 265k generated tokens
  (24.1 → 21.8 ms/token including prefill).
- Verdict: **no degradation detectable at this suite's resolution; the +3
  is within noise.  The variant is a keeper** — quality-gated, not md5:
  transcripts differ from the Q8 baseline by design (this is the
  non-bit-exact track).

Follow-ups left open: DSpark-on-Q4K (verify floor and prop_logits both
shrink; re-run the 25-case GPQA dspark sweep on this model), the MXFP4
version of the same track (needs the dense encoder in quants.c; ~−4.9 ms
estimate), imatrix calibration if COMPSEC-class precision ever matters,
and the dense Q4_K kernel's dequant overhead (the gap between −2.4
realized and −3.5–4 byte-scaled).

**Follow-up results (Aug 27):**

- **DSpark-on-Q4K measured NEGATIVE — do not compose them yet.**  GPQA
  25-case plain-vs-dspark ×3 on the Q4_K model: aggregate **−16.1%**
  (dspark slower; 5W/20L), vs the Q8 model's −3.2%.  Plain decode got the
  full +15% (188.0 s vs 220.5 s over the sweep) but dspark barely moved
  (218.2 vs 227.6 s): the verify/propose small-N nrow kernels
  (kernel_mul_mv_q8_0_nrow_*, round 6) are Q8-specific, so every
  re-quantized tensor falls back to the generic dense batch path in the
  verify — verify_layer improved only ~11% (55.4 → 49.2 s) against
  plain's ~15%, and avg_accept dropped to 0.348.  Acceptance fixture on
  the Q4K target stays output_match=1 ×5 (correctness fine; it is pure
  cost).  Practical guidance: Q4_K for plain decode, Q8 if you want
  DSpark; composing them needs a q4_K nrow small-N kernel family (the
  round-6 precedent, applied to the q4_K dense matvec).
- **Imatrix calibration for the dense families: built, screens NEUTRAL.**
  The existing collector covered routed MoE only, so it grew dense-family
  hooks (ds4.c ds4_imatrix_collector): sum(x²) over the materialized
  prefill buffers — attn_norm (→ attn_q_a + attn_kv entries), qr_norm
  (→ attn_q_b), heads viewed as n_out_group group_dim slices
  group-averaged (→ attn_output_a, count ×n_out_group), attn_low
  (→ attn_output_b); 256 rows/chunk subsample to bound readback;
  output.weight stays uncalibrated (head input not materialized in
  prefill).  gguf-requantize-dense grew --imatrix/--imatrix-strict
  (exact-name match, nval must equal ne[0], feeds
  ds4q_quantize_chunk — the q4_K path consumes it).  Collector bug
  caught by the tool's miss report: the save header said N_LAYER×7 but
  writes 8 entries/layer (3 MoE + 5 dense), truncating the last 43
  entries at load — fixed (header patched in place on the collected
  .dat; no recollection needed).  Calibration: 231 prompts / 262,144
  tokens on the Q8 baseline (433 MB .dat, 344 entries).  Variant
  …-Q4KAttn-…-Q4KOut-imatrix-…: 215/216 re-quantized tensors calibrated
  (only output.weight misses).  Screen (92 nothink): **61/92 — identical
  to no-imatrix**, flips perfectly symmetric (5 P→F / 5 F→P); per-source
  GPQA 16→17, SuperGPQA 20→21, AIME 8→7, COMPSEC 16→15.  **Canonical
  (thinking, 16k): 81/92 vs no-imatrix's 85/92 — MEASURED NEGATIVE, do
  not use for this variant** (11 answer flips, 6 P→F vs 2 F→P, GPQA
  Diamond 24/24 → 19/24; wall/token unchanged).  The screen's neutrality
  did not survive canonical resolution.  Suspects, unranked: the
  group-averaged heads vector for attn_output_a, the shared attn_norm
  vector for q_a+kv, the 256-row subsample, or the absolute-scale
  interplay with the q4_K weighted search.  The no-imatrix Q4_K variant
  stays the keeper (85/92).  The infrastructure (dense-family collector
  hooks + tool --imatrix) is landable as-is for future lower-bpw tracks
  (q2_k/iq2 head, denser MXFP4), where calibration is expected to matter.

**Q4_K prefill cost attribution + dequant-diet negative (Aug 27).**  The
variant's prefill penalty localizes cleanly by commit-only stage counters
(2.2k-token prompt): q_path **+22.5 ms (+5.0%)** and output_proj
**+26.2 ms (+3.7%)** — exactly the re-quantized projection families;
routed_moe/shared/router/hc all flat, whole-model counter delta +1.4%
(wall −1.1% at 2.2k tokens, −3% at the ~10-token lighthouse prompt,
**−1.66% at 37.6k tokens** interleaved ×3: base 564.63-564.77 vs
q4k 555.20-555.63 t/s — the long-context regime prices the trade at
~1.1 s on a 67 s prefill).
An instruction-diet twin of dequantize_dense_q4_K (sixteen byte loads →
four aligned uint4 loads + float4 lanes, identical element values and
arithmetic) was built bit-exact — the prefill variant harness confirmed
1,034,240 floats identical across 8 runs at 8192 tokens — and measured
**PERF-NEUTRAL in both regimes** (8k-token mul_mm: −0.006%; ~10-token
mv_ext: within noise).  Reverted per the no-noise rule.  Conclusion: the
q4_K dense prefill cost is structural in the mul_mm staging for 256-wide
blocks (thread→block mapping / sa write pattern), not dequant ALU;
recovering it means staging-mapping surgery on the shared template, or
accepting ~1–3% prefill as the price of +12.6% decode.  Also examined
and passed on: antirez/ds4 PR #864 (IQ2 half-LUT + mpp 2×16 tile split)
helps IQ2/Q2_K on M5+ only — its kernel family is Metal-4-gated
(ds4_metal.m pre-M5 disable) and not instantiated for MXFP4; the MXFP4
analog of its LUT trick was already landed here in earlier rounds
(dequantize_mxfp4_half_scale/_half_lut).

**Q4_K_M: data-driven mixed variant (Aug 27 night).**  Leave-one-out
ablation ranked the dense families by NLL sensitivity (each variant
restores one family to q8_0, scored on the 100-case official-continuation
fixture; total delta to recover = 0.0357 NLL, baseline 0.1467 vs keeper
0.1824):

| family restored to Q8 | NLL | share of delta | bytes share |
|---|---|---|---|
| attn_kv | 0.1739 | **23.8%** | 2% |
| attn_q_a | 0.1783 | 11.5% | 4% |
| attn_output_b | 0.1788 | 10.1% | 31% |
| attn_q_b | 0.1797 | 7.6% | 31% |
| output (head) | 0.1801 | 6.4% | 10% |
| attn_output_a | engine refuses | — | 31% |

The MLA input bottlenecks (kv: 4096→512, q_a: 4096→1024 lora) carry the
drift; the head is NOT a major driver (the prior was wrong — measured).
**Engine constraint discovered: attn_output_a/out_b are a fused pair in
the grouped attention-output kernel and cannot mix quant types** —
out_a-at-Q8 with out_b-at-Q4_K fails at load ("logits failed at target
token 1"); promote together or not at all.

**Variant built: `…-Q4KM-Attn-Q8Shared-Q4KOut-chat-v2-mxfp4-0731.gguf`**
(attn_kv + attn_q_a back at q8_0; q_b/out_a/out_b/head at q4_K; 130
tensors changed, copied tensors checksum-verified).  Metrics vs baseline
(keeper in parens): continuation NLL **0.1726** (+17.7% rel. vs
baseline's +24.4%) — 27.4% of the delta recovered; KL(base||var) mean
**0.0312** (0.0364), median 0.00034 (0.00053); top-1 agreement **94.58%**
(94.19%); base-top-1 in top-5 still 100%.  Decode is a wash — 51.44 vs
50.72 median interleaved ×3 (the +124 MB of promotions is below noise;
keeper read 51.47 in its own session).  Screen 61/92 (same as keeper —
the screen's resolution).  **Canonical (thinking, 16k): 81/92 vs the
keeper's 85/92 — MEASURED NEGATIVE ON THE GATE; the uniform Q4_K stays
the keeper** (AIME 24→22, GPQA Diamond 24→21, COMPSEC 15→16; 11 flips,
6 P→F vs 2 F→P; wall 6014 s, 21.5 ms/tok).

**Methodological finding — the surrogate metrics do not rank variants at
this quality level.**  Across three canonicals: uniform Q4_K has the
WORST continuation NLL of the variants (0.1824) and the BEST eval
(85/92); Q4_K_M improves every distribution metric (NLL 0.1726, KL
0.0312, top-1 94.58%) and scores 81; imatrix likewise 81.  NLL/KL on
100 official nothink continuations and canonical thinking-mode eval
disagree in both directions at these deltas — the canonical eval is the
only arbiter for accept/reject, and per-case flip counts (6:2, 11 total)
sit at the edge of binomial noise for a 92-case suite.  Do not use the
distribution metrics as a acceptance proxy for future variants; use them
only to characterize shape (tail structure, containment) after the gate.

Tool fixes this round: `--tensor-suffix SUF=TYPE` (indexer-excluded
suffix match, after prefix overrides, before family defaults) and a
precedence bug where an override setting a tensor BACK to its source
type was silently undone by the family default (the `target == t->type`
guard needed an explicit `overridden` flag).  Measurement gotcha: the
scorer prints an avg_nll per case — grep `'summary cases'`, not the
first `avg_nll=` match.  Tool determinism re-verified: a fresh full-Q4_K
build is byte-identical to the keeper.

**Distribution metrics, baseline vs the keeper variant (Aug 27).**
score_official grew `--dump-logits FILE`/`--dump-max N` (per-position
full-vocab F32 rows over the same teacher-forced official-continuation
stream; "LGD1" header).  2048 paired positions over the 100-case flash
fixture:

- top-1 argmax agreement **94.19%** (1929/2048); baseline's top-1 is in
  the variant's top-5 and top-20 at **100%** — every disagreement is a
  near-tie reordering, never a missing candidate.
- KL(base||q4k) mean 0.0364 / median 0.00053 / p90 0.104; KL(q4k||base)
  mean 0.0387 (symmetric — no directional mass loss); JSD (base 2) mean
  0.0087 / median 0.00014 / max 0.265.
- Shape: ~90% of positions are virtually identical (median top-1 prob
  gap 1e-6); the means are carried by a ~10% tail of genuine near-ties —
  exactly the positions that flip greedy streams (17/25 GPQA streams
  diverge) while leaving sampled/graded behavior intact.
- Companion NLL (official continuation tokens): 0.1467 → 0.1824 (+24%
  relative, both very low in absolute nats).
- Analysis script pattern + dump format documented here; the imatrix
  variant GGUF was deleted after its canonical loss (reproducible via
  the tool + the collected .dat).

### Metal decode stage GPU counters

The end-and-wait stage profiler (`DS4_METAL_DECODE_STAGE_PROFILE=1`) adds a
synchronization per stage boundary and changes the schedule, so its numbers
are inflated by per-boundary waits.  The stage-counter diagnostic keeps the
production token mostly intact: every stage boundary commits the open batch
command buffer without waiting, so the GPU queue stays fed, and each stage's
GPU busy span is printed after the token:

```
DS4_METAL_DECODE_STAGE_PROFILE=1 DS4_METAL_STAGE_COUNTERS=1 ./ds4 -m ds4flash.gguf \
    -p "Write a short story." -c 8192 -n 24 --temp 0
```

Both env vars are required: the first arms the boundary macros, the second
switches them from end-and-wait to commit-only sampling.  The concurrent
shared-expert/routed-MoE overlap stays armed under counters (only the
serializing profiler disables it), but the per-stage command buffers queue in
order, so overlapped stages report their serialized costs.  The per-token
`total-cb-busy` line matches the production GPU-busy time (about 22.5 ms on
M3 Ultra at a short context), which is the check that the attribution is
faithful.  M3 Ultra decode at a short context attributes the token roughly as:
routed MoE 5.9 ms, attention output projections 4.8 ms, Q lora path 5.1 ms
(Q-A/KV/compressor quad kernel 41 us + Q-B matvec 59 us per layer), attention
core plus inverse RoPE 2.1 ms, router/shared gate-up 1.9 ms, and about 3.4 ms
of per-layer HC pre/post bookkeeping, with the remaining dense Q8_0 matvecs
streaming at 590-650 GB/s, i.e. at the memory wall.

### Metal decode raw-layer gathered attention A/B (45 t/s round, part 1)

Decode attention has two schedules with a per-layer-parity split: ratio-4
layers use the gathered path (fused KV staging with pad fusion + packed32
attention with fused inverse RoPE, two dispatches), while ratio-0/128 layers
with `n_comp == 0` used a five-dispatch raw path (ring copy, standalone pad,
vec, plain reduce, standalone RoPE tail).  Commit-only stage counters showed
the raw layers at ~65 µs versus ~32 µs — about 0.7 ms/token, hidden in the
averaged ledger.  Decode now routes `n_comp == 0` layers through the gathered
path (the staging kernel already handles `n_comp == 0`; the packed32 gate's
`n_comp != 0` term is relaxed) whenever `use_mask == 0`.  Same kernels, same
reduction topology; the routing change is packaging only.  Rollbacks:
`DS4_METAL_DISABLE_DECODE_RAW_GATHERED_ATTN` (raw path) and
`DS4_METAL_DISABLE_DECODE_RAW_PACKED32` (staged vec+reduce instead of
packed32 on raw layers).

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_DECODE_RAW_GATHERED_ATTN \
  --include-selection --tokens 512
```

Balanced M3 Ultra A/B at the harness's 2048-token prefix: 43.18/43.10 t/s
(+0.16%; only layers 0–1 qualify once odd layers hold compressed rows), all
529 frontier rows / 68,389,120 logits / 528 selected ids bit-identical.  In
the campaign CLI regime (short prompt, `-n 128`), interleaved runs gave
44.59/44.67 t/s new versus 43.52/43.40 t/s rollback (+2.7%), transcripts
md5-identical (`db0c504c…`).

### Metal greedy chain decode A/B (45 t/s round, part 2)

The classic one-shot decode loop serialized every token through the host:
`waitUntilCompleted`, a 517 KiB logits readback, a CPU argmax, then the next
token's encode — about 0.5 ms/token of GPU idle at the boundary.  Chained
greedy decode (`metal_graph_greedy_chain`, engaged by
`generate_metal_graph_raw_swa` when resident, non-quality, non-streaming,
greedy) keeps the token id on-device: each token's graph ends with the GPU
argmax writing the next id into a device ring, and the next token's embedding
gathers it from the ring.  Encoding runs two tokens ahead of the host's
confirm cursor, so command buffers are always committed before the GPU drains
the previous token; the host lags only to print and check stop tokens (one
shared-event wait per token, hidden by the encode-ahead).  The hash-layer
router select reads the id from the ring through the existing
`use_token_buffer` kernel argument; the host-side hash override is skipped
(the resident fixed-route MoE never consumes it).  Bit-exactness: all kernels
and inputs are unchanged, and the GPU argsort top-1 reproduces the CPU argmax
including lowest-index ties, so the transcript is identical.
`DS4_DISABLE_GREEDY_CHAIN=1` restores the classic loop;
`DS4_GREEDY_CHAIN_DEBUG` / `DS4_GREEDY_CHAIN_DUMP_IDS` are diagnostics.
Interleaved M3 Ultra CLI runs: 45.47/45.55 t/s chained versus 44.73/44.73
classic (+1.75%), transcripts md5-identical; at a 2K-token prefix 44.12/43.40
(+1.7%); a 1024-token run and an early-stop prompt matched md5 exactly.

Combined round state: 43.46 → 45.51 t/s (+4.7%), bit-exact, `make test`
44/44.

### Session greedy chain (ds4-eval) + two headroom probes (Aug 22, round 3)

**Session chain decode** brings the round-2 chain to the session API used by
`ds4-eval`.  `ds4_session_chain_greedy_supported` /
`ds4_session_eval_chain_greedy` (ds4.c, next to `ds4_session_eval`) drive
`metal_graph_greedy_chain` on an existing session graph: seed = CPU argmax of
`s->logits` (identical to the classic temp-0 sample), pos0 =
`checkpoint.len`, approved tokens pushed into `s->checkpoint` by a trampoline
only after the caller's callback approves them; on full completion
`logits_out = s->logits` preserves the session logits invariant (on early
stop the logits are stale — eval only stops early on stop-token/quit/switch,
where they are never read).  Guards mirror the CLI set plus session state:
no GLM/CPU/distributed/TP/multi-tier, `support_kind == DS4_SUPPORT_NONE`, no
ssd-streaming/quality/CPU-router/steering, same env kill switch
(`DS4_DISABLE_GREEDY_CHAIN=1` forces the classic loop everywhere).

`ds4-eval` decodes in bursts (eval_chain_on_token carries the classic loop's
per-token bookkeeping verbatim).  Bursts are capped to stay out of the
think-close controller window (`remaining - soft_limit` while thinking, len>1
closes use the hard window) — inside the window the classic step runs, so
forced (non-argmax) closes behave identically.  Verified bit-exact: 3-case
traces (`--questions 3 -n 4096`, think and `--nothink`) identical except
volatile timestamp/seed/elapsed fields; per-case answers, token counts, and
think_close records match.  Speed: **45.22–45.24 t/s chained vs
44.44–44.59 classic** (+1.5%) on the eval cases; CLI oracle unchanged
(`db0c504c…`, 45.90 t/s), `make test` 44/44.

**Probe 1 — MoE down-sum6+HC4 tail fusion** (landed, default ON; rollback
`DS4_METAL_DISABLE_DECODE_MOE_HC_FUSION`): layer tail reordered to
pair-SwiGLU → plain shared-down matvec → new
`kernel_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_hc4_f32`
(metal/moe.metal) doing the down-sum6 + `shared_out` add + HC4 expand in one
dispatch, eliminating the `routed_out` f32 materialization and one kernel
boundary per layer.  Bit-exact (md5 n=128/512, harness bit-identical,
wrap-safe); **speed-neutral** — the old path already fused HC into the
shared-down kernel, so only a 16 KiB round trip was removed against the
26 MB expert stream.  Kept: one less dispatch and a simpler tail.

**Probe 2 — KV-staging direct read** (landed, default OFF; opt-in
`DS4_METAL_ENABLE_DECODE_RAW_DIRECT_KV`, hard-off
`DS4_METAL_DISABLE_DECODE_RAW_DIRECT_KV`, loud
`DS4_METAL_REQUIRE_DECODE_RAW_DIRECT_KV`): packed32 attention variant reading
the raw F32 ring and comp F16 caches directly (per-row source selection in
QK and PV loops, in-place tail, wrap-safe).  Bit-exact everywhere including a
probe-verified ring-wrap run — **but ~7% slower** (42.1 vs 45.3 t/s): the F32
raw rows (2048 B) are re-read and re-converted by all 64 head threadgroups
every token, ~2× raw-region traffic versus the one-time staging conversion
that amortizes across heads (+30 µs/layer on `attn_inv_rope`).  Confirms the
round-1 lesson: more parallel traffic throttles this GPU.  May still win on
devices with different L2/ALU balance.

Round-3 verdict: the handoff's remaining-headroom list is exhausted — HC
fusion neutral, direct-KV negative, and the concurrent shared-expert stream
(IQ2-gated) is contraindicated by the same throttling evidence.  The
remaining wall−GPU gap is ~0.1 ms/token; further gains need a cheaper MoE or
attention core, not packaging.

### Metal decode head attribution + chain-mode stage counters (Aug 22, 50 t/s campaign item 0)

The ledger's last unattributed ~0.8 ms is now measured; two tooling gaps had
hidden it.  The output head had no stage boundary after the vocab matvec, so
the final `end_commands` command buffer absorbed it, and the greedy chain
never reported counters at all (it also refused to engage under
`DS4_METAL_DECODE_STAGE_PROFILE`).  A `logits` output-stage boundary
(`DS4_METAL_OUTPUT_STAGE_PROFILE=1`) now closes the classic ledger, and the
chain reports per-token counters itself: each token ends with an `argmax`
sample covering the GPU argmax + event-signal tail, and a token's stages are
reported only at its confirm wait (GPUEndTime is unset in flight), with the
encode-ahead tokens' samples compacted for their own later reports.  The
chain now engages under commit-only counters; end-and-wait profiling still
forces the classic loop.

```
DS4_METAL_STAGE_COUNTERS=1 DS4_METAL_DECODE_STAGE_PROFILE=1 \
  DS4_METAL_OUTPUT_STAGE_PROFILE=1 ./ds4 -m ds4flash.gguf \
  -p "Write a short story." -c 8192 -n 24 --temp 0
```

M3 Ultra MXFP4 short-context chain decode (mean of 17 non-ratio-boundary
tokens): the head costs **0.90 ms/token** — Q8_0 logits matvec 0.769
(129280×4096, ~563 MB at ~730 GB/s: at the wall), GPU argmax 0.086 (a full
bitonic descending argsort for top-1 over 505 KiB — ~10× a dedicated
blockwise reduce, so A5's logits-epilogue top-1 fusion can recover most of
it), HC collapse stages 0.045.  The ledger now sums to the total-cb-busy
line: routed_moe 6.86, q_path 5.17, attn_output 4.79, router 1.92, attention
core 1.37, HC pre ×2 1.69, head 0.90 — 22.7 ms with the ~0.7 ms counter
commit tax versus the 21.96 ms production wall.

### Metal prefill stage counters (Aug 22, 50 t/s campaign P2)

The commit-only stage counters now cover prefill.  The batch layer encode
already carried per-stage boundaries; counter mode now labels them
`lNN:stage`, `metal_graph_layer_stage_profile_start` no longer drains per
layer under counters, and `metal_graph_prefill_layer_major` resets/reports
per chunk in both the single-buffer and split schedules (split-path samples
are timing-complete because each layer still ends in `end_commands`).  The
full env set:

```
DS4_METAL_STAGE_COUNTERS=1 DS4_METAL_LAYER_STAGE_PROFILE=1 \
  DS4_METAL_Q_STAGE_PROFILE=1 DS4_METAL_INDEXER_STAGE_PROFILE=1 \
  DS4_METAL_OUTPUT_STAGE_PROFILE=1 ./ds4 -m ds4flash.gguf --prompt-file …
```

M3 Ultra MXFP4 warm-page ledgers (one-shot; the one-shot CLI passes its
progress callback as `display_progress` unconditionally, so ≥32-token
prefills take the per-layer-drain split schedule regardless of TTY — the
P1 target; busy spans below are still faithful):

- n=6 tokens, 337 ms/pass — the per-layer fixed-cost map: routed_moe 1.82
  ms/layer (36 distinct experts × ~14 MB ≈ 0.50 GB ⇒ 0.63 ms byte floor;
  ~3× above it — the small-N GEMM waste P4 targets), hc_pre ×2 1.70,
  output_proj 0.82, shared gate/up+down 1.24, indexer_setup 0.93 per
  ratio-4 layer, compressor 0.57, q_b 0.29, attention 0.26, router 0.25,
  kv_path 0.19.
- n=637, 1391 ms (32 ms/layer): routed_moe 44.4%, output_proj 13.9%,
  attention 9.0%, hc_pre 6.6%, q_b 6.2%, shared_gate_up 4.0%, kv_path 4.0%.
- n=3092, 5184 ms (121 ms/layer): routed_moe 37.6% at **20.6 TFLOPS
  effective** — the compute-bound estimate (~19 TFLOPS ≈ 70% of FP16 MMA
  peak) confirmed from stage data; attention 13.7%, output_proj 15.4%.

Per-layer cost model from the three points: ~8.5 ms fixed + ~36 µs/token
marginal.  Decode oracles unchanged (`db0c504c…` n=128), `make test` passes.

### A1 probe — HC producer tail restructure: measured neutral, reverted (Aug 22)

The fused decode HC producer (`kernel_dsv4_hc_rms_norm_mix_f16_cluster2_
pre_norm`, ~19.6 µs/call ×86/token) was restructured per roadmap A1: the
4×4 Sinkhorn comb parallelized one-row-per-lane (double-buffered shmem
staging, one simdgroup barrier per iteration, redundant per-lane column
normalization in the original order) and the all-thread device seq_cst
fences shrunk to single-lane release/acquire.  Two independent findings:

- **Not bit-exact**: the serial comb's four unrolled rows compile to a
  row-position-dependent reduction tree (cross-row SIMD reassociation); the
  per-lane version lands rows 0/3 one ulp off (rows 1/2 exact), amplified to
  ~6e-8 over the 20 iterations — deterministic, dump-bisected to the comb
  slice (mixes and collapse outputs bitwise identical).  A per-lane rewrite
  cannot reproduce the serial kernel's compiled arithmetic from source.
- **Speed-neutral anyway**: interleaved CLI A/B, 45.90/45.87 vs
  45.92/45.91 t/s — the completion protocol + serial Sinkhorn hold no
  measurable latency; the stage's ~20 µs is dispatch/dependency floor plus
  the phase-1/2 streams, not the tail.  (The one fused dispatch boundary
  costs ~9 µs; cf. hc_flat_norm.)

Reverted; no rollback env retained because the variant was not bit-exact.
A1's 0.7–1.0 ms estimate does not exist in the tail.

### A3 probe — Q norm/RoPE deferral into the attention consumer: blocked by fast-math (Aug 22)

Roadmap A3 (mirror of the landed inverse-RoPE deferral: drop the per-layer
per-head Q RMS-norm + RoPE dispatch by having each packed32 attention
threadgroup redo its head's 512-wide norm + rotation on the raw q_b row) was
built and bisected.  Proven bit-exact: the deferral plumbing (skip at the q_b
site, standalone re-run at the attention site; rollback env
`DS4_METAL_DISABLE_DECODE_Q_NORM_ROPE_DEFER`, fallback-forcing diagnostic
`DS4_METAL_FORCE_DECODE_Q_NORM_ROPE_DEFER_FALLBACK`) and the in-kernel norm
tree emulation (4 simdgroups + zero-padded 32-slot plane + final 32-lane
simd_sum — bitwise on all 64 heads, all layer ratios).  Under
`DS4_METAL_MATH_SAFE=1` the fused path matched the rollback transcript
exactly.  Under the production fast-math library the YaRN rope tail
(three mul-add blend/rotation expressions) contracts differently inside the
attention kernel than inside the standalone kernel: 1-ulp drifts on
compressed layers, and source-level pinning (strict/fma/lerp forms, a
noinline shared helper) either missed ~26 elements or cascaded the whole
kernel's contraction (norm dots included).  Not landable bit-exactly;
reverted.  The (divergent) fused build also measured slower in a single
probe (43.77 vs ~45.5 t/s), so the win may not have been there regardless.
Fusion rule for this codebase: only contraction-free code (explicit
reduction trees, builtins, single muls) may move between kernels
bit-exactly; mul-add chains and transcendental-adjacent blends may not.

### A4 — router_project_select port to pre-M5/MXFP4: closed by analysis (Aug 22)

The M5 kernel folds the top-6 select into the router matvec's last
threadgroup, but computes no shared-expert work.  Traced the MXFP4 resident
decode FFN flow: today 5 dispatches — router+shared gate/up fused (the
shared expert free-rides as extra threadgroups in the router matvec),
select, early shared-down (independent of the MoE), MoE pair-SwiGLU, and
the landed sum6+HC4 tail (folds the shared_out add + HC expand).  Porting
the fused select keeps the count at 5 (router+select, MoE pair, plain sum6,
shared gate/up late, shared-down+HC expand) while losing both overlap slots
and disqualifying the sum6+HC4 fusion (`fuse_moe_down_hc` gates on
`router_shared_done != 0`).  The M5 win is inseparable from the IQ2-gated
parallel-FFN concurrent encoder (`parallel_full_ffn_eligible`), which does
not exist for MXFP4.  No code; not measured because the structure is
dispatch-neutral by direct reading.  The overlap machinery (selected-id
async loads) never engages for resident decode either way.

### Metal commit-only prefill split for display progress (Aug 22, 50 t/s campaign P1)

The `callback_split` schedule (`display_progress != NULL && n_tokens >= 32`)
forced 43 per-layer `end_commands` drains per chunk solely so the progress
bar updated — and because the one-shot CLI passes its progress callback as
`display_progress` unconditionally, even piped one-shot runs paid them.
When the split exists only for progress (`n_tokens <= 2048`, not
streaming/throttle/imatrix/split-profile), each layer now commits without
waiting via `ds4_gpu_flush_commands_progress`, and the bar rides each
command buffer's completion handler (GPU-true timing, no host drain).
>2048-token chunks keep draining (they also bound transient memory).
Rollback: `DS4_METAL_DISABLE_PREFILL_FLUSH_PROGRESS=1`.

Zero arithmetic change: md5 `ed17c76a…` on a 637-token prompt (-n 32) is
identical with/without the flush, the `db0c504c…` oracle holds, `make test`
passes, SSD streaming smoke OK.  Interleaved one-shot M3 Ultra prefill A/B:
**+2.7% at ~250 tokens** (411.5 vs 400.7 t/s), **+1.8% at 637** (541.9 vs
532.0), control at 3092 unchanged (650.7 vs 650.6) — ~17–20 ms per chunk of
drain idle removed.

### A2 probe — MXFP4 dequant constant-space LUT: measured negative, gated off (Aug 22)

The prescribed inner-loop variant: read the 16-entry MXFP4 value LUT
straight from constant space (no threadgroup staging, one less barrier per
threadgroup) in the two decode fixed-route kernels (pair-SwiGLU static,
sum6+HC4 static; templated `LUT_CONST` shares the same bodies).
Bit-identical (harness: 529 rows / 68,389,120 logits / 528 ids exact),
but **−2.7% decode** (42.41 vs 43.59 t/s at the harness's 2048-token
prefix): divergent constant-cache gathers lose to the threadgroup-staged
LUT on this GPU.  The other prescribed variants are blocked structurally:
uchar loads can't vectorize (17-byte block stride ⇒ unaligned), and
register select chains raise ops/byte.  Kept gated OFF (opt-in
`DS4_METAL_ENABLE_MXFP4_CONST_LUT`), matching the direct-KV precedent.
The routed-MoE dequant stays issue-rate-bound; no safe lever found here.

### Metal prefill chunk-size sweep (Aug 22, 50 t/s campaign P5)

`metal_prefill_variant_bench` grew a `--prefill-chunk N` option (the harness
used to hardcode 4096).  Balanced A/B at the 8192-token prefix (8 runs each,
control and a no-op candidate on the same path): **615.9 t/s at chunk 2048,
640.4 at 4096, 645.0 at 8192**.  The 4096 default is near-optimal: −4% at
2048, +0.7% at 8192 — not worth doubling the transient-buffer envelope the
4096 choice was designed to bound.  At a 32768-token prefix the blocks skew
with thermal soak (4096: 418–544 t/s within one process; 8192: 572–573),
consistent with no large chunk lever at long context either.
Recommendation: keep 4096.

### Long-context prefill decay profile (Aug 22, 50 t/s campaign P6)

Profiled with the P2 commit-only counters on a ~59k-token prompt (15 chunks
of 4096): per-chunk GPU busy grows 6.81 → 10.51 s monotonically.  The decay
is **not** the attention core (1.29×); it is the indexer machinery, which is
O(n_comp) per chunk: `score` 95→1782 ms/chunk (18.7×), `compressor`
0→1700 ms, `indexer_setup` 102→1142 ms, `topk` 20→275 ms.  (Chunk-0 routed
MoE reads ~25% high from first-touch page faults; steady-state compute
stages are flat per chunk.)  Follow-on target: the indexer score pass
(`ds4_gpu_indexer_scores_batch_tensor`) at long prefixes and the
compressor's per-chunk work.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control,
or `--help` to compare explicit split schedules.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

The pre-M5 one-token Flash attention-output LOW projection also has an exact
fixed-shape Q8_0 kernel. Compare it with the generic rollback using:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_ATTN_OUT_LOW_Q8_STATIC \
  --include-selection \
  --tokens 512
```

Balanced M3 Ultra A/B runs favored the fixed-shape kernel by 0.53%, 0.53%,
and 0.58% at a 2K-token prefix (43.03/42.80, 42.96/42.73, and 43.24/43.00
tok/s) and by 0.48% at an 8K-token prefix (38.44/38.26 tok/s). An independent
IQ2/Q2-model run at 2K gave 44.47/44.23 tok/s (+0.56%). All 1,909 compared
rows, 246,795,520 full-vocabulary logits, and 1,904 selected token IDs were
bit-identical.
Performance was measured on M3 Ultra; the exact host gate covers the shared
M1-M4 Flash shape and otherwise retains the generic kernel.

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

For resident pre-M5 MXFP4 prefills of 32 through 2047 tokens, the exact
scatter map, compact pair tile, pair/down tail culls, and down half-LUT now
use the same defaults as the established 2K+ path. Compare the complete
short-prefill extension with its aggregate rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 256 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_SMALL_PREFILL
```

Balanced M3 Ultra A/B medians for the tuned path versus the rollback were
112.43/89.25 tok/s at 32 tokens (+26.0%), 111.95/89.30 at 33 (+25.4%),
180.48/142.59 at 64 (+26.6%), 275.72/219.98 at 128 (+25.3%),
389.27/316.93 at 256 (+22.8%), 512.94/439.39 at 512 (+16.7%),
173.94/157.08 at 1024 (+10.7%), and 632.30/593.70 at 2047 (+6.5%). Every
one of the 64 measured runs produced bit-identical full-vocabulary logits.
Performance was measured on M3 Ultra; the guarded default also covers the
shared resident M1-M4 path.

### Metal batch indexer-query pruning A/B

On zero-prefix ratio-4 layers, the indexer query and its per-head weights are
not consumed until the compressed cache grows beyond the 512-row top-k.  The
resident pre-M5 path now skips the otherwise dead Q projection, RoPE, QAT, and
weight projection for batches of at least 32 tokens while the final compressed
count remains at or below top-k.  The indexer compressor and its persistent
cache/state updates are unchanged.  Compare the pruned path with its rollback:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 2048 \
  --warmup-tokens 2048 \
  --repeats 4 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_BATCH_INDEXER_QUERY_PRUNE
```

Balanced M3 Ultra A/B throughput for the pruned path versus rollback was
115.00/112.98 tok/s at 32 tokens (+1.79%), 277.42/273.81 at 128 (+1.32%),
509.19/502.28 at 512 (+1.38%), 606.13/597.65 at 1024 (+1.42%), and
660.17/651.29 at 2048 (+1.36%).  The last eligible prefix, 2051 tokens, gained
1.41%; 2052 tokens was flat, confirming that the query path remains enabled
when the 513th compressed row first makes top-k selection necessary.  All 56
prefill runs and 7,239,680 compared full-vocabulary logits were bit-identical.
A 2051-token prefix followed across the row-513 transition also matched three
full-vocabulary rows and two selected token IDs exactly.  Performance was
measured on M3 Ultra; the guarded path covers resident single-device M1-M4.

### Metal batch Q/KV finalizer A/B

The M3 resident Flash prefill path now follows vLLM's horizontal Q/KV
finalization schedule while retaining DS4's existing Q-head PSO. It dispatches
that exact Q RMSNorm+RoPE kernel concurrently with a KV-only kernel that folds
KV RoPE, the FP8 round trip, F16 rounding, and raw-ring insertion together.
The default covers 128 through 4096 tokens per dispatch; longer contexts use
the normal 4096-token chunks. A 32-token dispatch regressed and 64 tokens did
not clear the 0.3% acceptance threshold. Compare the retained path with its
serial rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 512 \
  --warmup-tokens 512 \
  --repeats 4 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_BATCH_QKV_FINALIZE
```

Balanced M3 Ultra A/B throughput for the concurrent path versus rollback was
273.60/272.63 tok/s at 128 tokens (+0.36%), 502.44/500.37 at 512 (+0.41%),
598.07/595.87 at 1024 (+0.37%), 651.27/649.11 at 2048 (+0.33%), and
615.06/612.37 at 8192 (+0.44%). All 56 prefill runs and 7,239,680 compared
full-vocabulary logits were bit-identical.
A 512-token prefill followed by decode also matched 73 full-vocabulary rows and
72 selected token IDs exactly, covering the persisted raw-cache state. The
schedule is based on vLLM's
[fused DeepSeek V4 finalizer](https://github.com/vllm-project/vllm/blob/c8de519917ce549f72132952116185e38b37c95d/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu#L382-L603),
but keeps Q and KV in separate Metal pipeline states to preserve DS4's exact Q
fast-math code generation.

### Metal prefill indexed-attention four-row staging A/B

The resident single-device pre-M5 indexed-attention prefill path now stages
four raw or F16-compressed K/V rows per threadgroup barrier while consuming
them in the original scalar-kernel row order. The specialization covers
non-quality batches of at least 32 tokens with 64 heads, 512-wide heads,
top-k 512, a 128-row raw window, ratio 4, and more than 512 compressed rows;
SSD streaming, TP2, decode, M5, and other shapes keep the original path.
Compare it with the
one-row staging rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 8192 \
  --warmup-tokens 4096 \
  --repeats 1 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_INDEXED_ATTN_PREFILL_RB4
```

Balanced M3 Ultra A/B throughput for four-row staging versus rollback was
631.53/626.50 tok/s at 4100 tokens (+0.80%),
620.39/614.76 at 8192 (+0.92%), 594.11/589.36 at 16384 (+0.81%), and
557.91/553.71 at 32768 (+0.76%). The 2048-token boundary, where compressed
rows do not yet exceed top-k, was flat. All 16 active-path prefill runs and
2,068,480 compared full-vocabulary logits were bit-identical. A direct forced
kernel oracle also matched all 1,048,576 attention-output floats exactly while
covering raw-ring wrap, visibility stops, and one- through three-row tails.
Separate 8K and 32K prefix-to-decode checks matched 106 full-vocabulary
frontiers, 13,703,680 floats, and 104 selected token IDs. Eight-row staging was
also exact but was 0.17-0.21% slower at the first two screened sizes, so only
the four-row specialization is retained. Performance was measured on M3
Ultra; the guarded default covers the shared resident M1-M4 path.

The same guarded path now groups sixteen heads per 256-thread workgroup by
having each SIMDgroup update two heads from every staged four-row block. This
halves K/V staging and workgroup count without changing either head's row
order or online-softmax arithmetic. Compare it with the accepted eight-head
RB4 rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 8192 \
  --warmup-tokens 4096 \
  --repeats 1 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_INDEXED_ATTN_PREFILL_HEADS16_DUAL_RB4
```

Balanced M3 Ultra A/B throughput for heads16 dual RB4 versus heads8 RB4 was
638.45/629.00 tok/s at 2052 tokens (+1.50%), 651.08/637.46 at 4100
(+2.14%), 640.33/626.12 at 8192 (+2.27%), and 613.72/600.81 at 16384
(+2.15%). The first matched 32768-token pair was 576.10/563.42 (+2.25%);
later slots in that process encountered severe system throttling and are not
used for the comparison. All 28 timed full-vocabulary rows were bit-identical.
The forced direct oracle matched all 1,048,576 attention-output floats and
proved both the dedicated rollback and new selector. Separate 8K and 32K
prefix-to-decode checks matched 94 full-vocabulary frontiers, 12,152,320
floats, and 92 selected token IDs.

### Metal batch MoE sum6-to-HC4 epilogue A/B

The resident single-device pre-M5 MXFP4 prefill path now consumes the six
routed expert-down rows directly in the HC4 expand/add/split epilogue. The
kernel preserves the original `s0 + s1 + ... + s5`, shared-expert add, and HC4
post/comb accumulation order while removing the standalone sum6 dispatch and
the routed-output F32 materialization. The default covers batches of 32 through
4096 tokens; longer prefixes use the normal 4096-token chunks. Debug, profiling,
steering, SSD, TP2, quality, decode, and other tensor shapes keep the original
path. Compare the fused path with its rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 8192 \
  --warmup-tokens 4096 \
  --repeats 4 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_BATCH_MOE_SUM6_HC_FUSION
```

Balanced M3 Ultra A/B throughput for the fused path versus rollback was
663.24/660.81 tok/s at 2048 tokens (+0.37%), 621.83/619.49 at 8192
(+0.38%), 596.50/594.76 at 16384 (+0.29%), and 560.30/558.54 at 32768
(+0.32%). The 32-token point was flat (-0.03%); 128, 512, and 4096 tokens
gained 0.19%, 0.18%, and 0.28%, respectively. All 64 prefill runs and
8,273,920 compared full-vocabulary logits were bit-identical. A direct oracle
also matched all 527,372 HC output floats exactly for a tail shape and the
production 32-by-4096 shape. Separate 8K and 32K prefix-to-decode checks
matched 146 full-vocabulary frontiers, 18,874,880 floats, and 144 selected
token IDs exactly. Performance was measured on M3 Ultra; the guarded default
covers the shared resident M1-M4 shape.

### Metal batch Q8 attention-output-to-HC4 epilogue A/B

The resident single-device pre-M5 Q8 attention-output path now feeds the
aligned 8192-to-4096 output-B matmul tile directly into the HC4 expand/split
epilogue. The specialization preserves the legacy Q8_0 dequantization and
simdgroup-MMA order, stages every 64-by-32 result through an 8 KiB F32
threadgroup tile, and then repeats the original scalar HC4 post/comb
accumulation order. This removes the global F32 `attn_out` write/read and the
standalone HC dispatch without removing the materialized F32 rounding
boundary. The default is deliberately limited to aligned batches of 512
through 4096 tokens; shorter batches, tails, debug/profiling/steering,
quality, SSD, TP2, M5, and other shapes retain the original path. Compare the
fused path with rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 8192 \
  --warmup-tokens 4096 \
  --repeats 2 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_BATCH_ATTN_OUT_HC_FUSION
```

Balanced M3 Ultra A/B throughput for the fused path versus rollback was
526.16/523.77 tok/s at 512 tokens (+0.46%), 668.43/663.66 at 2048
(+0.72%), 626.47/621.49 at 8192 (+0.80%), and 599.57/596.92 at 16384
(+0.44%). All 44 retained timing runs and 5,688,320 compared
full-vocabulary logits were bit-identical. A direct production-shape oracle
also matched all 262,144 low-rank values and 524,288 HC outputs exactly and
confirmed that the fused path did not touch the dead `attn_out` buffer.
Separate 8K and 32K prefix-to-decode checks matched 14 full-vocabulary
frontiers, 1,809,920 floats, and 12 selected token IDs exactly. A standalone
32K timing process was discarded because sustained thermal throttling changed
slot time from 58 to 142 seconds; the 32K run is correctness evidence only.
Performance was measured on M3 Ultra; the guarded default covers the shared
resident M1-M4 path.

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.
