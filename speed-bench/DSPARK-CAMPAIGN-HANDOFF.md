# DSpark campaign — handoff (rounds 1–10, sessions of Aug 22–23 + 26, M3 Ultra)

Repo `/Users/ifioravanti/github/ds4`, branch `perf/decode-50tps`, M3 Ultra.
Rounds 5–9 are committed on this branch (code + docs); round 10 (Aug 26)
left two tooling landings uncommitted in the working tree (the propose-chain
stage-counter map in ds4.c and the ds4-eval `--source` filter) plus the
round-10 records in README/handoff and a documenting comment at the f16
small-out nrow gate in ds4_metal.m.

**Record of truth:** `speed-bench/README.md`, section "DSpark speculation on
M3 Ultra" (rounds 1–10, every measurement, every negative result). Read it
first. This file is the quick-start; the README is the evidence.

## Where the campaign stands

Goal: beat the ~45.5 t/s plain-decode baseline decisively with DSpark
speculation. Current state (all validated, all measured interleaved):

- Verify pass: 66.6 → **54.3 ms** production (rounds 7–8; GPQA row-weighted
  average 47.3 ms over 58 verifies). Floor ≈ 45 ms (≈ 2 plain decode tokens;
  routed-expert bytes are irreducible).
- Fixture e2e (32 tok, temp 0): **all five prompts net-positive**;
  python_reverse ~59 t/s vs ~47 plain. c_add trips a pre-existing scheduler
  guard (accepted_draft 6 < 8 at 32 tokens — byte-identical class
  before/after all changes; tolerated).
- 128 tokens: medium code 47.8 vs 46.4 plain (**+3.2%**); story 42.1 vs
  46.2 (**−9%, structural** — corrected map below).
- GPQA Diamond (round 10 correction): **case 1's 0.6 s wall win does NOT
  generalize.**  25-case sweep ×3 medians: aggregate **−3.2% (dspark
  slower)**, 10W/13L/2T; 17/25 token streams diverge from plain (non-strict
  verify numerics; 3 answer flips); losses concentrate in low-yield cases.
  DSpark's reliable win band is high-yield code-class content.
- Economics: break-even ≈ 3 accepted/cycle (propose ~9 ms + verify ~54 ms
  vs 21.5 ms/token). Code yields 4.4 → wins; GPQA mixed (case 1 1.80,
  25-case aggregate 0.47); story 0.31 → loses. M3 Ultra decode already runs
  at ~730 of 800 GB/s, and the 6-row verify reads ~12.8 distinct
  experts/layer vs decode's 6.
- **Round 10 closed the round-9 item 1 (propose chain) NEGATIVE**: the
  chain is kernel-bound (FFN 48% of 3.57 ms GPU-busy/round), not
  dispatch-bound; a rope-merge + copy-fold dispatch bundle measured
  perf-neutral and was reverted. Also caught: the DSpark scheduler is
  wall-clock-driven — any propose-timing perturbation flips the schedule
  basin (avg_accept 4.4↔1.5) with byte-identical outputs, so single-run
  fixture t/s is not a build comparator. Details in the README round-10
  addendum.

## Round 9 (Aug 23): item 1 closed by measurement, item 3 landed

Handoff item 1 ("session bookkeeping trim, ~1 ms/token, no numerics risk")
was executed profiling-first and **its premise measured false**. No code
landed; the tree is net-unchanged; the full ritual re-ran green.

Corrected story machinery tax (**1.15 ms/token**, not ~1.8, and it is not
bookkeeping), interleaved ×3, 128-token lighthouse, conf-1.0 probe
(`--dspark-confidence 1.0`, drafting dead, streak escalation active:
112 skips / 7 proposes / 119 cycles):

| slice | ms/token | why it is not cuttable |
|---|---|---|
| chain-execution loss | **0.62** | `support_kind != NONE` rejects `ds4_session_chain_greedy_supported`; every DSpark token runs the classic wait/readback/argmax loop |
| propose machinery | 0.24 | ~20 ms first-propose markov-head first-touch amortized over 119 cycles + 6 explores; the explores are the scheduler's regime probes |
| capture + ring-maintain | ~0.05 | capture is ~free (3 target layers 40/41/42, one weighted-sum dispatch each; GPU stage-busy identical 2798 vs 2796 ms); it feeds `ring_maintain`'s support-KV alignment — gating either deadlocks streak recovery |
| unattributed residual | ≤0.29 | outside every DS4_DSPARK_STATS timer, GPU streams identical; submission-side or thermal, not actionable at this size |

Scoped, not built: **chain restore between bursts** — measured ROI is
story-only (~+0.58 ms/token on ~93% of tokens → 42.1 → ~43.2, still −6.5%
vs plain); code gains ~0 (every cycle bursts, the chain never gets ahead);
GPQA +~0.06. Emit-path surgery + ahead-encode/verify KV interaction risk
not justified by a story-only partial recovery.

## Landed this campaign (rounds 5–8; each has a rollback env)

| Change | Effect | Rollback |
|---|---|---|
| Chained GPU markov+argmax (propose) | prop_markov 1.33 → 0.65 ms/round | `DS4_DSPARK_NO_GPU_MARKOV=1` |
| nrow Q8 small-N matvec (verify+propose) | verify 74.3 → 68.9 ms | `DS4_METAL_DISABLE_Q8_NROW_MATMUL=1` |
| attention out-A grouped nrow | output_proj 9.0 → 6.2 ms/verify | same family |
| small-out nrow (router/HC proj, Q8+F16) | hc_proj 10.3 → 2.4, router mm 2.6 → ~0.5 | `DS4_METAL_DISABLE_F16_NROW_SMALLOUT=1`, `DS4_METAL_DISABLE_Q8_NROW_SMALLOUT=1` |
| fused batch router select | −0.7 ms/verify | `DS4_METAL_DISABLE_ROUTER_SELECT_BATCH_FUSION=1` |
| scheduler no-draft streak escalation (3→6→12→24→32) | story 39.9 → 42.1, code neutral | `DS4_DSPARK_SCHEDULER_NO_DRAFT_STREAK_CAP=0` |
| verify emit-chain fusion (round 8: exact pool + comp_row_finalize per-compressor modes, reuses decode-proven kernels) | verify 283.4 → 271.7 ms/5 passes (−2.4 ms/verify) | `DS4_METAL_DISABLE_DSPARK_COMP_FINALIZE_FUSE=1` |
| grouped routed MoE (opt-in; measured NEGATIVE 74 → 92 ms) | off by default | `DS4_DSPARK_ENABLE_VERIFY_GROUPED_MOE=1` to retry |

ds4-eval wiring: `--dspark/--dspark-confidence/--dspark-strict` work in
ds4-eval; support GGUF passed via `--mtp`. Greedy only (temp ≤ 0), else the
burst branch never engages and you pay support load for nothing.

## Validation ritual (every item, all must hold)

```bash
make ds4 ds4-eval                       # Metal kernels compile from embedded
                                        # source at startup; ds4_metal.o deps
                                        # on metal/*.metal — no lib step
./ds4 -m ds4flash.gguf -p "Write a short story about a lighthouse keeper." \
    -c 8192 -n 128 --temp 0 | md5       # MUST: db0c504c8203618552e685bd2c701e4f
make dspark-verify-depth \
    DS4_DSPARK_SUPPORT=gguf/DeepSeek-V4-Flash-DSpark-support.gguf
                                        # MUST: worst_argmax_gap=0.000
make dspark-acceptance \
    DS4_DSPARK_SUPPORT=gguf/DeepSeek-V4-Flash-DSpark-support.gguf
                                        # MUST: output_match=1 all five
                                        # (c_add guard trip = pre-existing)
make test                               # MUST: 44/44
```

## Measurement protocol + gotchas (each cost real time)

- **One ds4 process at a time** (instance lock). Serialize everything, and
  wait for the PID to fully exit — teardown of the 145 GB mapping lingers
  past process end and the lock outlives the pipe.
- **Sleep 15–30 s between timing runs** — thermal transients flip readings.
- Stats go to **stderr**; grep with `2>&1` before the pipe.
- Support model: `gguf/DeepSeek-V4-Flash-DSpark-support.gguf`; main model
  symlink `ds4flash.gguf` → the MXFP4 0731 GGUF. Pass `DS4_DSPARK_SUPPORT=`
  to the make targets.
- First propose of a process pays ~45 ms markov-head first-touch (PSO +
  page faults) — visible in 32-token fixtures, amortizes in real sessions.
- The verify router is **F16** 4096→256, not Q8.
- Wall time: interleaved A/B ×3, `verify=` from the DSpark stats as arbiter.
  Stage maps: `DS4_METAL_DECODE_STAGE_PROFILE=1 DS4_METAL_STAGE_COUNTERS=1`,
  aggregate stripping `lNN:` prefixes. Counter mode inflates (~26
  µs/boundary) and **shared_down misreads ~13 ms since the fused router
  select landed — artifact, treat as ~1.2**.
- **t/s deltas do not subtract** — convert to ms/token first. Round 9 also
  caught per-token stats timers (target_ms) exceeding wall (async prior
  work drains inside the next eval's wait): trust interleaved wall plus
  commit-only GPU stage-busy, not single host timers.
- `--mtp` without `--dspark` detects but does NOT activate the support
  model (chain still runs, prefill stays plain) — not a valid probe config.

## Measured NEGATIVE — do not retry without a new insight

- Expert-major grouped routed MoE at ~2.3 pairs/expert (L2 already dedups).
- Per-row routed MoE through single-token kernels; per-row HC pre; the
  dual-row HC-pre producer kernel (bit-exact, zero gain — hc_pre sharing
  was overestimated by the stage decomposition).
- Per-row base-head matmul through the multi-session rows-exact path
  (re-reads the 562 MB head per row).
- Static scheduler skip knobs (recover story, collapse code yield) — the
  streak shape exists because the pause must track regime persistence.
- Confidence sweep on GPQA: 0.4 and 0.8 both worse than the 0.6 default.
- **Round 9: bookkeeping trim as scoped** — the ~1 ms/token of trimmable
  host bookkeeping does not exist (map above); capture/ring gating on skip
  cycles deadlocks streak recovery; the propose explores are load-bearing.

## Open work, ranked (round-10 ranking)

1. **Chain restore between DSpark bursts** — recovers the 0.62 ms/token
   classic-loop loss, but measured ROI is story-only (see round-9 section);
   code/GPQA gain ≈ 0. Only if story matters enough for a partial recovery.
   The transition design must truncate the KV to the confirmed pos when a
   propose cycle follows chained skips (chain ahead-encodes 2 tokens with
   decode-kernel numerics; verify re-encodes with batch-kernel numerics).
2. **Whole-loop compressor/indexer fusion** — ceiling ~2–3 ms/verify;
   A3-class fast-math risk with only empirical gates; round-8 fusion
   already took the cheap 60%.
3. **Scheduler-freeze env** (new, tooling) — a DS4_DSPARK_SCHEDULER_FREEZE
   style knob pinning the schedule decisions would make fixture A/Bs
   deterministic; round 10 showed the wall-clock-driven scheduler flips
   basins on any timing perturbation (README round-10 addendum).
4. Support-model FFN kernel slivers — shared gate+up dual-output nrow
   (~30–50 µs/stage, the only untried dispatch-level item; low ceiling).
   The propose chain is otherwise closed (kernel-bound; round 10).
5. Kernel track otherwise closed: verify map (counter mode) routed_moe
   22.8 (floor), output_proj 6.1, compressor 5.0, q_path 4.2, hc_pre 3.4,
   attention 2.9, indexer_setup 2.7, router 2.7, hc_proj 2.4, kv_path 1.2.

Round-10 closures: propose chain (was item 1 — kernel-bound, fusion
perf-neutral, reverted); GPQA multi-case sweep (was item 4 — 25-case band
is parity-to-negative, recorded in README).

## If you take item 1 (chain restore)

Entry points: `ds4_session_chain_greedy_supported` (the
`support_kind != NONE` rejection that forces every DSpark token through
the classic wait/readback/argmax loop) and the speculative burst driver in
`ds4_session_prepare_dspark_draft_impl` / the eval decode loop's
`spec_burst_ok` branch. Design constraints from round 9's scoping:

- Capture (layers 40/41/42) and `metal_graph_dspark_ring_maintain` must
  keep running during chained skips — gating them deadlocks the streak
  recovery propose.
- When a propose cycle follows chained tokens, the chain is up to 2
  encodes ahead: those KV rows carry decode-kernel numerics while verify
  re-encodes with batch-kernel numerics. Truncate the KV to the confirmed
  pos and let verify re-encode (matches today's semantics); do NOT let
  the two numerics families mix in one cache window.
- Validate the transition cost first: story content has ~8 propose
  transitions per 128 tokens; the +0.58 ms/token recovery only nets if a
  transition (drain + KV rewind + cache_ends_at rebuild) costs ≲ a few ms.
- Full ritual plus interleaved A/B ×2 on the 128-token story pair; abort
  criteria: any c_add/net_saved class change, stream divergence, or a
  code/GPQA regression. And per the round-10 scheduler finding: compare
  builds by output + counter maps + x3 medians, never single fixture runs.

## Round-10 tooling landed (uncommitted — commit with the records)

- ds4.c: propose-chain stage-counter map (driver bracket +
  `ds%u:%s`-labeled boundaries). Run with
  `DS4_METAL_STAGE_COUNTERS=1 DS4_DSPARK_STAGE_PROFILE=1`; normalize per
  round by counting `ds0:attn_hc_pre` samples (rounds that reach the
  chain) — totals mix full and early-failed attempts.
- ds4_eval.c/ds4_help.c: `--source SUBSTR` (case-insensitive substring
  over the embedded case sources; mutually exclusive with
  `--case-sequence`). `--source "GPQA Diamond"` = 25 cases,
  "SuperGPQA" = 25, "AIME2025" = 25, "COMPSEC" = 17 of 92.
- ds4_metal.m: comment at the f16 small-out nrow gate documenting the
  round-10 negative (do not lower `out_dim >= 24`).
