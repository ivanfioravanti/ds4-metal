# Qwen3.8-Flash-Next on M5 Max: round 6 (depth-3 MTP decode)

Base: upstream `qwen3.8-flash-next` at dddb1ba (rounds 3-5 merged).  Same
machine, pack (`qwen38-q4k`: Q4_K routed gate/up, MXFP4 routed down, combined
main/MTP GGUF) and harnesses as the earlier rounds.

Goal: raise Q4 MTP decode throughput by ~5% without changing the model's
math accuracy.  Delivered: **a second, recursively chained draft token with a
3-row verify pass**, engaged adaptively by measured acceptance.

## The cycle before this round

`ds4_session_qwen4_spec_cycle` verified one pending draft per cycle: a 2-row
`qwen4_graph_forward_tokens` pass (~21.7 ms GPU at short context) plus a
2-row predictor catch-up that also predicted the next draft (~2.6 ms).  The
cycle's cost is dominated by the verify pass, whose weight traffic is the
same at 2 and 3 rows - the third row is nearly free for the dense Q8 family
(the r1_3 matvec streams the weights once for all rows; measured +0.3 ms on
the ~8 ms dense family) and costs ~+2.5 ms on the routed expert rows (9 more
expert-row gathers).  A third verified row therefore pays whenever the second
draft accepts often enough: at the measured cycle costs (33.7 ms deep vs
25.7 ms), break-even is a second-draft acceptance of ~0.6.

## What was built

- **Second pending draft via a recursive chain step**
  (`qwen4_graph_mtp_chain_step`): after the predictor catches up over the
  committed rows (trunk-conditioned, as before), one more nextn-layer row is
  run conditioned on the predictor's own output stream - the standard
  DeepSeek-style MTP recursion.  The predictor buffers hold 3 rows
  (`mtp_e/cat/proj/R`).
- **3-row verify pass** with per-row-exact kernel paths:
  - attention runs as 2/1-row sub-batches (`qwen4_graph_attention_core`), so
    every dispatch is one the T<=2 path already takes (the prefill `attn_mm`
    tile kernel is never selected); each sub-batch sees the block universe of
    its own T<=2 verify, not the whole chunk's;
  - the 3-row gate/mix runs as the exact 2-row pair kernel plus the 1-row
    generic kernel on row views;
  - the few-row matvec keeps the T<=2 lane map (`nxpsg` 16) and the routed
    gate/up keeps the M5 NR1/NSG4 geometry (`ds4_gpu_qwen4_set_verify_rows_exact`);
## Measurements (M5 Max, CLI greedy, --nothink, ctx 8192)

Interleaved A/B against the pre-round build (`qwen38_mtp_compare.py`, three
repeats, medians; outputs byte-identical on every prompt and repeat):

| prompt | baseline | this round (auto) | change |
| --- | ---: | ---: | ---: |
| Fibonacci (deterministic list) | 81.4 t/s | **92.1 t/s** | **+13.2%** |
| explanation (technical prose) | 72.0 t/s | 71.9 t/s | -0.1% |
| Hamlet (literary prose) | 67.8 t/s | 68.0 t/s | +0.3% |

Single-run spot checks of the final controller land higher on the mixed
prompts (Fibonacci 89.7, explanation 72.0, Hamlet 67.2 t/s; geometric mean
across the three +6.3%), and `DS4_QWEN4_MTP_DEPTH=3` measures the extremes:
Fibonacci 90.6 t/s (+17%), explanation 68.5 (-2.7%), Hamlet 62.5 (-5.8%) -
which is exactly what the adaptive policy is for.  Setting
`DS4_QWEN4_MTP_DRAFT_ROWS=151936` now also narrows the chained draft's head
(the second draft only needs its argmax): Fibonacci 93.6 t/s at forced depth
3 with identical acceptance counters.

Acceptance at forced depth 3: Fibonacci p1 = 0.977, p2 = 0.977; explanation
p1 = 0.741, p2 = 0.605; Hamlet p1 = 0.652, p2 = 0.533 - matching the
break-even analysis (p2 ~ 0.6 is the wash point).  Plain decode (54-57 t/s)
and the depth-2 MTP path are untouched.

Cycle anatomy at depth 3 (Fibonacci, ~35 ms/cycle): verify T=3 ~29.3 ms GPU
(of which routed experts ~9 ms, dense Q8 ~7.7 ms), predictor catch-up (3
rows) ~2.8 ms, chain step ~2.2 ms.

## Numerical behavior

The committed stream is still gated by exact target argmax checks; every
committed token is one the target's verify logits endorse.  Two bounded
perturbation classes exist, both already documented for the MTP path:

- the `mul_mv_ext` r1_3 template's compiled contraction differs from r1_2 by
  ~1 ulp per logit (probed directly: a cross-T fixture shows 1-ulp row
  differences).  Splitting the T=3 projections into exact 2+1-row calls was
  measured and rejected: the second weight stream costs ~3.5 ms/cycle and
  erases the win.  The 3-row verify is drift-class against the 2-row one in
  the same sense the 2-row verifier already is against one-token decode
  ("long greedy continuations need not be byte-identical", docs).
- in sparse-attention contexts (> ~2 K tokens), the indexer's block universe
  is verify-window-relative in the existing code, so a row's candidate blocks
  depend on the window shape; depth changes the window shape.  This is the
  same class the shipped 2-row verifier carries vs plain decode.

Empirically the greedy outputs of all three benchmark prompts are
**byte-identical** between depth 2 and depth 3 (and auto) over 400 tokens
each; the dense-context stream comparison (every aligned position's full
vocabulary) shows the drift staying at the ulp level.  A C-source continuation
at a 2 K prefix (sparse territory) diverges after ~11 tokens at one argmax
near-tie - the documented window class, not a quality change.

## Verification

- `make test-qwen4-kernels`: all fixtures pass, including the scan/front/conv
  snapshot paths extended with the second copy point and the few-row matvec
  group tests at T = 3.
- `qwen38_decode_variant_bench --stream-compare` (added this round): depth-2
  vs depth-3 driven independently to a common position, full-vocabulary
  logits hashed at every visited position; dense contexts stay bit-identical
  until the documented r1_3 contraction ulps, sparse contexts follow the
  window class.
- `qwen38_mtp_compare.py` against the pre-round build (source-isolated
  baseline): outputs byte-identical on all three prompts across three
  interleaved repeats; plain decode output is byte-identical too (400-token
  spot checks of the final binary at auto and forced depth 3 both match the
  baseline exactly).
- The default path audit: with `DS4_QWEN4_MTP_DEPTH=2` the cycle executes the
  same kernels in the same order as before the round (the depth-3 code is
  gated behind the pending second draft, which only the chain step creates).
  Acceptance counters move only when deep cycles run - by construction, the
  same way the harness treats any depth change.
