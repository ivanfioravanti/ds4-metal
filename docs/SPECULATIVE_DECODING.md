# Speculative Decoding

[README](../README.md)

Speculation proposes future tokens with a smaller draft block, then checks
them with the main model. An accepted prefix advances generation by several
tokens in one verification pass. It does not accelerate prefill.

It is opt-in. Gains depend on the prompt, model, backend, and context length;
poor acceptance can make it slower. Measure your workload rather than assuming
that a draft model always helps.

## DeepSeek V4.1 Flash: DSpark

V4.1 uses its own three-stage drafter. The V4 Flash support files below are
incompatible. This experimental path supports resident Metal inference and
Metal tensor parallelism. V4.1 DSpark cannot be combined with SSD streaming;
image requests fall back to ordinary target decoding.

Convert the draft shards from **the same source revision as the target GGUF**.
For the initial V4.1 Flash checkpoint:

```sh
revision=df42c109f1defefcbfcedbe7d905718a12266e40
hf download deepseek-ai/DeepSeek-V4.1-Flash --revision "$revision" \
  --local-dir /tmp/ds41-draft-source \
  --include config.json inference/config.json model.safetensors.index.json \
  model-00044-of-00048.safetensors model-00045-of-00048.safetensors \
  model-00046-of-00048.safetensors
make -C gguf-tools
uv run --with numpy python gguf-tools/deepseek41_dspark.py --hf /tmp/ds41-draft-source \
  --source-revision "$revision" \
  --out gguf/DeepSeek-V4.1-Flash-DSpark-support.gguf
./ds4 --metal -m gguf/DeepSeek-V4.1-Flash-Q4.gguf --dspark \
  --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-support.gguf --nothink
```

The converter requires NumPy and the native quantization library built by the
GGUF tooling. Use `--dry-run` to validate the source tensors before conversion.
It emits about 7.88 GiB of support weights, sharing the target embedding and
output head. The loader checks the checkpoint revision and draft layout.

For two-Mac TP, add the usual RDMA coordinator flags to the command above.
Only the coordinator needs `--dspark`, `--mtp-model`, and the support file.
The worker runs its ordinary target-model command. Both hosts must run this
version of the engine so that they can verify and commit the same prefix.

The drafter uses the last 128 target feature rows, proposes up to five tokens,
and filters them with `--dspark-confidence` (Metal default 0.6). The target
verifies the seed and proposals in a causal batch and restores the accepted
prefix, including window KV and unfinished compressor state. After loading a
saved target session, ordinary decode rebuilds the draft feature window.
The saved-session format does not include draft state.

Unlike the older DSpark paths described below, V4.1 always uses target sampling
at non-zero temperature, including target correction on a rejected proposal.
The live oracle checks greedy and sampled tokens, RNG state, full logits, and
live target state against ordinary decoding. Exact agreement is qualified by
those tests, not a promise across different hardware or execution settings.
`--quality` and `--dspark-strict` disable speculative acceptance.

This is opt-in experimental support, not a guaranteed speedup. Small verifier
batches and rejected drafts can cost more than ordinary decode; benchmark both
modes with the same prompt and sampling settings.

On two M3 Ultra Macs (80 GPU cores and 512 GiB each), the September 16,
2026 optimization round used a 7,956-token cold prefix, 512 greedy output
tokens, 327,680 allocated context, and confidence 0.6. Rates below are median
overall generation throughput from three runs. Ordinary and optimized DSpark
runs alternated; the initial DSpark column is the earlier three-run baseline.

| Configuration | Ordinary decode | Initial DSpark | Optimized DSpark |
| --- | ---: | ---: | ---: |
| Single Mac | 25.58 | 22.18 | 25.85 |
| Two Macs, TP RDMA | 30.09 | 20.92 | 28.95 |

Rates are tokens/s. The single-Mac gain over ordinary decode is modest
(about 1.1%); TP still trails ordinary decode by about 3.8%. DSpark remains
opt-in. Confidence 0.6 beat 0.45, 0.75 and 0.9 after optimization; shorter
draft caps and shallower command-buffer queues did not improve this workload.

The drafter now rejects low-confidence rows before vocabulary projection and
projects only requested proposal rows. Draft and verifier attention outputs
are batched; small Q8 batches share weight reads while retaining scalar
reductions. Small-batch HC and normalization fuse their existing BF16
boundaries. Solo verification submits layers while the CPU encodes subsequent
work. Rollback saves only overwritten raw-KV ring rows, reducing verifier
storage by about 59.5 MiB. TP verification reuses a two-slot RDMA receive
window and ordinary decode's checked-payload publication and GPU release
polling, avoiding per-gate control handshakes and shared-event releases.

Proposal and acceptance counts were unchanged. In TP, 318 draft cycles
accepted 194 extra tokens; 153 cycles proposed nothing. Generating 512 tokens
therefore evaluated 552 target rows, including rejected proposals. Improving
small-batch target verification remains more useful than merely raising the
reported acceptance percentage.

`make test-q8-decode-rows` checks exact Q8 output and buffer guards with solo
and TP thread-group sizes. The full DSpark oracle checks greedy and sampled
output, logits, RNG, and live target state on single-Mac, RDMA and TCP paths.

The matching cold HTTP sweeps covered 512 through 256K
nominal context with 128 output tokens, temperature 0.7, and one unseeded trial per configuration
and context. At the final 254,680-token prompt:

| Configuration | Prior ordinary prefill | Optimized DSpark prefill | Prior ordinary decode | Initial DSpark decode | Optimized DSpark decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single Mac | 629.17 | 628.52 | 20.66 | 17.12 | 20.42 |
| TP RDMA | 645.15 | 644.55 | 23.37 | 18.04 | 22.06 |

Optimized TP reached 30.48–31.89 tokens/s at nominal 512–16K contexts in
this chat workload. Its longer accepted draft batches differ from the raw
greedy benchmark above. Ordinary HTTP results are from the prior sweep.

Rates are tokens/s. Small differences in this single-trial HTTP sweep should
not be interpreted as statistically established gains.

### Verification follow-up

The follow-up kept the same target/draft weights, confidence 0.6, and proposal
and acceptance counts. Three rotating repetitions compared ordinary decode,
the preceding DSpark path with the new diagnostic switches disabled, and the
optimized path. The raw prompt had 7,956 tokens, generation had 512 greedy
tokens, and context allocation was 327,680.

| Configuration | Ordinary tokens/s | DSpark before tokens/s | DSpark after tokens/s | Verification before seconds | Verification after seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single Mac | 25.52 | 25.84 | 26.62 | 11.75 | 11.19 |
| TP RDMA | 30.09 | 28.97 | 30.22 | 10.18 | 9.50 |

Rates are median overall generation throughput. Verification time includes
prefix commit but excludes ordinary one-token steps and sampled correction
evaluation. The existing `verify` counter now reports this separately from
`target`, which includes both verification and ordinary target work.
The TP lead over ordinary decoding is marginal (0.43%); the three-run ranges
overlap. The single-Mac lead over ordinary decoding is 4.31%.

The verifier batches Q/K preparation, independent compressor/indexer
projections, and attention-output rounding/rotation. Cache publication,
pooling and index scoring stay causal. Small Q8 batches fuse their exact BF16
output boundary. TP overlaps independent Q/K projection and normalization
levels and vectorizes the checked payload checksum. Tiny Q4 MoE batches use
the scalar decoder's smaller thread-group layout. Final HC/norm inputs are
batched, redundant frontier copies are omitted, and raw-window undo plus KV
publication use one integer compute dispatch instead of two blit encoders.

The short oracle checks 448 greedy and 64 sampled outputs, RNG, full logits,
and live state, including forced six-row verification and rejected suffixes
across ring wraparound. Single-Mac, RDMA and TCP checks passed. The optional
`DS4_TEST_V41_LONG_CONTEXT=1` oracle allocates 65,536 context and forces six-row
batches at prefixes 16,383 and 32,767; both single-Mac and RDMA checks passed.
Supply a prompt containing more than 32,767 model tokens for that test.
Exact Q8/BF16 operator checks and buffer guards passed with both thread-group
layouts. Normal resident and SSD decode each passed 130 exact-state checks.
CPU-only syntax and targeted regressions passed; `make test` retained the
same nine pre-existing assertions with no new failures. Runtime validation
was on M3 Ultra, with no CUDA hardware run.

The HTTP sweep also exposed a terminal-boundary replay: after a speculative
stop token, the server rebuilt the entire 127,224-token prompt before sending
the final response, adding about 181 seconds after generation had finished.
Terminal rewinds now leave the shortened checkpoint invalid and defer its
rebuild until a later sync needs it. Resampling and continuing generation
still rebuild immediately. Six forced speculative stop boundaries per
topology, including follow-up requests, matched ordinary greedy responses;
the server regression suite passed. The final HTTP sweep includes this fix.

The matching cold HTTP sweep again covered 512 through 256K nominal context,
128 output tokens, temperature 0.7, and one unseeded request per point. At the
final 254,680-token prompt:

| Configuration | Prior ordinary decode | DSpark before this follow-up | DSpark after | DSpark prefill after |
| --- | ---: | ---: | ---: | ---: |
| Single Mac | 20.66 | 20.42 | 20.45 | 628.57 |
| TP RDMA | 23.37 | 22.06 | 23.44 | 645.37 |

Rates are tokens/s. The ordinary HTTP comparator is the prior sweep; unlike
the repeated native comparison, small HTTP differences are not established
statistical gains.

### Draft and snapshot follow-up

The next round batches the draft's attention-output BF16/rotation and final
HC reduction/normalization, and uses the existing fused HC expansion boundary.
Draft-token selection adds the vocabulary and Markov logits on the GPU and
uses the existing lowest-index argmax, reading back one token instead of two
full vocabulary rows. The verifier captures both recurrent pooling buffers
with one integer compute dispatch instead of two blit encoders. Target and
draft weights, confidence 0.6, proposal width and sampling remain unchanged.

Three rotating repetitions compare ordinary decode, the preceding DSpark
path with these five diagnostic switches disabled, and the optimized path.
The cold raw prefix has 7,956 tokens, generation has 512 greedy tokens, and
context allocation is 327,680. Acceptance counts are checked against the
preceding path. Median measurements:

| Configuration | Ordinary tokens/s | DSpark before tokens/s | DSpark after tokens/s | Draft before seconds | Draft after seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single Mac | 25.51 | 26.60 | 26.83 | 2.278 | 2.103 |
| TP RDMA | 29.97 | 30.21 | 30.49 | 2.291 | 2.132 |

Drafting time fell 7.7% on one Mac and 6.9% with TP. Verification/commit time
stayed effectively flat: single-Mac 11.19 to 11.19 seconds, TP 9.50 to 9.52
seconds per 512 outputs.

Full-logit, live-cache, RNG and output checks passed on single-Mac and RDMA,
including forced six-row verification, rejected suffixes and ring wraparound.
The long oracle checks prefixes 16,383 and 32,767; TCP fallback and normal
resident/SSD regression checks are also included. The full regression suite
retains the nine known assertions with no new failures. Runtime validation
is on M3 Ultra; CUDA hardware was not available.

The cold HTTP sweep uses the same ten prompts from 512 through 256K nominal
context, 128 output-token cap, temperature 0.7, and one unseeded request per
point. At the final 254,680-token prompt:

| Configuration | Previous DSpark decode | New DSpark decode | New prefill |
| --- | ---: | ---: | ---: |
| Single Mac | 20.45 | 19.83 | 628.51 |
| TP RDMA | 23.44 | 23.45 | 644.57 |

The new TP 256K request stopped naturally after 98 output tokens; all other
new points reached the 128-token cap.

Rates are tokens/s. The HTTP results are single sampled requests and use
previous sweeps as comparators; small differences are not established
statistical gains. Use the rotating greedy measurements for the controlled
before/after comparison.

## DeepSeek Flash: DSpark

DSpark is a separate support GGUF, not a standalone language model. It proposes
up to five future tokens. Match its checkpoint to the main model:

| Main checkpoint | Download | Support file |
| --- | --- | --- |
| Flash 0731 | `ds4f-dspark` | `gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf` |
| Flash Vision Experimental | `ds4f-vision-dspark` | `gguf/DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf` |

For the 0731 Q2 model:

```sh
./download_model.sh ds4f-q2
./download_model.sh ds4f-dspark
./ds4 --dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
```

For Vision Experimental, substitute its matching main model and support file.
Do not mix the two checkpoints. DSpark is not supported for PRO.
The same flags work in `ds4-agent` and non-batched `ds4-server` requests.

The support file adds about 5.6 GiB of weights plus runtime state. On Metal,
the main model can be resident or SSD-streamed. DSpark replaces the legacy
one-stage MTP drafter for that run; the two are not stacked.

Resident M5 paths batch supported verifier expert rows, including two-Mac TP.
On DGX Spark, resident Q2 also batches the seed with longer drafts and uses
small-batch Q8 and expert kernels. No extra flags are needed.
The scheduler can back off when drafting is unproductive. Defaults select the
fast paths; diagnostic environment variables are not needed for normal use.
Recorded comparisons are in [the QA guide](../QA_BEFORE_RELEASES.md).

For the tested Strix Halo coding configuration, use `--dspark --dspark-confidence 0.7` with the default five-token draft cap and scheduler. Client sampling is temperature `1.0`, `top_p=0.95`, `min_p=0`, and `top_k=0`; high reasoning was also checked on coding and tool-use requests. This uses opportunistic sampling as described below; exact-mode throughput is not qualified by these measurements. `--mtp-draft` controls legacy autoregressive MTP, not the DSpark draft width.

## GLM: built-in MTP

GLM's draft block is already in its main GGUF:

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --mtp
```

`--mtp-timing` also enables it and prints acceptance and timing counters.
The current GLM cycle commits up to two tokens. No external support file is
needed, and ordinary decode remains the default.

## Qwen3.8: built-in MTP

Both Qwen downloads include MTP and native BF16 n-grams in the main GGUF:

```sh
./download_model.sh qwen38-q4k
./ds4 --mtp
```

Ordinary decode uses the same file with `--mtp` omitted. For non-zero
temperature, add `--mtp-exact-sampling` to preserve the target sampling
distribution. See [Qwen setup](QWEN38_FLASH_NEXT.md) for Metal and CUDA.

The cycle drafts one token ahead by default and engages a **second, chained
draft** (one extra nextn-layer step conditioned on the predictor's own
stream, verified in a 3-row pass) while recent first-draft acceptance is
perfect, disengaging after repeated second-draft rejections.
`DS4_QWEN4_MTP_DEPTH=2` or `=3` fixes the depth;
`0` (default) is the adaptive policy.

## Sampling and reproducibility

For the older DeepSeek Flash and GLM paths, at temperature zero, accepted drafts
must match the target's greedy continuation. At non-zero temperature, the default mode is opportunistic:
ordinary tokens use the requested sampling settings, but matching greedy
drafts are accepted directly. Sampling resumes when the proposed suffix does
not match. This is deliberately more deterministic than ordinary sampling.

Use `--mtp-exact-sampling` to preserve the ordinary target sampling
distribution. Exact mode accepts greedy proposals with their target
probability and samples from the remaining distribution on rejection.

When a verified block crosses a tool sampling-mode boundary (for example
entering tool-call syntax during server decoding), the server rewinds to the
block start and re-evaluates the boundary token so the next sample uses the
new mode. Under exact sampling that rewind restores a pre-verify snapshot of
the recurrent state instead of resetting the graph, so long retained
contexts are not replayed at every boundary.

Accepted tokens keep the state produced by the batched verifier. Floating-point
reduction order can differ from one-token decode, so long greedy continuations
need not be byte-identical. For DeepSeek comparisons against the ordinary
target-only path, use `--quality` or `--dspark-strict`; these disable the
speculative acceptance path. They do not promise identical output across
different hardware or execution configurations.

Session-batched serving uses ordinary target decoding instead of combining
DSpark/MTP with the session batch. See [serving](SERVER.md#multiple-sessions).
