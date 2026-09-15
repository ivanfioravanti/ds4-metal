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

On two M3 Ultra Macs (80 GPU cores and 512 GiB each), a V4.1 Flash Q4 trial
with a 7,956-token cold prefix and 512 greedy generated tokens measured these
median overall decode rates across three alternating runs (327,680 allocated
context, confidence 0.6):

| Configuration | Ordinary decode | DSpark |
| --- | ---: | ---: |
| Single Mac | 25.63 tokens/s | 22.18 tokens/s |
| Two Macs, TP RDMA | 30.12 tokens/s | 20.92 tokens/s |

These measurements favor ordinary decode. Draft acceptance alone is insufficient:
the draft and small-batch target verifier must together cost less than serial
target decoding. V4.1 DSpark remains disabled by default.

The matching cold HTTP sweep covered 512 through 256K nominal context with
128 output tokens, temperature 0.7, and one unseeded trial per configuration
and context. At the final 254,680-token prompt:

| Configuration | Ordinary prefill | DSpark prefill | Ordinary decode | DSpark decode |
| --- | ---: | ---: | ---: | ---: |
| Single Mac | 629.17 | 628.95 | 20.66 | 17.12 |
| TP RDMA | 645.15 | 644.40 | 23.37 | 18.04 |

Rates are tokens/s. Small differences in this single-trial HTTP sweep should
not be interpreted as statistically established gains.

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
