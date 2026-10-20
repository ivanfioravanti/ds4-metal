# V4.1 scalar decode on Metal

## What changed

Short Q8 projections expose more independent output rows while keeping their accumulators and reductions. Independent hyperconnection work runs beside query and concurrent FFN projections. Query dispatch also publishes KV entries. A hierarchical scalar router reduces synchronization, and long-context index scoring skips blocks already excluded by the mask.

F16 tensors that were released as FP8 32x32 blocks (the Engram projections and the indexer query) are re-encoded once to their exact e4m3fn codes and read at one byte per weight, using the F16 kernel's lane walk. A tensor that does not round-trip keeps F16. The router reads a BF16 copy of its BF16-valued F32 weights, and its matvec and selection run in one dispatch. The attention reduce applies the inverse rope tail, so the low projection no longer applies it. The Engram add issues its loads before it accumulates, and the non-causal indexer sort uses the shuffle network.

The active concurrent shared-down path is covered, not only its fallback. Query RoPE and shared-down output fusions remain. All accepted optimizations select automatically for supported shapes; `DS4_METAL_DISABLE_V41_Q8_SHORT`, `DS4_METAL_DISABLE_V41_DEFERRED_HC`, `DS4_METAL_DISABLE_V41_DEFERRED_FFN_HC`, `DS4_METAL_DISABLE_V41_QUERY_KV`, `DS4_METAL_DISABLE_V41_ROUTER_HIER`, `DS4_METAL_DISABLE_V41_INDEX_MASKED`, `DS4_METAL_DISABLE_V41_FP8_F16`, `DS4_METAL_DISABLE_V41_ROUTER_BF16`, `DS4_METAL_DISABLE_V41_FUSE_ROUTER`, `DS4_METAL_DISABLE_V41_REDUCE_ROPE`, and `DS4_METAL_DISABLE_V41_TOPK_SHUFFLE` are diagnostic rollback switches (set to `1`).

## Results

M3 Ultra, 80 GPU cores, 512 GB, native MXFP4 main weights. Control: af7c02b. Quiet-machine checks preceded alternating runs; prefix snapshots excluded prefill/model loading from decode timing. Four measured runs per arm.

| Standard serial, no drafter | Control mean / best | Changed mean / best |
|---|---:|---:|
| 8k, 128 generated tokens | 34.52 / 34.53 | 36.96 / 36.97 t/s |
| 64k, 128 generated tokens | 33.47 / 33.50 | 35.98 / 35.99 t/s |

A separate integration test includes drafter capture with drafting disabled, 512 generated tokens, and identical accounting in both arms: 8k **33.3483→35.7105 t/s**, 62k **32.3824→34.7713 t/s** (means). These are not interchangeable protocols. Output tokens and complete final vocabulary logits match in each paired comparison. Rates describe the complete scalar bundle, not each commit independently.

The FP8, router, reduce-rope, Engram and sort changes, against control a74d62a: 2k **38.0→40.0 t/s**, 8k 36.7→38.9, 32k 36.0→38.4 (`ds4-bench`, 128 generated tokens). Greedy output text matches the control. With DSpark it matches too once `DS4_DSPARK_DISABLE_ADAPTIVE_DRAFTING=1` is set, because adaptive drafting reacts to measured time.

To reproduce the standard sweep with a warmed, resident model:

```sh
make ds4-bench
./ds4-bench -m MODEL.gguf --metal --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128 --csv results.csv
```

Use the same model, corpus, context and hardware for control/candidate. Keep first-use loading separate from generation. Reuse identity-checked prefix state for repeated decode comparisons where available.

## Validation

Production-shape Q8, deferred HC, hierarchical router and masked-score gates accompany the implementation. The accepted integration passed randomized exactness checks and native full-logit comparisons. Each extracted code commit builds Metal host code and CPU engine objects. Actual CUDA/distributed inference was not run.

```sh
make tests/test_v41_q8_short tests/test_v41_q8_short_fused \
  tests/test_v41_deferred_hc tests/test_v41_deferred_ffn_hc \
  tests/test_v41_router_hier tests/test_v41_index_masked
# Each gate accepts its fresh-draw count as its first argument.
./tests/test_v41_router_hier 50000
```

## Experiments not retained

- Removing duplicated HC normalization changed one split value by one ULP and changed native tokens. Keep the original complete calculation.
- Separate expert-slot dispatches preserved 256 million compared values but measured −0.031 ±0.035 t/s (standard error). More threadgroups alone did not improve throughput.
- Router/shared-gate fusion was superseded by placing complete HC work in the existing concurrent FFN stage.
- Enabling a host pipeline-lookup cache measured −0.00583 t/s, standard error 0.02129. No established gain.

Rejected runtime branches are absent. Individual improvements were measured at different stages and must not be added as if they were independent.
