# Qwen3.8-Flash-Next Q4 fast pack

This branch adds the versioned `ds4.qwen4.fast-pack` v3 Q4 format and the
Qwen3.8-Flash-Next model identity. The model architecture and pack manifest are
DS4-specific, but every quantized tensor uses a standard GGML/GGUF block type.
An arbitrary GGUF or an Unsloth quant is not accepted. Legacy custom v1/v2 packs are
deliberately rejected instead of being interpreted as the v3 block recipe.
The same loader also accepts the fixed v4 mixed-Q2 profile documented below;
the version is part of the quantization contract, not a generic format switch.

## Pack conversion

The converter reads the immutable official BF16 checkpoint, not a quantized
derivative. It writes one
`Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf`
base artifact, an external CPU-mapped
`Qwen3.8-Flash-Next-PLE-Q4_1.gguf`, and optional vision and MTP sidecars.

```sh
uv run --script gguf-tools/qwen4_pack.py \
  --src /path/to/Qwen3.8-Flash-Next-metadata \
  --out /path/to/qwen3.8-flash-next-q4 \
  --remote-repo Qwen/Qwen3.8-Flash-Next \
  --source-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --tokenizer-template /path/to/qwen-tokenizer-template.gguf
```

The source directory may contain the whole official checkpoint or only its
`config.json` and `model.safetensors.index.json`. With `--remote-repo`, missing
official files are fetched from the pinned revision and one source shard at a
time is staged, converted into its final random-write offsets, fsynced, and
removed. The output directory therefore needs space for the final pack plus at
most the largest source shard, not the complete 360 GB BF16 checkpoint. Before
preallocation, conversion admits the exact selected artifact sizes, the largest
remote stage (when needed), and the same 32 GiB safety reserve used by the GLM
converter. A
conversion state file resumes automatically at the last completed source
shard. State v4 authenticates every completed base/vision/MTP tensor range,
PLE row range, and PLE auxiliary range before skipping it; a damaged or older
unauthenticated partial conversion is rejected. `--fresh` deliberately
replaces a partial conversion.

Write v3 to a new output directory. The converter refuses to overwrite a
directory containing a finalized manifest, even with `--fresh`, so a failed
migration cannot destroy an existing pack. Legacy v1/v2 output directories
must not be reused.

The converter records the source revision, exact model geometry, quantization
recipes, per-artifact SHA-256 digests, and a deterministic tensor-manifest
digest in `qwen3.8-flash-next-q4.manifest.json`.

The quantization layout follows the DeepSeek/GLM strategy:

- routed gate/up/down experts: standard `Q4_K`;
- shared experts, output, GDN, QSA, PLE projections, and ordinary dense
  projections: standard `Q8_0`;
- token embedding, router, hyper-connection, norms, convolutions, and other
  sensitive control tensors: source BF16/F32;
- PLE n-gram rows: standard `Q4_1`, five 32-value blocks per 160-value row.

The routed down projection has a logical input width of 640. Its physical
`Q4_K` row is zero-padded to 768 so each row contains three complete 256-value
blocks; the runtime validates both widths and never exposes the padded tail as
model state. The external PLE uses `Q4_1` instead of padding every 160-value row
to a 256-value `Q4_K` block. This keeps each sparse SSD row at 100 bytes while
removing the former custom weight/scale/bias planes.

### 64 GiB mixed-Q2 profile

Pack v4 is the smaller-backbone profile for a 64 GiB Mac. It quantizes routed
expert gate/up tensors as standard `IQ2_XXS` and routed down tensors as
standard `Q2_K`; embeddings, control tensors, Q8 dense projections, shared
experts, output, and the SSD-backed `Q4_1` PLE keep the v3 precision choices.
The base metadata must declare `routed=mixed-q2`,
`routed_gate_up=IQ2_XXS`, and `routed_down=Q2_K`. The loader checks those
values and every routed tensor type before allocating Metal caches.

```sh
uv run --script gguf-tools/qwen4_pack.py \
  --profile q2 \
  --src /path/to/Qwen3.8-Flash-Next-metadata \
  --out /path/to/qwen3.8-flash-next-q2 \
  --remote-repo Qwen/Qwen3.8-Flash-Next \
  --source-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --tokenizer-template /path/to/qwen-tokenizer-template.gguf \
  --imatrix /path/to/imatrix_unsloth.gguf_file \
  --threads 4 \
  --no-mtp
```

The v4 artifacts and resumable state are distinct from v3:
`Qwen3.8-Flash-Next-IQ2XXSGateUp-Q2KDown-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf`,
`Qwen3.8-Flash-Next-Q2-PLE-Q4_1.gguf`,
`qwen3.8-flash-next-q2-vision.gguf`,
`qwen3.8-flash-next-q2.manifest.json`, and
`qwen3.8-flash-next-q2.conversion-state.json`. The current geometry projects
43.139 GB (40.176 GiB) of base tensor payload, 32.00 GB for the SSD-only PLE, and 0.48 GB
for optional vision. The PLE mapping is sparse and is not made resident as part
of the base mapping.

`IQ2_XXS` requires the pinned Qwen GGUF importance matrix from
`unsloth/Qwen3.8-Flash-Next-GGUF` revision
`c8b5954a88c2775c546b92593eda40ea041d3176`. The converter requires SHA-256
`a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`, validates
all gate/up/down sum and count tensors for 48 layers by 512 experts, normalizes
every positive-count expert independently, and records the dataset, chunk
geometry, digest, revision, and fallback provenance in both conversion identity
and the final manifest. The verified matrix has 24 zero-count part-experts:
16 gate/up and eight down. Only those entries use the deterministic
per-expert input-column weight-energy fallback; missing or malformed entries,
negative/nonfinite statistics, and any other imatrix digest fail closed.

Routed down remains logically 640 columns and physically 768 columns. The
converter quantizes each expert independently with its normalized 640-value
importance vector, zero-pads both source rows and importance to 768, and emits
three standard `Q2_K` blocks per row. `--threads` controls a bounded ordered
expert executor; changing the worker count does not change artifact bytes or
the resumable conversion identity. Use a low value while another conversion is
active, then resume with more workers later.

Q2 MTP is a matching-profile
sidecar contract in the runtime, but conversion remains disabled until a
separately calibrated MTP-expert importance matrix is available. A v3/Q4 MTP
sidecar is rejected with a v4/Q2 base, and the converter requires explicit
`--no-mtp` even for `--dry-run`. Vision and PLE sidecars likewise carry the v4
pack ID/version even though their tensor recipes are unchanged.

The small visual position grid remains BF16 because the patch kernel performs
direct four-row lookup and bilinear interpolation rather than a projection.
The official visual FC2 width is 4304, so conversion appends 16 zero columns
before `Q8_0` quantization and the Metal graph pads the matching activation
rows to 4320. This preserves the declared dense recipe without changing FC2.
The visual MLP uses Metal's precise `tanh` intrinsic for
`gelu_pytorch_tanh`; the fast intrinsic can produce non-finite values in the
negative tail reached by the official FC1 weights.
If a format correction affects only that sidecar, `--rebuild-vision` rebuilds
it atomically from the official visual source shard and merges the new tensor
records and checksum into an already-finalized pack; it does not rewrite the
base GGUF or the 32 GB PLE table.

Conversion is deterministic for a fixed checkpoint, tokenizer template, DS4
native quantizer source, NumPy version, and source revision. Q4_K pins the
upstream Apple reference accumulation order on every host and has a checked-in
exact-byte BF16 fixture. The converter uses the same local `libds4quants`
pattern as the GLM tooling; build it with
`make -C gguf-tools quants-shared` or select it with `--quants-library`.
`--dry-run` validates and prints the partition without writing the pack.
`--no-vision` and `--no-mtp` omit optional sidecars.

## Loading and diagnostics

The single base GGUF is passed with `--model`; the manifest locates and
verifies the external PLE and any requested optional sidecars before Metal
cache allocation.

```sh
./ds4 --inspect \
  --model /pack/Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf \
  --ple /pack/Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
  --vision /pack/qwen3.8-flash-next-q4-vision.gguf \
  --mtp-model /pack/qwen3.8-flash-next-q4-mtp.gguf \
  --prefill-chunk auto
```

Loading rejects legacy v1/v2 manifests, missing, duplicate, or extra tensors
and artifacts, geometry drift, wrong GGUF block recipes, unsafe artifact
names, a mismatched tensor-manifest digest, and sidecars from another pack.
Like the DeepSeek and GLM loaders, the default trusted-local policy checks
artifact sizes, GGUF structure and metadata without scanning every payload byte.
Set `DS4_QWEN4_VERIFY=always` to additionally require every selected artifact's
manifest SHA-256 before graph allocation. The v3 directories are exact: 1,211 base tensors, 333
vision tensors, 32 MTP tensors, and four PLE tensors. The summary reports the
admitted prefill cap, QSA microtile size, active Metal specializations, PLE
staging mode, partial-RoPE behavior, and sidecar state.

Single-base packaging reduces the number of files. Normal trusted-local startup
does not stream the 73.6 GiB base GGUF or external 29.8 GiB PLE sidecar merely
to recompute their converter-recorded hashes. Strict verification remains
available with `DS4_QWEN4_VERIFY=always`.

Automatic prefill admits 8192 tokens only for an uncached suffix of at least 8192
tokens with native kernels and sufficient transient memory. Prefix-cache
resumes use 2048. Explicit `2048`, `4096`, or `8192` requests are strict and
fail with a memory or geometry diagnostic instead of silently falling back.

PLE retains an SSD-backed CPU mapping and open descriptor; rows are read on
demand. For each graph call, DS4 hashes the two- and three-token histories and
submits the
first 16-row-per-token gather before zero-based transformer layer 0 starts.
Qwen consumes that staged data at zero-based layer 1 (the second transformer
layer), so layer 0 computation masks the first SSD read. Longer calls continue
through two alternating BF16 staging buffers while the GPU consumes the prior
512-token tile; the full 51B-parameter table is never made resident in unified
memory.

The production default remains the sparse `pread` gather. Set
`DS4_QWEN4_PLE_GATHER=mmap` to run the experimental demand-paged gather, or set
it explicitly to `pread` as the kill switch. The choice is parsed once when the
two-slot stager is created; both modes preserve row order, BF16 output, double
buffering, and the layer-0 overlap. `mmap` reuses the reference mapping decoder
and applies `POSIX_MADV_RANDOM` (or `MADV_RANDOM`) to the mapping. The startup
diagnostic, runtime summary, and cold-evidence record report the chosen mode and
whether that advice succeeded. Mapping the 29.8 GiB sidecar only reserves
virtual address space: selected pages are faulted on demand, and DS4 neither
locks nor scans the complete mapping.

The mapped pack is immutable for the lifetime of the server. Replacing,
truncating, or modifying the same inode can invalidate checksum evidence and
can deliver `SIGBUS` while an mmap gather touches the changed range. Keep the
validated artifact unchanged and switch back to `pread` if mmap page-fault or
memory-pressure behavior is unsuitable. This experiment deliberately adds no
row cache, deduplication, page coalescing, `DONTNEED`, or pack-format change.

Under `DS4_QWEN4_VERIFY=always`, the PLE checksum is streamed with macOS
`F_NOCACHE` before graph allocation. DS4 records whether enabling and clearing
`F_NOCACHE` actually succeeded, retains that exact validated file descriptor, and records whether
disabling sequential read-ahead for the default sparse runtime `pread` workload
succeeded. This keeps startup validation from turning a nominally cold PLE
benchmark into a warm 29.8 GiB page-cache benchmark and avoids a
validate/reopen path race. The retained descriptor is not a snapshot: the pack
must not be modified while it is validated or served. Startup, benchmark, and
runtime evidence bind the descriptor to its device, inode, size, modification
time, change time, and manifest SHA-256.

Text prefix-cache payloads use a Qwen-specific serializer. It persists the
token frontier and logits together with every FP32 Gated DeltaNet/PLE state,
the live BF16 QSA rows, pooled sparse-index rows, and matching optional MTP
history. Restore rejects a different Qwen geometry or MTP configuration before
touching the live frontier. Multimodal requests remain outside the disk prefix
cache because their image embeddings and three-axis position table are
request-owned.

## Kernel and pack tests

```sh
make test-qwen4
```

The Metal fixture checks `Q4_K` routed-expert and `Q8_0` projection parity,
Gated DeltaNet R4/R2/R1
outputs and FP32 states at 1, 5, 17, 2048, and 8192 tokens with masks, exact
single-query QSA scoring and ordered top-k for FP32/FP16/BF16 inputs through
32768 pooled rows, streaming QSA top-k without a full score sheet, multimodal
three-axis partial RoPE, and dense-causal QSA. The host fixture covers pack
manifests, checksums, prefill policy, Qwen image resizing/patch order, contracted
  M-RoPE tables, n-gram hashing, `Q4_1` PLE dequantization, bitwise pread/mmap parity for
duplicate, reordered, and VM-page-crossing rows, and the two-slot asynchronous
PLE stager lifecycle.

Before handing off or releasing this branch on Apple Metal, run the complete
model-independent regression gate:

```sh
make test-qwen4-release
```

This target runs, in order:

- `test-qwen4-release-core`: extractor and agent tests, model-free server tests
  (including batched-session scheduling, cache, cancellation, API, GLM/Qwen
  prompt, and tool-state fixtures), layer-pack and CPU placement tests, shared
  GPU argument/CLI tests, sampling, Q4_K/MXFP4 CPU dot products, and a compile
  check of the normal CLI/server/benchmark/agent binaries;
- the existing `test-qwen4` aggregate without substituting or omitting any of
  its host, Metal, pack-conversion, or acceptance-validator fixtures;
- the model-free Metal kernel group from `ds4_test`;
- the synthetic GLM 5.3 KDA/DSA primitive suite; and
- the non-Qwen Metal MXFP4 exactness suite, including the checked-in half-LUT
  verification.

Every input in this aggregate is a checked-in deterministic fixture or a small
synthetic tensor. It intentionally does not run the model-backed Metal session
oracle, GLM vision engine, 16K golden generation, 29.8 GiB PLE sidecar, network
server, or full-model performance/quality benchmarks. Those remain separate
acceptance gates because they require checkpoint artifacts or dedicated
hardware; this target must not turn their absence into a silent skip.

For GPU traces, capture `tests/test_qwen4_metal`; its command encoders are named
for the Qwen `Q4_K` MoE and `Q8_0` projection stages, `Qwen Gated DeltaNet R4`,
`Qwen QSA M=1 ... index scoring`, `Qwen QSA streaming top-k`, and the
`Qwen vision ...` encoder stages. Full-model acceptance runs should remain
uncaptured because capture changes residency and timing.

Set `DS4_QWEN4_PROFILE=1` on a small uncaptured run to print one consolidated
per-forward timing line for embedding, total PLE work, CPU gather time,
layer-1 blocking time, dense projections,
Gated DeltaNet recurrence, QSA, routed experts, cache/state updates, optional
MTP, and total latency. The Metal encoder labels above provide the matching GPU
trace intervals without forcing synchronization in normal runs.

Decode and prefill use standard `Q8_0` blocks for main projections and standard
`Q4_K` blocks for routed experts. Qwen-specific dispatch code retains the
model's 512-expert/top-10 routing, exact slot-order accumulation, GDN/QSA
fusions, and the existing reference fallbacks; sharing a qtype does not force
DeepSeek's top-6 kernel geometry onto Qwen.

The routed `Q4_K` expert down projection maps each of the ten selected experts
to its own work partition during decode, treats activation columns 640..767 as
zero, and applies route weights in the original slot order.

An Xcode 27 Metal System Trace of the fixture recorded 76 labeled encoders, 76
submissions, and 76 target GPU intervals with all command buffers completing.
The synthetic trace is dominated by the deliberately large 32K exact-top-k and
long R4/R2/R1 parity cases, so it is label/scheduling evidence only; hotspot
decisions use uncaptured full-graph stage timings.

For full-graph chunk parity, run `ds4-bench` twice with identical text and
`--prefill-chunk 2048`/`8192`, adding `--dump-frontier-state-dir DIR` to each
run. Compare the resulting payloads with
`uv run --script speed-bench/qwen4_state_compare.py LEFT RIGHT`. The comparator
checks token and BF16 QSA/index caches exactly and applies the established
floating-point tolerances to logits, FP32 recurrence, PLE, and optional MTP
state while ignoring only the expected scratch-chunk header field.

After parity is established, add `--gen-tokens 128` and
`--dump-frontier-generation-dir DIR` to the 16K run. The emitted JSON records
the exact fixed-length greedy token IDs and byte pieces (EOS is deliberately
excluded by the benchmark), providing the deterministic 16K+128 golden without
depending on terminal text decoding.

No v3 generation golden is checked in yet. The former vector authenticated an
legacy v1 pack and was removed with that format. After reference parity is
established on a real v3 pack, record a new portable 16K+128 vector keyed by
the source revision, DS4 pack ID, tensor-manifest digest, prompt-file digest,
and 16K frontier rather than by a machine-specific model path.

`speed-bench/qwen4_acceptance.py` validates the final DS4 result JSON.
It enforces an idle Apple M3 Ultra/512 GiB declaration, one discarded warmup,
odd sample counts of at least three, unique prompts, `cached_tokens=0`, batch
size one, MTP disabled for the core cases, median aggregation, complete
latency/memory reporting, and the automatic 8K-versus-2K regression rule.

Generate the cache-distinct benchmark corpus with exact tokenizer-level context
lengths before running DS4. The generator gives every
measured request (and the discarded warmup) a different stable first token, so
the live session cannot reuse even a short common prefix. Cold-PLE and MTP use
dedicated prompts rather than warmed copies of the end-to-end inputs.

```sh
uv run --script speed-bench/qwen4_prompts.py \
  --tokenizer /path/to/official/tokenizer.json \
  --corpus speed-bench/promessi_sposi.txt \
  --out /path/to/qwen4-acceptance-prompts
```

Run the persistent DS4 samples with the resumable harness:

```sh
uv run --script speed-bench/qwen4_benchmark.py \
  --ds4-server ./ds4-server \
  --ds4-model /pack/Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf \
  --ple /pack/Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
  --prompt-dir /path/to/qwen4-acceptance-prompts \
  --output /path/to/qwen4-acceptance.json \
  --phase core \
  --phase chunks
```

Before issuing any warmup, the harness starts a fresh DS4 process and sends
exactly one dedicated 10K PLE request through the strict 8192
prefill path, followed by one generated token. After DS4 has computed all
160,000 n-gram row IDs (100 bytes per five-block `Q4_1` row, or 16,000,000 logical bytes),
but before its first PLE staging submission, it
plans the exact unique file pages touched by the selected contiguous `Q4_1`
row ranges and samples those pages with Darwin `mincore`. In the default
`pread` mode these remain the exact pages addressed by its sparse row reads;
the evidence retains all existing `F_NOCACHE` and runtime read-ahead fields.
It additionally records the gather mode and mapping-advice result so an mmap
A/B cannot be mislabeled as pread. Acceptance recomputes
the cold fraction from the page counts and requires at least 95%; a `cold`
label alone is not evidence. The harness resolves the containing mount with
`df -P`, binds it to `diskutil info -plist`, and requires a local block device
with `SolidState=true`; external SSDs are valid, rotational and network-backed
volumes are not. Darwin `proc_pid_rusage` V2 snapshots retain the process start
time and require at least 16,000,000 bytes of disk reads from that same process
lifetime. The residency probe duration is recorded separately: the result keeps
the raw observed latency/rate and a derived diagnostic with probe time removed.
The harness also reads DS4's internal prefill/decode timers, samples process RSS, requires the
exact raw token count and fixed generation length, rejects any nonzero cache
reuse, persists after every sample, and resumes completed prompt IDs. Add
`--phase mtp --ds4-mtp /pack/qwen3.8-flash-next-q4-mtp.gguf` to measure the
matching Qwen3.8 MTP sidecar. The MTP phase starts DS4 with
`--mtp-draft 4 --mtp-timing`; the core phase does not load an MTP sidecar. For every successful
DS4 MTP request the harness parses the request-local Qwen timing lines and
records total cycles, drafted and accepted draft tokens, target tokens, cycle
time, maximum per-cycle draft depth, and verifier modes. Acceptance requires
nonzero cycles, drafts, and accepts, never permits accepts to exceed drafts or
the configured depth four, observes the block verifier, proves
`target_tokens = cycles + accepted`, and requires that target-token sum to
equal the API's fixed 500-token completion.
The dedicated cold-evidence invocation uses `DS4_QWEN4_VERIFY=always` before it creates this record
and must retain the same PLE device/inode/size/times, manifest SHA-256, pack ID,
tensor-manifest digest, PLE SHA-256, and SSD descriptor for the complete phase.
The `chunks` phase restarts DS4 at explicit 2K, explicit 8K, and automatic
admission, records three cache-distinct, uncached-prefix 10K samples for each,
and writes the automatic selection plus the 8K/2K regression comparison into
the same result document. These are not described as disk-cold PLE samples;
only the DS4-only first-request phase carries that claim.

## Integration status

Pack conversion, strict loading, single-base Metal mapping, adaptive admission,
SSD-overlapped PLE staging, chat/token identities, the end-to-end 48-layer
native graph, optional Q4 MTP, and the native Qwen vision/M-RoPE path are
implemented. Qwen MTP recursively drafts up to 16 tokens on device and verifies
the proposed suffix with one target-model graph pass. Full accepts rebuild the
canonical MTP history and commit the target state directly; partial accepts
restore one transaction checkpoint and replay only the accepted prefix. The
checkpoint covers Gated DeltaNet, PLE, and MTP state, while rolled-back QSA tails
remain hidden behind their restored logical lengths. The current
correctness-first transaction reserves about 129 MiB per MTP session and copies
about 114 MiB once per multi-token speculative cycle; state-buffer swapping is
the follow-up after real-model parity is established. Admission also prices the
row-scaled target-hidden MTP capture (80 MiB at 2K or 320 MiB at 8K), including
the live/checkpoint tensors and CPU logits rows. Native batched-server mode,
which already disables speculative MTP, does not allocate these optional
per-session verifier tensors. Kernel and host fixtures pass on the target M3
Ultra. The bounded,
resumable official BF16 converter validates the complete pinned Hub tensor
directory before any output allocation. Remaining acceptance work is producing
and validating the pack, full-model reference/golden runs, and the native DS4
throughput measurements. New conversion emits v3 standard GGML blocks. Legacy
v1/v2 packs are rejected, and v3 acceptance is not claimed until a real pack
passes inspection and regenerates the model-backed golden identity.

## Follow-up: 64 GiB Q2 profile

The Q4 fast pack is the correctness and performance baseline. After it passes,
the next branch should test a 2-bit backbone aimed at a 64 GiB MacBook while
leaving the 51B-parameter n-gram table on SSD. The PLE table is sparse at
runtime: each token hashes its two- and three-token history to 16 small rows,
and those rows enrich the representation at the second transformer layer. It
must therefore remain an on-demand sidecar rather than consume unified memory.

The decisive constraint for that profile is the same overlap implemented here:
submit the 16-row lookup before layer 0 and consume it at layer 1. The Q2 work
should first preserve Q4 token/logit parity tolerances and measure SSD stalls,
quality loss, resident memory, and first-token latency against this branch. It
should not trade away the established Q4 acceptance surface merely to reach the
64 GiB footprint.

The Q4 base is not the 64 GiB deliverable even when PLE stays on disk. A Q2
profile should follow the established DeepSeek/GLM recipe family—typically
imatrix-guided `IQ2_XXS` gate/up with `Q2_K` down, retaining higher precision
for shared, attention, output, and control paths. Its converter must compute
the exact emitted size and the runtime must prove the complete context/scratch
admission under memory pressure.

PLE itself contains 320,001,536 rows of 160 values (about 51.2 billion scalar
values) and occupies 29.8 GiB as standard `Q4_1`. Each row is five contiguous
20-byte blocks, so one sparse read obtains its codes, scale, and minimum without
touching distant tensor planes. The Q2 follow-up should keep this on-demand GGUF
sidecar unchanged unless a separately versioned PLE format proves both quality
and SSD-latency gains.
