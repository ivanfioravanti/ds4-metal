**Qwen3.8 Q4: native Metal optimization results**

The minimal v2 prefill and decode bundle is accepted. Final ordinary and
fixed-depth-two MTP performance and I/O audits passed, alongside official
quality and focused correctness gates. Accepted runtime patch SHA256:
`5058ab0a4da45d2d22e7142c89f2009d5d2c80592a23856acc8e93a920f623c1`.
Historical CSV `accepted=no` fields remain unchanged: analyzers collect
evidence, while adoption is a separate manual decision.

The measured machine is an **Apple M5 Ultra with 256 GiB unified memory** and a
32 KiB threadgroup-memory limit. The model is the 177,280,286,720-byte GGUF v3
`Qwen3.8-Flash-Next Q40Routed (converted fast-pack)`, architecture `qwen4exp`:
embedding 2560, expert width 640, 512 experts/top10, vocabulary 248320, 24 query
heads/two KV heads, and one predictor layer. Routed gate/up weights remain
Q4_K and down weights MXFP4. The original revision is
`0aaea5a238fb41a35106a551e73c8409dfb751ac`.

The implementation uses C, Objective-C and Metal with no MLX dependency. It
retains model weights, operand precision, equations, attention selection,
public APIs, checkpoint formats, full-vocabulary drafts and acceptance rules.
The accepted bundle combines existing 32-token gate/up dispatch, norm/halo reuse,
bounded n-gram prefetch/overlap, ordinary T1 fusion and query grouping. Automatic
short-row overlap is limited to T≤2; the accepted prefill rule remains T≥128.
Depth policy is unchanged and no T3 optimization is adopted. Device/model
guards preserve the recorded default scope; compatible norm/halo and query
operators can also reach other quantizations, without a full-model Q2 claim.

All timing used one large model process at a time, capacity 8192, code/prose at
2K/8K/32K/64K, and three external ABBA/BAAB/ABBA blocks. Original and candidate
ran serially in their matching runtime/source trees with clean environments.
Fresh prefill warmed the complete prefix; continued prefill warmed its initial
prefix and exact append. Decode used 16 warmup tokens and 128 measured tokens,
including selection. Paired 95% intervals use block log-speed ratios.
Diagnostic I/O runs are excluded from throughput. Native builds used the
project's `-O3 -ffast-math -g -mcpu=native` defaults, C99 and Objective-C/Metal;
independent numerical oracles retained their stricter existing build settings
and tolerances.

**Twenty attempted experiment families.** Nineteen produced usable measured
outcomes, including rejected, inconclusive and explicitly exploratory variants.
D09 is blocked by reproduced baseline correctness failures, with no usable
depth-three performance result. Optional subvariants were not exhausted.

| Family | Outcome |
|---|---|
| P01 Chunk sizing | Keep 8192. Smaller fresh chunks regressed; smaller-capacity append gains exclude migration cost. |
| P02 Gate/up tiles | Existing MID32 retained; isolated +3.517%. Larger tile/cap candidates not retained. |
| P03 Down tiles | Keep original dispatch; tested alternatives slower or inconclusive. |
| P04 Traversal | Groups 1/2/4 showed no gain; early measurements are exploratory. |
| P05 K staging | Lookahead lost its marginal gain; full 64/128 staging regressed. MID128 exceeded device memory. |
| P06 Dense prefill | Tile64 and short-append Q8 unpack/reuse regressed. |
| P07 Routing/HC | HC norm reuse retained; ordered lists slower, compact stride inconclusive. |
| P08 Convolution/GDN | Halo reuse retained; GDN scheduling alternatives not retained. |
| P09 Sparse preparation | Key groups4/8 inconclusive; selection/masking unchanged. |
| P10 N-gram prefill | Overlap and bounded next-chunk prefetch retained; coalescing rejected; worker tuning not separately completed. |
| D01 Submission/sync | Tested flush changes regressed; original schedule retained. |
| D02 Q4_K GEMV | NSG alternatives inconclusive; NR2 slower. |
| D03 MXFP4 GEMV | Alternatives inconclusive or slower; existing prefetch retained. |
| D04 Dense/head GEMV | Exact and changed-reduction candidates slower or inconclusive; nothing retained. |
| D05 Fusion | Accepted: isolated +0.666%, conditional +1.007%; included in final bundle. |
| D06 Attention | Accepted query4 preserves per-query arithmetic; key-partition alternatives slower. |
| D07 Recurrent traffic | Small isolated GDN gain disappeared in the bundle; not retained. |
| D08 N-gram decode | Accepted overlap at T≤2; isolated ordinary +1.177%, fixed2 MTP +0.857%; no row cache. |
| D09 Draft depth | Baseline-blocked before usable timing; no depth3 gain or policy change claimed. |
| D10 MTP selection/rollback | Disabling GPU selection or swap recovery regressed; existing behavior retained. |

**Accepted prefill versus original.** Gain% [95% interval], with continuations
ending at the indicated context. There are 24 cells/288 processes and 864 exact
full-logit/checkpoint frontiers with selected-token parity. Fresh and append256
intervals are positive. Append32 is inconclusive; some intervals extend below
−2%, so absence of every possible short-append regression is not established.

| Corpus/context | Fresh | Append32 | Append256 |
|---|---:|---:|---:|
| Code2K | 11.478 [10.807,12.153] | −1.155 [−4.337,2.134] | 6.072 [3.601,8.602] |
| Code8K | 10.038 [9.608,10.470] | 1.001 [−3.150,5.329] | 5.861 [5.703,6.020] |
| Code32K | 16.535 [16.422,16.649] | −0.379 [−1.897,1.161] | 6.008 [4.036,8.017] |
| Code64K | 19.945 [19.797,20.092] | 1.940 [−3.588,7.786] | 5.574 [5.430,5.717] |
| Prose2K | 10.095 [9.680,10.511] | 0.223 [−5.217,5.975] | 6.427 [6.234,6.620] |
| Prose8K | 8.761 [8.624,8.898] | −0.392 [−2.980,2.265] | 5.471 [4.649,6.300] |
| Prose32K | 25.355 [25.235,25.475] | −0.343 [−2.201,1.551] | 6.030 [2.913,9.241] |
| Prose64K | 28.763 [28.626,28.901] | −0.842 [−5.151,3.662] | 5.800 [4.702,6.911] |

[Prefill CSV](final-prefill-latency-matrix/results.csv) and
[chart](final-prefill-latency-matrix/results.svg).

**Final ordinary decode versus original.** All eight intervals are positive;
96 serial runs and 12,288 cycles passed the independent audit, exact logits and
full checkpoints. No swap activity or primary-workload regression was observed.
The separate comparison with the accepted-prefill baseline also passed all
eight cells, with gains1.929–2.160%; those gains are not added to this table.

| Corpus/context | Original → candidate tokens/s | Gain% [95% interval] |
|---|---:|---:|
| Code2K | 73.218 → 74.818 | 2.185 [2.134,2.237] |
| Code8K | 73.235 → 74.736 | 2.050 [1.969,2.131] |
| Code32K | 72.562 → 73.983 | 1.959 [1.547,2.372] |
| Code64K | 72.378 → 73.816 | 1.987 [1.687,2.287] |
| Prose2K | 73.177 → 74.735 | 2.129 [1.978,2.281] |
| Prose8K | 73.256 → 74.683 | 1.948 [1.710,2.187] |
| Prose32K | 72.613 → 74.050 | 1.979 [1.835,2.123] |
| Prose64K | 72.347 → 73.720 | 1.898 [1.539,2.258] |

[Ordinary CSV](final-ordinary-matrix/results.csv) and
[chart](final-ordinary-matrix/results.svg).

**Final fixed-depth-two MTP versus original.** The separate audit passed 96 runs,
7,572 cycles and 7,668 public-frontier hash rows, with exact full public state,
trajectory/cycle plan and timed final state. Seven intervals are positive;
code32K is inconclusive. No primary regression or swap activity was observed.
This does not measure adaptive-depth policy or claim raw equivalence to ordinary
decode.

| Corpus/context | Original → candidate tokens/s | Gain% [95% interval] |
|---|---:|---:|
| Code2K | 101.517 → 102.016 | 0.492 [0.185,0.800] |
| Code8K | 89.673 → 90.089 | 0.464 [0.219,0.710] |
| Code32K | 105.336 → 105.835 | 0.474 [−0.030,0.981], inconclusive |
| Code64K | 98.728 → 99.293 | 0.573 [0.210,0.937] |
| Prose2K | 82.670 → 83.062 | 0.474 [0.121,0.828] |
| Prose8K | 76.400 → 76.763 | 0.475 [0.142,0.808] |
| Prose32K | 80.874 → 81.285 | 0.509 [0.107,0.913] |
| Prose64K | 81.743 → 82.291 | 0.671 [0.370,0.974] |

[Fixed2 MTP CSV](final-mtp-depth2-matrix/results.csv) and
[chart](final-mtp-depth2-matrix/results.svg).

CSV latency is warm execution, not server/cold TTFT. Mean ordinary latency per
token spans 13.651–13.822 ms originally and 13.366–13.565 ms for candidate. MTP
committed-token means span 9.493–13.089 and 9.449–13.027 ms; cycles include the
next draft and selection. Sampled resident maxima reach 84.24/84.40 GiB ordinary
and 89.17/89.33 GiB MTP. Matched-cell increases are at most 160.4 MiB resident and
175.4 MiB physical footprint. Sampling follows pool drain (and MTP recovery),
not continuous peak tracking. Prefetch adds two bounded buffers totaling 160 MiB
at capacity 8192 and drains pending reads on cancellation. Storage stays
disk-backed. Final diagnostic I/O and independent audits passed all 32 processes
and 96 scope markers. All 32 final fingerprints and 1,278 MTP public-frontier
records match the corresponding uninstrumented runs. No errors, short/dropped/
unfinished reads, scope failures or swap samples were observed. Ordinary runs
read 2,048 rows/655,360 bytes each; MTP runs read 2,080–2,912 rows/
665,600–931,840 bytes, matching each original/candidate pair.

Ordinary cumulative native-read duration is 0.110985–0.131047 seconds, with
active-read union 0.022741–0.026661 seconds. MTP duration is
0.093677–0.146228 seconds and union 0.018535–0.028529 seconds. Concurrent syscall
sums/unions do not measure physical SSD service or critical-path stalls, and
instrumented throughput is excluded. Two valid-copy controls passed and six
malformed evidence fixtures were rejected as expected. Selected paired data:
[prefill](io-prefill-paired-audit.csv),
[ordinary](io-final-ordinary-matrix-audit/paired.csv), and
[fixed2 MTP](io-final-mtp-depth2-matrix-audit/paired.csv).

**Correctness and limitations.** All eight final official teacher-forced quality
comparisons passed, covering short default/quality and long
fresh/append1/append256 default/quality against pinned original scores. All 100
short and 12 long cases retain finite NLL/token-count checks; aligned API subsets
are 98 short/12 long. Existing tolerances were not relaxed. Eight focused final
gates passed: n-gram failure/cancellation/recovery, query primitives, fusion,
forced rejection/recovery, sampled rewind/restore, sparse boundaries, SSD
experts and command-buffer memory. Exact sampled one-row fallback and adaptive
greedy CLI text parity passed separately. The portable synthetic attention-query
gate also passed strict Metal validation in about four seconds (compile/run
about six seconds); it does not load the large model.

The broad `make test` **failed with 19 assertions**. Every assertion, 14 printed
mismatch details and five error verdicts match the original exactly; eight
remaining recipes passed separately. This is baseline classification, not a
passing broad suite. D09's ordinary-reference top-eight gate and separate
depth3 own-replay failures also reproduce on original. Future predictor `pos3`
read-before-staging is a confirmed source hazard, but its role in the replay
failure is unproven. No thresholds were weakened or depth3 performance adopted.
Non-Apple GPU/CUDA/test-hook and CPU preprocessing preserves original C tokens;
public headers, checkpoint serialization and 77 backend files remain unchanged.
This is source evidence, not physical CUDA/distributed execution.

**Reproduction overview.** From the repository root on the measured Mac,
the portable native targets are:

```sh
make -j4 all metal-prefill-variant-bench metal-decode-schedule-bench
make test-benchmark-harnesses test-qwen4-ngrams test-qwen4-prefill-candidates
MTL_DEBUG_LAYER=1 make test-qwen4-attention-query-groups
```

Example single-process checks, after setting absolute model and corpus paths:

```sh
qwen_model=/absolute/path/Qwen3.8-Flash-Next-Q4.gguf
qwen_corpus=/absolute/path/frozen-code-or-prose.txt
./speed-bench/metal_prefill_variant_bench --single-run \
  --model "$qwen_model" --prompt-file "$qwen_corpus" \
  --prefix-tokens 8192 --prefill-chunk 8192 --warmup-tokens 8192 \
  --ctx 8195 --compare-state
./speed-bench/metal_decode_schedule_bench --single-run \
  --model "$qwen_model" --prompt-file "$qwen_corpus" \
  --prefix-tokens 8192 --ctx 8449 --prefill-chunk 8192 \
  --warmup 16 --tokens 128 --include-selection
```

The measured code corpus is `ds4.c` from the original revision; prose is the
tracked `speed-bench/promessi_sposi.txt`. The included original-build helper,
native launcher and process runner reproduce prefill/ordinary protocol runs:

```sh
qwen_report=speed-bench/qwen38-m5-ultra
qwen_revision=0aaea5a238fb41a35106a551e73c8409dfb751ac
qwen_original=/tmp/qwen-original-reproduction
git show "${qwen_revision}:ds4.c" > "$qwen_report/original-code.txt"
shasum -a 256 -c "$qwen_report/corpus.sha256"
mkdir -p "$qwen_report/.build"
cc -O2 -Wall -Wextra -Werror -std=c11 \
  "$qwen_report/process-abba.c" -lm -o "$qwen_report/.build/process-abba"
bash "$qwen_report/build-original.sh" "$qwen_original" "$PWD"
sh "$qwen_report/process-matrix.sh" --phase prefill \
  --launcher "$qwen_original/run-benchmark" --model "$qwen_model" \
  --out /tmp/qwen-repro-prefill
sh "$qwen_report/process-matrix.sh" --phase ordinary \
  --launcher "$qwen_original/run-benchmark" --model "$qwen_model" \
  --out /tmp/qwen-repro-ordinary
```

Use a new original/output directory. The matrix commands above print commands
by default; add `--run` to collect each matrix serially. The native analyzer and
corpus manifest are included. For a direct continued-prefill sample, set
`--initial-tokens` to final-context minus 32 or 256 and warm the actual
prefix/append. Avoid concurrent builds/model work or diagnostic synchronization
during timing.

Separate same-depth MTP wrapper, launcher and analyzer sources are included,
but their historical provenance pins the original object aggregate
`822fe7e1d0491f5b4a830147e79d2724667a576ce9f430bfc2da02eb392bd17d`.
A rebuild in a different path or with another compiler can correctly fail that
identity check. Reproduction then requires a separately reviewed, newly pinned
measurement series; do not silently relax the check. The strict decode
harness's ordinary-reference MTP mode is not a substitute for this protocol.

Full raw logs, exact recorded argv, hashes, scorer TSVs, audit evidence and
historical built/temporary orchestration remain in the local evidence archive.
The branch ships the selected results and protocol sources, not those local
executables or an automatic recreation of their identities. Accepted scope
retains the inconclusive code32K MTP result, the 19 baseline-matched broad-test
failures and D09's baseline block; it does not claim every workload improved or
every possible experiment variant was exhausted.
