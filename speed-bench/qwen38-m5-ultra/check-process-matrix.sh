#!/bin/sh
# Entirely synthetic CPU-only launcher in an isolated tiny repository.
# Exercises execution, original/candidate environment isolation and immutable
# resume without opening a GGUF or invoking either benchmark executable.
set -eu
tools=$(cd "$(dirname "$0")" && pwd -P)
tmp=$(mktemp -d /tmp/ds4-process-matrix-test.XXXXXX)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
repo=$tmp/repo
original=$tmp/original
root=$repo/speed-bench/qwen38-m5-ultra
mkdir -p "$root/.build" "$original/speed-bench"
cp "$tools/process-matrix.sh" "$tools/process-abba.c" "$root/"
cp "$tools/.build/process-abba" "$root/.build/"
printf 'fixture\n' > "$repo/ds4.c"
printf 'fixture\n' > "$repo/ds4flash.gguf"
printf 'code fixture\n' > "$root/original-code.txt"
printf 'prose fixture\n' > "$repo/speed-bench/promessi_sposi.txt"
cp "$repo/ds4.c" "$original/"
for file in metal_prefill_variant_bench.c metal_decode_schedule_bench.c benchmark_pool.h benchmark_pool.m metal_prefill_variant_bench metal_decode_schedule_bench; do
    printf 'fixture\n' > "$repo/speed-bench/$file"
    cp "$repo/speed-bench/$file" "$original/speed-bench/"
done
(cd "$original" && shasum -a 256 ds4.c > ORIGINAL_RUNTIME_SHA256.txt)
cat > "$original/run-benchmark" <<'FIXTURE'
#!/bin/sh
set -eu
runtime=original;seconds=2
if [ "$1" = --candidate ]; then
    runtime=candidate;seconds=1;shift
    [ "${DS4_QWEN4_HC_NORM_REUSE:-}" = 1 ] || exit 41
else
    [ -z "${DS4_QWEN4_HC_NORM_REUSE:-}" ] || exit 42
fi
mode=$1;shift
[ -z "${DS4_QWEN4_PROFILE:-}" ] || exit 43
initial=0;prefix=0;mtp=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output) log=$2;shift 2;;
        --prefix-tokens) prefix=$2;shift 2;;
        --initial-tokens) initial=$2;shift 2;;
        --mtp) mtp=1;shift;;
        --single-run|--compare-state|--include-selection) shift;;
        *) shift 2;;
    esac
done
hash=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
{
    printf 'build_provenance={"runtime":"%s","runtime_files_sha256":"%s","binary_sha256":"%s","protocol_source_hashes":{"fixture":"%s"},"command_args":["--single-run"]}\n' "$runtime" "$hash" "$hash" "$hash"
    if [ "$mode" = prefill ]; then
        tokens=$((prefix-initial))
        printf 'metal-prefill-variant-bench: model=fixture prefix=%s compare_state=yes single_run=yes\n' "$prefix"
        for step in 0 1 2; do
            printf 'hashes run=1 continuation=%s position=%s logits_sha256=%s checkpoint_sha256=%s checkpoint_bytes=8\n' "$step" "$((prefix+step))" "$hash" "$hash"
        done
    else
        mode=ordinary; [ "$mtp" -eq 0 ] || mode=mtp_greedy
        tokens=128
        printf 'configuration mode=%s prefix=%s single_run=yes\n' "$mode" "$prefix"
        printf 'hashes run=1 position=%s logits_sha256=%s checkpoint_sha256=%s checkpoint_bytes=8\n' "$((prefix+tokens))" "$hash" "$hash"
    fi
    if [ "$mode" = prefill ]; then
        printf 'run=1 mode=%s pattern=single slot=1 variant=control tokens=%s seconds=%s first_token_ms=%s.125 selection_ms=0.125 first_token_id=42\n' "$mode" "$tokens" "$seconds" "$((seconds*1000))"
    else
        printf 'run=1 mode=%s pattern=single slot=1 variant=control tokens=%s seconds=%s first_cycle_ms=10\n' "$mode" "$tokens" "$seconds"
    fi
    printf 'aggregate mode=%s variant=control runs=1\n' "$mode"
} > "$log"
FIXTURE
chmod +x "$original/run-benchmark"
cd "$repo"
git init -q
git add ds4.c
git -c user.name=Fixture -c user.email=fixture@example.invalid commit -qm Fixture
shasum -a 256 speed-bench/qwen38-m5-ultra/original-code.txt speed-bench/promessi_sposi.txt > "$root/corpus.sha256"
run_fixture() {
    DS4_QWEN4_PROFILE=1 DS4_QWEN4_HC_NORM_REUSE=99 sh "$root/process-matrix.sh" \
        --run --out "$tmp/batch" --launcher "$original/run-benchmark" \
        --corpus code --context 2048 --schedule append32 --candidate-env DS4_QWEN4_HC_NORM_REUSE=1 "$@" > "$tmp/runner.log" 2>&1
}
run_fixture --phase all
grep -q '^format=external-abba-v3$' "$tmp/batch/protocol.txt"
grep -q '^prefill_first_token_latency=sync_then_immediate_initial_argmax$' "$tmp/batch/protocol.txt"
[ "$(wc -l < "$tmp/batch/exit-codes.tsv" | tr -d ' ')" -eq 37 ]
[ "$(awk -F, 'NR>1 && $3=="positive_paired_ci" && $4=="no" {n++} END {print n+0}' "$tmp/batch/results.csv")" -eq 3 ]
run_fixture --phase prefill
[ "$(wc -l < "$tmp/batch/exit-codes.tsv" | tr -d ' ')" -eq 37 ]
[ "$(grep -c 'resume:' "$tmp/runner.log")" -eq 12 ]
if run_fixture --phase prefill --candidate-env DS4_QWEN4_PREFILL_REUSE=1; then
    echo 'Changed protocol unexpectedly resumed' >&2;exit 1
fi
printf 'changed\n' >> "$repo/ds4.c"
if run_fixture --phase prefill; then
    echo 'Changed source unexpectedly resumed' >&2;exit 1
fi
echo 'process matrix v3: first-token latency protocol, three phases, clean environments, partial selection, resume and immutable checks passed'
