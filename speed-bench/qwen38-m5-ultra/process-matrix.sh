#!/bin/sh
# Sequential original-vs-candidate processes; dry-run unless --run is present.
# Every cell has ABBA, BAAB, ABBA external blocks. Completed immutable cells
# can be resumed or selected without rerunning their samples.
set -eu
run=0
phase=prefill
corpus_select=all
context_select=all
schedule_select=all
out=
model=ds4flash.gguf
launcher=
candidate_env=
blocks=3
mtp_depth=2
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run) run=1; shift ;;
        --out|--phase|--corpus|--context|--schedule|--model|--launcher|--candidate-env|--blocks|--mtp-depth)
            [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --out) out=$2 ;;
                --phase) phase=$2 ;;
                --corpus) corpus_select=$2 ;;
                --context) context_select=$2 ;;
                --schedule) schedule_select=$2 ;;
                --model) model=$2 ;;
                --launcher) launcher=$2 ;;
                --blocks) blocks=$2 ;;
                --mtp-depth) mtp_depth=$2 ;;
                --candidate-env)
                    # Values are passed as env arguments, never evaluated as shell code.
                    case "$2" in DS4_QWEN4_*=*|DS4_METAL_*=*) ;; *) echo 'Candidate settings must be DS4_QWEN4_* or DS4_METAL_* assignments.' >&2; exit 2;; esac
                    name=${2%%=*}; value=${2#*=}
                    case "$name" in *[!A-Z0-9_]*) echo 'Invalid environment variable name.' >&2; exit 2;; esac
                    case "$value" in ''|*[!a-zA-Z0-9_.+-]*) echo 'Candidate values must be simple numeric/identifier strings.' >&2; exit 2;; esac
                    case "$name" in
                        DS4_QWEN4_MTP_DEPTH)
                            echo 'Use --mtp-depth for the same fixed MTP depth in both builds.' >&2; exit 2;;
                        DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS|DS4_METAL_GRAPH_TOKEN_SECOND_SPLIT_LAYERS)
                            echo 'The decode harness sets these schedule controls; inherited overrides would be ignored.' >&2; exit 2;;
                        DS4_QWEN4_MTP_DRAFT_ROWS|DS4_QWEN4_MTP_DRAFT_VOCAB|DS4_QWEN4_SPEC_FORCE_ACCEPT)
                            echo 'This matrix requires full-vocabulary drafts and unchanged acceptance rules.' >&2; exit 2;;
                        *PROFILE*|*TIMING*|*DIAGNOSTIC*)
                            echo 'Collect synchronized attribution separately from throughput.' >&2; exit 2;;
                    esac
                    candidate_env="${candidate_env}${candidate_env:+
}$2"
                    ;;
            esac
            shift 2 ;;
        --help)
            cat <<'USAGE'
Usage: sh process-matrix.sh [--run] --out DIRECTORY [selectors/settings]
  --phase prefill|ordinary|mtp|all           default prefill
  --corpus code|prose|all                   default all
  --context 2048|8192|32768|65536|all        default all
  --schedule fresh|append32|append256|all   default all; prefill only
  --candidate-env NAME=VALUE               repeatable; default is candidate defaults
  --blocks N                               3..32; default 3
  --mtp-depth 2|3                           identical in both runtimes; default 2
  --launcher PATH                          default from /tmp/ds4-qwen-original-location
  --model PATH                             default ds4flash.gguf
No GPU work without --run. Run from the repository root with frozen corpora.
An existing batch resumes only if sources, binaries, corpora, and protocol match.
Selectors may change on resume; the immutable protocol/settings may not.
Original is A/control, candidate is B. Both get the same harness arguments.
Candidate environment settings apply only to B. Ambient DS4 settings are cleared.
USAGE
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$phase" in prefill|ordinary|mtp|all) ;; *) echo 'Invalid phase' >&2; exit 2;; esac
case "$corpus_select" in code|prose|all) ;; *) echo 'Invalid corpus' >&2; exit 2;; esac
case "$context_select" in 2048|8192|32768|65536|all) ;; *) echo 'Invalid context' >&2; exit 2;; esac
case "$schedule_select" in fresh|append32|append256|all) ;; *) echo 'Invalid schedule' >&2; exit 2;; esac
case "$mtp_depth" in 2|3) ;; *) echo 'Invalid MTP depth' >&2; exit 2;; esac
case "$blocks" in ''|*[!0-9]*) echo 'Invalid block count' >&2; exit 2;; esac
[ "$blocks" -ge 3 ] && [ "$blocks" -le 32 ] || { echo 'Blocks must be 3..32' >&2; exit 2; }
[ -n "$out" ] || { echo '--out is required' >&2; exit 2; }
repo=$(pwd -P)
root=$repo/speed-bench/qwen38-m5-ultra
[ -f "$repo/ds4.c" ] || { echo 'Run from repository root' >&2; exit 2; }
absolute() { case "$1" in /*) printf '%s\n' "$1";; *) printf '%s/%s\n' "$repo" "$1";; esac; }
out=$(absolute "$out")
model=$(absolute "$model")
code=$root/original-code.txt
prose=$repo/speed-bench/promessi_sposi.txt
if [ -z "$launcher" ]; then
    [ -r /tmp/ds4-qwen-original-location ] || { echo 'Use --launcher PATH for the immutable native launcher.' >&2; exit 2; }
    launcher=$(cat /tmp/ds4-qwen-original-location)/run-benchmark
fi
launcher=$(absolute "$launcher")
original=$(dirname "$launcher")
analyzer=$root/.build/process-abba
for file in "$model" "$code" "$prose" "$launcher" "$original/ORIGINAL_RUNTIME_SHA256.txt"; do
    [ -f "$file" ] || { echo "Missing input: $file" >&2; exit 2; }
done
# Canonical ordering makes repeated settings unambiguous and resume signatures stable.
candidate_env=$(printf '%s\n' "$candidate_env" | sed '/^$/d' | LC_ALL=C sort)
if [ -n "$candidate_env" ] && [ "$(printf '%s\n' "$candidate_env" | cut -d= -f1 | uniq -d | wc -l | tr -d ' ')" -ne 0 ]; then
    echo 'Duplicate candidate environment variable.' >&2; exit 2
fi
protocol() {
    printf 'format=external-abba-v3\nmodel=%s\nlauncher=%s\nblocks=%s\nmtp_depth=%s\n' "$model" "$launcher" "$blocks" "$mtp_depth"
    printf 'chunk=8192\nprefill_warmup_fresh=actual_full_prefix\nprefill_warmup_continued=actual_initial_prefix_then_exact_timed_append\ndecode_warmup=16\ndecode_tokens=128\ndecode_selection=yes\n'
    printf 'prefill_throughput_region=sync_to_host_logits\nprefill_first_token_latency=sync_then_immediate_initial_argmax\nprefill_selection_latency=initial_argmax_after_sync\n'
    printf 'prefill_selection_policy=greedy_argmax_excluding_eos\nprefill_resource_frontier=after_initial_selection\n'
    printf 'prefill_state=checkpoint_and_logits_at_frontier_plus_2_tokens\nenvironment=clean_PATH_HOME_TMPDIR_C_locale\n'
    printf 'home=%s\ntmpdir=%s\n' "$HOME" "${TMPDIR:-/tmp}"
    printf 'candidate_settings_begin\n%s\ncandidate_settings_end\n' "$candidate_env"
    # Detect accidental model replacement without hashing the 165 GiB tensor payload.
    stat -L -f 'model_stat=device:%d inode:%i bytes:%z mtime:%m' "$model"
}
source_files() {
    for base in "$repo" "$original"; do
        while IFS= read -r row; do
            relative=$(printf '%s\n' "$row" | cut -c67-)
            printf '%s\0' "$base/$relative"
        done < "$original/ORIGINAL_RUNTIME_SHA256.txt"
        for file in speed-bench/metal_prefill_variant_bench.c speed-bench/metal_decode_schedule_bench.c \
            speed-bench/benchmark_pool.h speed-bench/benchmark_pool.m \
            speed-bench/metal_prefill_variant_bench speed-bench/metal_decode_schedule_bench; do
            printf '%s\0' "$base/$file"
        done
    done
    printf '%s\0' "$launcher" "$original/ORIGINAL_RUNTIME_SHA256.txt" "$code" "$prose" \
        "$root/process-matrix.sh" "$root/process-abba.c" "$analyzer"
}
verify_batch() {
    protocol > "$out/protocol.check"
    cmp -s "$out/protocol.txt" "$out/protocol.check" || { echo 'Batch protocol/model/settings changed; use a new directory.' >&2; exit 2; }
    rm "$out/protocol.check"
    shasum -a 256 -c "$out/source.sha256" >/dev/null || { echo 'Immutable batch source/binary/corpus check failed; stopped.' >&2; exit 2; }
}
if [ "$run" -eq 1 ]; then
    [ -x "$analyzer" ] || { echo 'Build .build/process-abba before collection.' >&2; exit 2; }
    mkdir "$root/matrix-active.lock"
    trap 'rmdir "$root/matrix-active.lock"' EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if [ -d "$out" ]; then
        [ -f "$out/protocol.txt" ] && [ -f "$out/source.sha256" ] || { echo 'Existing directory is not an initialized immutable batch.' >&2; exit 2; }
        verify_batch
    else
        mkdir -p "$out"
        shasum -a 256 -c "$root/corpus.sha256" >/dev/null
        protocol > "$out/protocol.txt"
        source_files | xargs -0 shasum -a 256 > "$out/source.sha256"
        git rev-parse HEAD > "$out/revision.txt"
        git diff -- ds4.c ds4_metal.m metal/qwen4.metal Makefile \
            speed-bench/metal_prefill_variant_bench.c speed-bench/metal_decode_schedule_bench.c > "$out/runtime.patch"
        printf 'case\tblock\tslot\tvariant\texit_code\tlog\n' > "$out/exit-codes.tsv"
        verify_batch
    fi
fi
run_cell() {
    mode=$1; corpus=$2; prefix=$3; schedule=$4
    case "$schedule" in fresh) initial=0;; append32) initial=$((prefix-32));; append256) initial=$((prefix-256));; esac
    case "$corpus" in code) prompt=$code;; prose) prompt=$prose;; esac
    cell=$mode-$corpus-$prefix-$schedule
    cell_dir=$out/$cell
    warmup=$prefix
    printf '%s: %s alternating external blocks, original versus candidate\n' "$cell" "$blocks"
    if [ "$run" -eq 1 ]; then
        mkdir -p "$cell_dir"
        printf 'case\tmode\tblock\tslot\tvariant\tlog\n' > "$cell_dir/manifest.tsv.new"
    fi
    block=1
    while [ "$block" -le "$blocks" ]; do
        if [ "$((block%2))" -eq 1 ]; then variants='control candidate candidate control'; else variants='candidate control control candidate'; fi
        slot=0
        for variant in $variants; do
            slot=$((slot+1))
            log=$cell_dir/block-$block-slot-$slot-$variant.log
            if [ "$variant" = candidate ]; then set -- "$launcher" --candidate; else set -- "$launcher"; fi
            if [ "$mode" = prefill ]; then
                set -- "$@" prefill --single-run --model "$model" --prompt-file "$prompt" \
                    --prefix-tokens "$prefix" --initial-tokens "$initial" --prefill-chunk 8192 \
                    --warmup-tokens "$warmup" --compare-state
            else
                set -- "$@" decode --single-run --model "$model" --prompt-file "$prompt" \
                    --prefix-tokens "$prefix" --prefill-chunk 8192 --ctx "$((prefix+257))" \
                    --warmup 16 --tokens 128 --include-selection
                if [ "$mode" = mtp_greedy ]; then set -- "$@" --mtp --mtp-depth "$mtp_depth"; fi
            fi
            set -- "$@" --output "$log"
            # A clean environment excludes synchronized profiling and all ambient controls.
            if [ "$variant" = candidate ] && [ -n "$candidate_env" ]; then
                old_ifs=$IFS; IFS='
'
                set -- $candidate_env "$@"
                IFS=$old_ifs
            fi
            set -- env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin "HOME=$HOME" "TMPDIR=${TMPDIR:-/tmp}" LANG=C LC_ALL=C "$@"
            if [ "$run" -eq 0 ]; then
                printf '  block=%s slot=%s %s: ' "$block" "$slot" "$variant"
                printf '%s ' "$@"; printf '\n'
                continue
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$mode" "$block" "$slot" "$variant" "$log" >> "$cell_dir/manifest.tsv.new"
            verify_batch
            if [ -f "$log.exit" ] && [ "$(cat "$log.exit")" = 0 ] && [ -s "$log" ]; then
                shasum -a 256 -c "$log.sha256" >/dev/null || { echo "Completed sample changed: $log" >&2; exit 2; }
                printf '  resume: block=%s slot=%s %s\n' "$block" "$slot" "$variant"
                continue
            fi
            # Preserve interrupted/rejected attempts; never overwrite collected evidence.
            if [ -e "$log" ]; then
                attempt=1
                while [ -e "$log.attempt-$attempt" ]; do attempt=$((attempt+1)); done
                mv "$log" "$log.attempt-$attempt"
                if [ -f "$log.exit" ]; then mv "$log.exit" "$log.exit.attempt-$attempt"; fi
            fi
            printf '%s\n' "$@" > "$log.arguments.txt"
            printf '%s\0' "$@" > "$log.argv"
            date -u '+%Y-%m-%dT%H:%M:%SZ' > "$log.started"
            result=0
            "$@" || result=$?
            printf '%s\n' "$result" > "$log.exit"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$block" "$slot" "$variant" "$result" "$log" >> "$out/exit-codes.tsv"
            if [ -f "$log" ]; then shasum -a 256 "$log" > "$log.sha256"; fi
            verify_batch
            [ "$result" -eq 0 ] || { echo "Process failed (exit $result): $log" >&2; exit "$result"; }
        done
        block=$((block+1))
    done
    if [ "$run" -eq 1 ]; then
        mv "$cell_dir/manifest.tsv.new" "$cell_dir/manifest.tsv"
        "$analyzer" "$cell_dir/manifest.tsv" "$cell_dir/result.csv" "$cell_dir/result.svg"
        printf 'collected_and_hash_verified; accepted=no\n' > "$cell_dir/status.txt"
        printf 'case\tmode\tblock\tslot\tvariant\tlog\n' > "$out/manifest.tsv"
        for manifest in "$out"/*/manifest.tsv; do
            [ -f "$manifest" ] || continue
            sed '1d' "$manifest" >> "$out/manifest.tsv"
        done
        "$analyzer" "$out/manifest.tsv" "$out/results.csv" "$out/results.svg"
    fi
}
for mode in prefill ordinary mtp_greedy; do
    requested_mode=$mode; [ "$mode" != mtp_greedy ] || requested_mode=mtp
    [ "$phase" = all ] || [ "$phase" = "$requested_mode" ] || continue
    for corpus in code prose; do
        [ "$corpus_select" = all ] || [ "$corpus_select" = "$corpus" ] || continue
        for prefix in 2048 8192 32768 65536; do
            [ "$context_select" = all ] || [ "$context_select" = "$prefix" ] || continue
            if [ "$mode" = prefill ]; then
                for schedule in fresh append32 append256; do
                    [ "$schedule_select" = all ] || [ "$schedule_select" = "$schedule" ] || continue
                    run_cell "$mode" "$corpus" "$prefix" "$schedule"
                done
            else
                run_cell "$mode" "$corpus" "$prefix" fresh
            fi
        done
    done
done
