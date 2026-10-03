#!/bin/sh
# Fixed-depth2 MTP external ABBA; dry-run by default, no adaptive-policy claim.
set -eu
run=0;out=;launcher=;model=ds4flash.gguf;corpus_choice=all;context_choice=all;blocks=3;candidate_env=
while [ "$#" -gt 0 ];do
    case "$1" in
        --run)run=1;shift;;
        --out|--launcher|--model|--corpus|--context|--blocks|--candidate-env)
            [ "$#" -ge 2 ] || exit 2
            case "$1" in
                --out)out=$2;;--launcher)launcher=$2;;--model)model=$2;;--corpus)corpus_choice=$2;;--context)context_choice=$2;;--blocks)blocks=$2;;
                --candidate-env)
                    case "$2" in DS4_QWEN4_DECODE_FUSIONS=1|DS4_QWEN4_ATTN_QUERY_GROUP=4|DS4_QWEN4_ATTN_QUERY_GROUP=6|DS4_QWEN4_NGRAM_OVERLAP=1|DS4_QWEN4_GDN_NSG=4) ;; *) echo 'Unsupported candidate bundle setting.' >&2;exit 2;;esac
                    candidate_env="${candidate_env}${candidate_env:+
}$2";;
            esac;shift 2;;
        *)echo 'usage: matrix.sh [--run] --launcher PINNED_RUN_MTP --out BATCH [--model MODEL] [--corpus code|prose|all] [--context 2048|8192|32768|65536|all] [--blocks 3..32] [--candidate-env NAME=VALUE]' >&2;exit 2;;
    esac
done
[ -n "$out" ] && [ -n "$launcher" ] || exit 2
case "$corpus_choice" in code|prose|all) ;; *)exit 2;;esac
case "$context_choice" in 2048|8192|32768|65536|all) ;; *)exit 2;;esac
case "$blocks" in ''|*[!0-9]*)exit 2;;esac
[ "$blocks" -ge 3 ] && [ "$blocks" -le 32 ] || exit 2
repo=$(pwd -P);root=$repo/speed-bench/qwen38-m5-ultra;tool=$root/external-mtp
absolute(){ case "$1" in /*)printf '%s\n' "$1";;*)printf '%s/%s\n' "$repo" "$1";;esac; }
out=$(absolute "$out");model=$(absolute "$model");launcher=$(absolute "$launcher");launchdir=$(dirname "$launcher")
analyzer=$tool/analyze-external;code=$root/original-code.txt;prose=$repo/speed-bench/promessi_sposi.txt
for file in "$repo/ds4.c" "$model" "$code" "$prose" "$launcher" "$launchdir/launcher.sha256";do [ -f "$file" ] || { echo "Missing input: $file" >&2;exit 2; };done
candidate_env=$(printf '%s\n' "$candidate_env" | sed '/^$/d' | LC_ALL=C sort)
[ -z "$(printf '%s\n' "$candidate_env" | cut -d= -f1 | uniq -d)" ] || exit 2
original=$(cat "$launchdir/original-root.txt");candidate=$(cat "$launchdir/candidate-root.txt")
obuild=$(cat "$launchdir/original-build.txt");cbuild=$(cat "$launchdir/candidate-build.txt")
[ "$candidate" = "$repo" ] || { echo 'Run from the configured candidate workspace.' >&2;exit 2; }
protocol(){
    printf 'format=external-mtp-abba-v1\nmode=mtp_same_depth\nbaseline_kind=original\nfixed_depth=2\nadaptive_policy_measured=no\n'
    printf 'launcher=%s\nmodel=%s\nblocks=%s\nchunk=8192\nwarmup=16\ntokens=128\nselection=non_eos_argmax\n' "$launcher" "$model" "$blocks"
    printf 'home=%s\ntmpdir=%s\nenvironment=clean_PATH_HOME_TMPDIR_C_locale\ncandidate_settings_begin\n%s\ncandidate_settings_end\n' "$HOME" "${TMPDIR:-/tmp}" "$candidate_env"
    stat -L -f 'model_stat=device:%d inode:%i bytes:%z mtime:%m' "$model"
}
source_files(){
    for base in "$original" "$candidate";do
        while IFS= read -r row;do relative=$(printf '%s\n' "$row" | cut -c67-);printf '%s\0' "$base/$relative";done < "$original/ORIGINAL_RUNTIME_SHA256.txt"
        for file in metal_decode_schedule_bench.c benchmark_pool.h benchmark_pool.m benchmark_pool.o;do printf '%s\0' "$base/speed-bench/$file";done
    done
    for build in "$obuild" "$cbuild";do
        for file in metal_decode_mtp_same_depth.c metal_decode_schedule_bench.c metal_decode_mtp_same_depth runtime.sha256 diagnostic.sha256 build-settings.txt;do printf '%s\0' "$build/$file";done
    done
    printf '%s\0' "$launcher" "$launchdir/launcher_config.h" "$launchdir/launcher.sha256" "$original/ORIGINAL_RUNTIME_SHA256.txt" \
        "$tool/matrix.sh" "$tool/analyze-external.c" "$tool/analyze-same-depth.c" "$analyzer" "$code" "$prose"
}
verify(){
    protocol > "$out/protocol.check";cmp -s "$out/protocol.txt" "$out/protocol.check" || { echo 'Protocol/model/settings changed.' >&2;exit 2; };rm "$out/protocol.check"
    shasum -a 256 -c "$out/source.sha256" >/dev/null
}
analyze_manifest(){
    manifest=$1;dir=$2;key=$(shasum -a 256 "$manifest" | cut -c1-16);stem=$dir/analysis-$key
    if [ -f "$stem.sha256" ];then shasum -a 256 -c "$stem.sha256" >/dev/null;cmp -s "$manifest" "$stem.tsv"
    else cp "$manifest" "$stem.tsv";"$analyzer" "$stem.tsv" "$stem.csv" "$stem.svg";shasum -a 256 "$stem.tsv" "$stem.csv" "$stem.svg" > "$stem.sha256";fi
    ln -sfn "$(basename "$stem.csv")" "$dir/results.csv";ln -sfn "$(basename "$stem.svg")" "$dir/results.svg"
}
if [ "$run" = 1 ];then
    [ -x "$analyzer" ] || { echo 'Build and CPU-test the external MTP analyzer first.' >&2;exit 2; }
    mkdir "$root/matrix-active.lock";trap 'rmdir "$root/matrix-active.lock"' EXIT;trap 'exit 129' HUP;trap 'exit 130' INT;trap 'exit 143' TERM
    shasum -a 256 -c "$launchdir/launcher.sha256" >/dev/null
    if [ -d "$out" ];then [ -f "$out/protocol.txt" ] && [ -f "$out/source.sha256" ] || exit 2;verify
    else
        mkdir -p "$out";shasum -a 256 -c "$root/corpus.sha256" >/dev/null;protocol > "$out/protocol.txt"
        source_files | xargs -0 shasum -a 256 > "$out/source.sha256"
        cp "$launchdir/launcher_config.h" "$out/launcher_config.h";git diff -- ds4.c ds4_metal.m metal > "$out/candidate-from-original.patch"
        printf 'case\tblock\tslot\tvariant\texit_code\tlog\n' > "$out/exit-codes.tsv";verify
    fi
fi
for corpus in code prose;do
    [ "$corpus_choice" = all ] || [ "$corpus_choice" = "$corpus" ] || continue
    case "$corpus" in code)prompt=$code;;prose)prompt=$prose;;esac
    for prefix in 2048 8192 32768 65536;do
        [ "$context_choice" = all ] || [ "$context_choice" = "$prefix" ] || continue
        cell=mtp-depth2-$corpus-$prefix;dir=$out/$cell
        if [ "$run" = 1 ];then mkdir -p "$dir";printf 'case\tblock\tslot\tvariant\tlog\n' > "$dir/manifest.tsv.new";fi
        block=1
        while [ "$block" -le "$blocks" ];do
            if [ "$((block%2))" = 1 ];then order='control candidate candidate control';else order='candidate control control candidate';fi
            slot=0
            for variant in $order;do
                slot=$((slot+1));log=$dir/block-$block-slot-$slot-$variant.log
                if [ "$variant" = candidate ];then set -- "$launcher" --candidate;else set -- "$launcher";fi
                set -- "$@" mtp --single-run --model "$model" --prompt-file "$prompt" --prefix-tokens "$prefix" --ctx "$((prefix+257))" \
                    --prefill-chunk 8192 --warmup 16 --tokens 128 --include-selection --mtp --mtp-depth 2 --output "$log"
                if [ "$variant" = candidate ] && [ -n "$candidate_env" ];then saved_ifs=$IFS;IFS='
'
                    set -- $candidate_env "$@";IFS=$saved_ifs
                fi
                set -- env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin "HOME=$HOME" "TMPDIR=${TMPDIR:-/tmp}" LANG=C LC_ALL=C "$@"
                printf '%s block=%s slot=%s variant=%s\n' "$cell" "$block" "$slot" "$variant"
                if [ "$run" = 0 ];then printf '  ';printf '%s ' "$@";printf '\n';continue;fi
                printf '%s\t%s\t%s\t%s\t%s\n' "$cell" "$block" "$slot" "$variant" "$log" >> "$dir/manifest.tsv.new";verify
                if [ -f "$log.exit" ] && [ "$(cat "$log.exit")" = 0 ] && [ -s "$log" ];then shasum -a 256 -c "$log.sha256" >/dev/null;continue;fi
                if [ -e "$log" ];then
                    attempt=1;while [ -e "$log.attempt-$attempt" ];do attempt=$((attempt+1));done
                    mv "$log" "$log.attempt-$attempt";for suffix in exit sha256 arguments.txt argv started;do [ ! -f "$log.$suffix" ] || mv "$log.$suffix" "$log.$suffix.attempt-$attempt";done
                fi
                printf '%s\n' "$@" > "$log.arguments.txt";printf '%s\0' "$@" > "$log.argv";date -u '+%Y-%m-%dT%H:%M:%SZ' > "$log.started"
                result=0;"$@" || result=$?;printf '%s\n' "$result" > "$log.exit"
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$block" "$slot" "$variant" "$result" "$log" >> "$out/exit-codes.tsv"
                [ ! -f "$log" ] || shasum -a 256 "$log" "$log.argv" "$log.arguments.txt" "$log.exit" > "$log.sha256"
                verify;[ "$result" = 0 ] || exit "$result"
            done
            block=$((block+1))
        done
        if [ "$run" = 1 ];then
            mv "$dir/manifest.tsv.new" "$dir/manifest.tsv";analyze_manifest "$dir/manifest.tsv" "$dir"
            printf 'case\tblock\tslot\tvariant\tlog\n' > "$out/manifest.tsv"
            for manifest in "$out"/mtp-depth2-*/manifest.tsv;do [ ! -f "$manifest" ] || sed '1d' "$manifest" >> "$out/manifest.tsv";done
            analyze_manifest "$out/manifest.tsv" "$out"
        fi
    done
done
