#!/bin/sh
# CPU-only parser fixtures. No runtime/model/Metal calls.
set -eu
analyzer=${1:-./.build/process-abba}
tmp=$(mktemp -d /tmp/ds4-process-abba-test.XXXXXX)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
hash=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
printf 'case\tmode\tblock\tslot\tvariant\tlog\n' > "$tmp/manifest.tsv"
for mode in prefill ordinary mtp_greedy; do
    for block in 1 2 3; do
        case "$block" in 2) variants='candidate control control candidate';; *) variants='control candidate candidate control';; esac
        slot=0
        for variant in $variants; do
            slot=$((slot+1))
            log=$tmp/$mode-$block-$slot.log
            runtime=original; seconds=2
            if [ "$variant" = candidate ]; then runtime=candidate; seconds=1; fi
            {
                printf 'build_provenance={"runtime":"%s","runtime_files_sha256":"%s","binary_sha256":"%s","protocol_source_hashes":{"test":"%s"},"command_args":["--single-run"]}\n' "$runtime" "$hash" "$hash" "$hash"
                if [ "$mode" = prefill ]; then
                    printf 'metal-prefill-variant-bench: model=fixture prefix=2048 compare_state=yes single_run=yes\n'
                    for step in 0 1 2; do
                        printf 'hashes run=1 continuation=%s position=%s logits_sha256=%s checkpoint_sha256=%s checkpoint_bytes=8\n' "$step" "$((2048+step))" "$hash" "$hash"
                    done
                else
                    printf 'configuration mode=%s prefix=2048 single_run=yes\n' "$mode"
                    printf 'hashes run=1 position=2176 logits_sha256=%s checkpoint_sha256=%s checkpoint_bytes=8\n' "$hash" "$hash"
                fi
                if [ "$mode" = prefill ]; then
                    printf 'run=1 mode=%s pattern=single slot=1 variant=control tokens=128 seconds=%s first_token_ms=%s.125 selection_ms=0.125 first_token_id=42\n' "$mode" "$seconds" "$((seconds*1000))"
                else
                    printf 'run=1 mode=%s pattern=single slot=1 variant=control tokens=128 seconds=%s first_cycle_ms=10\n' "$mode" "$seconds"
                fi
                if [ "$mode" = prefill ]; then
                    printf 'telemetry run=1 process_valid=1 swap_valid=1 vm_valid=1 resident_bytes=1024 footprint_bytes=512 disk_read_bytes=128 disk_write_bytes=0 global_swap_used_bytes=0 global_swapin_pages=0 global_swapout_pages=0\n'
                else
                    printf 'memory run=1 process_valid=1 swap_valid=1 vm_valid=1 resident_bytes=1024 footprint_bytes=512 disk_read_bytes=128 disk_write_bytes=0 global_swap_used_bytes=0 global_swapins_pages=0 global_swapouts_pages=0\n'
                fi
                printf 'aggregate mode=%s variant=control runs=1\n' "$mode"
            } > "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$mode" "$block" "$slot" "$variant" "$log" >> "$tmp/manifest.tsv"
        done
    done
done
"$analyzer" "$tmp/manifest.tsv" "$tmp/report.csv" "$tmp/chart.svg"
[ "$(awk -F, 'NR>1 && $3=="positive_paired_ci" && $4=="no" && $9==100 && $12=="yes" && $13=="yes" {n++} END {print n+0}' "$tmp/report.csv")" -eq 3 ]
awk -F, 'NR==1 {fields=NF;for(i=1;i<=NF;i++) col[$i]=i;next} NF!=fields || $(col["original_valid_process_rows"])!=6 || $(col["original_max_resident_bytes"])!=1024 || $(col["candidate_timed_disk_read_bytes"])!=768 {exit 1}' "$tmp/report.csv"
awk -F, '
 NR==1 {for(i=1;i<=NF;i++)c[$i]=i;next}
 $2=="prefill" {
   if($(c["original_mean_first_selected_token_ms"])!=2000.125||$(c["candidate_mean_first_selected_token_ms"])!=1000.125||
      $(c["original_mean_initial_selection_ms"])!=0.125||$(c["original_mean_prompt_token_ms"])!=15.625||
      $(c["original_mean_committed_token_ms"])!=""||$(c["original_mean_first_decode_step_ms"])!=""||
      $(c["original_mean_first_mtp_cycle_ms"])!=""||$(c["latency_scope"])!="warm_sync_then_immediate_initial_argmax")exit 1
 }
 $2!="prefill" {
   if($(c["original_mean_first_selected_token_ms"])!=""||$(c["original_mean_initial_selection_ms"])!=""||
      $(c["original_mean_prompt_token_ms"])!=""||$(c["original_mean_committed_token_ms"])!=15.625)exit 1;
   if($2=="ordinary"&&($(c["original_mean_first_decode_step_ms"])!=10||$(c["original_mean_first_mtp_cycle_ms"])!=""))exit 1;
   if($2=="mtp_greedy"&&($(c["original_mean_first_mtp_cycle_ms"])!=10||$(c["original_mean_first_decode_step_ms"])!=""))exit 1
 }
' "$tmp/report.csv"
cp "$tmp/prefill-1-2.log" "$tmp/saved.log"
expect_rejected() {
    if "$analyzer" "$tmp/manifest.tsv" "$tmp/rejected.csv" "$tmp/rejected.svg" >/dev/null 2>&1; then
        echo "Parser accepted invalid fixture: $1" >&2; exit 1
    fi
    grep -q "$2" "$tmp/rejected.csv"
}
sed 's/checkpoint_sha256=0/checkpoint_sha256=1/' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'checkpoint drift' cross_build_hash_mismatch
sed '/continuation=2/d' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'missing continuation hash' missing_or_invalid_single_run
sed 's/"test":"0/"test":"1/' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'different harness protocol' incompatible_protocol_or_arguments
sed 's/seconds=1/seconds=nan/' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'nonfinite elapsed time' missing_or_invalid_single_run
sed 's/ first_token_ms=[^ ]*//' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'missing first-token latency (legacy protocol)' missing_or_invalid_single_run
sed 's/ selection_ms=[^ ]*//' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'missing initial selection latency' missing_or_invalid_single_run
sed 's/first_token_ms=1000.125/first_token_ms=nan/' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'nonfinite first-token latency' missing_or_invalid_single_run
sed 's/selection_ms=0.125/selection_ms=-0.125/' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'negative selection latency' missing_or_invalid_single_run
sed 's/first_token_ms=1000.125/first_token_ms=1001.125/' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'inconsistent sync plus selection timing' missing_or_invalid_single_run
sed 's/ first_token_id=42//' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'missing selected token' missing_or_invalid_single_run
sed 's/first_token_id=42/first_token_id=43/' "$tmp/saved.log" > "$tmp/prefill-1-2.log"
expect_rejected 'selected token differs despite equal logits hashes' first_selected_token_mismatch
cp "$tmp/saved.log" "$tmp/prefill-1-2.log"
cp "$tmp/manifest.tsv" "$tmp/full.tsv"
sed '/prefill.*prefill.*3.*4/d' "$tmp/full.tsv" > "$tmp/manifest.tsv"
expect_rejected 'incomplete ABBA block' incomplete_external_blocks
echo 'process ABBA v3 parser: exact hashes, three modes, sync throughput, selected-token latency/identity, distinct latency scopes and malformed/missing evidence passed'
