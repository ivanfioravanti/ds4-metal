#!/bin/bash
# Compile only this diagnostic; use prebuilt runtime objects. No model run.
set -euo pipefail
if [[ $# != 2 ]]; then
    printf 'usage: %s RUNTIME_TREE NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
source_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
runtime=$(cd -- "$1" && pwd -P)
parent=$(cd -- "$(dirname -- "$2")" && pwd -P)
output="$parent/$(basename -- "$2")"
[[ ! -e $output && ! -L $output ]] || { printf 'Output must be new.\n' >&2; exit 2; }
cmp "$source_dir/metal_decode_schedule_bench.c" "$runtime/speed-bench/metal_decode_schedule_bench.c"
core=(ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o ds4_engram.o)
for file in "${core[@]}" speed-bench/benchmark_pool.o; do
    [[ -r "$runtime/$file" ]] || { printf 'Missing prebuilt object: %s\n' "$file" >&2; exit 2; }
done
mkdir -- "$output"
cp "$source_dir/metal_decode_mtp_same_depth.c" "$source_dir/metal_decode_schedule_bench.c" "$output/"
cc=${CC:-cc}
cflags=(-O3 -ffast-math -g -mcpu=native -Wall -Wextra -Werror -std=c99)
{
    printf 'Separate same-depth MTP diagnostic; included strict source unchanged.\n'
    printf 'Runtime source/cwd: %s\nCompiler: %s\nCFLAGS: %s\n' "$runtime" "$cc" "${cflags[*]}"
    printf 'Only the wrapper is compiled; original/workspace runtime objects remain untouched.\n'
    "$cc" --version
} > "$output/build-settings.txt"
(
    cd -- "$runtime"
    shasum -a 256 ds4.c ds4_metal.m ds4.h ds4_gpu.h "${core[@]}" \
        speed-bench/benchmark_pool.h speed-bench/benchmark_pool.m speed-bench/benchmark_pool.o \
        speed-bench/metal_decode_schedule_bench.c metal/*.metal
) > "$output/runtime.sha256"
(
    cd -- "$runtime"
    "$cc" "${cflags[@]}" -I. -Ispeed-bench -c "$output/metal_decode_mtp_same_depth.c" -o "$output/metal_decode_mtp_same_depth.o"
    "$cc" "${cflags[@]}" -o "$output/metal_decode_mtp_same_depth" \
        "$output/metal_decode_mtp_same_depth.o" speed-bench/benchmark_pool.o "${core[@]}" \
        -lm -pthread -framework Foundation -framework Metal
    shasum -a 256 -c "$output/runtime.sha256"
) > "$output/build.log" 2>&1
shasum -a 256 "$output/metal_decode_mtp_same_depth.c" "$output/metal_decode_schedule_bench.c" \
    "$output/metal_decode_mtp_same_depth" > "$output/diagnostic.sha256"
printf 'Built only %s/metal_decode_mtp_same_depth; no model launched.\n' "$output"
printf 'Execute with cwd %s and absolute model/corpus paths after independent gates.\n' "$runtime"
