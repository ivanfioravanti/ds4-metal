#!/bin/bash
# Rebuild the pinned original runtime with the current measurement protocol.
# This only creates NEW_OUTPUT; it never builds or changes the candidate tree.
set -euo pipefail

revision=0aaea5a238fb41a35106a551e73c8409dfb751ac
if [[ $# != 2 ]]; then
    printf 'usage: %s NEW_OUTPUT_DIRECTORY CANDIDATE_WORKSPACE\n' "$0" >&2
    exit 2
fi
if [[ $(uname -s) != Darwin ]]; then
    printf 'The original Metal benchmark build requires macOS.\n' >&2
    exit 2
fi
case "$1$2" in
    *$'\n'*|*$'\r'*) printf 'Directory names must not contain newlines.\n' >&2; exit 2 ;;
esac
script_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
candidate=$(cd -- "$2" && pwd -P)
parent=$(cd -- "$(dirname -- "$1")" && pwd -P)
output="$parent/$(basename -- "$1")"
if [[ -e $output || -L $output ]]; then
    printf 'Output must be a new directory: %s\n' "$output" >&2
    exit 2
fi
cc=${CC:-cc}
jobs=${JOBS:-8}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { printf 'JOBS must be positive.\n' >&2; exit 2; }
command -v "$cc" >/dev/null
git -C "$candidate" cat-file -e "$revision^{commit}"
protocol=(metal_prefill_variant_bench.c metal_decode_schedule_bench.c benchmark_pool.h benchmark_pool.m)
for file in "${protocol[@]}"; do
    [[ -r "$candidate/speed-bench/$file" ]] || { printf 'Missing protocol: %s\n' "$file" >&2; exit 2; }
done
core=(ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o ds4_engram.o)
cflags=(-O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99)
objcflags=(-O3 -ffast-math -g -mcpu=native -Wall -Wextra -fobjc-arc)
ldflags=(-lm -pthread -framework Foundation -framework Metal)

mkdir -- "$output"
git -C "$candidate" archive --format=tar "$revision" | tar -xf - -C "$output"
printf '%s\n' "$revision" > "$output/ORIGINAL_REVISION"
{
    printf 'Original runtime git revision: %s\n' "$revision"
    printf 'Enhanced harnesses copied from: %s\n' "$candidate"
    printf 'Compiler: %s\nCFLAGS: %s\nOBJCFLAGS: %s\n' "$cc" "${cflags[*]}" "${objcflags[*]}"
    printf 'Harness pool: -fno-objc-arc\n'
    "$cc" --version
} > "$output/BENCHMARK_BUILD.txt"

(
    cd -- "$output"
    make -j"$jobs" CC="$cc" CFLAGS="${cflags[*]}" OBJCFLAGS="${objcflags[*]}" "${core[@]}"
    # Runtime and its Makefile stay as archived. Only the protocol is overlaid.
    for file in "${protocol[@]}"; do cp "$candidate/speed-bench/$file" "speed-bench/$file"; done
    "$cc" "${cflags[@]}" -fno-objc-arc -c speed-bench/benchmark_pool.m -o speed-bench/benchmark_pool.o
    for bench in metal_prefill_variant_bench metal_decode_schedule_bench; do
        "$cc" "${cflags[@]}" -I. -c "speed-bench/$bench.c" -o "speed-bench/$bench.o"
        "$cc" "${cflags[@]}" -o "speed-bench/$bench" "speed-bench/$bench.o" \
            speed-bench/benchmark_pool.o "${core[@]}" "${ldflags[@]}"
    done
    while IFS= read -r file; do
        shasum -a 256 "$file"
        chmod a-w "$file"
    done < "$script_dir/original-runtime-files.txt" > ORIGINAL_RUNTIME_SHA256.txt
    chmod a-w ORIGINAL_REVISION ORIGINAL_RUNTIME_SHA256.txt BENCHMARK_BUILD.txt
    cp "$script_dir/run-benchmark.c" "$script_dir/runner_metal_paths.inc" .
    # A forced include avoids command-line macro quoting for paths with spaces.
    escaped_output=$(printf '%s' "$output" | sed 's/\\/\\\\/g; s/"/\\"/g')
    escaped_candidate=$(printf '%s' "$candidate" | sed 's/\\/\\\\/g; s/"/\\"/g')
    printf '#define ORIGINAL_ROOT "%s"\n#define WORKSPACE_ROOT "%s"\n' \
        "$escaped_output" "$escaped_candidate" > runner_config.h
    "$cc" -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations \
        -include runner_config.h run-benchmark.c -o run-benchmark
    # Help initializes no engine or Metal device and also verifies the manifest.
    ./run-benchmark prefill --help
    ./run-benchmark decode --help
) > "$output/ORIGINAL_BUILD.log" 2>&1
printf 'Original runtime and current protocol ready: %s\n' "$output"
printf 'Run: "%s/run-benchmark" [--candidate] prefill|decode OPTIONS --output ABSOLUTE_LOG\n' "$output"
