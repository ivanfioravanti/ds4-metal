#!/bin/sh
# Build ONLY the separate native launcher, during an authorized CPU window.
# Diagnostic wrapper builds must already exist; runtime/model files stay intact.
set -eu
[ "$#" = 6 ] && [ "$1" = --build ] || { echo 'usage: prepare-launcher.sh --build ORIGINAL_TREE ORIGINAL_BUILD CANDIDATE_TREE CANDIDATE_BUILD NEW_TMP_OUTPUT' >&2;exit 2; }
shift
tool=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
original=$(CDPATH= cd -- "$1" && pwd -P);original_build=$(CDPATH= cd -- "$2" && pwd -P)
candidate=$(CDPATH= cd -- "$3" && pwd -P);candidate_build=$(CDPATH= cd -- "$4" && pwd -P)
parent=$(CDPATH= cd -- "$(dirname -- "$5")" && pwd -P);out=$parent/$(basename -- "$5")
case "$out" in /private/tmp/*|/tmp/*) ;; *) echo 'Use a new temporary launcher directory.' >&2;exit 2;;esac
[ ! -e "$out" ] && [ ! -L "$out" ] || exit 2
for path in "$original" "$original_build" "$candidate" "$candidate_build" "$out";do
    printf '%s\n' "$path" | LC_ALL=C grep -Eq '^[A-Za-z0-9_./+:-]+$' || { echo 'Launcher configuration paths must contain no whitespace or C quoting characters.' >&2;exit 2; }
done
for build in "$original_build" "$candidate_build";do
    shasum -a 256 -c "$build/diagnostic.sha256" >/dev/null
done
(cd "$original" && shasum -a 256 -c ORIGINAL_RUNTIME_SHA256.txt && shasum -a 256 -c "$original_build/runtime.sha256") >/dev/null
(cd "$candidate" && shasum -a 256 -c "$candidate_build/runtime.sha256") >/dev/null
for file in metal_decode_mtp_same_depth.c metal_decode_schedule_bench.c;do cmp "$original_build/$file" "$candidate_build/$file";done
for file in benchmark_pool.h benchmark_pool.m;do cmp "$original/speed-bench/$file" "$candidate/speed-bench/$file";done
hash(){ shasum -a 256 "$1" | cut -c1-64; }
[ "$(hash "$original_build/metal_decode_mtp_same_depth.c")" = 1555a405f6bd197eb961ba4bef7aee9784132fec96eae802e013851e2281b75c ] || exit 2
[ "$(hash "$original_build/metal_decode_schedule_bench.c")" = 8637e88c9dd7ed785a1e3ebcf34b8b3c6d03cee402295d801c34e64de2095f73 ] || exit 2
mkdir "$out"
cp "$tool/run-mtp.c" "$tool/runner_metal_paths.inc" "$out/"
{
    printf '#define ORIGINAL_ROOT "%s"\n#define ORIGINAL_BUILD "%s"\n#define CANDIDATE_ROOT "%s"\n#define CANDIDATE_BUILD "%s"\n' "$original" "$original_build" "$candidate" "$candidate_build"
    printf '#define ORIGINAL_MANIFEST_SHA256 "%s"\n' "$(hash "$original/ORIGINAL_RUNTIME_SHA256.txt")"
    printf '#define ORIGINAL_BUILD_RUNTIME_SHA256 "%s"\n#define CANDIDATE_BUILD_RUNTIME_SHA256 "%s"\n' "$(hash "$original_build/runtime.sha256")" "$(hash "$candidate_build/runtime.sha256")"
    printf '#define ORIGINAL_BINARY_SHA256 "%s"\n#define CANDIDATE_BINARY_SHA256 "%s"\n' "$(hash "$original_build/metal_decode_mtp_same_depth")" "$(hash "$candidate_build/metal_decode_mtp_same_depth")"
    printf '#define MTP_WRAPPER_SHA256 "%s"\n#define STRICT_PROTOCOL_SHA256 "%s"\n' "$(hash "$original_build/metal_decode_mtp_same_depth.c")" "$(hash "$original_build/metal_decode_schedule_bench.c")"
    printf '#define POOL_H_SHA256 "%s"\n#define POOL_M_SHA256 "%s"\n' "$(hash "$original/speed-bench/benchmark_pool.h")" "$(hash "$original/speed-bench/benchmark_pool.m")"
} > "$out/launcher_config.h"
printf '%s\n' "$original" > "$out/original-root.txt";printf '%s\n' "$candidate" > "$out/candidate-root.txt"
printf '%s\n' "$original_build" > "$out/original-build.txt";printf '%s\n' "$candidate_build" > "$out/candidate-build.txt"
cc -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations "$out/run-mtp.c" -o "$out/run-mtp" > "$out/build.log" 2>&1
"$out/run-mtp" --verify-only mtp > "$out/original-verification.log"
"$out/run-mtp" --candidate --verify-only mtp > "$out/candidate-verification.log"
shasum -a 256 "$out/run-mtp.c" "$out/runner_metal_paths.inc" "$out/launcher_config.h" "$out/run-mtp" \
    "$out/original-root.txt" "$out/candidate-root.txt" "$out/original-build.txt" "$out/candidate-build.txt" \
    "$out/original-verification.log" "$out/candidate-verification.log" > "$out/launcher.sha256"
printf 'Separate external MTP launcher ready: %s/run-mtp; no model launched.\n' "$out"
