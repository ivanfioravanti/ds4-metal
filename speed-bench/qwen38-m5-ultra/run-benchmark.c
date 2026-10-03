/* Native provenance/orchestration only; runtime original objects are never
 * rebuilt here. Compile: cc -O2 -Wall -Wextra -Werror
 * -Wno-deprecated-declarations -include runner_config.h
 * run-benchmark.c -o run-benchmark (see build-original.sh). */
#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifndef ORIGINAL_ROOT
#error "Define ORIGINAL_ROOT as the immutable original build directory"
#endif
#ifndef WORKSPACE_ROOT
#error "Define WORKSPACE_ROOT as the candidate workspace directory"
#endif
#ifndef REVISION
#define REVISION "0aaea5a238fb41a35106a551e73c8409dfb751ac"
#endif
static const struct { const char *key, *relative; } metal_sources[] = {
#include "runner_metal_paths.inc"
};
static const char *protocols[] = {
    "speed-bench/metal_prefill_variant_bench.c", "speed-bench/metal_decode_schedule_bench.c",
    "speed-bench/benchmark_pool.h", "speed-bench/benchmark_pool.m",
};
static void die(const char *message) { fprintf(stderr, "benchmark-runner: %s (%s)\n", message, strerror(errno)); exit(1); }
static void joined(char out[PATH_MAX], const char *root, const char *file) {
    if (snprintf(out, PATH_MAX, "%s/%s", root, file) >= PATH_MAX) { errno = ENAMETOOLONG; die(file); }
}
static void hex_digest(const unsigned char *bytes, char out[65]) {
    for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", bytes[i]);
}
static void file_digest(const char *path, char out[65]) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die(path);
    unsigned char buffer[65536], bytes[32];
    CC_SHA256_CTX state;
    CC_SHA256_Init(&state);
    size_t n;
    while ((n = fread(buffer, 1, sizeof(buffer), fp))) CC_SHA256_Update(&state, buffer, (CC_LONG)n);
    if (ferror(fp)) die(path);
    fclose(fp);
    CC_SHA256_Final(bytes, &state);
    hex_digest(bytes, out);
}
static void json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') { putchar('\\'); putchar(*p); }
        else if (*p < 32) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}
static char *absolute_path(const char *cwd, const char *value) {
    if (value[0] == '/') return strdup(value);
    char path[PATH_MAX];
    joined(path, cwd, value);
    return strdup(path);
}
int main(int argc, char **argv) {
    int at = 1;
    const bool candidate = at < argc && !strcmp(argv[at], "--candidate");
    if (candidate) at++;
    if (at >= argc || (strcmp(argv[at], "prefill") && strcmp(argv[at], "decode"))) {
        fprintf(stderr, "usage: %s [--candidate] prefill|decode BENCHMARK_OPTIONS\n", argv[0]);
        return 2;
    }
    const bool decode = !strcmp(argv[at++], "decode");
    const char *runtime = candidate ? WORKSPACE_ROOT : ORIGINAL_ROOT;
    const char *program = decode ? "speed-bench/metal_decode_schedule_bench" : "speed-bench/metal_prefill_variant_bench";
    char cwd[PATH_MAX], path[PATH_MAX];
    if (!getcwd(cwd, sizeof(cwd))) die("getcwd");
    char **args = calloc((size_t)argc + 1, sizeof(*args));
    if (!args) die("allocate arguments");
    int n_args = 1;
    args[0] = (char *)program;
    char *output = NULL;
    for (; at < argc; at++) {
        const bool out = !strcmp(argv[at], "--output");
        const bool input = !strcmp(argv[at], "--model") || !strcmp(argv[at], "-m") || !strcmp(argv[at], "--prompt-file");
        if (out || input) {
            if (at + 1 >= argc) { errno = EINVAL; die("missing path argument"); }
            char *absolute = absolute_path(cwd, argv[at + 1]);
            if (!absolute) die("allocate path");
            if (out) output = absolute;
            else { args[n_args++] = argv[at]; args[n_args++] = absolute; }
            at++;
        } else args[n_args++] = argv[at];
    }
    if (output) {
        int fd = open(output, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0 || dup2(fd, STDOUT_FILENO) < 0 || dup2(fd, STDERR_FILENO) < 0) die("open output");
        close(fd);
    }
    joined(path, ORIGINAL_ROOT, "ORIGINAL_RUNTIME_SHA256.txt");
    FILE *manifest = fopen(path, "r");
    if (!manifest) die(path);
    char line[PATH_MAX + 68], hash[65];
    CC_SHA256_CTX combined;
    CC_SHA256_Init(&combined);
    size_t runtime_files = 0;
    while (fgets(line, sizeof(line), manifest)) {
        size_t length = strlen(line);
        if (length < 68 || line[64] != ' ' || line[65] != ' ' || line[length - 1] != '\n') { errno = EINVAL; die("malformed original manifest"); }
        line[length - 1] = '\0';
        joined(path, runtime, line + 66);
        file_digest(path, hash);
        if (!candidate && strncmp(hash, line, 64)) { errno = EINVAL; die("immutable original runtime changed"); }
        CC_SHA256_Update(&combined, line + 66, (CC_LONG)strlen(line + 66));
        CC_SHA256_Update(&combined, hash, 64);
        runtime_files++;
    }
    if (ferror(manifest)) die("read original manifest");
    fclose(manifest);
    unsigned char combined_bytes[32];
    char runtime_hash[65];
    CC_SHA256_Final(combined_bytes, &combined);
    hex_digest(combined_bytes, runtime_hash);
    for (size_t i = 0; i < sizeof(metal_sources) / sizeof(metal_sources[0]); i++) {
        joined(path, runtime, metal_sources[i].relative);
        if (setenv(metal_sources[i].key, path, 1)) die("pin Metal source");
    }
    printf("build_provenance={\"runtime\":\"%s\",\"original_revision\":\"%s\",\"runtime_files\":%zu,\"runtime_files_sha256\":\"%s\",\"cwd\":",
           candidate ? "candidate" : "original", REVISION, runtime_files, runtime_hash);
    json_string(runtime);
    printf(",\"program\":"); json_string(program);
    joined(path, runtime, program); file_digest(path, hash);
    printf(",\"binary_sha256\":\"%s\",\"protocol_source_hashes\":{", hash);
    for (size_t i = 0; i < sizeof(protocols) / sizeof(protocols[0]); i++) {
        if (i) putchar(',');
        json_string(protocols[i]); putchar(':');
        joined(path, runtime, protocols[i]); file_digest(path, hash); json_string(hash);
    }
    printf("},\"metal_sources_pinned\":%zu,\"command_args\":[", sizeof(metal_sources) / sizeof(metal_sources[0]));
    for (int i = 1; i < n_args; i++) { if (i > 1) putchar(','); json_string(args[i]); }
    printf("]}\n");
    fflush(NULL);
    if (chdir(runtime)) die("chdir runtime");
    execv(program, args);
    die(program);
}
