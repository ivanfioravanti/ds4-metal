#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { return isatty(fileno(fp)) != 0; }

/* Differential coverage for the existing 32/64-token gate/up tiles.
 * Run on a Metal 4 tensor device: the startup log must say tensor ops enabled.
 * The independent arithmetic references remain in test_qwen4_kernels.c. */
enum { GUARD = 16, D = 256, E = 8, S = 3, STRIDE = 4, O = 260 };
static uint32_t seed = 0x739ac123u;
static uint32_t rnd(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return seed;
}
static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", what); exit(1); }
}
static void env(const char *name, unsigned value) {
    char text[32]; snprintf(text, sizeof(text), "%u", value);
    require(setenv(name, text, 1) == 0, name);
}
static uint32_t bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

typedef struct { unsigned mid, tails, tiles; } variant;
static const variant variants[] = {
    {64,1,8}, {32,1,8},
    /* Caps force looped tiles; both token and output-row tails are guarded. */
    {64,0,1}, {32,0,1}, {64,0,3}, {32,0,3}, {64,1,32}, {32,1,32},
};

static void quant_fill(uint8_t *p, uint64_t bytes, unsigned type) {
    const unsigned block = type == 12 ? 144 : 17;
    for (uint64_t off = 0; off < bytes; off += block) {
        for (unsigned i = 0; i < block; i++) p[off+i] = (uint8_t)rnd();
        if (type == 12) {
            const uint16_t d = 0x1800u + (rnd() % 5u) * 0x100u, m = 0x1400;
            memcpy(p + off, &d, 2); memcpy(p + off + 2, &m, 2);
        } else p[off] = 118u + rnd() % 5u;
    }
}

static int compare_output(const float *ref, const float *got, size_t n, unsigned width,
                          unsigned T, unsigned F, unsigned run, const char *stage) {
    size_t different = 0, first = 0;
    for (size_t i = 0; i < n + GUARD; i++) {
        const int live = i < n && (i / width) % STRIDE < S;
        if (live && (bits(got[i]) & 0x7f800000u) == 0x7f800000u) {
            fprintf(stderr, "FAIL nonfinite %s T=%u F=%u run=%u i=%zu\n", stage, T, F, run, i);
            return 0;
        }
        if (bits(ref[i]) != bits(got[i])) { if (!different) first = i; different++; }
    }
    if (different) fprintf(stderr, "FAIL exact %s T=%u F=%u run=%u differences=%zu first=%zu ref=%08x got=%08x\n",
                           stage, T, F, run, different, first, bits(ref[first]), bits(got[first]));
    return different == 0;
}

typedef struct { unsigned ff; uint64_t gate, up, down; } weights;

static int check_mm(const void *map, uint64_t bytes, weights w, unsigned T) {
    const unsigned F = w.ff;
    const size_t xn = (size_t)T * D, mn = (size_t)T * STRIDE * F, on = (size_t)T * STRIDE * O;
    const size_t maxn = mn > on ? mn : on;
    float *hx = malloc(xn * 4), *refm = malloc((mn + GUARD) * 4), *refo = malloc((on + GUARD) * 4);
    float *got = malloc((maxn + GUARD) * 4);
    int32_t *ids = malloc((size_t)T * S * 4);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(xn * 4), *mid = ds4_gpu_tensor_alloc((mn + GUARD) * 4);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((on + GUARD) * 4), *gi = ds4_gpu_tensor_alloc((size_t)T * S * 4);
    ds4_gpu_tensor *lists = ds4_gpu_tensor_alloc((size_t)E * T * 4), *counts = ds4_gpu_tensor_alloc(E * 4);
    require(hx && refm && refo && got && ids && x && mid && out && gi && lists && counts, "matrix allocation");
    for (size_t i = 0; i < xn; i++) hx[i] = ((int)(rnd() % 257) - 128) / 1024.0f;
    for (unsigned t = 0; t < T; t++) {
        ids[t*S] = 0; ids[t*S+1] = 1 + t % 3; ids[t*S+2] = 4 + t % 3;
    }
    require(ds4_gpu_tensor_write(x, 0, hx, xn * 4) &&
            ds4_gpu_tensor_write(gi, 0, ids, (size_t)T * S * 4), "matrix inputs");
    env("DS4_QWEN4_MOE_MM_NAX", 2);
    env("DS4_QWEN4_MOE_MM_SPECIALIZE", 1);
    int passed = 1;
    for (unsigned run = 0; run < sizeof(variants) / sizeof(*variants); run++) {
        const variant v = variants[run];
        env("DS4_QWEN4_MOE_MID_NAX_WIDTH", v.mid);
        env("DS4_QWEN4_MOE_TAILS", v.tails);
        env("DS4_QWEN4_MOE_MID_TILES", v.tiles); env("DS4_QWEN4_MOE_DOWN_TILES", v.tiles);
        const float poison = -12345.25f;
        for (size_t i = 0; i < maxn + GUARD; i++) got[i] = poison;
        require(ds4_gpu_tensor_write(mid, 0, got, (mn + GUARD) * 4) &&
                ds4_gpu_tensor_write(out, 0, got, (on + GUARD) * 4), "matrix poison");
        require(ds4_gpu_begin_commands(), "begin matrix commands");
        int ok = ds4_gpu_qwen4_moe_build_lists_tensor(lists, counts, gi, T, S, E, T) &&
                 ds4_gpu_qwen4_moe_mm_mid_tensor(mid, x, lists, counts, map, bytes, w.gate, w.up, 12, E, T, S, STRIDE, D, F, T) &&
                 ds4_gpu_qwen4_moe_mm_down_tensor(out, mid, lists, counts, map, bytes, w.down, 39, E, T, S, STRIDE, F, O, T);
        require(ds4_gpu_end_commands() && ok, "matrix dispatch");
        for (unsigned stage = 0; stage < 2; stage++) {
            const size_t n = stage ? on : mn;
            const unsigned width = stage ? O : F;
            float *ref = stage ? refo : refm;
            require(ds4_gpu_tensor_read(stage ? out : mid, 0, got, (n + GUARD) * 4), "matrix readback");
            for (size_t i = 0; i < n + GUARD; i++) {
                const int live = i < n && (i / width) % STRIDE < S;
                require(live ? bits(got[i]) != bits(poison) : bits(got[i]) == bits(poison), "matrix writes and guards");
            }
            if (!run) memcpy(ref, got, (n + GUARD) * 4);
            ok &= compare_output(ref, got, n, width, T, F, run, stage ? "down" : "mid");
        }
        printf("%s MM T=%u F=%u O=%u run=%u mid=%u tails=%u tiles=%u\n",
               ok ? "PASS" : "FAIL", T, F, O, run, v.mid, v.tails, v.tiles);
        fflush(stdout);
        passed &= ok;
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(gi);
    ds4_gpu_tensor_free(lists); ds4_gpu_tensor_free(counts);
    free(hx); free(refm); free(refo); free(got); free(ids);
    return passed;
}

int main(void) {
    require(ds4_gpu_init(), "Metal initialization");
    weights cases[] = {{320,0,0,0}, {448,0,0,0}};
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t bytes = 0;
    for (unsigned i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        weights *w = &cases[i];
        const uint64_t gu = (uint64_t)E * w->ff * (D / 256) * 144;
        const uint64_t dw = (uint64_t)E * O * (w->ff / 32) * 17;
        w->gate = bytes; w->up = align_up(bytes + gu, page);
        w->down = align_up(w->up + gu, page); bytes = align_up(w->down + dw, page) + page;
    }
    void *map = NULL;
    require(posix_memalign(&map, page, bytes) == 0, "weight allocation");
    memset(map, 0, bytes);
    for (unsigned i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        const weights w = cases[i];
        const uint64_t gu = (uint64_t)E * w.ff * (D / 256) * 144;
        quant_fill((uint8_t *)map + w.gate, gu, 12);
        quant_fill((uint8_t *)map + w.up, gu, 12);
        quant_fill((uint8_t *)map + w.down, (uint64_t)E * O * (w.ff / 32) * 17, 39);
    }
    require(ds4_gpu_set_model_map(map, bytes), "weight map");
    const unsigned sizes[] = {1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 257, 513};
    int ok = 1;
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(*sizes); i++)
        ok &= check_mm(map, bytes, cases[i % 2], sizes[i]);
    /* Release the runtime's no-copy weight buffers before their host storage. */
    ds4_gpu_cleanup(); free(map);
    return ok ? 0 : 1;
}
