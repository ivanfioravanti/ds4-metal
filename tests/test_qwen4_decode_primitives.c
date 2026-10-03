/* Qwen decode primitive regression using the existing fixture encoders and
 * independent reference tolerances. The attention target checks query groups
 * 12/6/4, dense and sparse boundaries, output/partial identity, and guards.
 * Build and run: make test-qwen4-attention-query-groups
 */
#define main ds4_existing_qwen_kernel_suite_main
#include "tests/test_qwen4_kernels.c"
#undef main

enum { DC_GUARD = 16 };
static const float dc_canary = 17381.25f;

typedef struct {
    ds4_gpu_tensor *allocation, *data;
    uint64_t count;
} dc_guarded;

static void *dc_malloc(uint64_t bytes) {
    void *p = malloc((size_t)(bytes ? bytes : 1));
    require_ok(p != NULL, "diagnostic allocation");
    return p;
}

static void dc_finite(const float *v, uint64_t n, const char *what) {
    for (uint64_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, v + i, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            fprintf(stderr, "%s nonfinite at %llu\n", what, (unsigned long long)i);
            exit(1);
        }
    }
}

static dc_guarded dc_alloc(uint64_t count) {
    dc_guarded b;
    b.count = count;
    b.allocation = ds4_gpu_tensor_alloc((count + 2u * DC_GUARD) * 4u);
    require_ok(b.allocation != NULL, "guard allocation");
    b.data = ds4_gpu_tensor_view(b.allocation, DC_GUARD * 4u, count * 4u);
    require_ok(b.data != NULL, "guard data view");
    return b;
}

static void dc_reset(dc_guarded *b) {
    require_ok(ds4_gpu_tensor_fill_f32(b->allocation, dc_canary, b->count + 2u * DC_GUARD),
               "guard fill");
}

static float *dc_read(dc_guarded *b, uint64_t written) {
    float *all = download(b->allocation, b->count + 2u * DC_GUARD);
    require_ok(written <= b->count, "written count in allocation");
    for (uint64_t i = 0; i < DC_GUARD; i++)
        require_ok(all[i] == dc_canary, "leading output guard");
    for (uint64_t i = DC_GUARD + written; i < b->count + 2u * DC_GUARD; i++)
        require_ok(all[i] == dc_canary, "trailing/unused output guard");
    dc_finite(all + DC_GUARD, written, "GPU output/partial");
    float *data = dc_malloc(b->count * 4u);
    memcpy(data, all + DC_GUARD, b->count * 4u);
    free(all);
    return data;
}

static void dc_free(dc_guarded *b) {
    ds4_gpu_tensor_free(b->data);
    ds4_gpu_tensor_free(b->allocation);
}

static void dc_unchanged(ds4_gpu_tensor *g, const void *expected, uint64_t bytes, const char *what) {
    void *actual = dc_malloc(bytes);
    require_ok(ds4_gpu_tensor_read(g, 0, actual, bytes), "read immutable input");
    require_ok(memcmp(actual, expected, (size_t)bytes) == 0, what);
    free(actual);
}

/* Drift is descriptive only. Numerical pass/fail always uses the CPU oracle. */
static void dc_drift(const char *name, const float *got, const float *control, uint64_t count) {
    double max_abs = 0, ss = 0;
    uint64_t changed = 0;
    for (uint64_t i = 0; i < count; i++) {
        const double d = (double)got[i] - control[i];
        if (fabs(d) > max_abs) max_abs = fabs(d);
        ss += d * d;
        changed += memcmp(got + i, control + i, 4) != 0;
    }
    printf("drift %s changed=%llu count=%llu max_abs=%.9g rms=%.9g\n", name,
           (unsigned long long)changed, (unsigned long long)count,
           max_abs, sqrt(ss / (double)count));
}

/* Full real head geometry, with 257 independently encoded row patterns.
 * Every output has an independent double reference; no multi-GB double shadow.
 * The non-power-of-two period also exposes row-group/tail indexing mistakes.
 */
static void dc_q8_case(arena_t *a, uint32_t K, uint32_t M, bool head) {
    const uint32_t patterns = M < 257u ? M : 257u;
    const uint64_t row_bytes = (uint64_t)(K / 32u) * 34u;
    double *shadow;
    a->used = 0;
    const uint64_t off = arena_q8_0(a, patterns, K, &shadow, 0.05f);
    require_ok(off + row_bytes * M <= a->size, "Q8 fixture arena size");
    for (uint32_t row = patterns; row < M; row++)
        memcpy(a->base + off + (uint64_t)row * row_bytes,
               a->base + off + (uint64_t)(row % patterns) * row_bytes, (size_t)row_bytes);
    a->used = off + row_bytes * M;
    float *x = dc_malloc((uint64_t)K * 4u);
    double *dots = dc_malloc((uint64_t)patterns * sizeof(double));
    double *ref = dc_malloc((uint64_t)M * sizeof(double));
    dc_guarded out = dc_alloc(M);
    const unsigned dense_nsg[] = {4, 1, 2, 8};
    const unsigned head_nsg[] = {8, 4, 16, 32};
    const unsigned *nsg = head ? head_nsg : dense_nsg;
    for (unsigned input = 0; input < 2; input++) {
        for (uint32_t k = 0; k < K; k++) {
            float v = frand();
            if (input) v *= (k % 7u == 0u ? 16.0f : 0.03125f);
            x[k] = v;
        }
        for (uint32_t row = 0; row < patterns; row++) {
            double dot = 0;
            for (uint32_t k = 0; k < K; k++) dot += shadow[(uint64_t)row * K + k] * (double)x[k];
            dots[row] = dot;
        }
        for (uint32_t row = 0; row < M; row++) ref[row] = dots[row % patterns];
        ds4_gpu_tensor *gx = upload(x, K);
        float *control = NULL;
        for (unsigned variant = 0; variant < 4; variant++) {
            char setting[24], name[128];
            snprintf(setting, sizeof(setting), "%u", nsg[variant]);
            unsetenv("DS4_QWEN4_OUTPUT_Q8_NSG");
            require_ok(setenv("DS4_METAL_Q8_MV_NSG", head ? "4" : setting, 1) == 0, "dense NSG env");
            if (head) require_ok(setenv("DS4_QWEN4_OUTPUT_Q8_NSG", setting, 1) == 0, "head NSG env");
            snprintf(name, sizeof(name), "Q8 M=%u K=%u input=%u requested_NSG=%u effective_NSG=%u", M, K, input,
                     nsg[variant], !head && M > 65536u ? 8u : nsg[variant]);
            float *first = NULL;
            for (unsigned repeat = 0; repeat < 2; repeat++) {
                dc_reset(&out);
                require_ok(ds4_gpu_qwen4_matmul_q8_0_tensor(out.data, a->base, a->size,
                           off, K, M, gx, 1), name);
                float *got = dc_read(&out, M);
                check_close(name, got, ref, M, 3e-5);
                if (repeat) {
                    require_ok(memcmp(got, first, (size_t)M * 4u) == 0, "same Q8 candidate exact repeat");
                    free(got);
                } else first = got;
            }
            if (!control) control = first;
            else { dc_drift(name, first, control, M); free(first); }
            dc_unchanged(gx, x, (uint64_t)K * 4u, "Q8 input unchanged");
        }
        free(control);
        ds4_gpu_tensor_free(gx);
    }
    unsetenv("DS4_QWEN4_OUTPUT_Q8_NSG");
    unsetenv("DS4_METAL_Q8_MV_NSG");
    dc_free(&out);
    free(ref); free(dots); free(x); free(shadow);
}

static void dc_attention_ref(double *ref, const float *q, const float *gate,
                             const uint16_t *k, const uint16_t *v,
                             const int32_t *sel, const uint32_t *counts,
                             uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D,
                             uint32_t pos, bool sparse, uint32_t stride, float scale) {
    double *scores = dc_malloc((uint64_t)(pos + T) * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        const uint32_t n = sparse ? counts[t] : pos + t + 1u;
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / (H / Hkv);
            double m = -1e300, total = 0;
            for (uint32_t j = 0; j < n; j++) {
                const uint32_t p = sparse ? (uint32_t)sel[(uint64_t)t * stride + j] : j;
                require_ok(p <= pos + t, "CPU oracle selected key is causal");
                double dot = 0;
                for (uint32_t d = 0; d < D; d++)
                    dot += (double)q[((uint64_t)t * H + h) * D + d] *
                           f16_to_f32(k[((uint64_t)p * Hkv + kvh) * D + d]);
                scores[j] = dot * (double)scale;
                if (scores[j] > m) m = scores[j];
            }
            for (uint32_t j = 0; j < n; j++) { scores[j] = exp(scores[j] - m); total += scores[j]; }
            for (uint32_t d = 0; d < D; d++) {
                double sum = 0;
                for (uint32_t j = 0; j < n; j++) {
                    const uint32_t p = sparse ? (uint32_t)sel[(uint64_t)t * stride + j] : j;
                    sum += scores[j] * f16_to_f32(v[((uint64_t)p * Hkv + kvh) * D + d]);
                }
                const uint64_t i = ((uint64_t)t * H + h) * D + d;
                ref[i] = n ? sum / total * sigmoid_d(gate[i]) : 0;
            }
        }
    }
    free(scores);
}

static void dc_attention_case(uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D,
                              uint32_t pos, bool sparse, uint32_t selected, float qscale) {
    const uint32_t stride = 2052, cap = pos + T;
    const uint64_t qn = (uint64_t)T * H * D, kn = (uint64_t)cap * Hkv * D;
    const float scale = 1.0f / sqrtf((float)D);
    float *q = rand_vec(qn, qscale), *gate = rand_vec(qn, 2.0f);
    uint16_t *k = dc_malloc(kn * 2u), *v = dc_malloc(kn * 2u);
    int32_t *sel = dc_malloc((uint64_t)T * stride * 4u);
    uint32_t *counts = dc_malloc((uint64_t)T * 4u);
    for (uint64_t i = 0; i < kn; i++) {
        k[i] = f32_to_f16(0.75f * frand()); v[i] = f32_to_f16(frand());
    }
    for (uint32_t t = 0; t < T; t++) {
        const uint32_t nvis = pos + t + 1u;
        counts[t] = sparse ? selected : 0;
        require_ok(selected <= stride && (!sparse || selected <= nvis), "attention selected fixture count");
        for (uint32_t i = 0; i < stride; i++) sel[(uint64_t)t * stride + i] = -1;
        /* Selected full blocks are sorted and distinct; last 0..3 keys are
         * the causal residual. Unused slots are poisoned, never valid keys. */
        const uint32_t tail = sparse && selected ? selected % 4u : 0u;
        const uint32_t blocks = selected / 4u, visible_blocks = nvis / 4u;
        for (uint32_t b = 0; b < blocks; b++) {
            const uint32_t block = (uint32_t)((uint64_t)b * visible_blocks / blocks);
            for (uint32_t d = 0; d < 4; d++) sel[(uint64_t)t * stride + 4u * b + d] = (int32_t)(4u * block + d);
        }
        for (uint32_t d = 0; d < tail; d++) sel[(uint64_t)t * stride + 4u * blocks + d] = (int32_t)(nvis - tail + d);
    }
    double *ref = dc_malloc(qn * sizeof(double));
    dc_attention_ref(ref, q, gate, k, v, sel, counts, T, H, Hkv, D, pos, sparse, stride, scale);
    ds4_gpu_tensor *gq = upload(q, qn), *ggate = upload(gate, qn);
    ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(kn * 2u), *gv = ds4_gpu_tensor_alloc(kn * 2u);
    ds4_gpu_tensor *gs = ds4_gpu_tensor_alloc((uint64_t)T * stride * 4u), *gc = ds4_gpu_tensor_alloc((uint64_t)T * 4u);
    require_ok(gk && gv && gs && gc, "attention input allocs");
    require_ok(ds4_gpu_tensor_write(gk, 0, k, kn * 2u) && ds4_gpu_tensor_write(gv, 0, v, kn * 2u) &&
               ds4_gpu_tensor_write(gs, 0, sel, (uint64_t)T * stride * 4u) &&
               ds4_gpu_tensor_write(gc, 0, counts, (uint64_t)T * 4u), "attention inputs upload");
    dc_guarded out = dc_alloc(qn);
    dc_guarded part = dc_alloc((uint64_t)T * H * 64u * (2u + D));
    const unsigned widths[] = {32, 64, 128, 256, 512, 0}; /* zero means no partial buffer */
    const unsigned groups[] = {12, 6, 4};
    float *partition_control = NULL;
    for (unsigned wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        const uint32_t nkeys = sparse ? stride : pos + T;
        uint32_t splits = widths[wi] ? (nkeys + widths[wi] - 1u) / widths[wi] : 1u;
        if (splits < 1u) splits = 1u;
        if (splits > 64u) splits = 64u;
        const uint32_t actual_width = (nkeys + splits - 1u) / splits;
        const uint64_t written = splits > 1u ? (uint64_t)T * H * splits * (2u + D) : 0u;
        char setting[24];
        snprintf(setting, sizeof(setting), "%u", widths[wi] ? widths[wi] : 32u);
        require_ok(setenv("DS4_QWEN4_ATTN_SPLIT_KEYS", setting, 1) == 0, "attention split env");
        float *group_control = NULL, *part_control = NULL;
        for (unsigned gi = 0; gi < 3; gi++) {
            char name[180];
            snprintf(setting, sizeof(setting), "%u", groups[gi]);
            require_ok(setenv("DS4_QWEN4_ATTN_QUERY_GROUP", setting, 1) == 0, "attention group env");
            snprintf(name, sizeof(name), "attention T=%u H=%u/%u D=%u pos=%u sparse=%u selected=%u qscale=%g split_request=%u splits=%u width=%u group=%u",
                     T, H, Hkv, D, pos, sparse, selected, qscale, widths[wi], splits, actual_width, groups[gi]);
            float *first = NULL, *first_part = NULL;
            for (unsigned repeat = 0; repeat < 2; repeat++) {
                dc_reset(&out); dc_reset(&part);
                require_ok(ds4_gpu_qwen4_attn_decode_tensor(out.data, gq, ggate, gk, gv, gs, gc,
                    widths[wi] ? part.data : NULL, T, H, Hkv, D, pos, sparse, stride, scale), name);
                float *got = dc_read(&out, qn), *p = dc_read(&part, written);
                check_close(name, got, ref, qn, 5e-4);
                if (repeat) {
                    require_ok(memcmp(first, got, qn * 4u) == 0, "same attention candidate exact repeat");
                    require_ok(memcmp(first_part, p, part.count * 4u) == 0, "same candidate exact partial repeat");
                    free(got); free(p);
                } else { first = got; first_part = p; }
            }
            if (!group_control) {
                group_control = first; part_control = first_part;
                if (!partition_control) {
                    partition_control = dc_malloc(qn * 4u);
                    memcpy(partition_control, first, qn * 4u);
                } else dc_drift(name, first, partition_control, qn);
            } else {
                require_ok(memcmp(group_control, first, qn * 4u) == 0, "query grouping output bit identity");
                require_ok(memcmp(part_control, first_part, part.count * 4u) == 0, "query grouping full partial bit identity");
                free(first); free(first_part);
            }
        }
        free(group_control); free(part_control);
    }
    dc_unchanged(gq, q, qn * 4u, "query unchanged");
    dc_unchanged(ggate, gate, qn * 4u, "gate unchanged");
    dc_unchanged(gk, k, kn * 2u, "key cache unchanged");
    dc_unchanged(gv, v, kn * 2u, "value cache unchanged");
    dc_unchanged(gs, sel, (uint64_t)T * stride * 4u, "selected list unchanged");
    dc_unchanged(gc, counts, (uint64_t)T * 4u, "selected counts unchanged");
    unsetenv("DS4_QWEN4_ATTN_QUERY_GROUP"); unsetenv("DS4_QWEN4_ATTN_SPLIT_KEYS");
    dc_free(&part); dc_free(&out);
    ds4_gpu_tensor_free(gc); ds4_gpu_tensor_free(gs); ds4_gpu_tensor_free(gv); ds4_gpu_tensor_free(gk);
    ds4_gpu_tensor_free(ggate); ds4_gpu_tensor_free(gq);
    free(partition_control); free(ref); free(counts); free(sel); free(v); free(k); free(gate); free(q);
}

int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "dense") && strcmp(argv[1], "head") &&
                     strcmp(argv[1], "attention") && strcmp(argv[1], "all"))) {
        fprintf(stderr, "Usage: %s dense|head|attention|all\n", argv[0]);
        return 2;
    }
    const bool all = !strcmp(argv[1], "all");
    arena_t a = {0};
    a.size = (uint64_t)768u << 20;
    a.base = mmap(NULL, a.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    require_ok(a.base != MAP_FAILED, "diagnostic model arena");
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(a.base, a.size), "diagnostic map registration");
    if (all || !strcmp(argv[1], "dense")) {
        const uint32_t shapes[][2] = {{2560,48},{2560,640},{2560,641},{2560,10240},
                                     {6144,2560},{10240,2560},{2560,65535},{2560,65536},{2560,65537}};
        for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
            dc_q8_case(&a, shapes[i][0], shapes[i][1], false);
    }
    if (all || !strcmp(argv[1], "head")) dc_q8_case(&a, 2560, 248320, true);
    if (all || !strcmp(argv[1], "attention")) {
        const uint32_t dense[] = {0,31,32,2047,2048,2049,2050,2051,2052};
        for (unsigned i = 0; i < sizeof(dense) / sizeof(dense[0]); i++)
            dc_attention_case(1,24,2,256,dense[i],false,0,1.0f);
        for (uint32_t pos = 2051; pos <= 2054; pos++)
            dc_attention_case(1,24,2,256,pos,true,2048u + ((pos + 1u) % 4u),1.0f);
        dc_attention_case(1,24,2,256,8191,true,2048,1.0f);
        dc_attention_case(1,24,2,256,32767,true,2048,1.0f);
        dc_attention_case(1,24,2,256,65535,true,2048,1.0f);
        dc_attention_case(1,24,2,256,8191,true,0,1.0f);
        dc_attention_case(1,24,2,256,8194,true,2051,64.0f);
        dc_attention_case(1,24,2,256,8194,true,2051,1e-5f);
        /* Diagnostic query grouping must leave unsupported shapes untouched. */
        dc_attention_case(2,24,2,256,2051,true,2048,1.0f);
        dc_attention_case(1,8,2,256,2051,true,2048,1.0f);
        dc_attention_case(1,24,2,128,2051,true,2048,1.0f);
    }
    ds4_gpu_cleanup();
    require_ok(munmap(a.base, (size_t)a.size) == 0, "diagnostic arena release");
    puts("PASS: native independent-reference, candidate-repeat, query-group exactness, and guard gates");
    return 0;
}
