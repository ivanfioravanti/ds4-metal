#include "ds4.h"
#include "benchmark_pool.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <unistd.h>

#define BENCH_NAME "metal-prefill-variant-bench"

enum {
    VARIANT_COUNT = 2,
    RUNS_PER_REPEAT = 4,
    DEFAULT_PREFIX_TOKENS = 8192,
    DEFAULT_WARMUP_TOKENS = 32,
    DEFAULT_REPEATS = 3,
    STATE_CONTINUATION_TOKENS = 2,
};

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *candidate_env;
    const char *candidate_value;
    const char *control_value;
    const char *output_path;
    const char *extra_env[16];    /* candidate-only NAME=VALUE settings */
    const char *control_env[16];  /* control-only NAME=VALUE settings */
    int n_extra;
    int n_control;
    bool interleave;
    bool single_run;
    int prefill_chunk;
    int candidate_chunk;
    int prefix_tokens;
    int initial_tokens;
    bool tolerate_drift;     /* report drift statistics instead of failing on a logit mismatch */
    bool compare_state;
    int warmup_tokens;
    int ctx;
    int repeats;
} bench_config;

typedef struct {
    double seconds;
    size_t runs;
    size_t tokens;
} variant_result;

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s --candidate-env NAME [options]\n"
            "\n"
            "  -m, --model PATH       GGUF path (default: ds4flash.gguf)\n"
            "  --prompt-file PATH     token source (default: ds4.c)\n"
            "  --candidate-env NAME   unset NAME for control, set it for candidate\n"
            "  --candidate-value TEXT candidate env value (default: 1)\n"
            "  --control-value TEXT   explicit control env value (default: unset)\n"
            "  --extra-env NAME=VALUE additional candidate-only setting; repeatable\n"
            "  --control-env NAME=VALUE control-only setting; repeatable\n"
            "  --prefill-chunk N      tokens per chunk (default: 4096)\n"
            "  --candidate-chunk N    Qwen candidate chunk size (control: --prefill-chunk)\n"
            "  --output PATH         write complete benchmark log to PATH\n"
            "  --prefix-tokens N      final prefill length (default: 8192)\n"
            "  --initial-tokens N     untimed live prefix before appending to that length\n"
            "  --tolerate-drift       report max/mean |delta|, top-1 agreement instead of failing on a mismatch\n"
            "  --compare-state       require exact checkpoint bytes after prefill and 2 known\n"
            "                         continuation tokens (untimed; same chunk or one-batch\n"
            "                         continued append, no interleave)\n"
            "  --warmup-tokens N      fresh warmup length (default: 32; min: 32); continued\n"
            "                         cases warm the actual initial prefix and exact append\n"
            "  --ctx N                session allocation (default: max lengths + 1)\n"
            "  --repeats N            alternating ABBA/BAAB pairs (default: 3)\n"
            "  --single-run           one warmed control sample with exact hashes; no paired CI\n"
            "  --interleave           one session per repeat, prefilled chunk by chunk with the\n"
            "                         variant alternating per chunk (and per repeat), so both\n"
            "                         variants share the machine state and every position\n",
            argv0);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s: %s requires an argument\n", BENCH_NAME, opt);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int_arg(const char *value, const char *opt, int minimum) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || value[0] == '\0' || !end || *end != '\0' ||
        parsed < minimum || parsed > INT_MAX) {
        fprintf(stderr,
                "%s: invalid value for %s: %s\n",
                BENCH_NAME,
                opt,
                value);
        exit(2);
    }
    return (int)parsed;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config cfg = {
        .model_path = "ds4flash.gguf",
        .prompt_path = "ds4.c",
        .candidate_env = NULL,
        .candidate_value = "1",
        .prefill_chunk = 4096,
        .prefix_tokens = DEFAULT_PREFIX_TOKENS,
        .warmup_tokens = DEFAULT_WARMUP_TOKENS,
        .ctx = 0,
        .repeats = DEFAULT_REPEATS,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            cfg.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            cfg.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--control-value")) {
            cfg.control_value = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--candidate-value")) {
            cfg.candidate_value = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prefill-chunk")) {
            cfg.prefill_chunk = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--candidate-chunk")) {
            cfg.candidate_chunk = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--output")) {
            cfg.output_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--candidate-env")) {
            cfg.candidate_env = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--extra-env") || !strcmp(arg, "--control-env")) {
            const char *spec = need_arg(&i, argc, argv, arg);
            const char **list = arg[2] == 'e' ? cfg.extra_env : cfg.control_env;
            int *n = arg[2] == 'e' ? &cfg.n_extra : &cfg.n_control;
            if (!strchr(spec, '=') || spec[0] == '=' || *n >= 16) {
                fprintf(stderr, "%s: %s needs NAME=VALUE (at most 16)\n", BENCH_NAME, arg);
                exit(2);
            }
            list[(*n)++] = spec;
        } else if (!strcmp(arg, "--interleave")) {
            cfg.interleave = true;
        } else if (!strcmp(arg, "--single-run")) {
            cfg.single_run = true;
        } else if (!strcmp(arg, "--prefix-tokens")) {
            cfg.prefix_tokens =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--warmup-tokens")) {
            cfg.warmup_tokens = parse_int_arg(
                need_arg(&i, argc, argv, arg), arg, DEFAULT_WARMUP_TOKENS);
        } else if (!strcmp(arg, "--tolerate-drift")) {
            cfg.tolerate_drift = true;
        } else if (!strcmp(arg, "--compare-state")) {
            cfg.compare_state = true;
        } else if (!strcmp(arg, "--initial-tokens")) {
            cfg.initial_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--ctx")) {
            cfg.ctx = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 2);
        } else if (!strcmp(arg, "--repeats")) {
            cfg.repeats =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else {
            fprintf(stderr, "%s: unknown option: %s\n", BENCH_NAME, arg);
            usage(stderr, argv[0]);
            exit(2);
        }
    }

    if (cfg.single_run) {
        if (cfg.interleave || cfg.candidate_chunk) {
            fprintf(stderr, "%s: --single-run requires the control prefill path\n", BENCH_NAME);
            exit(2);
        }
        cfg.repeats = 1;
        if (!cfg.candidate_env) cfg.candidate_env = "DS4_BENCH_NOOP";
    }
    if (cfg.candidate_chunk && !cfg.candidate_env)
        cfg.candidate_env = "DS4_QWEN4_PREFILL_CHUNK";
    if (cfg.candidate_chunk && (cfg.interleave || cfg.candidate_chunk > 65536 || cfg.prefill_chunk > 65536)) {
        fprintf(stderr, "%s: chunk comparison requires non-interleaved chunks <= 65536\n", BENCH_NAME);
        exit(2);
    }
    const bool one_batch_append = cfg.initial_tokens > 0 && cfg.prefix_tokens > cfg.initial_tokens &&
        cfg.prefix_tokens - cfg.initial_tokens <= cfg.prefill_chunk &&
        cfg.prefix_tokens - cfg.initial_tokens <= cfg.candidate_chunk;
    if (cfg.compare_state && (cfg.interleave ||
            (cfg.candidate_chunk && cfg.candidate_chunk != cfg.prefill_chunk && !one_batch_append))) {
        fprintf(stderr, "%s: --compare-state requires identical chunks or a continued append that fits both capacities, without interleaving\n", BENCH_NAME);
        exit(2);
    }
    if (!cfg.candidate_env || cfg.candidate_env[0] == '\0' ||
        strchr(cfg.candidate_env, '=') != NULL) {
        fprintf(stderr, "%s: --candidate-env requires a valid name\n", BENCH_NAME);
        exit(2);
    }
    if (cfg.initial_tokens >= cfg.prefix_tokens) {
        fprintf(stderr, "%s: --initial-tokens must be less than --prefix-tokens\n", BENCH_NAME);
        exit(2);
    }
    const int continuation = cfg.compare_state ? STATE_CONTINUATION_TOKENS : 0;
    if (cfg.prefix_tokens > INT_MAX - continuation) {
        fprintf(stderr, "%s: requested token length is too large\n", BENCH_NAME);
        exit(2);
    }
    const int longest =
        cfg.prefix_tokens + continuation > cfg.warmup_tokens
            ? cfg.prefix_tokens + continuation
            : cfg.warmup_tokens;
    if (longest == INT_MAX) {
        fprintf(stderr, "%s: requested token length is too large\n", BENCH_NAME);
        exit(2);
    }
    if (cfg.ctx == 0) cfg.ctx = longest + 1;
    if (cfg.ctx <= longest) {
        fprintf(stderr,
                "%s: --ctx must exceed prefix, continuation and warmup lengths "
                "(minimum %d)\n",
                BENCH_NAME,
                longest + 1);
        exit(2);
    }
    return cfg;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr,
                "%s: failed to open %s: %s\n",
                BENCH_NAME,
                path,
                strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: failed to seek %s\n", BENCH_NAME, path);
        fclose(fp);
        return NULL;
    }
    const long len = ftell(fp);
    if (len < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: failed to size %s\n", BENCH_NAME, path);
        fclose(fp);
        return NULL;
    }
    char *text = malloc((size_t)len + 1u);
    if (!text) {
        fprintf(stderr, "%s: out of memory reading %s\n", BENCH_NAME, path);
        fclose(fp);
        return NULL;
    }
    if (fread(text, 1, (size_t)len, fp) != (size_t)len) {
        fprintf(stderr, "%s: failed to read %s\n", BENCH_NAME, path);
        free(text);
        fclose(fp);
        return NULL;
    }
    text[len] = '\0';
    fclose(fp);
    return text;
}

static int apply_env_list(const char *const *list, int n, bool set) {
    for (int i = 0; i < n; i++) {
        const char *eq = strchr(list[i], '=');
        char name[128];
        const size_t len = (size_t)(eq - list[i]);
        if (len >= sizeof(name)) return 1;
        memcpy(name, list[i], len);
        name[len] = '\0';
        if ((set ? setenv(name, eq + 1, 1) : unsetenv(name)) != 0) return 1;
    }
    return 0;
}

static int select_variant(const bench_config *cfg, int variant) {
    /* Control: candidate settings unset (or set to --control-value),
     * control-only settings applied.  Candidate: control-only settings
     * unset, candidate settings applied. */
    int env_rc = apply_env_list(cfg->control_env, cfg->n_control, variant == 0);
    env_rc |= apply_env_list(cfg->extra_env, cfg->n_extra, variant != 0);
    env_rc |= variant == 0
            ? (cfg->control_value ? setenv(cfg->candidate_env, cfg->control_value, 1) : unsetenv(cfg->candidate_env))
            : setenv(cfg->candidate_env, cfg->candidate_value, 1);
    if (cfg->candidate_chunk) {
        char chunk[32];
        snprintf(chunk, sizeof(chunk), "%d", variant ? cfg->candidate_chunk : cfg->prefill_chunk);
        env_rc |= setenv("DS4_QWEN4_PREFILL_CHUNK", chunk, 1);
    }
    if (env_rc != 0) {
        fprintf(stderr,
                "%s: failed to select %s with %s: %s\n",
                BENCH_NAME,
                variant == 0 ? "control" : "candidate",
                cfg->candidate_env,
                strerror(errno));
        return 1;
    }
    return 0;
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool g_tolerate_drift = false;
static bool g_last_exact = true;

/* Drift statistics over one full-vocabulary row: max and mean |delta|, the
 * top-1 ids of both rows, and the reference top-1 margin (top-1 minus top-2). */
static void report_drift(const float *reference, const float *observed, int vocab, size_t run) {
    double max_d = 0.0, sum_d = 0.0;
    int r0 = 0, o0 = 0, r1 = -1;
    for (int i = 0; i < vocab; i++) {
        const double d = fabs((double)reference[i] - (double)observed[i]);
        if (d > max_d) max_d = d;
        sum_d += d;
        if (reference[i] > reference[r0]) { r1 = r0; r0 = i; }
        else if (i != r0 && (r1 < 0 || reference[i] > reference[r1])) r1 = i;
        if (observed[i] > observed[o0]) o0 = i;
    }
    printf("run=%zu drift max_abs=%.6g mean_abs=%.6g top1_reference=%d top1_observed=%d top1_agree=%s reference_margin=%.6g\n",
           run, max_d, sum_d / (double)vocab, r0, o0, r0 == o0 ? "yes" : "no",
           r1 >= 0 ? (double)reference[r0] - (double)reference[r1] : 0.0);
}

static int compare_logits(
        const float *reference,
        const float *observed,
        int          vocab,
        size_t       run) {
    const size_t bytes = (size_t)vocab * sizeof(reference[0]);
    /* Bit tests remain valid under the project's -ffast-math build. A pair
     * of identical NaNs must never establish a correctness reference. */
    for (int i = 0; i < vocab; i++) {
        if ((float_bits(reference[i]) & 0x7f800000u) == 0x7f800000u ||
            (float_bits(observed[i]) & 0x7f800000u) == 0x7f800000u) {
            g_last_exact = false;
            fprintf(stderr, "%s: nonfinite logit at run %zu id=%d\n", BENCH_NAME, run, i);
            return 1;
        }
    }
    g_last_exact = memcmp(reference, observed, bytes) == 0;
    if (g_last_exact) return 0;
    if (g_tolerate_drift) { report_drift(reference, observed, vocab, run); return 0; }

    size_t first = SIZE_MAX;
    size_t differing = 0;
    for (int i = 0; i < vocab; i++) {
        if (memcmp(&reference[i], &observed[i], sizeof(float)) != 0) {
            if (first == SIZE_MAX) first = (size_t)i;
            differing++;
        }
    }
    fprintf(stderr,
            "%s: raw logit mismatch at run %zu: differing=%zu/%d\n",
            BENCH_NAME,
            run,
            differing,
            vocab);
    if (first != SIZE_MAX) {
        fprintf(stderr,
                "%s: first mismatch id=%zu reference=%a (0x%08x) "
                "observed=%a (0x%08x)\n",
                BENCH_NAME,
                first,
                reference[first],
                (unsigned)float_bits(reference[first]),
                observed[first],
                (unsigned)float_bits(observed[first]));
    }
    return 1;
}

static int prepare_run_session(ds4_session **out, ds4_engine *engine,
                               const bench_config *cfg, const ds4_tokens *initial,
                               int variant, char *err, size_t errlen);

typedef struct {
    double seconds, first_token_ms, selection_ms;
    int first_token_id;
} prefill_timing;

/* Sync throughput excludes selection. The warm request's first-token clock
 * includes that immediately following CPU argmax, before any telemetry. */
static int sync_and_select(ds4_session *session, const ds4_tokens *tokens,
                           int eos, prefill_timing *timing, char *err, size_t errlen) {
    const double start = now_sec();
    const int rc = ds4_session_sync(session, tokens, err, errlen);
    const double synced = now_sec();
    if (rc != 0) return rc;
    const int selected = ds4_session_argmax_excluding(session, eos);
    const double selected_at = now_sec();
    if (selected < 0) {
        snprintf(err, errlen, "first-token selection returned no token");
        return 1;
    }
    if (synced < start || selected_at < synced) {
        snprintf(err, errlen, "nonmonotonic first-token clock");
        return 1;
    }
    *timing = (prefill_timing){synced - start, (selected_at - start) * 1000.0,
                              (selected_at - synced) * 1000.0, selected};
    return 0;
}

static int warm_variant(
        ds4_engine         *engine,
        const bench_config *cfg,
        const ds4_tokens   *all_tokens,
        int                 variant,
        char               *err,
        size_t              errlen) {
    ds4_session *session = NULL;
    /* Continued prefills can select kernels absent from a fresh short prefix,
     * especially at the sparse-attention boundary. Warm the measured append
     * at its actual position, using the same control-prefix construction. */
    ds4_tokens initial = {
        .v = all_tokens->v,
        .len = cfg->initial_tokens,
        .cap = cfg->initial_tokens,
    };
    const int target = initial.len ? cfg->prefix_tokens : cfg->warmup_tokens;
    ds4_tokens warmup = {
        .v = all_tokens->v,
        .len = target,
        .cap = target,
    };
    if (prepare_run_session(&session, engine, cfg, &initial, variant, err, errlen) != 0) {
        fprintf(stderr,
                "%s: failed to prepare %s warmup session: %s\n",
                BENCH_NAME,
                variant == 0 ? "control" : "candidate",
                err[0] ? err : "unknown error");
        ds4_session_free(session);
        return 1;
    }
    int rc = ds4_session_sync(session, &warmup, err, errlen);
    if (rc == 0 && ds4_session_argmax_excluding(session, ds4_token_eos(engine)) < 0) {
        snprintf(err, errlen, "warmup first-token selection returned no token");
        rc = 1;
    }
    if (rc != 0) {
        fprintf(stderr,
                "%s: %s warmup failed: %s\n",
                BENCH_NAME,
                variant == 0 ? "control" : "candidate",
                err[0] ? err : "unknown error");
    }
    if (rc == 0) {
        printf("warmup variant=%s schedule=%s initial_tokens=%d target_tokens=%d append_tokens=%d chunk=%d selection_warmed=yes\n",
               variant == 0 ? "control" : "candidate", initial.len ? "continued" : "fresh",
               initial.len, target, target - initial.len,
               variant && cfg->candidate_chunk ? cfg->candidate_chunk : cfg->prefill_chunk);
    }
    ds4_session_free(session);
    return rc != 0;
}

/* Prefix arithmetic is always the control's. Chunk capacities are fixed at
 * session creation, so a chunk comparison restores that prefix into a new
 * session of the requested capacity before starting the timer. */
static int prepare_run_session(ds4_session **out, ds4_engine *engine,
                               const bench_config *cfg, const ds4_tokens *initial,
                               int variant, char *err, size_t errlen) {
    *out = NULL;
    if (select_variant(cfg, initial->len ? 0 : variant) != 0 ||
        ds4_session_create(out, engine, cfg->ctx) != 0) return 1;
    if (initial->len && ds4_session_sync(*out, initial, err, errlen) != 0) return 1;
    if (initial->len && cfg->candidate_chunk && cfg->candidate_chunk != cfg->prefill_chunk) {
        ds4_session_snapshot prefix = {0};
        if (ds4_session_save_snapshot(*out, &prefix, err, errlen) != 0) {
            ds4_session_snapshot_free(&prefix);
            return 1;
        }
        ds4_session_free(*out);
        *out = NULL;
        const int failed = select_variant(cfg, variant) != 0 ||
            ds4_session_create(out, engine, cfg->ctx) != 0 ||
            ds4_session_load_snapshot(*out, &prefix, err, errlen) != 0;
        ds4_session_snapshot_free(&prefix);
        if (failed) return 1;
    }
    return select_variant(cfg, variant) != 0;
}

static int compare_snapshot_bytes(const ds4_session_snapshot *reference,
                                  const ds4_session_snapshot *observed,
                                  size_t run, int continuation) {
    if (reference->len != observed->len) {
        fprintf(stderr, "%s: run=%zu continuation=%d checkpoint length mismatch reference=%llu observed=%llu\n",
                BENCH_NAME, run, continuation, (unsigned long long)reference->len,
                (unsigned long long)observed->len);
        return 1;
    }
    if (memcmp(reference->ptr, observed->ptr, (size_t)reference->len) == 0) return 0;
    uint64_t first = 0;
    while (first < reference->len && reference->ptr[first] == observed->ptr[first]) first++;
    fprintf(stderr, "%s: run=%zu continuation=%d checkpoint mismatch byte=%llu reference=%02x observed=%02x\n",
            BENCH_NAME, run, continuation, (unsigned long long)first,
            reference->ptr[first], observed->ptr[first]);
    return 1;
}

static void report_hashes(const float *logits, int vocab, const ds4_session_snapshot *snapshot,
                          size_t run, int continuation, int position) {
    char logits_hash[65], checkpoint_hash[65];
    ds4_bench_sha256(logits, (size_t)vocab * sizeof(float), logits_hash);
    ds4_bench_sha256(snapshot->ptr, (size_t)snapshot->len, checkpoint_hash);
    printf("hashes run=%zu continuation=%d position=%d logits_sha256=%s checkpoint_sha256=%s checkpoint_bytes=%llu\n",
           run, continuation, position, logits_hash, checkpoint_hash, (unsigned long long)snapshot->len);
}

/* This deliberately ignores --tolerate-drift: checkpoint validation is exact.
 * Payloads include logits and live recurrent/cache state, not GPU scratch.
 * Save/readback and known-token continuations all occur after the timer. */
static int verify_state(ds4_session *session, const bench_config *cfg,
                        const ds4_tokens *tokens, ds4_session_snapshot *references,
                        float *logits, int vocab, size_t run, char *err, size_t errlen) {
    for (int step = 0; step <= STATE_CONTINUATION_TOKENS; step++) {
        if (step && ds4_session_eval(session, tokens->v[cfg->prefix_tokens + step - 1], err, errlen) != 0) {
            fprintf(stderr, "%s: run=%zu continuation=%d failed: %s\n", BENCH_NAME, run, step, err);
            return 1;
        }
        if (ds4_session_pos(session) != cfg->prefix_tokens + step) {
            fprintf(stderr, "%s: run=%zu continuation=%d has unexpected position\n", BENCH_NAME, run, step);
            return 1;
        }
        if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
            fprintf(stderr, "%s: run=%zu continuation=%d logits unavailable\n", BENCH_NAME, run, step);
            return 1;
        }
        for (int i = 0; i < vocab; i++) {
            if ((float_bits(logits[i]) & 0x7f800000u) == 0x7f800000u) {
                fprintf(stderr, "%s: run=%zu continuation=%d nonfinite logit id=%d\n", BENCH_NAME, run, step, i);
                return 1;
            }
        }
        ds4_session_snapshot observed = {0};
        if (ds4_session_save_snapshot(session, &observed, err, errlen) != 0) {
            fprintf(stderr, "%s: run=%zu continuation=%d snapshot failed: %s\n", BENCH_NAME, run, step, err);
            ds4_session_snapshot_free(&observed);
            return 1;
        }
        if (cfg->single_run) report_hashes(logits, vocab, &observed, run, step, cfg->prefix_tokens + step);
        int failed = 0;
        if (!references[step].ptr) {
            references[step] = observed;
            memset(&observed, 0, sizeof(observed));
        } else failed = compare_snapshot_bytes(&references[step], &observed, run, step);
        ds4_session_snapshot_free(&observed);
        if (failed) return 1;
        printf("state run=%zu continuation=%d position=%d bytes=%llu exact=yes\n", run, step,
               cfg->prefix_tokens + step, (unsigned long long)references[step].len);
    }
    return 0;
}

int main(int argc, char **argv) {
    static const int orders[2][RUNS_PER_REPEAT] = {
        {0, 1, 1, 0},
        {1, 0, 0, 1},
    };
    const bench_config cfg = parse_options(argc, argv);
    if (cfg.output_path && !freopen(cfg.output_path, "w", stdout)) {
        perror(cfg.output_path);
        return 1;
    }
    if (cfg.output_path) {
        if (dup2(fileno(stdout), STDERR_FILENO) < 0) { perror("dup2"); return 1; }
        setvbuf(stdout, NULL, _IOLBF, 0);
    }
    g_tolerate_drift = cfg.tolerate_drift;
    char *text = read_text(cfg.prompt_path);
    if (!text) return 1;
    if (select_variant(&cfg, 0) != 0) {
        free(text);
        return 1;
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = DS4_BACKEND_METAL,
        .context_size = cfg.ctx,
        /* Qwen reads its chunk environment when each session is created. */
        .prefill_chunk = cfg.candidate_chunk ? 0u : (uint32_t)cfg.prefill_chunk,
        .power_percent = 100,
        .warm_weights = true,
    };
    ds4_engine *engine = NULL;
    ds4_tokens tokens = {0};
    float *reference = NULL;
    float *observed = NULL;
    ds4_session_snapshot state_reference[STATE_CONTINUATION_TOKENS + 1] = {{0}};
    variant_result results[VARIANT_COUNT] = {0};
    char err[256] = {0};
    bool have_reference = false;
    void *engine_pool = ds4_bench_pool_create();
    void *run_pool = NULL;
    size_t exact_runs = 0;
    double paired_sum = 0, paired_sum_sq = 0;
    int rc = 1;

    if (ds4_engine_open(&engine, &opt) != 0) goto done;
    if (cfg.candidate_chunk && !ds4_engine_is_qwen4(engine)) {
        fprintf(stderr, "%s: --candidate-chunk requires Qwen\n", BENCH_NAME);
        goto done;
    }
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    text = NULL;

    const int final_tokens = cfg.prefix_tokens + (cfg.compare_state ? STATE_CONTINUATION_TOKENS : 0);
    const int needed =
        final_tokens > cfg.warmup_tokens
            ? final_tokens
            : cfg.warmup_tokens;
    if (tokens.len < needed) {
        fprintf(stderr,
                "%s: prompt has %d tokens; need %d\n",
                BENCH_NAME,
                tokens.len,
                needed);
        goto done;
    }
    const int vocab = ds4_engine_vocab_size(engine);
    const int eos = ds4_token_eos(engine);
    if (vocab < 1 || eos < 0 || eos >= vocab) {
        fprintf(stderr, "%s: valid vocabulary and EOS token required for first-token selection\n", BENCH_NAME);
        goto done;
    }
    const size_t logit_bytes = (size_t)vocab * sizeof(float);
    reference = malloc(logit_bytes);
    observed = malloc(logit_bytes);
    if (!reference || !observed) {
        fprintf(stderr, "%s: logit buffer allocation failed\n", BENCH_NAME);
        goto done;
    }

    fprintf(stderr,
            "%s: model=%s prompt=%s prefix=%d initial=%d warmup=%d ctx=%d repeats=%d "
            "candidate_env=%s candidate_value=%s control_value=%s prefill_chunk=%d candidate_chunk=%d compare_state=%s continuation_tokens=%d single_run=%s\n",
            BENCH_NAME,
            cfg.model_path,
            cfg.prompt_path,
            cfg.prefix_tokens,
            cfg.initial_tokens,
            cfg.warmup_tokens,
            cfg.ctx,
            cfg.repeats,
            cfg.candidate_env,
            cfg.candidate_value,
            cfg.control_value ? cfg.control_value : "<unset>",
            cfg.prefill_chunk, cfg.candidate_chunk, cfg.compare_state ? "yes" : "no",
            cfg.compare_state ? STATE_CONTINUATION_TOKENS : 0, cfg.single_run ? "yes" : "no");
    if (cfg.initial_tokens && cfg.candidate_chunk && cfg.candidate_chunk != cfg.prefill_chunk) {
        printf("prefix_setup source_variant=control source_chunk=%d snapshot_restore=both_variants timing=excluded append_tokens=%d\n",
               cfg.prefill_chunk, cfg.prefix_tokens - cfg.initial_tokens);
    }

    for (int variant = 0; variant < (cfg.single_run ? 1 : VARIANT_COUNT); variant++) {
        err[0] = '\0';
        run_pool = ds4_bench_pool_create();
        const int warm_rc = warm_variant(engine, &cfg, &tokens, variant, err, sizeof(err));
        ds4_bench_pool_drain(run_pool);
        run_pool = NULL;
        if (warm_rc != 0) {
            goto done;
        }
    }

    ds4_tokens prefix = {
        .v = tokens.v,
        .len = cfg.prefix_tokens,
        .cap = cfg.prefix_tokens,
    };
    ds4_tokens initial = { .v = tokens.v, .len = cfg.initial_tokens, .cap = cfg.initial_tokens };
    const int timed_tokens = cfg.prefix_tokens - cfg.initial_tokens;
    size_t run = 0;
    if (cfg.interleave) {
        /* Sustained load drifts the GPU by several percent within a minute,
         * and single fresh-session runs carry positional outliers, so
         * measure both variants inside the same session: every chunk after
         * --initial-tokens is timed under the variant of its parity, and the
         * next repeat swaps the parity.  Final logits must match across
         * repeats, which pins every chunk of both variants. */
        const int chunk = cfg.prefill_chunk;
        if (timed_tokens % chunk != 0 || cfg.initial_tokens % chunk != 0) {
            fprintf(stderr, "%s: --interleave needs --initial-tokens and --prefix-tokens as multiples of --prefill-chunk\n", BENCH_NAME);
            goto done;
        }
        /* The first session that reaches a context length pays first-use
         * costs on its buffers for every chunk after the first (measured at
         * -25% on an M5 Max), so pass 0 walks the whole prefix untimed. */
        for (int repeat = -1; repeat < cfg.repeats; repeat++) {
            run_pool = ds4_bench_pool_create();
            const bool timed = repeat >= 0;
            ds4_session *session = NULL;
            if (select_variant(&cfg, 0) != 0 || ds4_session_create(&session, engine, cfg.ctx) != 0) {
                fprintf(stderr, "%s: failed to create interleaved session %d\n", BENCH_NAME, repeat + 1);
                goto done;
            }
            err[0] = '\0';
            if (initial.len && ds4_session_sync(session, &initial, err, sizeof(err)) != 0) {
                fprintf(stderr, "%s: initial prefix failed: %s\n", BENCH_NAME, err);
                ds4_session_free(session);
                goto done;
            }
            for (int c = 0; c < timed_tokens / chunk; c++) {
                const int variant = (c + repeat) & 1;
                ds4_tokens part = { .v = tokens.v, .len = cfg.initial_tokens + (c + 1) * chunk,
                                    .cap = cfg.initial_tokens + (c + 1) * chunk };
                if (select_variant(&cfg, variant) != 0) { ds4_session_free(session); goto done; }
                prefill_timing timing = {0};
                const int sync_rc = sync_and_select(session, &part, eos, &timing, err, sizeof(err));
                const double seconds = timing.seconds;
                if (sync_rc != 0 || ds4_session_pos(session) != part.len) {
                    fprintf(stderr, "%s: interleaved chunk %d failed: %s\n", BENCH_NAME, c + 1, err[0] ? err : "position");
                    ds4_session_free(session);
                    goto done;
                }
                if (timed) {
                    results[variant].seconds += seconds;
                    results[variant].runs++;
                    results[variant].tokens += (size_t)chunk;
                }
                printf("run=%zu repeat=%d pattern=%s slot=%d variant=%s tokens=%d seconds=%.9f tokens_per_second=%.4f exact=%s first_token_ms=%.9f selection_ms=%.9f first_token_id=%d\n",
                       run + 1, repeat + 1, !timed ? "warm" : repeat & 1 ? "BABA" : "ABAB", c + 1,
                       variant == 0 ? "control" : "candidate", chunk, seconds,
                       seconds > 0.0 ? (double)chunk / seconds : 0.0,
                       timed ? "pending" : "untimed", timing.first_token_ms, timing.selection_ms, timing.first_token_id);
                fflush(stdout);
                if (timed) run++;
            }
            memset(observed, (repeat & 1) == 0 ? 0xa5 : 0x5a, logit_bytes);
            if (ds4_session_copy_logits(session, observed, vocab) != vocab) {
                fprintf(stderr, "%s: failed to copy logits after repeat %d\n", BENCH_NAME, repeat + 1);
                ds4_session_free(session);
                goto done;
            }
            if (!have_reference) {
                memcpy(reference, observed, logit_bytes);
                have_reference = true;
            } else if (compare_logits(reference, observed, vocab, (size_t)repeat + 1) != 0) {
                ds4_session_free(session);
                goto done;
            }
            if (timed && g_last_exact) exact_runs++;
            printf("repeat=%d final logits exact=%s\n", repeat + 1, g_last_exact ? "yes" : "no");
            ds4_session_free(session);
            ds4_bench_pool_drain(run_pool);
            run_pool = NULL;
        }
    } else
    for (int repeat = 0; repeat < cfg.repeats; repeat++) {
        double block_seconds[2] = {0};
        const int *order = orders[repeat & 1];
        const char *pattern = cfg.single_run ? "single" : (repeat & 1) == 0 ? "ABBA" : "BAAB";
        for (int slot = 0; slot < (cfg.single_run ? 1 : RUNS_PER_REPEAT); slot++) {
            run_pool = ds4_bench_pool_create();
            const int variant = order[slot];
            ds4_session *session = NULL;
            err[0] = '\0';
            if (prepare_run_session(&session, engine, &cfg, &initial, variant, err, sizeof(err)) != 0) {
                fprintf(stderr,
                        "%s: failed to prepare run %zu %s session: %s\n",
                        BENCH_NAME,
                        run + 1,
                        variant == 0 ? "control" : "candidate", err);
                if (session) ds4_session_free(session);
                goto done;
            }

            err[0] = '\0';
            ds4_bench_memory before, after;
            ds4_bench_memory_sample(getpid(), &before);
            prefill_timing timing = {0};
            const int sync_rc = sync_and_select(session, &prefix, eos, &timing, err, sizeof(err));
            const double seconds = timing.seconds;
            ds4_bench_memory_sample(getpid(), &after);
            block_seconds[variant] += seconds;
            if (sync_rc != 0) {
                fprintf(stderr,
                        "%s: run %zu %s prefill failed: %s\n",
                        BENCH_NAME,
                        run + 1,
                        variant == 0 ? "control" : "candidate",
                        err[0] ? err : "unknown error");
                ds4_session_free(session);
                goto done;
            }
            if (ds4_session_pos(session) != cfg.prefix_tokens) {
                fprintf(stderr,
                        "%s: run %zu ended at position %d; expected %d\n",
                        BENCH_NAME,
                        run + 1,
                        ds4_session_pos(session),
                        cfg.prefix_tokens);
                ds4_session_free(session);
                goto done;
            }

            memset(observed, (run & 1u) == 0 ? 0xa5 : 0x5a, logit_bytes);
            if (ds4_session_copy_logits(session, observed, vocab) != vocab) {
                fprintf(stderr,
                        "%s: failed to copy logits at run %zu\n",
                        BENCH_NAME,
                        run + 1);
                ds4_session_free(session);
                goto done;
            }
            if (!have_reference) {
                if (variant != 0) {
                    fprintf(stderr,
                            "%s: first measured run must be control\n",
                            BENCH_NAME);
                    ds4_session_free(session);
                    goto done;
                }
                memset(reference, 0x5a, logit_bytes);
                if (ds4_session_copy_logits(session, reference, vocab) != vocab ||
                    compare_logits(reference, observed, vocab, run + 1) != 0) {
                    fprintf(stderr,
                            "%s: failed to establish control reference\n",
                            BENCH_NAME);
                    ds4_session_free(session);
                    goto done;
                }
                have_reference = true;
            } else if (compare_logits(
                           reference, observed, vocab, run + 1) != 0) {
                ds4_session_free(session);
                goto done;
            }
            if (cfg.single_run && !cfg.compare_state) {
                ds4_session_snapshot snapshot = {0};
                if (ds4_session_save_snapshot(session, &snapshot, err, sizeof(err)) != 0) {
                    fprintf(stderr, "%s: snapshot hash failed: %s\n", BENCH_NAME, err);
                    ds4_session_snapshot_free(&snapshot);
                    ds4_session_free(session);
                    goto done;
                }
                report_hashes(observed, vocab, &snapshot, run + 1, 0, ds4_session_pos(session));
                ds4_session_snapshot_free(&snapshot);
            }
            if (cfg.compare_state && verify_state(session, &cfg, &tokens, state_reference,
                                                   observed, vocab, run + 1, err, sizeof(err)) != 0) {
                ds4_session_free(session);
                goto done;
            }

            results[variant].seconds += seconds;
            results[variant].runs++;
            results[variant].tokens += (size_t)timed_tokens;
            if (g_last_exact) exact_runs++;
            printf("run=%zu repeat=%d pattern=%s slot=%d variant=%s "
                   "tokens=%d seconds=%.9f tokens_per_second=%.4f exact=%s first_token_ms=%.9f selection_ms=%.9f first_token_id=%d\n",
                   run + 1,
                   repeat + 1,
                   pattern,
                   slot + 1,
                   variant == 0 ? "control" : "candidate",
                   timed_tokens,
                   seconds,
                   seconds > 0.0
                       ? (double)timed_tokens / seconds
                       : 0.0,
                   g_last_exact ? "yes" : "no", timing.first_token_ms, timing.selection_ms, timing.first_token_id);
            fflush(stdout);
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0) {
                printf("resources run=%zu maxrss=%ld minor_faults=%ld major_faults=%ld input_blocks=%ld output_blocks=%ld swaps=%ld\n",
                       run + 1, usage.ru_maxrss, usage.ru_minflt, usage.ru_majflt,
                       usage.ru_inblock, usage.ru_oublock, usage.ru_nswap);
            }
            ds4_session_free(session);
            ds4_bench_pool_drain(run_pool);
            run_pool = NULL;
            ds4_bench_memory released;
            ds4_bench_memory_sample(getpid(), &released);
            printf("telemetry run=%zu process_valid=%d swap_valid=%d vm_valid=%d resident_bytes=%llu footprint_bytes=%llu released_resident_bytes=%llu disk_read_bytes=%llu disk_write_bytes=%llu global_swap_used_bytes=%llu global_swapin_pages=%llu global_swapout_pages=%llu\n",
                   run + 1, before.process_valid && after.process_valid && released.process_valid,
                   after.swap_usage_valid, before.vm_statistics_valid && after.vm_statistics_valid,
                   (unsigned long long)after.resident_bytes, (unsigned long long)after.physical_footprint_bytes,
                   (unsigned long long)released.resident_bytes,
                   (unsigned long long)(after.disk_read_bytes - before.disk_read_bytes),
                   (unsigned long long)(after.disk_write_bytes - before.disk_write_bytes),
                   (unsigned long long)after.swap_used_bytes,
                   (unsigned long long)(after.swapins_pages - before.swapins_pages),
                   (unsigned long long)(after.swapouts_pages - before.swapouts_pages));
            run++;
        }
        if (!cfg.single_run) {
            const double delta = log(block_seconds[0] / block_seconds[1]);
            paired_sum += delta;
            paired_sum_sq += delta * delta;
            printf("paired repeat=%d gain_percent=%.6f\n", repeat + 1, 100.0 * expm1(delta));
        }
    }

    if (!cfg.single_run && !cfg.interleave && cfg.repeats >= 3) {
        /* Treat each balanced four-run block as one observation, not four
         * independent samples. Student-t interval on log throughput ratios. */
        static const double t95[] = {0, 12.706205, 4.302653, 3.182446, 2.776445,
            2.570582, 2.446912, 2.364624, 2.306004, 2.262157, 2.228139,
            2.200985, 2.178813, 2.160369, 2.144787, 2.131450, 2.119906,
            2.109816, 2.100922, 2.093024, 2.085964, 2.079614, 2.073874,
            2.068658, 2.063899, 2.059539, 2.055529, 2.051831, 2.048407,
            2.045230, 2.042273};
        const double n = cfg.repeats, mean = paired_sum / n;
        const double variance = fmax(0.0, (paired_sum_sq - paired_sum * mean) / (n - 1));
        const int df = cfg.repeats - 1;
        const double radius = t95[df < 30 ? df : 30] * sqrt(variance / n);
        printf("paired_ci blocks=%d gain_percent=%.6f low95_percent=%.6f high95_percent=%.6f all_logits_exact=%s\n",
               cfg.repeats, 100 * expm1(mean), 100 * expm1(mean - radius),
               100 * expm1(mean + radius), exact_runs == run ? "yes" : "no");
    }

    for (int variant = 0; variant < (cfg.single_run ? 1 : VARIANT_COUNT); variant++) {
        printf("aggregate variant=%s runs=%zu tokens=%zu seconds=%.6f "
               "tokens_per_second=%.4f\n",
               variant == 0 ? "control" : "candidate",
               results[variant].runs,
               results[variant].tokens,
               results[variant].seconds,
               results[variant].seconds > 0.0
                   ? (double)results[variant].tokens /
                         results[variant].seconds
                   : 0.0);
    }
    if (cfg.single_run) {
        printf("single_summary exact_runs=%zu exact_floats=%zu vocab=%d\n",
               exact_runs, exact_runs * (size_t)vocab, vocab);
    } else {
        const double control_tps =
            results[0].seconds > 0.0
                ? (double)results[0].tokens / results[0].seconds
                : 0.0;
        const double candidate_tps =
            results[1].seconds > 0.0
                ? (double)results[1].tokens / results[1].seconds
                : 0.0;
        printf("candidate_delta_percent=%.4f exact_runs=%zu "
               "exact_floats=%zu vocab=%d\n",
               control_tps > 0.0
                   ? (candidate_tps / control_tps - 1.0) * 100.0
                   : 0.0,
               exact_runs,
               exact_runs * (size_t)vocab,
               vocab);
    }
    rc = 0;

done:
    ds4_bench_pool_drain(run_pool);
    free(text);
    free(reference);
    free(observed);
    for (int step = 0; step <= STATE_CONTINUATION_TOKENS; step++)
        ds4_session_snapshot_free(&state_reference[step]);
    ds4_tokens_free(&tokens);
    if (engine) ds4_engine_close(engine);
    ds4_bench_pool_drain(engine_pool);
    return rc;
}
