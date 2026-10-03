#include "ds4.h"
#include "benchmark_pool.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define FIRST_SPLIT_ENV "DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS"
#define SECOND_SPLIT_ENV "DS4_METAL_GRAPH_TOKEN_SECOND_SPLIT_LAYERS"

enum {
    VARIANT_COUNT = 2,
    DEFAULT_PREFIX_TOKENS = 2048,
    DEFAULT_CTX = 4096,
    DEFAULT_WARMUP = 16,
    DEFAULT_MEASURED = 128,
};

typedef struct {
    int first;
    int second;
} decode_schedule;

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *candidate_env;
    const char *candidate_value;
    const char *control_value;
    const char *output_path;
    int prefix_tokens;
    int ctx;
    int warmup;
    int measured;
    int repeats;
    int mtp_depth;
    uint32_t prefill_chunk;
    bool mtp;
    bool single_run;
    bool include_selection;
    decode_schedule control;
    decode_schedule candidate;
} bench_config;

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "\n"
            "  -m, --model PATH       GGUF path (default: ds4flash.gguf)\n"
            "  --prompt-file PATH     token source (default: ds4.c)\n"
            "  --prefix-tokens N      prefill length (default: 2048)\n"
            "  --ctx N                session allocation (default: 4096)\n"
            "  --warmup N             untimed steps per variant (default: 16)\n"
            "  --tokens N             measured tokens per run (default: 128)\n"
            "  --repeats N            alternating ABBA/BAAB blocks (minimum/default: 3)\n"
            "  --single-run           one warmed control sample with exact hashes; no paired CI\n"
            "  --prefill-chunk N      prefix prefill chunk (default: engine default)\n"
            "  --control-first N      control first split (default: 2)\n"
            "  --control-second N     control second split (default: 32)\n"
            "  --candidate-first N    candidate first split (default: 1; control with --candidate-env)\n"
            "  --candidate-second N   candidate second split (default: 32; control with --candidate-env)\n"
            "  --candidate-env NAME   dispatch-time environment control\n"
            "  --candidate-value V    candidate value (default: 1; unset removes NAME)\n"
            "  --control-value V      control value (default: unset)\n"
            "  --mtp                  benchmark greedy MTP separately from ordinary decode\n"
            "  --mtp-depth N          fixed MTP depth, 2 or 3 (default: 2)\n"
            "  --output PATH          write results and diagnostics to PATH\n"
            "  --include-selection    include one non-EOS argmax in each timed step\n",
            argv0);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr,
                "metal-decode-schedule-bench: %s requires an argument\n",
                opt);
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
                "metal-decode-schedule-bench: invalid value for %s: %s\n",
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
        .prefix_tokens = DEFAULT_PREFIX_TOKENS,
        .ctx = DEFAULT_CTX,
        .warmup = DEFAULT_WARMUP,
        .measured = DEFAULT_MEASURED,
        .repeats = 3,
        .mtp_depth = 2,
        .include_selection = false,
        .control = {.first = 2, .second = 32},
        .candidate = {.first = 1, .second = 32},
    };
    bool candidate_first_explicit = false;
    bool candidate_second_explicit = false;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            cfg.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            cfg.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--candidate-env")) {
            cfg.candidate_env = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--candidate-value")) {
            cfg.candidate_value = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--control-value")) {
            cfg.control_value = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--output")) {
            cfg.output_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--repeats")) {
            cfg.repeats = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--single-run")) {
            cfg.single_run = true;
        } else if (!strcmp(arg, "--prefill-chunk")) {
            cfg.prefill_chunk = (uint32_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--mtp")) {
            cfg.mtp = true;
        } else if (!strcmp(arg, "--mtp-depth")) {
            cfg.mtp_depth = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 2);
        } else if (!strcmp(arg, "--include-selection")) {
            cfg.include_selection = true;
        } else if (!strcmp(arg, "--prefix-tokens")) {
            cfg.prefix_tokens =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--ctx")) {
            cfg.ctx = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--warmup")) {
            cfg.warmup =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--tokens") || !strcmp(arg, "--measured")) {
            cfg.measured =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--control-first")) {
            cfg.control.first =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--control-second")) {
            cfg.control.second =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--candidate-first")) {
            cfg.candidate.first =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
            candidate_first_explicit = true;
        } else if (!strcmp(arg, "--candidate-second")) {
            cfg.candidate.second =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
            candidate_second_explicit = true;
        } else {
            fprintf(stderr,
                    "metal-decode-schedule-bench: unknown option: %s\n",
                    arg);
            usage(stderr, argv[0]);
            exit(2);
        }
    }

    if (cfg.mtp_depth > 3) {
        fprintf(stderr, "metal-decode-schedule-bench: --mtp-depth must be 2 or 3\n");
        exit(2);
    }
    if (!cfg.single_run && cfg.repeats < 3) {
        fprintf(stderr, "metal-decode-schedule-bench: at least 3 paired blocks are required\n");
        exit(2);
    }
    if (cfg.single_run) cfg.repeats = 1;
    if (cfg.candidate_value && !strcmp(cfg.candidate_value, "unset")) cfg.candidate_value = NULL;
    if (cfg.control_value && !strcmp(cfg.control_value, "unset")) cfg.control_value = NULL;
    if (cfg.candidate_env && (!cfg.candidate_env[0] || strchr(cfg.candidate_env, '='))) {
        fprintf(stderr, "metal-decode-schedule-bench: invalid environment name\n");
        exit(2);
    }
    /* These controls are cached before the second variant can run. Silently
     * toggling them in one engine would measure the control twice. */
    static const char *const cached[] = {
        "DS4_QWEN4_NO_FUSE", "DS4_QWEN4_NO_IDX_SELECT", "DS4_QWEN4_NO_STATE_POOL",
        "DS4_QWEN4_MTP_DRAFT_VOCAB", "DS4_QWEN4_TIMING", "DS4_QWEN4_BATCH_TIMING",
    };
    for (size_t i = 0; i < sizeof(cached) / sizeof(cached[0]); i++) {
        if (!cfg.single_run && cfg.candidate_env && !strcmp(cfg.candidate_env, cached[i])) {
            fprintf(stderr, "metal-decode-schedule-bench: %s is initialization-only or diagnostic; use separate process controls\n", cached[i]);
            exit(2);
        }
    }
    if (cfg.mtp && cfg.candidate_env &&
        (!strcmp(cfg.candidate_env, "DS4_QWEN4_MTP_DRAFT_ROWS") ||
         !strcmp(cfg.candidate_env, "DS4_QWEN4_MTP_DRAFT_VOCAB") ||
         !strcmp(cfg.candidate_env, "DS4_QWEN4_SPEC_FORCE_ACCEPT"))) {
        fprintf(stderr, "metal-decode-schedule-bench: MTP requires full-vocabulary drafts and normal acceptance\n");
        exit(2);
    }
    if (cfg.mtp && cfg.candidate_env && !strcmp(cfg.candidate_env, "DS4_QWEN4_MTP_DEPTH") &&
        (!cfg.control_value ||
         (strcmp(cfg.control_value, "2") && strcmp(cfg.control_value, "3")) ||
         (!cfg.single_run && (!cfg.candidate_value ||
          (strcmp(cfg.candidate_value, "2") && strcmp(cfg.candidate_value, "3")))))) {
        fprintf(stderr, "metal-decode-schedule-bench: MTP depth comparisons require explicit fixed 2/3 values\n");
        exit(2);
    }

    /*
     * An environment-gated comparison isolates that feature by default.
     * Explicit candidate split arguments can still combine a feature and
     * schedule experiment when desired.
     */
    if (cfg.candidate_env) {
        if (!candidate_first_explicit) {
            cfg.candidate.first = cfg.control.first;
        }
        if (!candidate_second_explicit) {
            cfg.candidate.second = cfg.control.second;
        }
    }

    const int64_t needed = (int64_t)cfg.prefix_tokens +
        (cfg.warmup > cfg.measured ? cfg.warmup : cfg.measured) + 1;
    if (needed > cfg.ctx) {
        fprintf(stderr,
                "metal-decode-schedule-bench: --ctx must exceed "
                "prefix + max(warmup, measured) (minimum %lld)\n",
                (long long)needed);
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
                "metal-decode-schedule-bench: failed to open %s: %s\n",
                path,
                strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to seek %s\n",
                path);
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to size %s\n",
                path);
        fclose(fp);
        return NULL;
    }
    char *text = malloc((size_t)len + 1u);
    if (!text) {
        fprintf(stderr,
                "metal-decode-schedule-bench: out of memory reading %s\n",
                path);
        fclose(fp);
        return NULL;
    }
    if (fread(text, 1, (size_t)len, fp) != (size_t)len) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to read %s\n",
                path);
        free(text);
        fclose(fp);
        return NULL;
    }
    text[len] = '\0';
    fclose(fp);
    return text;
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int select_variant(const bench_config *cfg, int variant) {
    const decode_schedule *schedule =
        variant == 0 ? &cfg->control : &cfg->candidate;
    char first[16];
    char second[16];
    snprintf(first, sizeof(first), "%d", schedule->first);
    snprintf(second, sizeof(second), "%d", schedule->second);
    if (setenv(FIRST_SPLIT_ENV, first, 1) != 0 ||
        setenv(SECOND_SPLIT_ENV, second, 1) != 0) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to select "
                "%s schedule %d/%d: %s\n",
                variant == 0 ? "control" : "candidate",
                schedule->first,
                schedule->second,
                strerror(errno));
        return 1;
    }
    const char *value = variant == 0 ? cfg->control_value : cfg->candidate_value;
    if (cfg->candidate_env &&
        (value ? setenv(cfg->candidate_env, value, 1) : unsetenv(cfg->candidate_env)) != 0) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to select candidate "
                "environment %s for %s: %s\n",
                cfg->candidate_env,
                variant == 0 ? "control" : "candidate",
                strerror(errno));
        return 1;
    }
    return 0;
}

/* Unset output diagnostics so A/B runs start clean. */
static int normalize_output_experiment_env(void) {
    static const char *const conflicting[] = {
        "DS4_METAL_REQUIRE_OUTPUT_HC_WEIGHTS4",
    };
    for (size_t i = 0; i < sizeof(conflicting) / sizeof(conflicting[0]); i++) {
        if (unsetenv(conflicting[i]) != 0) {
            fprintf(stderr,
                    "metal-decode-schedule-bench: failed to unset %s: %s\n",
                    conflicting[i],
                    strerror(errno));
            return 1;
        }
    }
    return 0;
}

/* Test IEEE exponent bits: isfinite may be folded away under -ffast-math. */
static int finite_logits(const float *row, int vocab, int frontier, const char *label) {
    for (int i = 0; i < vocab; i++) {
        const uint32_t bits = float_bits(row[i]);
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
            fprintf(stderr, "metal-decode-schedule-bench: nonfinite logit %s frontier=%d id=%d bits=0x%08x\n",
                    label, frontier, i, bits);
            return 1;
        }
    }
    return 0;
}

static int compare_logits(ds4_session *session, const float *expected,
                          float *observed, int vocab, int frontier) {
    memset(observed, 0xa5, (size_t)vocab * sizeof(float));
    if (ds4_session_copy_logits(session, observed, vocab) != vocab) return 1;
    if (finite_logits(expected, vocab, frontier, "expected") ||
        finite_logits(observed, vocab, frontier, "observed")) return 1;
    if (!memcmp(expected, observed, (size_t)vocab * sizeof(float))) return 0;
    int first = -1, differing = 0;
    for (int i = 0; i < vocab; i++) {
        if (float_bits(expected[i]) != float_bits(observed[i])) {
            if (first < 0) first = i;
            differing++;
        }
    }
    fprintf(stderr, "metal-decode-schedule-bench: raw logit mismatch frontier=%d differing=%d/%d first=%d expected=%a (0x%08x) observed=%a (0x%08x)\n",
            frontier, differing, vocab, first, expected[first], float_bits(expected[first]),
            observed[first], float_bits(observed[first]));
    return 1;
}

static int compare_snapshot(ds4_session *session, const ds4_session_snapshot *expected,
                            ds4_session_snapshot *observed, char *err, size_t errlen) {
    if (ds4_session_save_snapshot(session, observed, err, errlen)) return 1;
    if (observed->len == expected->len && !memcmp(observed->ptr, expected->ptr, (size_t)expected->len)) return 0;
    uint64_t first = 0, common = observed->len < expected->len ? observed->len : expected->len;
    while (first < common && observed->ptr[first] == expected->ptr[first]) first++;
    snprintf(err, errlen, "raw checkpoint mismatch: expected_bytes=%llu observed_bytes=%llu first_offset=%llu",
             (unsigned long long)expected->len, (unsigned long long)observed->len,
             (unsigned long long)first);
    return 1;
}

typedef struct {
    double seconds;
    int tokens;
} cycle_sample;

/* The timing pass replays the independently captured greedy trajectory. No
 * logit copies, checkpoint serialization, or diagnostic GPU waits occur
 * inside it. MTP must commit exactly the corresponding target token slice. */
static int run_segment(ds4_session *session, const bench_config *cfg,
                       const int *tokens, const float *rows, float *observed,
                       int vocab, int eos, int start, int count, bool verify,
                       cycle_sample *samples, int *n_samples, char *err, size_t errlen) {
    int done = 0, cycles = 0;
    while (done < count) {
        const int offset = start + done;
        if (verify && compare_logits(session, rows + (size_t)offset * vocab, observed, vocab, offset)) return 1;
        int accepted[3], n = 1, selected = -1;
        const double t0 = now_sec();
        if (cfg->mtp) {
            const int cap = count - done < 3 ? count - done : 3;
            n = ds4_session_eval_speculative_argmax(session, tokens[offset], cap, eos,
                                                     accepted, cap, err, errlen);
        } else {
            if (ds4_session_eval(session, tokens[offset], err, errlen)) return 1;
            accepted[0] = tokens[offset];
        }
        if (cfg->include_selection) selected = ds4_session_argmax_excluding(session, eos);
        const double elapsed = now_sec() - t0;
        if (n < 1 || n > count - done || n > 3) {
            if (!err[0]) snprintf(err, errlen, "invalid speculative count %d", n);
            return 1;
        }
        if (memcmp(accepted, tokens + offset, (size_t)n * sizeof(int))) {
            snprintf(err, errlen, "accepted token mismatch at frontier %d", offset);
            return 1;
        }
        if (cfg->include_selection && selected != tokens[offset + n]) {
            snprintf(err, errlen, "selected token mismatch at frontier %d", offset + n);
            return 1;
        }
        if (samples) samples[cycles] = (cycle_sample){elapsed, n};
        cycles++;
        done += n;
    }
    if (n_samples) *n_samples = cycles;
    if (verify && compare_logits(session, rows + (size_t)(start + count) * vocab,
                                 observed, vocab, start + count)) return 1;
    return 0;
}

static double student_t95(int df) {
    static const double t[] = {0, 12.706205, 4.302653, 3.182446, 2.776445,
        2.570582, 2.446912, 2.364624, 2.306004, 2.262157, 2.228139,
        2.200985, 2.178813, 2.160369, 2.144787, 2.131450, 2.119906,
        2.109816, 2.100922, 2.093024, 2.085964, 2.079614, 2.073874,
        2.068658, 2.063899, 2.059539, 2.055529, 2.051831, 2.048407,
        2.045230, 2.042273};
    return t[df < 30 ? df : 30]; /* Conservative beyond 30 degrees of freedom. */
}

static int validate_environment(const bench_config *cfg) {
    static const char *const diagnostics[] = {
        "DS4_QWEN4_TIMING", "DS4_QWEN4_BATCH_TIMING", "DS4_QWEN4_MOE_PROFILE",
        "DS4_METAL_GRAPH_DUMP_PREFIX", "DS4_METAL_DECODE_STAGE_PROFILE",
        "DS4_METAL_LAYER_STAGE_PROFILE", "DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE",
        "DS4_METAL_Q8_PREFILL_PROFILE", "DS4_TOKEN_TIMING", "DS4_MTP_TIMING",
    };
    for (size_t i = 0; i < sizeof(diagnostics) / sizeof(diagnostics[0]); i++) {
        if (getenv(diagnostics[i]) || (cfg->candidate_env && !strcmp(cfg->candidate_env, diagnostics[i]))) {
            fprintf(stderr, "metal-decode-schedule-bench: remove diagnostic %s before timing\n", diagnostics[i]);
            return 1;
        }
    }
    if (cfg->mtp && (getenv("DS4_QWEN4_MTP_DRAFT_ROWS") ||
                     getenv("DS4_QWEN4_MTP_DRAFT_VOCAB") ||
                     getenv("DS4_QWEN4_SPEC_FORCE_ACCEPT"))) {
        fprintf(stderr, "metal-decode-schedule-bench: MTP requires unmodified full-vocabulary drafts and acceptance\n");
        return 1;
    }
    if (cfg->mtp) {
        char value[16];
        snprintf(value, sizeof(value), "%d", cfg->mtp_depth);
        if (setenv("DS4_QWEN4_MTP_DEPTH", value, 1)) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    const bench_config cfg = parse_options(argc, argv);
    if (cfg.output_path) {
        if (!freopen(cfg.output_path, "w", stdout) || dup2(fileno(stdout), STDERR_FILENO) < 0) {
            fprintf(stderr, "metal-decode-schedule-bench: cannot open output\n");
            return 1;
        }
        setvbuf(stdout, NULL, _IOLBF, 0);
        setvbuf(stderr, NULL, _IOLBF, 0);
    }
    if (validate_environment(&cfg) || normalize_output_experiment_env() || select_variant(&cfg, 0)) return 1;
    char *text = read_text(cfg.prompt_path);
    if (!text) return 1;
    void *outer_pool = ds4_bench_pool_create(), *run_pool = NULL;
    ds4_engine_options opt = {
        .model_path = cfg.model_path, .backend = DS4_BACKEND_METAL,
        .context_size = cfg.ctx, .prefill_chunk = cfg.prefill_chunk,
        .power_percent = 100, .warm_weights = true,
        .glm_mtp = cfg.mtp,
    };
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    ds4_session_snapshot base = {0}, reference_state = {0}, observed_state = {0};
    int *trajectory = NULL;
    float *rows = NULL, *observed = NULL;
    cycle_sample *samples = NULL;
    char err[512] = {0};
    int rc = 1;
    const int total = cfg.measured;
    const int reference_steps = cfg.warmup > total ? cfg.warmup : total;
    const char *mode = cfg.mtp ? "mtp_greedy" : "ordinary";
    if (ds4_engine_open(&engine, &opt)) goto done;
    if (cfg.mtp && ds4_engine_mtp_draft_tokens(engine) <= 1) {
        snprintf(err, sizeof(err), "model has no enabled MTP support");
        goto done;
    }
    ds4_tokenize_text(engine, text, &prompt);
    free(text);
    text = NULL;
    if (prompt.len < cfg.prefix_tokens) {
        snprintf(err, sizeof(err), "prompt has %d tokens; need %d", prompt.len, cfg.prefix_tokens);
        goto done;
    }
    const int vocab = ds4_engine_vocab_size(engine), layers = ds4_engine_layer_count(engine);
    const int eos = ds4_token_eos(engine);
    if (cfg.control.first > layers || cfg.control.second > layers ||
        cfg.candidate.first > layers || cfg.candidate.second > layers) {
        snprintf(err, sizeof(err), "split layers must be between 0 and %d", layers);
        goto done;
    }
    if (vocab < 1 || (size_t)(reference_steps + 1) > SIZE_MAX / sizeof(float) / (size_t)vocab) goto done;
    trajectory = malloc((size_t)(reference_steps + 1) * sizeof(int));
    rows = malloc((size_t)(reference_steps + 1) * vocab * sizeof(float));
    observed = malloc((size_t)vocab * sizeof(float));
    samples = malloc((size_t)cfg.measured * sizeof(*samples));
    if (!trajectory || !rows || !observed || !samples) {
        snprintf(err, sizeof(err), "allocation failed");
        goto done;
    }
    ds4_tokens prefix = {.v = prompt.v, .len = cfg.prefix_tokens, .cap = cfg.prefix_tokens};
    if (ds4_session_create(&session, engine, cfg.ctx) ||
        ds4_session_sync(session, &prefix, err, sizeof(err)) ||
        ds4_session_save_snapshot(session, &base, err, sizeof(err))) goto done;
    printf("configuration mode=%s model=%s prompt=%s prefix=%d ctx=%d warmup=%d measured=%d repeats=%d prefill_chunk=%u control=%d/%d candidate=%d/%d candidate_env=%s control_value=%s candidate_value=%s include_selection=%s mtp_depth=%d single_run=%s\n",
           mode, cfg.model_path, cfg.prompt_path, cfg.prefix_tokens, cfg.ctx,
           cfg.warmup, cfg.measured, cfg.repeats, cfg.prefill_chunk,
           cfg.control.first, cfg.control.second, cfg.candidate.first, cfg.candidate.second,
           cfg.candidate_env ? cfg.candidate_env : "(none)",
           cfg.control_value ? cfg.control_value : "(unset)",
           cfg.candidate_value ? cfg.candidate_value : "(unset)",
           cfg.include_selection ? "yes" : "no", cfg.mtp_depth, cfg.single_run ? "yes" : "no");
    /* Establish every target row through ordinary decoding, independently of
     * speculative cycle boundaries. The entire reference pass is untimed. */
    run_pool = ds4_bench_pool_create();
    for (int i = 0; i <= reference_steps; i++) {
        if (ds4_session_copy_logits(session, rows + (size_t)i * vocab, vocab) != vocab ||
            finite_logits(rows + (size_t)i * vocab, vocab, i, "reference")) goto done;
        trajectory[i] = ds4_session_argmax_excluding(session, eos);
        if (trajectory[i] < 0) goto done;
        if (!cfg.mtp && i == total && ds4_session_save_snapshot(session, &reference_state, err, sizeof(err))) goto done;
        if (i < reference_steps && ds4_session_eval(session, trajectory[i], err, sizeof(err))) goto done;
    }
    ds4_bench_pool_drain(run_pool);
    run_pool = NULL;
    /* Fixed MTP depth is required: its adaptive policy counters are not part
     * of the public checkpoint. Restoring a snapshot invalidates pending
     * drafts; both validation and timing start at that same clean frontier
     * after an independent warmup and restore. */
    const int validation_variants = cfg.single_run ? (cfg.mtp ? 1 : 0) : VARIANT_COUNT;
    for (int variant = 0; variant < validation_variants; variant++) {
        run_pool = ds4_bench_pool_create();
        if (select_variant(&cfg, variant) ||
            ds4_session_load_snapshot(session, &base, err, sizeof(err)) ||
            run_segment(session, &cfg, trajectory, rows, observed, vocab, eos,
                        0, cfg.warmup, true, NULL, NULL, err, sizeof(err)) ||
            ds4_session_load_snapshot(session, &base, err, sizeof(err)) ||
            run_segment(session, &cfg, trajectory, rows, observed, vocab, eos,
                        0, cfg.measured, true, NULL, NULL, err, sizeof(err))) goto done;
        if (cfg.mtp && variant == 0) {
            if (ds4_session_save_snapshot(session, &reference_state, err, sizeof(err))) goto done;
        } else if (compare_snapshot(session, &reference_state, &observed_state, err, sizeof(err))) goto done;
        printf("validation mode=%s variant=%s all_public_frontier_logits_exact=yes accepted_tokens_exact=yes checkpoint_exact=%s checkpoint_reference=%s reference_rows=%d vocab=%d\n",
               mode, variant ? "candidate" : "control",
               cfg.mtp && variant == 0 ? "reference_capture" : "yes",
               cfg.mtp ? "control_mtp" : "ordinary", reference_steps + 1, vocab);
        fflush(stdout);
        ds4_bench_pool_drain(run_pool);
        run_pool = NULL;
    }
    double seconds[VARIANT_COUNT] = {0}, paired_sum = 0, paired_sq = 0;
    size_t runs = 0;
    for (int block = 0; block < cfg.repeats; block++) {
        const int abba[] = {0, 1, 1, 0}, baab[] = {1, 0, 0, 1};
        const int *order = block & 1 ? baab : abba;
        double block_seconds[VARIANT_COUNT] = {0};
        for (int slot = 0; slot < (cfg.single_run ? 1 : 4); slot++) {
            const int variant = order[slot];
            run_pool = ds4_bench_pool_create();
            if (select_variant(&cfg, variant) ||
                ds4_session_load_snapshot(session, &base, err, sizeof(err)) ||
                run_segment(session, &cfg, trajectory, rows, observed, vocab, eos,
                            0, cfg.warmup, false, NULL, NULL, err, sizeof(err))) goto done;
            /* Warm kernels without shifting the advertised context length. */
            if (ds4_session_load_snapshot(session, &base, err, sizeof(err))) goto done;
            struct rusage before = {0}, after = {0};
            ds4_bench_memory memory_before = {0}, memory_after = {0}, memory_drained = {0};
            ds4_bench_memory_sample(getpid(), &memory_before);
            getrusage(RUSAGE_SELF, &before);
            int n_samples = 0;
            const double t0 = now_sec();
            if (run_segment(session, &cfg, trajectory, rows, observed, vocab, eos,
                            0, cfg.measured, false, samples, &n_samples, err, sizeof(err))) goto done;
            const double elapsed = now_sec() - t0;
            getrusage(RUSAGE_SELF, &after);
            ds4_bench_memory_sample(getpid(), &memory_after);
            if (ds4_session_pos(session) != cfg.prefix_tokens + total ||
                compare_logits(session, rows + (size_t)total * vocab, observed, vocab, total) ||
                compare_snapshot(session, &reference_state, &observed_state, err, sizeof(err))) goto done;
            if (cfg.single_run) {
                char logits_hash[65], checkpoint_hash[65];
                ds4_bench_sha256(observed, (size_t)vocab * sizeof(float), logits_hash);
                ds4_bench_sha256(observed_state.ptr, (size_t)observed_state.len, checkpoint_hash);
                printf("hashes run=%zu position=%d logits_sha256=%s checkpoint_sha256=%s checkpoint_bytes=%llu\n",
                       runs + 1, ds4_session_pos(session), logits_hash, checkpoint_hash,
                       (unsigned long long)observed_state.len);
            }
            ds4_bench_pool_drain(run_pool);
            run_pool = NULL;
            ds4_bench_memory_sample(getpid(), &memory_drained);
            runs++;
            seconds[variant] += elapsed;
            block_seconds[variant] += elapsed;
            printf("run=%zu mode=%s repeat=%d pattern=%s slot=%d variant=%s tokens=%d cycles=%d seconds=%.9f tokens_per_second=%.6f first_cycle_ms=%.6f mean_token_ms=%.6f exact=yes\n",
                   runs, mode, block + 1, cfg.single_run ? "single" : block & 1 ? "BAAB" : "ABBA", slot + 1,
                   variant ? "candidate" : "control", cfg.measured, n_samples, elapsed,
                   cfg.measured / elapsed, 1000 * samples[0].seconds, 1000 * elapsed / cfg.measured);
            for (int i = 0, offset = 0; i < n_samples; i++) {
                printf("cycle run=%zu cycle=%d offset=%d tokens=%d seconds=%.9f\n",
                       runs, i + 1, offset, samples[i].tokens, samples[i].seconds);
                offset += samples[i].tokens;
            }
            printf("resources run=%zu maxrss=%ld minor_faults=%ld major_faults=%ld input_blocks=%ld output_blocks=%ld swaps=%ld user_seconds=%.6f system_seconds=%.6f\n",
                   runs, after.ru_maxrss, after.ru_minflt - before.ru_minflt,
                   after.ru_majflt - before.ru_majflt, after.ru_inblock - before.ru_inblock,
                   after.ru_oublock - before.ru_oublock, after.ru_nswap - before.ru_nswap,
                   after.ru_utime.tv_sec - before.ru_utime.tv_sec + (after.ru_utime.tv_usec - before.ru_utime.tv_usec) / 1e6,
                   after.ru_stime.tv_sec - before.ru_stime.tv_sec + (after.ru_stime.tv_usec - before.ru_stime.tv_usec) / 1e6);
            printf("memory run=%zu process_valid=%d swap_valid=%d vm_valid=%d resident_bytes=%llu footprint_bytes=%llu disk_read_bytes=%llu disk_write_bytes=%llu global_swap_used_bytes=%llu global_swapins_pages=%llu global_swapouts_pages=%llu\n",
                   runs, memory_drained.process_valid && memory_before.process_valid && memory_after.process_valid,
                   memory_after.swap_usage_valid, memory_before.vm_statistics_valid && memory_after.vm_statistics_valid,
                   (unsigned long long)memory_drained.resident_bytes,
                   (unsigned long long)memory_drained.physical_footprint_bytes,
                   (unsigned long long)(memory_after.disk_read_bytes - memory_before.disk_read_bytes),
                   (unsigned long long)(memory_after.disk_write_bytes - memory_before.disk_write_bytes),
                   (unsigned long long)memory_after.swap_used_bytes,
                   (unsigned long long)(memory_after.swapins_pages - memory_before.swapins_pages),
                   (unsigned long long)(memory_after.swapouts_pages - memory_before.swapouts_pages));
            fflush(stdout);
        }
        if (!cfg.single_run) {
            const double delta = log(block_seconds[0] / block_seconds[1]);
            paired_sum += delta;
            paired_sq += delta * delta;
            printf("paired mode=%s repeat=%d gain_percent=%.6f\n", mode, block + 1, 100 * expm1(delta));
        }
    }
    if (!cfg.single_run) {
    const double n = cfg.repeats, mean = paired_sum / n;
    const double variance = fmax(0.0, (paired_sq - paired_sum * mean) / (n - 1));
    const double radius = student_t95(cfg.repeats - 1) * sqrt(variance / n);
    printf("paired_ci mode=%s blocks=%d gain_percent=%.6f low95_percent=%.6f high95_percent=%.6f all_logits_exact=yes all_checkpoints_exact=yes\n",
           mode, cfg.repeats, 100 * expm1(mean), 100 * expm1(mean - radius), 100 * expm1(mean + radius));
    }
    for (int variant = 0; variant < (cfg.single_run ? 1 : VARIANT_COUNT); variant++) {
        const size_t n_runs = cfg.single_run ? 1u : (size_t)cfg.repeats * 2u;
        const size_t n_tokens = (size_t)cfg.measured * n_runs;
        printf("aggregate mode=%s variant=%s runs=%zu tokens=%zu seconds=%.9f tokens_per_second=%.6f\n",
               mode, variant ? "candidate" : "control", n_runs,
               n_tokens, seconds[variant], n_tokens / seconds[variant]);
    }
    rc = 0;
done:
    if (rc) fprintf(stderr, "metal-decode-schedule-bench: failed: %s\n", err[0] ? err : "see diagnostic above");
    free(text);
    free(trajectory);
    free(rows);
    free(observed);
    free(samples);
    ds4_session_snapshot_free(&base);
    ds4_session_snapshot_free(&reference_state);
    ds4_session_snapshot_free(&observed_state);
    if (session) ds4_session_free(session);
    ds4_tokens_free(&prompt);
    if (engine) ds4_engine_close(engine);
    ds4_bench_pool_drain(run_pool);
    ds4_bench_pool_drain(outer_pool);
    return rc;
}
