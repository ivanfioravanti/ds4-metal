/* A benchmark must reject nonfinite rows even under -ffast-math, and must
 * distinguish diagnostic drift from an exact comparison. No model needed. */
#include "../ds4.h"
#include <time.h>
static int mock_create(ds4_session **out, ds4_engine *engine, int ctx);
static void mock_free(ds4_session *session);
static int mock_sync(ds4_session *session, const ds4_tokens *tokens, char *err, size_t errlen);
static int mock_save(ds4_session *session, ds4_session_snapshot *snapshot, char *err, size_t errlen);
static int mock_load(ds4_session *session, const ds4_session_snapshot *snapshot, char *err, size_t errlen);
static void mock_snapshot_free(ds4_session_snapshot *snapshot);
static int mock_argmax(ds4_session *session, int eos);
static int mock_eos(ds4_engine *engine);
static int mock_clock(clockid_t clock, struct timespec *out);
#define ds4_session_create mock_create
#define ds4_session_free mock_free
#define ds4_session_sync mock_sync
#define ds4_session_save_snapshot mock_save
#define ds4_session_load_snapshot mock_load
#define ds4_session_snapshot_free mock_snapshot_free
#define ds4_session_argmax_excluding mock_argmax
#define ds4_token_eos mock_eos
#define clock_gettime mock_clock
#define main prefill_benchmark_main
#include "../speed-bench/metal_prefill_variant_bench.c"
#undef main
#include <assert.h>
#include <sys/wait.h>

static struct {
    int active, creates, syncs, saved, loaded, fail_sync;
    int selections, fail_selection, selected_position, clocks, event_count;
    char events[16];
    int from[2], to[2], variant[2];
} mock;

static void event(char value) {
    assert(mock.event_count + 1 < (int)sizeof(mock.events));
    mock.events[mock.event_count++] = value;
    mock.events[mock.event_count] = '\0';
}

static int mock_clock(clockid_t clock, struct timespec *out) {
    assert(clock == CLOCK_MONOTONIC && mock.clocks < 3);
    const long ns[] = {0, 500000000, 750000000};
    *out = (struct timespec){1, ns[mock.clocks++]};
    event('C');
    return 0;
}

static int mock_eos(ds4_engine *engine) { (void)engine; return 3; }

static int mock_argmax(ds4_session *session, int eos) {
    assert(mock.active == 1 && eos == 3);
    event('A');
    mock.selections++;
    mock.selected_position = *(int *)session;
    return mock.fail_selection ? -1 : 2;
}

static int mock_create(ds4_session **out, ds4_engine *engine, int ctx) {
    (void)engine;
    assert(ctx > 0 && mock.active == 0);
    *out = (ds4_session *)calloc(1, sizeof(int));
    assert(*out);
    mock.active++;
    mock.creates++;
    return 0;
}

static void mock_free(ds4_session *session) {
    if (session) mock.active--;
    free(session);
}

static int mock_sync(ds4_session *session, const ds4_tokens *tokens, char *err, size_t errlen) {
    assert(mock.active == 1 && mock.syncs < 2);
    event('S');
    const int i = mock.syncs++;
    mock.from[i] = *(int *)session;
    mock.to[i] = tokens->len;
    mock.variant[i] = atoi(getenv("DS4_BENCH_WARMUP_TEST"));
    if (mock.fail_sync == mock.syncs) {
        snprintf(err, errlen, "mock read failure");
        return 1;
    }
    *(int *)session = tokens->len;
    return 0;
}

static int mock_save(ds4_session *session, ds4_session_snapshot *snapshot, char *err, size_t errlen) {
    (void)err; (void)errlen;
    snapshot->len = sizeof(int);
    snapshot->ptr = malloc(sizeof(int));
    assert(snapshot->ptr);
    memcpy(snapshot->ptr, session, sizeof(int));
    mock.saved++;
    return 0;
}

static int mock_load(ds4_session *session, const ds4_session_snapshot *snapshot, char *err, size_t errlen) {
    (void)err; (void)errlen;
    assert(snapshot->len == sizeof(int));
    memcpy(session, snapshot->ptr, sizeof(int));
    mock.loaded++;
    return 0;
}

static void mock_snapshot_free(ds4_session_snapshot *snapshot) {
    free(snapshot->ptr);
    memset(snapshot, 0, sizeof(*snapshot));
}

static void test_warmup(void) {
    int token = 1;
    ds4_tokens tokens = {.v = &token, .len = 8224, .cap = 8224};
    bench_config cfg = {.candidate_env = "DS4_BENCH_WARMUP_TEST", .control_value = "0",
        .candidate_value = "1", .prefix_tokens = 8224, .warmup_tokens = 512,
        .prefill_chunk = 8192, .ctx = 9000};
    char err[80] = {0};
    assert(warm_variant(NULL, &cfg, &tokens, 1, err, sizeof(err)) == 0);
    assert(mock.active == 0 && mock.creates == 1 && mock.syncs == 1);
    assert(mock.selections == 1 && mock.selected_position == 512 && !strcmp(mock.events, "SA"));
    assert(mock.from[0] == 0 && mock.to[0] == 512 && mock.variant[0] == 1);
    cfg.initial_tokens = 8192;
    for (int variant = 0; variant < 2; variant++) {
        memset(&mock, 0, sizeof(mock));
        assert(warm_variant(NULL, &cfg, &tokens, variant, err, sizeof(err)) == 0);
        assert(mock.active == 0 && mock.creates == 1 && mock.syncs == 2);
        assert(mock.selections == 1 && mock.selected_position == 8224 && !strcmp(mock.events, "SSA"));
        assert(mock.from[0] == 0 && mock.to[0] == 8192 && mock.variant[0] == 0);
        assert(mock.from[1] == 8192 && mock.to[1] == 8224 && mock.variant[1] == variant);
    }
    cfg.candidate_chunk = 1024;
    memset(&mock, 0, sizeof(mock));
    assert(warm_variant(NULL, &cfg, &tokens, 1, err, sizeof(err)) == 0);
    assert(mock.active == 0 && mock.creates == 2 && mock.saved == 1 && mock.loaded == 1);
    assert(mock.selections == 1 && mock.selected_position == 8224);
    assert(mock.from[1] == 8192 && mock.to[1] == 8224 && mock.variant[1] == 1);
    for (int fail = 1; fail <= 2; fail++) {
        memset(&mock, 0, sizeof(mock));
        mock.fail_sync = fail;
        assert(warm_variant(NULL, &cfg, &tokens, 1, err, sizeof(err)) != 0);
        assert(mock.active == 0 && mock.syncs == fail);
        assert(mock.selections == 0);
    }
    memset(&mock, 0, sizeof(mock));
    mock.fail_selection = 1;
    assert(warm_variant(NULL, &cfg, &tokens, 1, err, sizeof(err)) != 0);
    assert(mock.active == 0 && mock.selections == 1 && strstr(err, "selection returned no token"));
    unsetenv("DS4_BENCH_WARMUP_TEST");
    unsetenv("DS4_QWEN4_PREFILL_CHUNK");
}

static void test_first_token_clock(void) {
    setenv("DS4_BENCH_WARMUP_TEST", "0", 1);
    for (int failure = 0; failure <= 2; failure++) {
        memset(&mock, 0, sizeof(mock));
        mock.fail_sync = failure == 1;
        mock.fail_selection = failure == 2;
        ds4_session *session = NULL;
        assert(mock_create(&session, NULL, 10) == 0);
        int token = 1;
        ds4_tokens tokens = {.v = &token, .len = 8, .cap = 8};
        prefill_timing timing = {0};
        char err[80] = {0};
        const int rc = sync_and_select(session, &tokens, 3, &timing, err, sizeof(err));
        assert(rc == (failure ? 1 : 0));
        assert(!strcmp(mock.events, failure == 1 ? "CSC" : "CSCAC"));
        assert(mock.selections == (failure == 1 ? 0 : 1));
        assert(mock.saved == 0 && mock.loaded == 0 && mock.syncs == 1);
        assert(*(int *)session == (failure == 1 ? 0 : 8));
        if (!failure) {
            assert(timing.seconds == 0.5 && timing.selection_ms == 250.0 && timing.first_token_ms == 750.0);
            assert(timing.first_token_ms == timing.seconds * 1000.0 + timing.selection_ms);
            assert(timing.first_token_id == 2 && mock.selected_position == 8);
        } else assert(strstr(err, failure == 1 ? "read failure" : "selection returned no token"));
        mock_free(session);
        assert(mock.active == 0);
    }
    unsetenv("DS4_BENCH_WARMUP_TEST");
}

static void test_chunk_state_options(void) {
    const struct { const char *initial, *prefix, *control, *candidate; bool allowed; } cases[] = {
        {"8192", "8224", "8192", "512", true},
        {"8192", "8448", "8192", "512", true},
        {"8192", "8448", "8192", "4096", true},
        {"8192", "8704", "8192", "512", true},
        {"8192", "8705", "8192", "512", false},
        {"8192", "8448", "128", "512", false},
        {"8192", "8448", "1024", "128", false},
        {"0", "8192", "8192", "512", false},
        {"0", "8192", "512", "512", true},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char *args[] = {"bench", "--compare-state", "--initial-tokens", (char *)cases[i].initial,
            "--prefix-tokens", (char *)cases[i].prefix, "--prefill-chunk", (char *)cases[i].control,
            "--candidate-chunk", (char *)cases[i].candidate};
        if (cases[i].allowed) {
            bench_config cfg = parse_options(10, args);
            assert(cfg.compare_state && !cfg.interleave);
        } else {
            fflush(NULL);
            const pid_t pid = fork();
            assert(pid >= 0);
            if (pid == 0) {
                (void)parse_options(10, args);
                _exit(0);
            }
            int status = 0;
            assert(waitpid(pid, &status, 0) == pid);
            assert(WIFEXITED(status) && WEXITSTATUS(status) == 2);
        }
    }
}

int main(void) {
    test_warmup();
    test_first_token_clock();
    test_chunk_state_options();
    char *ordinary[] = {"bench", "--candidate-env", "DS4_BENCH_NOOP"};
    bench_config cfg = parse_options(3, ordinary);
    assert(cfg.repeats == 3 && !cfg.single_run);
    char *single[] = {"bench", "--single-run"};
    cfg = parse_options(2, single);
    assert(cfg.single_run && cfg.repeats == 1 && !cfg.interleave);
    float a[] = {1.0f, -2.0f}, b[] = {1.0f, -2.0f};
    assert(compare_logits(a, b, 2, 1) == 0 && g_last_exact);
    b[1] = -1.0f;
    assert(compare_logits(a, b, 2, 2) != 0 && !g_last_exact);
    g_tolerate_drift = true;
    assert(compare_logits(a, b, 2, 3) == 0 && !g_last_exact);
    b[1] = a[1];
    assert(compare_logits(a, b, 2, 4) == 0 && g_last_exact);
    const uint32_t nonfinite[] = {0x7fc00000u, 0x7f800000u, 0xff800000u};
    for (size_t i = 0; i < sizeof(nonfinite) / sizeof(*nonfinite); i++) {
        memcpy(a, &nonfinite[i], 4);
        memcpy(b, &nonfinite[i], 4);
        assert(compare_logits(a, b, 2, 5 + i) != 0 && !g_last_exact);
    }
    puts("prefill benchmark reporting tests passed");
    return 0;
}
