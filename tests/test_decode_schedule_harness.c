/* CPU-only harness contract test; no model or Metal runtime is linked.
 * cc -std=c99 -Wall -Wextra -I. tests/test_decode_schedule_harness.c -lm -o /tmp/test_decode_schedule_harness
 */
#define main decode_bench_main
#include "../speed-bench/metal_decode_schedule_bench.c"
#undef main
#include <assert.h>
#include <sys/wait.h>

struct ds4_engine { bool mtp; };
struct ds4_session { int pos; int active; };
static int live_sessions, maximum_sessions, evals, copies, saves;
static int hash_calls;
static int inject_drift;
void *ds4_bench_pool_create(void) { return NULL; }
void ds4_bench_pool_drain(void *pool) { (void)pool; }
void ds4_bench_memory_sample(pid_t pid, ds4_bench_memory *out) { (void)pid; memset(out, 0, sizeof(*out)); }
/* Hash correctness is checked against known SHA-256 vectors by the helper's
 * test; this mock verifies that the harness exports both final artifacts. */
void ds4_bench_sha256(const void *data, size_t bytes, char hex[65]) {
    assert(data && bytes);
    memset(hex, 'a', 64);
    hex[64] = '\0';
    hash_calls++;
}
static float mock_logit(int pos, int id) { return (float)(pos * 8 + id); }

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    *out = calloc(1, sizeof(**out));
    (*out)->mtp = opt->glm_mtp;
    return 0;
}
void ds4_engine_close(ds4_engine *e) { free(e); }
int ds4_engine_mtp_draft_tokens(ds4_engine *e) { return e->mtp ? 2 : 0; }
int ds4_engine_vocab_size(ds4_engine *e) { (void)e; return 4; }
int ds4_engine_layer_count(ds4_engine *e) { (void)e; return 48; }
int ds4_token_eos(ds4_engine *e) { (void)e; return 3; }
void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e; (void)text;
    out->len = out->cap = 16;
    out->v = calloc(16, sizeof(int));
}
void ds4_tokens_free(ds4_tokens *t) { free(t->v); memset(t, 0, sizeof(*t)); }
int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx) {
    (void)e; (void)ctx;
    *out = calloc(1, sizeof(**out));
    live_sessions++;
    if (maximum_sessions < live_sessions) maximum_sessions = live_sessions;
    return 0;
}
void ds4_session_free(ds4_session *s) { live_sessions--; free(s); }
int ds4_session_sync(ds4_session *s, const ds4_tokens *t, char *err, size_t len) {
    (void)err; (void)len; s->pos = t->len; return 0;
}
int ds4_session_eval(ds4_session *s, int token, char *err, size_t len) {
    (void)err; (void)len; assert(token == 2);
    if (inject_drift == 2 && getenv("DS4_BENCH_TEST") && !strcmp(getenv("DS4_BENCH_TEST"), "4")) s->active = 1;
    s->pos++; evals++; return 0;
}
int ds4_session_pos(ds4_session *s) { return s->pos; }
int ds4_session_argmax_excluding(ds4_session *s, int eos) {
    (void)s; assert(eos == 3); return 2;
}
int ds4_session_copy_logits(ds4_session *s, float *out, int cap) {
    assert(cap == 4); copies++;
    for (int i = 0; i < cap; i++) out[i] = mock_logit(s->pos, i);
    if (inject_drift == 1 && getenv("DS4_BENCH_TEST") && !strcmp(getenv("DS4_BENCH_TEST"), "4")) out[0] += 0.125f;
    if (inject_drift == 3) {
        const uint32_t bits = UINT32_C(0x7fc00000);
        memcpy(&out[0], &bits, sizeof(bits));
    }
    return cap;
}
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t len) {
    (void)err; (void)len; saves++;
    snap->ptr = realloc(snap->ptr, sizeof(*s));
    snap->len = snap->cap = sizeof(*s);
    memcpy(snap->ptr, s, sizeof(*s));
    return 0;
}
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t len) {
    (void)err; (void)len;
    assert(snap->len == sizeof(*s));
    memcpy(s, snap->ptr, sizeof(*s));
    return 0;
}
void ds4_session_snapshot_free(ds4_session_snapshot *s) { free(s->ptr); memset(s, 0, sizeof(*s)); }
int ds4_session_eval_speculative_argmax(ds4_session *s, int first, int maximum, int eos,
                                       int *accepted, int capacity, char *err, size_t len) {
    assert(first == 2 && eos == 3);
    int n = atoi(getenv("DS4_QWEN4_MTP_DEPTH"));
    if (n > maximum) n = maximum;
    if (n > capacity) n = capacity;
    for (int i = 0; i < n; i++) {
        accepted[i] = 2;
        if (ds4_session_eval(s, 2, err, len)) return -1;
    }
    return n;
}

static void test_parse_failure(const char *name, const char *value) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (!pid) {
        char *argv[] = {"bench", (char *)name, (char *)value};
        (void)parse_options(3, argv);
        _exit(0);
    }
    int status;
    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 2);
}

static void test_single_mtp_options(void) {
    char *depth[] = {"bench", "--mtp", "--single-run", "--candidate-env", "DS4_QWEN4_MTP_DEPTH", "--control-value", "3"};
    assert(parse_options(7, depth).single_run);
    const char *forbidden[] = {"DS4_QWEN4_MTP_DRAFT_VOCAB", "DS4_QWEN4_MTP_DRAFT_ROWS", "DS4_QWEN4_SPEC_FORCE_ACCEPT"};
    for (size_t i = 0; i < sizeof(forbidden) / sizeof(*forbidden); i++) {
        pid_t pid = fork();
        assert(pid >= 0);
        if (!pid) {
            char *argv[] = {"bench", "--mtp", "--single-run", "--candidate-env", (char *)forbidden[i], "--control-value", "1"};
            (void)parse_options(7, argv);
            _exit(0);
        }
        int status;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 2);
    }
}

static void test_benchmark(bool mtp, int drift, bool single) {
    char output[] = "/tmp/ds4-decode-harness-XXXXXX";
    int fd = mkstemp(output);
    assert(fd >= 0);
    close(fd);
    int out_fd = dup(STDOUT_FILENO), err_fd = dup(STDERR_FILENO);
    assert(out_fd >= 0 && err_fd >= 0);
    char *argv[27] = {"bench", "--prompt-file", "AGENT.md", "--prefix-tokens", "2", "--ctx", "256",
        "--warmup", "4", "--tokens", "128", "--repeats", "3", "--candidate-env", "DS4_BENCH_TEST",
        "--control-value", "2", "--candidate-value", "4", "--include-selection", "--output", output,
        "--mtp", "--mtp-depth", "3"};
    inject_drift = drift;
    copies = saves = evals = hash_calls = 0;
    int argc = mtp ? 25 : 22;
    if (single) argv[argc++] = "--single-run";
    int result = decode_bench_main(argc, argv);
    fflush(stdout); fflush(stderr);
    assert(dup2(out_fd, STDOUT_FILENO) >= 0 && dup2(err_fd, STDERR_FILENO) >= 0);
    close(out_fd); close(err_fd);
    assert(result == (drift ? 1 : 0));
    char *log = read_text(output);
    assert(log);
    if (drift) {
        assert(strstr(log, drift == 1 ? "raw logit mismatch" :
                          drift == 2 ? "raw checkpoint mismatch" : "nonfinite logit"));
        assert(!strstr(log, "paired_ci"));
    } else {
        if (single) {
            assert(strstr(log, "repeat=1 pattern=single slot=1 variant=control"));
            assert(!strstr(log, "paired") && !strstr(log, "run=2 mode="));
            assert(strstr(log, "logits_sha256=") && strstr(log, "checkpoint_sha256="));
            assert(strstr(log, "hashes run=1 position=130 "));
            assert(hash_calls == 2 && strstr(log, "runs=1 tokens=128"));
        } else {
            assert(strstr(log, "repeat=1 pattern=ABBA slot=1 variant=control"));
            assert(strstr(log, "repeat=2 pattern=BAAB slot=1 variant=candidate"));
            assert(strstr(log, "repeat=3 pattern=ABBA slot=4 variant=control"));
            assert(strstr(log, "paired_ci") && strstr(log, "all_checkpoints_exact=yes"));
            assert(strstr(log, "tokens=768"));
        }
        assert(strstr(log, mtp ? "mode=mtp_greedy" : "mode=ordinary"));
        assert(evals == 128 + (single ? (mtp ? 2 : 1) : 14) * 132);
        /* Only the reference/validation passes and final row copies read
         * logits; measured token cycles never do so. */
        assert(single && !mtp ? copies == 130 : copies < evals / 2);
        assert(saves == (single ? 3 : mtp ? 15 : 16));
    }
    assert(maximum_sessions == 1 && live_sessions == 0);
    free(log);
    assert(unlink(output) == 0);
    unsetenv("DS4_BENCH_TEST");
    unsetenv("DS4_QWEN4_MTP_DEPTH");
}

int main(void) {
    test_parse_failure("--repeats", "2");
    test_parse_failure("--candidate-env", "DS4_QWEN4_NO_FUSE");
    test_parse_failure("--mtp-depth", "4");
    test_single_mtp_options();
    test_benchmark(false, 0, false);
    test_benchmark(true, 0, false);
    test_benchmark(false, 1, false);
    test_benchmark(false, 2, false);
    test_benchmark(false, 3, false);
    test_benchmark(false, 0, true);
    test_benchmark(true, 0, true);
    puts("decode schedule harness: PASS (ordinary/MTP blocks, one session, exact drift rejection, no timed logit copies)");
    return 0;
}
