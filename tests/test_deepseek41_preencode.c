/* DeepSeek V4.1 decode with the next token's first command buffer pre-encoded, against the same decode without it,
 * bit for bit.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/V4.1.gguf DS4_TEST_DSPARK=/path/to/V4.1-DSpark.gguf \
 *       make test-deepseek41-preencode
 *
 * One --dspark-strict engine (verify rows on the decode arithmetic, so a script is repeatable) runs the same greedy
 * script twice, with DS4_GPU_V41_PREENCODE off and then on: plain steps,
 * DSpark cycles (accepted and rejected drafts), a snapshot, steps, the snapshot restored into the session, steps.
 * The logits after every operation must match, and the second run must have committed pre-encoded slots. */
#include "ds4.h"
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 4096
#define MAX_RECORDS 128

static const char *prompt_text =
    "The history of the printing press begins in the fifteenth century, when";

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

typedef struct {
    int n, tokens[MAX_RECORDS], cycles, partial;
    float *logits;
} script_log;

static void record(ds4_session *s, script_log *log, int V, int token) {
    if (log->n == MAX_RECORDS) fail("too many records");
    if (ds4_session_copy_logits(s, log->logits + (size_t)log->n * V, V) != V) fail("copy logits");
    log->tokens[log->n++] = token;
}

static void steps(ds4_session *s, script_log *log, int V, int n) {
    char err[256] = {0};
    for (int i = 0; i < n; i++) {
        const int token = ds4_session_argmax(s);
        if (ds4_session_eval(s, token, err, sizeof(err)) != 0) fail(err);
        record(s, log, V, token);
    }
}

/* 16 steps, DSpark cycles over 32 tokens, snapshot, 8 steps, restore, 8 steps */
static void script(ds4_engine *e, const ds4_tokens *prompt, script_log *log, int V) {
    char err[256] = {0};
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, TEST_CTX) != 0 || ds4_session_sync(s, prompt, err, sizeof(err)) != 0)
        fail(err[0] ? err : "session");
    record(s, log, V, -1);
    steps(s, log, V, 16);
    for (int produced = 0; produced < 32;) {
        int accepted[17];
        const int n = ds4_session_eval_speculative_argmax(s, ds4_session_argmax(s), 64 - produced, -1, accepted, 17,
                                                          err, sizeof(err));
        if (n <= 0) fail(err[0] ? err : "speculative cycle");
        record(s, log, V, accepted[n - 1]);
        produced += n;
        log->cycles++;
        log->partial += n >= 2 && n < 17;
    }
    ds4_session_snapshot snap = {0};
    if (ds4_session_save_snapshot(s, &snap, err, sizeof(err)) != 0) fail(err);
    steps(s, log, V, 8);
    if (ds4_session_load_snapshot(s, &snap, err, sizeof(err)) != 0) fail(err);
    record(s, log, V, -2);
    steps(s, log, V, 8);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(s);
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL"), *support = getenv("DS4_TEST_DSPARK");
    if (!model || !model[0] || !support || !support[0]) {
        fprintf(stderr, "SKIP: DS4_TEST_MODEL or DS4_TEST_DSPARK is not set\n");
        return 0;
    }
    setenv("DS4_DSPARK_DISABLE_ADAPTIVE_DRAFTING", "1", 1);   /* every cycle drafts */
    ds4_engine_options opt = {
        .model_path = model,
        .mtp_path = support,
        .dspark = true,
        .dspark_strict = true,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    const int V = ds4_engine_vocab_size(e);
    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, prompt_text, &prompt);
    script_log log[2] = {{0}, {0}};
    uint64_t slots[2];
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (int on = 0; on < 2; on++) {
        log[on].logits = malloc((size_t)MAX_RECORDS * V * sizeof(float));
        if (!log[on].logits) fail("alloc");
        ds4_gpu_test_set_v41_off(on ? 0 : DS4_GPU_V41_PREENCODE);
        const uint64_t before = ds4_gpu_test_kernel_runs("pre-encoded slot");
        script(e, &prompt, &log[on], V);
        slots[on] = ds4_gpu_test_kernel_runs("pre-encoded slot") - before;
    }
    ds4_gpu_test_set_v41_off(0);
    int bad = log[0].n != log[1].n;
    if (bad) fprintf(stderr, "records: %d vs %d\n", log[0].n, log[1].n);
    for (int i = 0; !bad && i < log[0].n; i++) {
        if (log[0].tokens[i] != log[1].tokens[i] ||
            memcmp(log[0].logits + (size_t)i * V, log[1].logits + (size_t)i * V, (size_t)V * sizeof(float))) {
            fprintf(stderr, "record %d (token %d vs %d): logits differ\n", i, log[0].tokens[i], log[1].tokens[i]);
            bad = 1;
        }
    }
    printf("pre-encode off/on: %d records, %d DSpark cycles (%d partly accepted), slots committed %llu / %llu\n",
           log[0].n, log[0].cycles, log[0].partial, (unsigned long long)slots[0], (unsigned long long)slots[1]);
    if (bad) fail("pre-encoded decode differs");
    if (slots[0] != 0 || slots[1] == 0) fail("pre-encode did not switch");
    if (log[0].partial == 0) fail("no DSpark cycle rejected part of its draft");
    for (int on = 0; on < 2; on++) free(log[on].logits);
    ds4_tokens_free(&prompt);
    ds4_engine_close(e);
    printf("OK\n");
    return 0;
}
