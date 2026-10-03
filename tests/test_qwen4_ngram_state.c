#define DS4_TEST_HOOKS
#include "../ds4.c"
#include <assert.h>

static void exact_snapshot(ds4_session *session, const ds4_session_snapshot *expected,
                            char *error, size_t errlen) {
    ds4_session_snapshot observed = {0};
    assert(ds4_session_save_snapshot(session, &observed, error, errlen) == 0);
    assert(observed.len == expected->len);
    if (memcmp(observed.ptr, expected->ptr, (size_t)observed.len)) {
        uint64_t first = 0;
        while (first < observed.len && observed.ptr[first] == expected->ptr[first]) first++;
        fprintf(stderr, "Qwen n-gram recovery checkpoint differs at byte %llu: %02x/%02x\n",
                (unsigned long long)first, observed.ptr[first], expected->ptr[first]);
        abort();
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) assert(isfinite(session->logits[i]));
    ds4_session_snapshot_free(&observed);
}

/* A read may fail after GPU work has modified recurrent state. Public eval
 * must refuse an invalid checkpoint, and restoring the saved frontier must
 * recover every live state byte before the same known-token continuation. */
static void check_direct_read_recovery(ds4_engine *engine, const ds4_tokens *prompt) {
    const char *name = "DS4_QWEN4_NGRAM_OVERLAP";
    const char *value = getenv(name);
    char *saved = value ? strdup(value) : NULL;
    assert(!value || saved);
    assert(setenv(name, "0", 1) == 0);
    char error[256] = {0};
    ds4_session_snapshot prefix = {0}, expected[2] = {{0}};
    ds4_session *control = NULL;
    int tokens[2];
    assert(ds4_session_create(&control, engine, 256) == 0);
    assert(ds4_session_sync(control, prompt, error, sizeof(error)) == 0);
    assert(ds4_session_save_snapshot(control, &prefix, error, sizeof(error)) == 0);
    for (unsigned step = 0; step < 2; step++) {
        tokens[step] = ds4_session_argmax(control);
        assert(tokens[step] >= 0);
        assert(ds4_session_eval(control, tokens[step], error, sizeof(error)) == 0);
        assert(ds4_session_save_snapshot(control, &expected[step], error, sizeof(error)) == 0);
    }
    ds4_session_free(control);
    for (int overlap = 0; overlap < 2; overlap++) {
        assert(setenv(name, overlap ? "1" : "0", 1) == 0);
        ds4_session *live = NULL;
        assert(ds4_session_create(&live, engine, 256) == 0);
        assert(ds4_session_load_snapshot(live, &prefix, error, sizeof(error)) == 0);
        const int fd = engine->model.ngram_fd;
        engine->model.ngram_fd = INT_MAX;
        const int failed = ds4_session_eval(live, tokens[0], error, sizeof(error));
        engine->model.ngram_fd = fd;
        assert(failed != 0 && !live->checkpoint_valid);
        assert(ds4_session_argmax(live) == -1);
        assert(live->checkpoint.len == prompt->len);
        assert(!memcmp(live->checkpoint.v, prompt->v, (size_t)prompt->len * sizeof(int)));
#ifdef __APPLE__
        if (overlap && engine->backend == DS4_BACKEND_METAL) {
            assert(live->qwen4_graph.pos == 0 && live->qwen4_graph.mtp_pos == 0);
            assert(!live->qwen4_graph.snap_valid && !live->qwen4_graph.snap0_valid && !live->qwen4_graph.snap2_valid);
        }
#endif
        const uint32_t failed_pos = live->qwen4_graph.pos;
        /* Restoring the FD alone must not authorize a stale-state decode. */
        assert(ds4_session_eval(live, tokens[0], error, sizeof(error)) != 0);
        assert(!live->checkpoint_valid && live->qwen4_graph.pos == failed_pos);
        assert(live->checkpoint.len == prompt->len);
        assert(!memcmp(live->checkpoint.v, prompt->v, (size_t)prompt->len * sizeof(int)));
        ds4_session_snapshot invalid = {0};
        assert(ds4_session_save_snapshot(live, &invalid, error, sizeof(error)) != 0);
        ds4_session_snapshot_free(&invalid);
        assert(ds4_session_load_snapshot(live, &prefix, error, sizeof(error)) == 0);
        exact_snapshot(live, &prefix, error, sizeof(error));
        for (unsigned step = 0; step < 2; step++) {
            assert(ds4_session_eval(live, tokens[step], error, sizeof(error)) == 0);
            assert(live->checkpoint.len == prompt->len + (int)step + 1);
            exact_snapshot(live, &expected[step], error, sizeof(error));
        }
        ds4_session_free(live);
        printf("Qwen n-gram direct recovery overlap=%d: invalid retry refused, restored state and 2 tokens exact\n", overlap);
    }
    ds4_session_snapshot_free(&prefix);
    for (unsigned i = 0; i < 2; i++) ds4_session_snapshot_free(&expected[i]);
    assert((saved ? setenv(name, saved, 1) : unsetenv(name)) == 0);
    free(saved);
}

#ifdef __APPLE__
typedef struct { ds4_session *session; int stop_at; bool stop; } prefetch_cancel;
static bool prefetch_cancelled(void *context) { return ((prefetch_cancel *)context)->stop; }
static void prefetch_progress(void *context, const char *event, int current, int total) {
    (void)total;
    prefetch_cancel *cancel = context;
    if (strcmp(event, "prefill_chunk")) return;
    assert(!cancel->session->qwen4_graph.prefetched_ngrams);
    assert(!cancel->session->qwen4_graph.prefetched_ngram_tokens);
    if (current >= cancel->stop_at) cancel->stop = true;
}

static void prefetch_sync(ds4_session *s, const ds4_tokens *corpus, int n, char *error, size_t errlen) {
    ds4_tokens prefix = *corpus;
    prefix.len = n;
    assert(ds4_session_sync(s, &prefix, error, errlen) == 0);
    assert(s->checkpoint_valid && s->checkpoint.len == n);
}

static void check_prefetch(ds4_engine *engine, const ds4_tokens *prompt) {
    if (engine->backend != DS4_BACKEND_METAL) return;
    const char *names[] = {"DS4_QWEN4_NGRAM_PREFETCH", "DS4_QWEN4_NGRAM_OVERLAP"};
    char *saved[2] = {0};
    for (unsigned i = 0; i < 2; i++) {
        const char *value = getenv(names[i]);
        saved[i] = value ? strdup(value) : NULL;
        assert(!value || saved[i]);
    }
    ds4_tokens corpus = {0};
    assert(prompt->len > 0);
    for (int i = 0; i < 100; i++) ds4_tokens_push(&corpus, prompt->v[i % prompt->len]);
    char error[256] = {0};
    for (int initial = 0; initial <= 33; initial += 33) {
        const int target = initial + 67, boundary = initial + 32;
        ds4_session_snapshot start = {0}, retained = {0}, expected = {0}, rebuilt = {0};
        ds4_session *control = NULL;
        for (unsigned i = 0; i < 2; i++) assert(setenv(names[i], "0", 1) == 0);
        assert(ds4_session_create(&control, engine, 256) == 0);
        assert(control->qwen4_graph.cap_tokens == 32);
        if (initial) {
            prefetch_sync(control, &corpus, initial, error, sizeof(error));
            assert(ds4_session_save_snapshot(control, &start, error, sizeof(error)) == 0);
        }
        prefetch_sync(control, &corpus, boundary, error, sizeof(error));
        assert(ds4_session_save_snapshot(control, &retained, error, sizeof(error)) == 0);
        prefetch_sync(control, &corpus, target, error, sizeof(error));
        assert(ds4_session_save_snapshot(control, &expected, error, sizeof(error)) == 0);
        /* A failure invalidates the prefix. Its sync retry replays the whole
         * prompt, which has different chunks from a 33-token continuation. */
        ds4_session_invalidate(control);
        prefetch_sync(control, &corpus, target, error, sizeof(error));
        assert(ds4_session_save_snapshot(control, &rebuilt, error, sizeof(error)) == 0);
        ds4_session_free(control);
        for (int recovery = 0; recovery < 3; recovery++) {
            assert(setenv(names[0], "1", 1) == 0 && setenv(names[1], "1", 1) == 0);
            ds4_session *live = NULL;
            assert(ds4_session_create(&live, engine, 256) == 0);
            if (initial) assert(ds4_session_load_snapshot(live, &start, error, sizeof(error)) == 0);
            prefetch_cancel cancel = {.session = live, .stop_at = boundary};
            if (recovery == 2) {
                ds4_session_set_cancel(live, prefetch_cancelled, &cancel);
                ds4_session_set_progress(live, prefetch_progress, &cancel);
            }
            /* First job succeeds. The queued next job fails while the first
             * GPU chunk executes. Cancellation must drain and ignore it. */
            qwen4_ngram_prefetch_test_fail_after = 1;
            ds4_tokens prefix = corpus;
            prefix.len = target;
            const int rc = ds4_session_sync(live, &prefix, error, sizeof(error));
            qwen4_ngram_prefetch_test_fail_after = -1;
            ds4_session_set_cancel(live, NULL, NULL);
            ds4_session_set_progress(live, NULL, NULL);
            assert(!live->qwen4_graph.prefetched_ngrams && !live->qwen4_graph.prefetched_ngram_tokens);
            assert(live->checkpoint.len == boundary && live->qwen4_graph.pos == (uint32_t)boundary);
            assert(!memcmp(live->checkpoint.v, corpus.v, (size_t)boundary * sizeof(int)));
            if (recovery == 2) {
                assert(rc == DS4_SESSION_SYNC_INTERRUPTED && live->checkpoint_valid);
                exact_snapshot(live, &retained, error, sizeof(error));
            } else {
                assert(rc != 0 && rc != DS4_SESSION_SYNC_INTERRUPTED && !live->checkpoint_valid);
                assert(ds4_session_eval(live, corpus.v[boundary], error, sizeof(error)) != 0);
                ds4_session_snapshot invalid = {0};
                assert(ds4_session_save_snapshot(live, &invalid, error, sizeof(error)) != 0);
                ds4_session_snapshot_free(&invalid);
                if (recovery == 1) {
                    assert(ds4_session_load_snapshot(live, &retained, error, sizeof(error)) == 0);
                    exact_snapshot(live, &retained, error, sizeof(error));
                }
            }
            prefetch_sync(live, &corpus, target, error, sizeof(error));
            exact_snapshot(live, recovery == 0 ? &rebuilt : &expected, error, sizeof(error));
            ds4_session_free(live);
            printf("Qwen prefetch initial=%d target=%d recovery=%s: queued error, final partial, checkpoint exact\n",
                   initial, target, recovery == 0 ? "sync" : recovery == 1 ? "restore" : "cancel");
        }
        ds4_session_snapshot_free(&start); ds4_session_snapshot_free(&retained);
        ds4_session_snapshot_free(&expected); ds4_session_snapshot_free(&rebuilt);
    }
    ds4_tokens_free(&corpus);
    for (unsigned i = 0; i < 2; i++) {
        assert((saved[i] ? setenv(names[i], saved[i], 1) : unsetenv(names[i])) == 0);
        free(saved[i]);
    }
}
#endif

static void check_batch(ds4_engine *engine, const ds4_tokens *prompt, bool speculative) {
    enum { N = 4 };
    ds4_session *sessions[N] = {0};
    ds4_decode_item items[N];
    ds4_tokens frontier[N] = {0};
    ds4_session_snapshot snapshot = {0};
    float *saved_logits = xmalloc(DS4_N_VOCAB * sizeof(float));
    int accepted[N][2], counts[N];
    char error[256] = {0};
    for (int i = 0; i < N; i++) {
        assert(ds4_session_create(&sessions[i], engine, 256) == 0);
        assert(ds4_session_sync(sessions[i], prompt, error, sizeof(error)) == 0);
    }
    for (int step = 0; step < 12; step++) {
        if (step == 8) {
            assert(ds4_session_load_snapshot(sessions[0], &snapshot, error, sizeof(error)) == 0);
            assert(!memcmp(sessions[0]->logits, saved_logits, DS4_N_VOCAB * sizeof(float)));
        }
        for (int i = 0; i < N; i++) {
            ds4_session *s = sessions[(i + step) % N];
            items[i] = (ds4_decode_item){s, ds4_session_argmax(s)};
            ds4_tokens_copy(&frontier[i], &s->checkpoint);
        }
        /* Interleave ordinary cycles with the predictor, as exact-sampling
         * requests and output limits do in the server. */
        if (speculative && step % 4 != 3) {
            if (step == 2) {
                /* Counting has not finished: force rejection of an EOS draft. */
                assert(items[0].session->glm_mtp_have);
                items[0].session->glm_mtp_draft = ds4_token_eos(engine);
            }
            assert(ds4_sessions_eval_batch_speculative_argmax(items, N, accepted,
                        counts, error, sizeof(error)) == 0);
            if (step == 2) assert(counts[0] == 1);
            for (int i = 0; i < N; i++) {
                ds4_session *s = items[i].session;
                assert(counts[i] >= 1 && counts[i] <= 2);
                assert(accepted[i][0] == items[i].token);
                assert(s->checkpoint.len == frontier[i].len + counts[i]);
                for (int j = 0; j < counts[i]; j++)
                    assert(s->checkpoint.v[frontier[i].len + j] == accepted[i][j]);
                if (engine->backend == DS4_BACKEND_METAL) {
                    assert(!s->qwen4_graph.snap0_valid && !s->qwen4_graph.snap2_valid);
                    if (counts[i] == 1) assert(!s->qwen4_graph.snap_valid);
                }
            }
        } else {
            assert(ds4_sessions_eval_batch(items, N, error, sizeof(error)) == 0);
            for (int i = 0; i < N; i++) {
                assert(items[i].session->checkpoint.len == frontier[i].len + 1);
                assert(!items[i].session->glm_mtp_have);
            }
        }
        for (int i = 0; i < N; i++) {
            ds4_session *s = items[i].session;
            assert(s->checkpoint_valid && s->qwen4_graph.pos == (uint32_t)s->checkpoint.len);
            for (uint32_t j = 0; j < DS4_N_VOCAB; j++) assert(isfinite(s->logits[j]));
        }
        if (step == 1) {
            assert(ds4_session_save_snapshot(sessions[0], &snapshot, error, sizeof(error)) == 0);
            memcpy(saved_logits, sessions[0]->logits, DS4_N_VOCAB * sizeof(float));
        }
        if (step == 6 && speculative) {
            for (int i = 0; i < N; i++) {
                ds4_session *s = items[i].session;
                if (counts[i] != 2) continue;
                int pos = s->checkpoint.len - 1;
                ds4_session_rewind(s, pos);
                assert(s->checkpoint_valid && s->qwen4_graph.pos == (uint32_t)pos);
                assert(!memcmp(s->logits, s->qwen4_verify_logits, DS4_N_VOCAB * sizeof(float)));
            }
        }
    }
    ds4_session_snapshot_free(&snapshot);
    free(saved_logits);
    for (int i = 0; i < N; i++) {
        items[i] = (ds4_decode_item){sessions[i], ds4_session_argmax(sessions[i])};
        ds4_tokens_copy(&frontier[i], &sessions[i]->checkpoint);
    }
    ds4_decode_item duplicate[2] = {items[0], items[0]};
    assert(ds4_sessions_eval_batch_speculative_argmax(duplicate, 2, accepted,
                counts, error, sizeof(error)) != 0);
    for (int i = 0; i < N; i++) assert(sessions[i]->checkpoint.len == frontier[i].len);

    const int fd = engine->model.ngram_fd;
    engine->model.ngram_fd = INT_MAX;
    int rc = speculative ? ds4_sessions_eval_batch_speculative_argmax(items, N, accepted,
                                counts, error, sizeof(error)) :
                           ds4_sessions_eval_batch(items, N, error, sizeof(error));
    engine->model.ngram_fd = fd;
    assert(rc != 0);
    for (int i = 0; i < N; i++) {
        assert(!sessions[i]->checkpoint_valid);
        assert(ds4_session_argmax(sessions[i]) == -1);
        assert(ds4_session_sync(sessions[i], &frontier[i], error, sizeof(error)) == 0);
        ds4_session *control = NULL;
        assert(ds4_session_create(&control, engine, 256) == 0);
        assert(ds4_session_sync(control, &frontier[i], error, sizeof(error)) == 0);
        assert(!memcmp(sessions[i]->logits, control->logits, DS4_N_VOCAB * sizeof(float)));
        ds4_session_free(control);
        items[i].token = ds4_session_argmax(sessions[i]);
        /* Test the logical limit independently of rounded GPU allocation. */
        sessions[i]->ctx_size = sessions[i]->checkpoint.len + 1;
    }
    assert(ds4_sessions_eval_batch_speculative_argmax(items, N, accepted, counts,
                error, sizeof(error)) == 0);
    for (int i = 0; i < N; i++) {
        assert(counts[i] == 1);
        assert(sessions[i]->checkpoint.len == sessions[i]->ctx_size);
        items[i].token = ds4_session_argmax(sessions[i]);
    }
    assert(ds4_sessions_eval_batch_speculative_argmax(items, N, accepted, counts,
                error, sizeof(error)) != 0);
    for (int i = 0; i < N; i++) {
        assert(sessions[i]->checkpoint_valid);
        ds4_tokens_free(&frontier[i]);
        ds4_session_free(sessions[i]);
    }
    printf("Qwen batch %s: mixed cycles, rejection, disk failure/recovery, context limit OK\n",
           speculative ? "MTP" : "ordinary");
}

static void check_arena_resize(ds4_engine *engine, const ds4_tokens *prompt) {
    ds4_session *small = NULL, *large = NULL;
    char error[256] = {0};
    assert(ds4_session_create(&small, engine, 128) == 0);
    assert(ds4_session_sync(small, prompt, error, sizeof(error)) == 0);
    assert(ds4_session_create(&large, engine, 512) == 0);
    assert(ds4_session_sync(large, prompt, error, sizeof(error)) == 0);
    if (engine->backend == DS4_BACKEND_METAL) {
        assert(!small->qwen4_graph.owns_scratch);
        assert(large->qwen4_graph.owns_scratch);
    }
    ds4_session_free(large);
    ds4_session_free(small);
    assert(ds4_session_create(&large, engine, 1024) == 0);
    assert(ds4_session_sync(large, prompt, error, sizeof(error)) == 0);
    if (engine->backend == DS4_BACKEND_METAL) assert(!large->qwen4_graph.owns_scratch);
    ds4_session_free(large);
    assert(engine->qwen4_arena_users == 0);
    puts("Qwen shared arena: live growth fallback and idle replacement OK");
}

/* A failed disk lookup must invalidate the recurrent frontier, including
 * speculative snapshots. Retrying must rebuild from the retained tokens. */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s QWEN_GGUF\n", argv[0]);
        return 2;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {.model_path = argv[1],
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .glm_mtp = true, .prefill_chunk = 32,
        .share_session_prefill_workspace = true, .placement_session_count_hint = 4};
    assert(ds4_engine_open(&engine, &opt) == 0);
    assert(ds4_engine_is_qwen4(engine) && engine->model.ngram_tensor);
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, "Count from one to ten.", DS4_THINK_NONE, &prompt);
    char error[256] = {0};
    for (int mode = 0; mode < 4; mode++) {
        ds4_session *live = NULL, *control = NULL;
        ds4_tokens frontier = {0};
        ds4_session_snapshot snapshot = {0};
        assert(ds4_session_create(&live, engine, 256) == 0);
        assert(ds4_session_create(&control, engine, 256) == 0);
        assert(ds4_session_sync(live, &prompt, error, sizeof(error)) == 0);
        int accepted[3];
        if (mode == 2) {
            assert(ds4_session_eval_speculative_argmax(live, ds4_session_argmax(live),
                3, -1, accepted, 3, error, sizeof(error)) == 1);
            assert(live->glm_mtp_have);
        }
        const ds4_tokens *current = ds4_session_tokens(live);
        for (int i = 0; i < current->len; i++) ds4_tokens_push(&frontier, current->v[i]);
        int token = ds4_session_argmax(live);
        const char *depth_env = getenv("DS4_QWEN4_MTP_DEPTH");
        const bool require_deep_failure = mode == 2 && depth_env && !strcmp(depth_env, "3");
        if (require_deep_failure) {
            /* Mirror the actual deep-verifier admission checks before the
             * injected I/O failure; a plain/two-row fallback is not coverage. */
            ds4_qwen4_gpu_graph *g = &live->qwen4_graph;
            assert(getenv("DS4_QWEN4_NO_MTP_BATCH") == NULL);
            assert(live->checkpoint_valid && g->pos == (uint32_t)live->checkpoint.len);
            assert(qwen4_spec_depth(live) == 3 && live->glm_mtp_have && live->glm_mtp_have2);
            assert(live->glm_mtp_parent == token && g->cap_tokens >= 3u && g->mtp_R);
            assert(g->pos + 3u <= g->ctx_cap && g->pos + 3u <= (uint32_t)live->ctx_size);
            assert(qwen4_graph_fused(g, 2u) && qwen4_graph_ensure_snap2(g));
            assert(g->snap2_ple_hist != NULL);
            fprintf(stderr, "Qwen n-gram mode2: T3 read-failure admission verified; two pending drafts, fused graph, capacity and snapshot present\n");
        }
        const int fd = engine->model.ngram_fd;
        engine->model.ngram_fd = INT_MAX;
        if (mode == 3) {
            for (int i = 0; i < 32; i++) ds4_tokens_push(&frontier,token);
            assert(ds4_session_sync(live,&frontier,error,sizeof(error)) != 0);
        } else if (mode == 0) {
            ds4_tokens_push(&frontier, token);
            assert(ds4_session_sync(live, &frontier, error, sizeof(error)) != 0);
        } else if (mode == 1) {
            assert(ds4_session_eval(live, token, error, sizeof(error)) != 0);
        } else {
            assert(ds4_session_eval_speculative_argmax(live, token,
                3, -1, accepted, 3, error, sizeof(error)) < 0);
            if (require_deep_failure) {
                assert(strstr(error, "Qwen3.8 mtp: verify failed") != NULL);
                assert(!live->glm_mtp_have && !live->glm_mtp_have2);
                fprintf(stderr, "Qwen n-gram mode2: admitted T3 verifier returned the injected read failure\n");
            }
        }
        engine->model.ngram_fd = fd;
        assert(!live->checkpoint_valid);
        assert(ds4_session_argmax(live) == -1);
        assert(ds4_session_save_snapshot(live, &snapshot, error, sizeof(error)) != 0);
        assert(ds4_session_sync(live, &frontier, error, sizeof(error)) == 0);
        assert(ds4_session_sync(control, &frontier, error, sizeof(error)) == 0);
        assert(!memcmp(live->logits, control->logits, DS4_N_VOCAB * sizeof(float)));
        token = ds4_session_argmax(control);
        assert(ds4_session_eval(live, token, error, sizeof(error)) == 0);
        assert(ds4_session_eval(control, token, error, sizeof(error)) == 0);
        assert(!memcmp(live->logits, control->logits, DS4_N_VOCAB * sizeof(float)));
        ds4_tokens_free(&frontier);
        ds4_session_free(control);
        ds4_session_free(live);
        printf("Qwen n-gram failure/recovery mode %d: exact logits OK\n", mode);
    }
    check_direct_read_recovery(engine, &prompt);
#ifdef __APPLE__
    check_prefetch(engine, &prompt);
#endif
    check_batch(engine, &prompt, false);
    check_batch(engine, &prompt, true);
    check_arena_resize(engine, &prompt);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
}
