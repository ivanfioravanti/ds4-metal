/* Live V4.1 DSpark oracle: one model mapping, two independent target sessions.
 * Run explicitly on a Metal host with the matching target and draft GGUFs. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s (%s)\n", \
    __FILE__, __LINE__, #x, err); goto done; } } while (0)

static bool same_frontier(ds4_session *a, ds4_session *b) {
    if (a->checkpoint.len != b->checkpoint.len ||
        memcmp(a->checkpoint.v, b->checkpoint.v, a->checkpoint.len * sizeof(int)) ||
        memcmp(a->logits, b->logits, DS4_N_VOCAB * sizeof(float)) ||
        memcmp(&a->ds41_graph.history, &b->ds41_graph.history, sizeof(a->ds41_graph.history))) return false;
    ds41_state_span as[54], bs[54];
    uint32_t n = ds41_state_spans(&a->ds41_graph, a->ds41_graph.pos, as);
    if (n != ds41_state_spans(&b->ds41_graph, b->ds41_graph.pos, bs)) return false;
    for (uint32_t i = 0; i < n; i++)
        if (as[i].bytes != bs[i].bytes || memcmp(ds4_gpu_tensor_contents(as[i].tensor),
                ds4_gpu_tensor_contents(bs[i].tensor), as[i].bytes)) return false;
    return true;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 8) {
        fprintf(stderr, "usage: %s TARGET.gguf DRAFT.gguf PROMPT.txt [LISTEN PORT RDMA_DEVICE_OR_tcp GID]\n", argv[0]);
        return 2;
    }
    char err[256] = {0}, *text = NULL;
    size_t bytes = 0;
    ds4_engine *e = NULL;
    ds4_tp *tp = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_tokens prompt = {0};
    int rc = 1;
    ds4_engine_options opt = {.model_path = argv[1], .mtp_path = argv[2],
        .backend = DS4_BACKEND_METAL, .context_size = 4096, .power_percent = 100, .dspark = true};
    if (argc == 8) {
        char *end = NULL;
        long port = strtol(argv[5], &end, 10);
        CHECK(end != argv[5] && !*end && port > 0 && port <= 65535);
        long gid = strtol(argv[7], &end, 10);
        CHECK(end != argv[7] && !*end && gid >= 0 && gid <= 255);
        opt.tp = (ds4_tp_options){.role = DS4_TP_LEADER, .listen_host = argv[4],
            .listen_port = (int)port, .transport = !strcmp(argv[6], "tcp") ?
                DS4_TP_TRANSPORT_TCP : DS4_TP_TRANSPORT_RDMA,
            .rdma_device = argv[6], .rdma_gid_index = (int)gid, .rdma_gid_index_set = true};
    }
    CHECK(imatrix_read_text_file(argv[3], &text, &bytes));
    CHECK(ds4_engine_open(&e, &opt) == 0);
    if (argc == 8) {
        ds4_tp_identity id = {.gguf_bytes = ds4_engine_model_bytes(e),
            .model_id = ds4_engine_model_id(e), .n_layer = ds4_engine_layer_count(e),
            .n_embd = ds4_engine_embd_dim(e), .n_vocab = ds4_engine_vocab_size(e),
            .quant_bits = ds4_engine_routed_quant_bits(e), .ctx_size = 4096};
        ds4_engine_tp_gate_schedule(e, &id.gate_slot_start, &id.gate_slot_step,
            &id.gates_per_token, id.gate_slot_mask);
        CHECK(ds4_tp_create(&tp, &opt.tp, &id, err, sizeof(err)));
        CHECK(ds4_engine_tp_bind(e, tp, err, sizeof(err)));
    }
    CHECK(ds4_session_create(&control, e, 4096) == 0);
    CHECK(ds4_session_create(&candidate, e, 4096) == 0);
    ds4_encode_chat_prompt(e, NULL, text, DS4_THINK_NONE, &prompt);
    CHECK(prompt.len > 2047);
    const int prefixes[] = {18, 127, 128, 129, 511, 2047, 127};
    const float confidence = e->dspark_confidence_threshold;
    unsigned cycles = 0, accepted_drafts = 0;
    for (unsigned p = 0; p < sizeof(prefixes) / sizeof(prefixes[0]); p++) {
        /* Force all five proposals across a ring wrap, including rejected
         * suffixes. Ordinary confidence filtering rarely exercises six rows. */
        const bool full_width = p == 6;
        e->dspark_confidence_threshold = full_width ? 0.f : confidence;
        ds4_session_invalidate(control);
        ds4_session_invalidate(candidate);
        prompt.len = prefixes[p];
        CHECK(ds4_session_sync(control, &prompt, err, sizeof(err)) == 0);
        CHECK(ds4_session_sync(candidate, &prompt, err, sizeof(err)) == 0);
        CHECK(same_frontier(control, candidate));
        CHECK(ds41_dspark_ready(&candidate->ds41_graph));
        for (int emitted = 0; emitted < 64;) {
            int tokens[6];
            int cap = 64 - emitted < 6 ? 64 - emitted : 6;
            int seed = ds4_session_argmax_ignoring_eos(control, DS4_THINK_NONE);
            int n = ds4_session_eval_speculative_argmax_ignoring_eos(candidate, seed,
                cap, -1, DS4_THINK_NONE, tokens, cap, err, sizeof(err));
            CHECK(n > 0 && n <= cap);
            if (full_width && cap == 6)
                CHECK(candidate->ds41_graph.verify && candidate->ds41_graph.verify->count == 6);
            for (int i = 0; i < n; i++) {
                CHECK(tokens[i] == ds4_session_argmax_ignoring_eos(control, DS4_THINK_NONE));
                CHECK(ds4_session_eval(control, tokens[i], err, sizeof(err)) == 0);
            }
            CHECK(same_frontier(control, candidate));
            emitted += n;
            accepted_drafts += n - 1;
            cycles++;
        }
        fprintf(stderr, "V4.1 DSpark prefix=%d%s: 64 identical greedy tokens and exact target frontiers PASS\n",
                prefixes[p], full_width ? " forced-six-row" : "");
    }
    e->dspark_confidence_threshold = confidence;
    CHECK(accepted_drafts > 0);
    ds4_session_invalidate(control);
    ds4_session_invalidate(candidate);
    prompt.len = 511;
    CHECK(ds4_session_sync(control, &prompt, err, sizeof(err)) == 0);
    CHECK(ds4_session_sync(candidate, &prompt, err, sizeof(err)) == 0);
    uint64_t control_rng = 42, candidate_rng = 42;
    for (int emitted = 0; emitted < 64;) {
        int tokens[6];
        int cap = 64 - emitted < 6 ? 64 - emitted : 6;
        int seed = ds4_sample_logits(control->logits, DS4_N_VOCAB, .7f, 50, .95f, .02f, &control_rng);
        CHECK(seed == ds4_sample_logits(candidate->logits, DS4_N_VOCAB, .7f, 50, .95f, .02f, &candidate_rng));
        int n = ds4_session_eval_speculative(candidate, seed, cap, -1,
            .7f, 50, .95f, .02f, &candidate_rng, tokens, cap, err, sizeof(err));
        CHECK(n > 0 && n <= cap);
        for (int i = 0; i < n; i++) {
            int next = i ? ds4_sample_logits(control->logits, DS4_N_VOCAB, .7f, 50, .95f, .02f, &control_rng) : seed;
            CHECK(tokens[i] == next);
            CHECK(ds4_session_eval(control, next, err, sizeof(err)) == 0);
        }
        CHECK(control_rng == candidate_rng && same_frontier(control, candidate));
        emitted += n;
    }
    fprintf(stderr, "V4.1 DSpark: 64 identical sampled tokens, RNG and target frontiers PASS\n");
    if (!tp) {
    FILE *payload = tmpfile();
    CHECK(payload && ds4_session_save_payload(candidate, payload, err, sizeof(err)) == 0);
    long payload_bytes = ftell(payload);
    rewind(payload);
    CHECK(ds4_session_load_payload(candidate, payload, (uint64_t)payload_bytes, err, sizeof(err)) == 0);
    fclose(payload);
    CHECK(same_frontier(control, candidate));
    CHECK(!ds41_dspark_ready(&candidate->ds41_graph));
    int seed = ds4_session_argmax(control), out[6];
    CHECK(ds4_session_eval_speculative_argmax(candidate, seed, 6, -1, out, 6, err, sizeof(err)) == 1);
    CHECK(out[0] == seed && ds4_session_eval(control, seed, err, sizeof(err)) == 0);
    CHECK(same_frontier(control, candidate));
    fprintf(stderr, "V4.1 DSpark: disk restore safely rebuilds draft features via ordinary decode PASS\n");
    }
    int stop_seed = ds4_session_argmax(control), stopped[6];
    CHECK(ds4_session_eval_speculative_argmax(candidate, stop_seed, 6, stop_seed,
        stopped, 6, err, sizeof(err)) == 1);
    CHECK(stopped[0] == stop_seed && ds4_session_eval(control, stop_seed, err, sizeof(err)) == 0);
    CHECK(same_frontier(control, candidate));
    CHECK(ds4_session_eval_speculative_argmax(candidate, stop_seed, 6, -1,
        NULL, 6, err, sizeof(err)) == 0);
    CHECK(same_frontier(control, candidate));
    /* Restrict the public context to one final token without reallocating it. */
    int original_ctx = candidate->ctx_size;
    candidate->ctx_size = candidate->checkpoint.len + 1;
    stop_seed = ds4_session_argmax(control);
    CHECK(ds4_session_eval_speculative_argmax(candidate, stop_seed, 6, -1,
        stopped, 6, err, sizeof(err)) == 1);
    CHECK(ds4_session_eval(control, stop_seed, err, sizeof(err)) == 0);
    CHECK(same_frontier(control, candidate));
    CHECK(ds4_session_eval_speculative_argmax(candidate, stop_seed, 6, -1,
        stopped, 6, err, sizeof(err)) == 0);
    candidate->ctx_size = original_ctx;
    fprintf(stderr, "V4.1 DSpark: EOS, null output and context cap PASS\n");
    fprintf(stderr, "V4.1 DSpark: %u cycles, %u accepted drafts PASS\n", cycles, accepted_drafts);
    rc = 0;
done:
    ds4_session_free(candidate);
    ds4_session_free(control);
    if (tp) (void)ds4_tp_send_stop(tp);
    ds4_engine_close(e);
    ds4_tp_free(tp);
    ds4_tokens_free(&prompt);
    free(text);
    return rc;
}
