#ifndef DS4_QWEN4_H
#define DS4_QWEN4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_QWEN4_ARCHITECTURE "qwen4-exp"
#define DS4_QWEN4_PLE_ARCHITECTURE "qwen4-exp-ple"
#define DS4_QWEN4_PACK_MANIFEST_VERSION UINT32_C(3)
#define DS4_QWEN4_PACK_SCHEMA "ds4.qwen4.fast-pack"
#define DS4_QWEN4_PACK_MANIFEST_FILE "qwen3.8-flash-next-q4.manifest.json"
#define DS4_QWEN4_PACK_BASE_SHARDS 1u
#define DS4_QWEN4_PACK_MAX_BASE_SHARDS DS4_QWEN4_PACK_BASE_SHARDS
#define DS4_QWEN4_PACK_MAX_ARTIFACTS 4u
#define DS4_QWEN4_PACK_PATH_CAP 256u
#define DS4_QWEN4_PACK_ID_CAP 65u

/* Qwen3.8-Flash-Next (HF model_type=qwen4_exp) fixed release geometry. */
enum {
    DS4_QWEN4_LAYERS = 48,
    DS4_QWEN4_HIDDEN = 2560,
    DS4_QWEN4_VOCAB = 248320,
    DS4_QWEN4_ATTN_HEADS = 24,
    DS4_QWEN4_KV_HEADS = 2,
    DS4_QWEN4_HEAD_DIM = 256,
    DS4_QWEN4_ROPE_DIM = 64,
    DS4_QWEN4_HC_COUNT = 4,
    DS4_QWEN4_HC_LOWRANK = 320,
    DS4_QWEN4_LINEAR_KEY_HEADS = 16,
    DS4_QWEN4_LINEAR_VALUE_HEADS = 48,
    DS4_QWEN4_LINEAR_HEAD_DIM = 128,
    DS4_QWEN4_LINEAR_CONV = 4,
    DS4_QWEN4_EXPERTS = 512,
    DS4_QWEN4_EXPERTS_USED = 10,
    DS4_QWEN4_EXPERT_FF = 640,
    DS4_QWEN4_EXPERT_FF_PADDED = 768,
    DS4_QWEN4_SHARED_EXPERT_FF = 640,
    DS4_QWEN4_FULL_ATTN_INTERVAL = 4,
    DS4_QWEN4_INDEXER_HEADS = 4,
    DS4_QWEN4_INDEXER_KV_HEADS = 1,
    DS4_QWEN4_INDEXER_HEAD_DIM = 128,
    DS4_QWEN4_INDEXER_BUDGET = 2048,
    DS4_QWEN4_INDEXER_RATIO = 4,
    DS4_QWEN4_QSA_MICROTILE = 512,
    DS4_QWEN4_PLE_LAYER = 1,
    DS4_QWEN4_PLE_EMBED_DIM = 2560,
    DS4_QWEN4_PLE_ROW_DIM = 160,
    DS4_QWEN4_PLE_CONV = 4,
    DS4_QWEN4_NGRAM_SIZE = 3,
    DS4_QWEN4_HEADS_PER_NGRAM = 8,
    DS4_QWEN4_NGRAM_HEADS = 16,
    DS4_QWEN4_NGRAM_MAX_SIZE = 8,
    DS4_QWEN4_NGRAM_MAX_HEADS = 32,
    /* PLE resets n-gram history at the model-config EOS (<|endoftext|>),
     * while generation stops at the tokenizer EOS (<|im_end|>). */
    DS4_QWEN4_NGRAM_EOS = 248044,
    DS4_QWEN4_EOS = 248046,
    DS4_QWEN4_VISION_START_TOKEN = 248053,
    DS4_QWEN4_VISION_END_TOKEN = 248054,
    DS4_QWEN4_IMAGE_TOKEN = 248056,
    DS4_QWEN4_VISION_LAYERS = 27,
    DS4_QWEN4_VISION_HIDDEN = 1152,
    DS4_QWEN4_VISION_INTERMEDIATE = 4304,
    /* The official FC2 input is not divisible by the Q8_0 block size.  The
     * fast pack appends 16 zero columns to preserve standard GGML blocks. */
    DS4_QWEN4_VISION_INTERMEDIATE_PADDED = 4320,
    DS4_QWEN4_VISION_HEADS = 16,
    DS4_QWEN4_VISION_HEAD_DIM = 72,
    DS4_QWEN4_VISION_PATCH = 16,
    DS4_QWEN4_VISION_TEMPORAL_PATCH = 2,
    DS4_QWEN4_VISION_MERGE = 2,
    DS4_QWEN4_VISION_POSITION_SIDE = 48,
    DS4_QWEN4_VISION_OUTPUT = 2560,
    DS4_QWEN4_MROPE_T = 11,
    DS4_QWEN4_MROPE_H = 11,
    DS4_QWEN4_MROPE_W = 10,
    /* Exact v3 GGUF tensor-directory cardinalities.  Tokenizer vocabulary
     * and merge tables are GGUF metadata and do not add tensor entries. */
    DS4_QWEN4_BASE_TENSOR_COUNT =
        1 + 25 * DS4_QWEN4_LAYERS + 6 + 3 + 1,
    DS4_QWEN4_PLE_TENSOR_COUNT = 4,
    DS4_QWEN4_VISION_TENSOR_COUNT =
        12 * DS4_QWEN4_VISION_LAYERS + 9,
    DS4_QWEN4_MTP_TENSOR_COUNT = 32,
};

#define DS4_QWEN4_CONTEXT_LENGTH UINT32_C(262144)
#define DS4_QWEN4_ROPE_THETA 10000000.0f
#define DS4_QWEN4_RMS_EPS 1.0e-6f
#define DS4_QWEN4_NGRAM_VOCAB_BASE UINT64_C(20000000)
#define DS4_QWEN4_NGRAM_VOCAB_DIVISOR UINT64_C(128)
#define DS4_QWEN4_NGRAM_SEED UINT64_C(1234)

typedef enum {
    DS4_QWEN4_PREFILL_AUTO = 0,
    DS4_QWEN4_PREFILL_2048 = 2048,
    DS4_QWEN4_PREFILL_4096 = 4096,
    DS4_QWEN4_PREFILL_8192 = 8192,
} ds4_qwen4_prefill_mode;

#define DS4_QWEN4_FAST_PATH_CAPS_VERSION UINT32_C(1)
#define DS4_QWEN4_FAST_PATH_REASON_CAP 192u
#define DS4_QWEN4_FAST_EXACT_Q8_PREFILL (UINT64_C(1) << 0)
#define DS4_QWEN4_FAST_EXACT_Q4_MOE     (UINT64_C(1) << 1)
#define DS4_QWEN4_FAST_GDN_R4           (UINT64_C(1) << 2)
#define DS4_QWEN4_FAST_QSA_STREAM_TOPK  (UINT64_C(1) << 3)
#define DS4_QWEN4_FAST_QSA_M1           (UINT64_C(1) << 4)
#define DS4_QWEN4_FAST_DECODE_Q4        (UINT64_C(1) << 5)
#define DS4_QWEN4_FAST_DECODE_Q8        (UINT64_C(1) << 6)
#define DS4_QWEN4_FAST_PLE_STAGER       (UINT64_C(1) << 7)
#define DS4_QWEN4_FAST_METAL_REQUIRED_MASK \
    (DS4_QWEN4_FAST_EXACT_Q8_PREFILL | DS4_QWEN4_FAST_EXACT_Q4_MOE | \
     DS4_QWEN4_FAST_GDN_R4 | DS4_QWEN4_FAST_QSA_STREAM_TOPK | \
     DS4_QWEN4_FAST_QSA_M1 | DS4_QWEN4_FAST_DECODE_Q4 | \
     DS4_QWEN4_FAST_DECODE_Q8)
#define DS4_QWEN4_FAST_REQUIRED_MASK \
    (DS4_QWEN4_FAST_METAL_REQUIRED_MASK | DS4_QWEN4_FAST_PLE_STAGER)

typedef struct {
    uint32_t version;
    uint32_t struct_size;
    uint64_t available_mask;
    uint64_t required_mask;
    uint64_t missing_mask;
    char missing_reason[DS4_QWEN4_FAST_PATH_REASON_CAP];
} ds4_qwen4_fast_path_caps;

typedef struct {
    uint32_t admitted_cap;
    uint64_t scratch_required;
    const char *reason;
} ds4_qwen4_prefill_admission;

typedef enum {
    DS4_QWEN4_ARTIFACT_BASE = 0,
    DS4_QWEN4_ARTIFACT_PLE = 1,
    DS4_QWEN4_ARTIFACT_VISION = 2,
    DS4_QWEN4_ARTIFACT_MTP = 3,
} ds4_qwen4_artifact_kind;

bool ds4_qwen4_artifact_tensor_count_valid(
        ds4_qwen4_artifact_kind kind,
        uint64_t count);

typedef struct {
    ds4_qwen4_artifact_kind kind;
    uint32_t shard_index;
    uint64_t bytes;
    char path[DS4_QWEN4_PACK_PATH_CAP];
    char sha256[65];
} ds4_qwen4_pack_artifact;

typedef struct {
    uint32_t version;
    uint32_t base_shard_count;
    uint32_t artifact_count;
    char pack_id[DS4_QWEN4_PACK_ID_CAP];
    char source_revision[DS4_QWEN4_PACK_ID_CAP];
    char tensor_manifest_sha256[65];
    ds4_qwen4_pack_artifact artifacts[DS4_QWEN4_PACK_MAX_ARTIFACTS];
} ds4_qwen4_pack_manifest;

int ds4_qwen4_pack_manifest_load(ds4_qwen4_pack_manifest *manifest,
                                 const char *path,
                                 char *error,
                                 size_t error_cap);

const ds4_qwen4_pack_artifact *ds4_qwen4_pack_find_artifact(
        const ds4_qwen4_pack_manifest *manifest,
        ds4_qwen4_artifact_kind kind,
        uint32_t shard_index);

/* Select the artifacts that must be verified for this invocation.  Base
 * shards and PLE are always selected; optional sidecars are selected only
 * when their matching command-line option was supplied. */
int ds4_qwen4_pack_validation_mask(
        const ds4_qwen4_pack_manifest *manifest,
        bool vision_requested,
        bool mtp_requested,
        uint32_t *mask_out,
        char *error,
        size_t error_cap);

int ds4_qwen4_sha256_file(const char *path,
                          char hex_out[65],
                          uint64_t *bytes_out,
                          char *error,
                          size_t error_cap);

int ds4_qwen4_pack_validate_artifact(const ds4_qwen4_pack_artifact *artifact,
                                     const char *path,
                                     char *error,
                                     size_t error_cap);

bool ds4_qwen4_parse_prefill_mode(const char *value,
                                  ds4_qwen4_prefill_mode *out,
                                  char *error,
                                  size_t error_cap);

uint32_t ds4_qwen4_select_prefill_chunk(ds4_qwen4_prefill_mode mode,
                                        bool fast_path_complete,
                                        uint32_t admitted_cap,
                                        uint32_t uncached_tokens,
                                        uint32_t cached_tokens,
                                        const char **reason);

/* Make the single startup-time scratch decision used by every session.
 * Auto admits only 2K or 8K; explicit modes must fit exactly. */
bool ds4_qwen4_admit_prefill(ds4_qwen4_prefill_mode mode,
                             bool fast_path_complete,
                             const char *missing_fast_path,
                             uint64_t scratch_available,
                             ds4_qwen4_prefill_admission *out,
                             char *error,
                             size_t error_cap);

/* Peak transient storage used by the native Qwen prefill graph.  The value
 * includes two HC streams, dense/GDN projections, routed-expert activation
 * rows, QSA microtile/heap storage, and the double-buffered PLE upload. */
uint64_t ds4_qwen4_prefill_scratch_bytes(uint32_t chunk);

typedef struct {
    uint32_t token_start;
    uint32_t token_count;
    uint32_t grid_width;
    uint32_t grid_height;
} ds4_qwen4_mrope_span;

/* Build the contracted Qwen multimodal position table in axis-major
 * [3, capacity] layout.  Text positions share one coordinate; image tokens
 * use temporal/row/column coordinates after the 2x2 spatial merger. */
bool ds4_qwen4_build_mrope_positions(
        int32_t                     *positions,
        uint32_t                     capacity,
        uint32_t                     prompt_len,
        const ds4_qwen4_mrope_span  *images,
        size_t                       image_count,
        int32_t                     *delta_out);

typedef struct {
    uint32_t ngram_size;
    uint32_t heads_per_ngram;
    uint32_t n_heads;
    uint32_t eos;
    int64_t multipliers[DS4_QWEN4_NGRAM_MAX_SIZE];
    int64_t vocab[DS4_QWEN4_NGRAM_MAX_HEADS];
    int64_t offsets[DS4_QWEN4_NGRAM_MAX_HEADS];
    uint64_t total_rows;
} ds4_qwen4_ngram_hash;

bool ds4_qwen4_ngram_hash_init(ds4_qwen4_ngram_hash *hash,
                               uint32_t unigram_vocab,
                               uint32_t ngram_size,
                               uint32_t heads_per_ngram,
                               uint64_t vocab_base,
                               uint64_t divisor,
                               uint64_t seed,
                               uint32_t ple_layer_index,
                               uint32_t eos);

bool ds4_qwen4_ngram_row_ids(const ds4_qwen4_ngram_hash *hash,
                             const uint32_t *previous,
                             size_t previous_count,
                             const uint32_t *tokens,
                             size_t token_count,
                             int64_t *rows,
                             size_t row_count);

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;
    uint64_t rows;
    uint32_t dim;
    uint32_t qtype;
    uint32_t block_size;
    uint32_t blocks_per_row;
    uint32_t row_bytes;
    uint64_t data_offset;
    uint64_t identity_device;
    uint64_t identity_inode;
    int64_t identity_mtime_sec;
    int64_t identity_mtime_nsec;
    int64_t identity_ctime_sec;
    int64_t identity_ctime_nsec;
    char sha256[65];
    bool validation_nocache_requested;
    bool validation_nocache_enabled;
    bool validation_nocache_cleared;
    bool runtime_readahead_requested;
    bool runtime_readahead_disabled;
} ds4_qwen4_ple_table;

typedef struct {
    uint64_t *pages;
    size_t count;
    uint64_t page_size;
} ds4_qwen4_ple_page_plan;

typedef struct {
    uint64_t page_size;
    uint64_t target_pages;
    uint64_t resident_target_pages;
    uint64_t cold_target_pages;
    uint64_t identity_device;
    uint64_t identity_inode;
    uint64_t identity_size;
    int64_t identity_mtime_sec;
    int64_t identity_mtime_nsec;
    int64_t identity_ctime_sec;
    int64_t identity_ctime_nsec;
    bool identity_stable;
} ds4_qwen4_ple_residency;

int ds4_qwen4_ple_table_open(ds4_qwen4_ple_table *table,
                             const char *path,
                             char *error,
                             size_t error_cap);

/* Validate the PLE checksum without populating the macOS file cache, then
 * retain that same descriptor for sparse runtime reads. */
int ds4_qwen4_pack_validate_ple_open(
        const ds4_qwen4_pack_artifact *artifact,
        const char *path,
        ds4_qwen4_ple_table *table,
        char *error,
        size_t error_cap);

void ds4_qwen4_ple_table_close(ds4_qwen4_ple_table *table);

/* Return the exact sorted set of file pages touched by sparse pread gathers
 * for the supplied PLE rows.  The caller releases plan->pages with the
 * matching free helper. */
bool ds4_qwen4_ple_plan_pages(const ds4_qwen4_ple_table *table,
                              const int64_t *rows,
                              size_t row_count,
                              uint64_t page_size,
                              ds4_qwen4_ple_page_plan *plan,
                              char *error,
                              size_t error_cap);
void ds4_qwen4_ple_page_plan_free(ds4_qwen4_ple_page_plan *plan);

/* Darwin-only benchmark proof.  It samples mincore residency for the exact
 * target pages before the caller submits the first sparse PLE gather. */
bool ds4_qwen4_ple_measure_residency(
        const ds4_qwen4_ple_table *table,
        const int64_t *rows,
        size_t row_count,
        ds4_qwen4_ple_residency *residency,
        char *error,
        size_t error_cap);

bool ds4_qwen4_ple_row_f32(const ds4_qwen4_ple_table *table,
                           uint64_t row,
                           float *out,
                           size_t out_count);

bool ds4_qwen4_ple_gather_bf16(const ds4_qwen4_ple_table *table,
                               const int64_t *rows,
                               size_t row_count,
                               uint16_t *out,
                               size_t out_count);

typedef enum {
    DS4_QWEN4_PLE_GATHER_PREAD = 0,
    DS4_QWEN4_PLE_GATHER_MMAP = 1,
} ds4_qwen4_ple_gather_mode;

typedef struct {
    ds4_qwen4_ple_gather_mode mode;
    bool random_advice_requested;
    bool random_advice_succeeded;
    uint64_t gather_count;
    uint64_t gather_rows;
    double gather_seconds;
    double wait_seconds;
} ds4_qwen4_ple_stager_stats;

const char *ds4_qwen4_ple_gather_mode_name(
        ds4_qwen4_ple_gather_mode mode);

/* Two-slot CPU staging queue.  Each slot owns a worker and a BF16 output
 * buffer, allowing the caller to submit the next PLE gather while Metal is
 * consuming the previous slot.  DS4_QWEN4_PLE_GATHER selects the default
 * pread path or the experimental demand-paged mmap path at initialization. */
typedef struct {
    void *impl;
} ds4_qwen4_ple_stager;

int ds4_qwen4_ple_stager_init(ds4_qwen4_ple_stager *stager,
                              const ds4_qwen4_ple_table *table,
                              size_t max_rows,
                              char *error,
                              size_t error_cap);
void ds4_qwen4_ple_stager_destroy(ds4_qwen4_ple_stager *stager);

bool ds4_qwen4_ple_stager_submit(ds4_qwen4_ple_stager *stager,
                                 uint32_t slot,
                                 const int64_t *rows,
                                 size_t row_count);

bool ds4_qwen4_ple_stager_wait(ds4_qwen4_ple_stager *stager,
                               uint32_t slot,
                               const uint16_t **data,
                               size_t *value_count);

bool ds4_qwen4_ple_stager_snapshot(
        ds4_qwen4_ple_stager *stager,
        ds4_qwen4_ple_stager_stats *stats);

#endif
