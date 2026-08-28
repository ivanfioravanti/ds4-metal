/*
 * gguf-requantize-dense -- GGUF -> GGUF dense re-quantizer.
 *
 * Re-encodes selected dense tensor families (attention projections, output
 * head, or arbitrary name prefixes) from their dequantized in-file values,
 * copying every other tensor and the whole metadata section verbatim.  This
 * is the tool the requant track was waiting on: deepseek4-quantize needs the
 * HF safetensors, which are not always on disk, and its MXFP4 path is
 * repack-only.  The reference for a re-encoded tensor here is the input
 * GGUF's values (double quantization: e.g. q8_0 -> q4_k adds only the
 * second-order error of the nearly lossless q8_0 stage on top of q4_k's
 * own), which is acceptable for quality-gated experiments.
 *
 * The GGUF reader/writer mirrors deepseek4-quantize.c byte for byte (header,
 * verbatim KV blob, tensor table, alignment-padded data section) so the
 * output stays load-compatible with the ds4 runtime without touching the
 * production quantizer.
 */

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quants.h"

/* GGUF v3 metadata value types (same numbering as deepseek4-quantize.c). */
#define GGUF_TYPE_UINT8   0
#define GGUF_TYPE_INT8    1
#define GGUF_TYPE_UINT16  2
#define GGUF_TYPE_INT16   3
#define GGUF_TYPE_UINT32  4
#define GGUF_TYPE_INT32   5
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_BOOL    7
#define GGUF_TYPE_STRING  8
#define GGUF_TYPE_ARRAY   9
#define GGUF_TYPE_UINT64  10
#define GGUF_TYPE_INT64   11
#define GGUF_TYPE_FLOAT64 12

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    die("%s '%s': %s", what, path ? path : "?", strerror(errno));
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory (%zu bytes)", n);
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die("out of memory (%zu x %zu)", n, sz);
    return p;
}

/* ---- GGUF read helpers ------------------------------------------------- */

static uint32_t read_u32_le_fp(FILE *fp, const char *what) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) die("short read (%s)", what);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint64_t read_u64_le_fp(FILE *fp, const char *what) {
    uint8_t b[8];
    if (fread(b, 1, 8, fp) != 8) die("short read (%s)", what);
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | b[i];
    return v;
}

static char *read_gguf_string_fp(FILE *fp) {
    uint64_t n = read_u64_le_fp(fp, "GGUF string length");
    if (n > (1u << 20)) die("unreasonable GGUF string length %" PRIu64, n);
    char *s = xmalloc((size_t)n + 1);
    if (n && fread(s, 1, (size_t)n, fp) != (size_t)n) die("short GGUF string read");
    s[n] = '\0';
    return s;
}

static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static void skip_bytes_fp(FILE *fp, uint64_t n) {
    if (fseeko(fp, (off_t)n, SEEK_CUR) != 0) die("GGUF seek failed");
}

static void skip_gguf_value_fp(FILE *fp, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t n = read_u64_le_fp(fp, "GGUF string length");
        skip_bytes_fp(fp, n);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = read_u32_le_fp(fp, "GGUF array type");
        uint64_t n = read_u64_le_fp(fp, "GGUF array count");
        if (elem_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len = read_u64_le_fp(fp, "GGUF array string length");
                skip_bytes_fp(fp, len);
            }
        } else {
            size_t sz = gguf_scalar_size(elem_type);
            if (!sz) die("unsupported GGUF array type %u", elem_type);
            skip_bytes_fp(fp, n * sz);
        }
        return;
    }
    size_t sz = gguf_scalar_size(type);
    if (!sz) die("unsupported GGUF value type %u", type);
    skip_bytes_fp(fp, sz);
}

/* ---- GGUF model --------------------------------------------------------- */

typedef struct {
    char *name;
    int n_dims;
    int64_t ne[DS4Q_MAX_DIMS];
    ds4q_type type;
    uint64_t old_offset;
    uint64_t new_offset;
    size_t size;
    size_t new_size;
    ds4q_type new_type;
} tensor_meta;

typedef struct {
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint8_t *kv_raw;      /* the full KV region, byte-for-byte */
    size_t kv_raw_len;
    size_t alignment;
    size_t data_offset;
    tensor_meta *tensors;
} gguf_file;

static size_t tensor_nbytes(ds4q_type type, const int64_t *ne, int n_dims) {
    size_t nbytes = ds4q_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) nbytes *= (size_t)ne[i];
    return nbytes;
}

static gguf_file open_gguf(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "GGUF", 4) != 0) {
        die("'%s' is not a GGUF file", path);
    }
    gguf_file g;
    memset(&g, 0, sizeof(g));
    g.alignment = 32;
    g.version = read_u32_le_fp(fp, "GGUF version");
    if (g.version != 3 && g.version != 2) die("unsupported GGUF version %u", g.version);
    g.n_tensors = read_u64_le_fp(fp, "GGUF tensor count");
    g.n_kv = read_u64_le_fp(fp, "GGUF KV count");
    if (g.n_tensors > 100000u) die("unreasonable tensor count %" PRIu64, g.n_tensors);

    off_t kv_start = ftello(fp);
    if (kv_start < 0) die("GGUF ftell failed");
    for (uint64_t i = 0; i < g.n_kv; i++) {
        char *key = read_gguf_string_fp(fp);
        uint32_t type = read_u32_le_fp(fp, "GGUF KV type");
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t a = read_u32_le_fp(fp, "GGUF alignment");
            if (a) g.alignment = a;
        } else {
            skip_gguf_value_fp(fp, type);
        }
        free(key);
    }
    off_t tensor_start = ftello(fp);
    if (tensor_start < 0 || tensor_start < kv_start) die("GGUF ftell failed");

    /* Keep every KV record verbatim: this tool adds no metadata. */
    g.kv_raw_len = (size_t)(tensor_start - kv_start);
    g.kv_raw = xmalloc(g.kv_raw_len);
    if (fseeko(fp, kv_start, SEEK_SET) != 0) die("GGUF seek failed");
    if (g.kv_raw_len && fread(g.kv_raw, 1, g.kv_raw_len, fp) != g.kv_raw_len) {
        die("GGUF KV read failed");
    }
    if (fseeko(fp, tensor_start, SEEK_SET) != 0) die("GGUF seek failed");

    g.tensors = xcalloc((size_t)g.n_tensors, sizeof(g.tensors[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        t->name = read_gguf_string_fp(fp);
        t->n_dims = (int)read_u32_le_fp(fp, "GGUF tensor rank");
        if (t->n_dims < 1 || t->n_dims > DS4Q_MAX_DIMS) die("bad GGUF tensor rank");
        for (int j = 0; j < t->n_dims; j++) {
            t->ne[j] = (int64_t)read_u64_le_fp(fp, "GGUF tensor dim");
        }
        t->type = (ds4q_type)read_u32_le_fp(fp, "GGUF tensor type");
        t->old_offset = read_u64_le_fp(fp, "GGUF tensor offset");
        t->size = tensor_nbytes(t->type, t->ne, t->n_dims);
    }
    off_t meta_end = ftello(fp);
    if (meta_end < 0) die("GGUF ftell failed");
    g.data_offset = ds4q_pad((size_t)meta_end, g.alignment);
    fclose(fp);
    return g;
}

/* ---- dequantize (the sources worth reading back) ------------------------ */

static bool can_dequant(ds4q_type type) {
    return type == DS4Q_TYPE_Q8_0 || type == DS4Q_TYPE_F16 ||
           type == DS4Q_TYPE_F32 || type == DS4Q_TYPE_BF16;
}

static float *tensor_to_f32(const tensor_meta *t, const uint8_t *raw) {
    const int64_t n_row = t->n_dims > 1 ? t->ne[1] : 1;
    for (int i = 2; i < t->n_dims; i++) {
        if (t->ne[i] != 1) die("tensor %s has rank > 2 extents", t->name);
    }
    const int64_t n_col = t->ne[0];
    float *out = xmalloc((size_t)n_row * (size_t)n_col * sizeof(float));
    if (t->type == DS4Q_TYPE_F32) {
        memcpy(out, raw, (size_t)n_row * (size_t)n_col * sizeof(float));
        return out;
    }
    if (t->type == DS4Q_TYPE_F16 || t->type == DS4Q_TYPE_BF16) {
        const uint16_t *src = (const uint16_t *)raw;
        for (size_t i = 0; i < (size_t)n_row * (size_t)n_col; i++) {
            out[i] = t->type == DS4Q_TYPE_F16 ? ds4q_f16_to_f32(src[i])
                                              : ds4q_bf16_to_f32(src[i]);
        }
        return out;
    }
    if (t->type == DS4Q_TYPE_Q8_0) {
        if (n_col % 32 != 0) die("q8_0 tensor %s has ne[0] not divisible by 32", t->name);
        const size_t blocks = (size_t)n_row * (size_t)(n_col / 32);
        const uint8_t *p = raw;
        float *o = out;
        for (size_t b = 0; b < blocks; b++) {
            const float scale = ds4q_f16_to_f32((uint16_t)(p[0] | (p[1] << 8)));
            const int8_t *qs = (const int8_t *)(p + 2);
            for (int j = 0; j < 32; j++) o[j] = scale * (float)qs[j];
            p += 34;
            o += 32;
        }
        return out;
    }
    die("cannot dequantize %s from %s", t->name, ds4q_type_name(t->type));
    return NULL;
}

/* ---- family classification ----------------------------------------------
 * Same suffixes as deepseek4-quantize's is_attention_projection(), but the
 * indexer's own q_b ("blk.N.indexer.attn_q_b.weight") also suffix-matches
 * and must stay untouched (F16Indexer recipe), so indexer names are
 * excluded explicitly. */

static bool is_attention_projection(const char *name) {
    if (strstr(name, "indexer")) return false;
    return strstr(name, ".attn_kv.weight") || strstr(name, ".attn_q_a.weight") ||
           strstr(name, ".attn_q_b.weight") || strstr(name, ".attn_output_a.weight") ||
           strstr(name, ".attn_output_b.weight");
}

static bool is_output_head(const char *name) {
    return strcmp(name, "output.weight") == 0;
}

/* ---- imatrix (llama.cpp legacy .dat, ds4 collector format) -------------- */

typedef struct {
    char *name;
    float *values;
    int64_t n_values;
} imatrix_entry;

typedef struct {
    imatrix_entry *entries;
    int n_entries;
    bool strict;
} imatrix_store;

static int32_t rq_read_i32(FILE *fp, const char *what) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) die("short read (%s)", what);
    return (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8) |
                     ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24));
}

static void imatrix_load(imatrix_store *im, const char *path, bool strict) {
    memset(im, 0, sizeof(*im));
    im->strict = strict;
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open imatrix", path);
    int32_t n = rq_read_i32(fp, "imatrix entry count");
    if (n < 1) die("imatrix has no entries");
    im->entries = xcalloc((size_t)n, sizeof(im->entries[0]));
    im->n_entries = n;
    for (int i = 0; i < n; i++) {
        int32_t len = rq_read_i32(fp, "imatrix name length");
        if (len <= 0 || len > 4096) die("bad imatrix name length");
        char *name = xmalloc((size_t)len + 1);
        if (fread(name, 1, (size_t)len, fp) != (size_t)len) die("short imatrix name read");
        name[len] = '\0';
        int32_t ncall = rq_read_i32(fp, "imatrix calls");
        int32_t nval = rq_read_i32(fp, "imatrix values");
        if (nval < 1) die("bad imatrix value count for %s", name);
        float *values = xmalloc((size_t)nval * sizeof(float));
        if (fread(values, sizeof(float), (size_t)nval, fp) != (size_t)nval) {
            die("short imatrix value read for %s", name);
        }
        if (ncall > 0) {
            for (int j = 0; j < nval; j++) values[j] /= (float)ncall;
        }
        for (int j = 0; j < nval; j++) {
            if (!isfinite(values[j])) die("non-finite imatrix value in %s", name);
        }
        im->entries[i] = (imatrix_entry){ .name = name, .values = values, .n_values = nval };
    }
    fclose(fp);
    fprintf(stderr, "loaded imatrix %s: %d entries\n", path, n);
}

static const float *imatrix_find(const imatrix_store *im, const tensor_meta *t) {
    for (int i = 0; i < im->n_entries; i++) {
        if (strcmp(im->entries[i].name, t->name) == 0) {
            if (im->entries[i].n_values != t->ne[0]) {
                die("imatrix size mismatch for %s: got %" PRId64 " expected %" PRId64,
                    t->name, im->entries[i].n_values, t->ne[0]);
            }
            return im->entries[i].values;
        }
    }
    if (im->strict) die("missing imatrix entry for %s", t->name);
    return NULL;
}

/* ---- CLI ---------------------------------------------------------------- */

static void usage(const char *argv0) {
    printf("usage: %s --in IN.gguf --out OUT.gguf [options]\n", argv0);
    printf("\nGGUF -> GGUF dense re-quantizer: re-encodes the requested tensor\n"
           "families from their dequantized in-file values and copies everything\n"
           "else (metadata and tensors) verbatim.\n\noptions:\n");
    printf("  --in FILE            input GGUF (the values are the reference)\n");
    printf("  --out FILE           output GGUF path\n");
    printf("  --attention-proj T   re-encode attn_q_a/q_b/kv/output_a/b projections\n");
    printf("  --output T           re-encode the output head (output.weight)\n");
    printf("  --tensor-type PFX=T  re-encode tensors whose name starts with PFX; repeatable\n");
    printf("  --tensor-suffix SUF=T re-encode tensors whose name ends with SUF (indexer excluded); repeatable\n");
    printf("  --dry-run            print the plan and exit\n");
    printf("  --overwrite          replace --out if it exists\n");
    printf("  --verify             after writing, checksum every copied tensor against --in\n");
    printf("  --imatrix FILE       importance matrix (ds4 --imatrix-out legacy .dat) for re-encoded tensors\n");
    printf("  --imatrix-strict    fail if a re-encoded tensor has no matching imatrix vector\n");
    printf("\nT: f16, f32, bf16, q8_0, q4_k, q2_k, iq2_xxs (sources: f32/f16/bf16/q8_0)\n");
}

typedef struct {
    const char *prefix;
    ds4q_type type;
} type_override;

/* Suffix overrides run after prefix overrides and before the family
 * defaults, so a family can be re-promoted (e.g. .attn_q_b.weight=q8_0
 * inside --attention-proj q4_K) for Q4_K_M-style mixed builds.
 * Indexer tensors are excluded: blk.N.indexer.attn_q_b.weight also
 * suffix-matches and must keep its recipe type. */
static bool name_has_suffix(const char *name, const char *suffix) {
    if (strstr(name, "indexer")) return false;
    const size_t nlen = strlen(name), slen = strlen(suffix);
    return nlen >= slen && strcmp(name + nlen - slen, suffix) == 0;
}

static ds4q_type parse_type(const char *s) {
    for (int t = 0; t < DS4Q_TYPE_COUNT; t++) {
        const char *name = ds4q_type_name((ds4q_type)t);  /* holes are NULL */
        if (name && strcmp(name, s) == 0) return (ds4q_type)t;
    }
    die("unknown tensor type '%s'", s);
    return DS4Q_TYPE_F32;
}

static void write_u32(FILE *fp, uint32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u32 failed");
}

static void write_u64(FILE *fp, uint64_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u64 failed");
}

static void write_gguf_string(FILE *fp, const char *s) {
    uint64_t n = strlen(s);
    write_u64(fp, n);
    if (n && fwrite(s, 1, (size_t)n, fp) != (size_t)n) die("write string failed");
}

static void write_padding(FILE *fp, size_t n) {
    static const uint8_t zeros[4096] = {0};
    while (n) {
        size_t chunk = n < sizeof(zeros) ? n : sizeof(zeros);
        if (fwrite(zeros, 1, chunk, fp) != chunk) die("write padding failed");
        n -= chunk;
    }
}

static uint64_t fnv1a64_bytes(const uint8_t *data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char **argv) {
    const char *in_path = NULL;
    const char *out_path = NULL;
    ds4q_type attn_type = DS4Q_TYPE_COUNT;   /* COUNT = not requested */
    ds4q_type out_type = DS4Q_TYPE_COUNT;
    type_override overrides[64];
    int n_overrides = 0;
    type_override suffix_overrides[64];
    int n_suffix_overrides = 0;
    bool dry_run = false;
    bool overwrite = false;
    bool verify = false;
    const char *imatrix_path = NULL;
    bool imatrix_strict = false;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(arg, "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(arg, "--attention-proj") && i + 1 < argc) attn_type = parse_type(argv[++i]);
        else if (!strcmp(arg, "--output") && i + 1 < argc) out_type = parse_type(argv[++i]);
        else if (!strcmp(arg, "--tensor-type") && i + 2 < argc) {
            char *spec = argv[++i];
            char *eq = strchr(spec, '=');
            if (!eq) die("--tensor-type expects PFX=TYPE");
            *eq = '\0';
            if (n_overrides >= (int)(sizeof(overrides) / sizeof(overrides[0]))) {
                die("too many --tensor-type overrides");
            }
            overrides[n_overrides].prefix = spec;
            overrides[n_overrides].type = parse_type(eq + 1);
            n_overrides++;
        }
        else if (!strcmp(arg, "--tensor-suffix") && i + 2 < argc) {
            char *spec = argv[++i];
            char *eq = strchr(spec, '=');
            if (!eq) die("--tensor-suffix expects SUF=TYPE");
            *eq = '\0';
            if (n_suffix_overrides >= (int)(sizeof(suffix_overrides) / sizeof(suffix_overrides[0]))) {
                die("too many --tensor-suffix overrides");
            }
            suffix_overrides[n_suffix_overrides].prefix = spec;
            suffix_overrides[n_suffix_overrides].type = parse_type(eq + 1);
            n_suffix_overrides++;
        }
        else if (!strcmp(arg, "--dry-run")) dry_run = true;
        else if (!strcmp(arg, "--overwrite")) overwrite = true;
        else if (!strcmp(arg, "--verify")) verify = true;
        else if (!strcmp(arg, "--imatrix") && i + 1 < argc) imatrix_path = argv[++i];
        else if (!strcmp(arg, "--imatrix-strict")) imatrix_strict = true;
        else { usage(argv[0]); die("unknown or incomplete argument '%s'", arg); }
    }
    if (!in_path || (!out_path && !dry_run)) {
        usage(argv[0]);
        die("--in and --out are required");
    }

    imatrix_store im = {0};
    if (imatrix_path) imatrix_load(&im, imatrix_path, imatrix_strict);
    gguf_file g = open_gguf(in_path);
    printf("input: %s\n  version=%u n_kv=%" PRIu64 " n_tensors=%" PRIu64
           " alignment=%zu data_offset=%zu\n",
           in_path, g.version, g.n_kv, g.n_tensors, g.alignment, g.data_offset);

    /* Build the plan. */
    size_t changed = 0;
    size_t old_bytes = 0, new_bytes = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        ds4q_type target = t->type;
        bool overridden = false;
        if (n_overrides > 0) {
            for (int k = 0; k < n_overrides; k++) {
                if (strncmp(t->name, overrides[k].prefix, strlen(overrides[k].prefix)) == 0) {
                    target = overrides[k].type;
                    overridden = true;
                    break;
                }
            }
        }
        if (!overridden && n_suffix_overrides > 0) {
            for (int k = 0; k < n_suffix_overrides; k++) {
                if (name_has_suffix(t->name, suffix_overrides[k].prefix)) {
                    target = suffix_overrides[k].type;
                    overridden = true;
                    break;
                }
            }
        }
        /* Family defaults must not re-apply after an explicit override
         * (a Q4_K_M promotion sets the target back to the q8_0 source
         * type, which the old target == t->type guard silently undid). */
        if (!overridden && is_attention_projection(t->name) && attn_type != DS4Q_TYPE_COUNT) {
            target = attn_type;
        }
        if (!overridden && is_output_head(t->name) && out_type != DS4Q_TYPE_COUNT) {
            target = out_type;
        }
        if (target != t->type) {
            if (!ds4q_can_quantize(target)) {
                die("%s: %s cannot be encoded by this build", t->name, ds4q_type_name(target));
            }
            if (!can_dequant(t->type)) {
                die("%s: source type %s cannot be dequantized", t->name, ds4q_type_name(t->type));
            }
            if (t->ne[0] % ds4q_block_size(target) != 0) {
                die("%s: ne[0] %" PRId64 " not divisible by %s block size",
                    t->name, t->ne[0], ds4q_type_name(target));
            }
        }
        t->new_type = target;
        t->new_size = target == t->type ? t->size : tensor_nbytes(target, t->ne, t->n_dims);
        if (target != t->type) {
            changed++;
            const char *im_state = !imatrix_path ? "" :
                (imatrix_find(&im, t) ? " [imatrix]" :
                 (printf("imatrix-miss: %s\n", t->name), " [no-imatrix]"));
            printf("type_change: %s %s -> %s (%.2f MiB -> %.2f MiB)%s\n",
                   t->name, ds4q_type_name(t->type), ds4q_type_name(target),
                   (double)t->size / 1048576.0, (double)t->new_size / 1048576.0,
                   im_state);
        }
        old_bytes += t->size;
        new_bytes += t->new_size;
    }
    printf("tensors: %" PRIu64 " (changed %zu)\n", g.n_tensors, changed);
    printf("tensor bytes: %.2f GiB -> %.2f GiB (delta %+.2f GiB)\n",
           (double)old_bytes / (1ull << 30), (double)new_bytes / (1ull << 30),
           ((double)new_bytes - (double)old_bytes) / (1ull << 30));
    if (changed == 0) {
        printf("nothing to do: no tensor type changes requested\n");
        return 0;
    }
    if (dry_run) return 0;

    /* Output layout (mirrors deepseek4-quantize's build_output_context). */
    size_t tensor_info = 0;
    size_t off = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        t->new_offset = off;
        off += ds4q_pad(t->new_size, g.alignment);
        tensor_info += sizeof(uint64_t) + strlen(t->name) + 4 +
                       (size_t)t->n_dims * 8 + 4 + 8;
    }
    const size_t meta_size = 4 + 4 + 8 + 8 + g.kv_raw_len + tensor_info;
    const size_t data_offset = ds4q_pad(meta_size, g.alignment);

    if (!overwrite) {
        FILE *probe = fopen(out_path, "rb");
        if (probe) {
            fclose(probe);
            die("output '%s' exists (use --overwrite)", out_path);
        }
    }
    FILE *in_fp = fopen(in_path, "rb");
    if (!in_fp) die_errno("open input", in_path);
    FILE *fp = fopen(out_path, "wb");
    if (!fp) die_errno("open output", out_path);

    if (fwrite("GGUF", 1, 4, fp) != 4) die("write GGUF magic failed");
    write_u32(fp, g.version);
    write_u64(fp, g.n_tensors);
    write_u64(fp, g.n_kv);
    if (fwrite(g.kv_raw, 1, g.kv_raw_len, fp) != g.kv_raw_len) die("write GGUF KV failed");
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const tensor_meta *t = &g.tensors[i];
        write_gguf_string(fp, t->name);
        write_u32(fp, (uint32_t)t->n_dims);
        for (int j = 0; j < t->n_dims; j++) write_u64(fp, (uint64_t)t->ne[j]);
        write_u32(fp, (uint32_t)t->new_type);
        write_u64(fp, t->new_offset);
    }
    long pos = ftell(fp);
    if (pos < 0 || (size_t)pos > data_offset) die("GGUF metadata larger than planned");
    write_padding(fp, data_offset - (size_t)pos);

    uint8_t *copy_buf = xmalloc(64 << 20);
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        if (fseeko(in_fp, (off_t)(g.data_offset + t->old_offset), SEEK_SET) != 0) {
            die_errno("seek input tensor", t->name);
        }
        if (t->new_type == t->type) {
            size_t left = t->size;
            while (left) {
                size_t chunk = left > (64 << 20) ? (64 << 20) : left;
                if (fread(copy_buf, 1, chunk, in_fp) != chunk) {
                    die_errno("read tensor", t->name);
                }
                if (fwrite(copy_buf, 1, chunk, fp) != chunk) die_errno("write tensor", out_path);
                left -= chunk;
            }
        } else {
            uint8_t *raw = xmalloc(t->size);
            if (fread(raw, 1, t->size, in_fp) != t->size) die_errno("read tensor", t->name);
            float *f32 = tensor_to_f32(t, raw);
            free(raw);
            uint8_t *enc = xmalloc(t->new_size);
            const int64_t n_row = t->n_dims > 1 ? t->ne[1] : 1;
            ds4q_quantize_init(t->new_type);
            const float *imat = imatrix_path ? imatrix_find(&im, t) : NULL;
            size_t written = ds4q_quantize_chunk(t->new_type, f32, enc, 0,
                                                 n_row, t->ne[0], imat);
            if (written != t->new_size) {
                die("%s: encoded %zu bytes, expected %zu", t->name, written, t->new_size);
            }
            free(f32);
            if (fwrite(enc, 1, t->new_size, fp) != t->new_size) die_errno("write tensor", out_path);
            free(enc);
            fprintf(stderr, "[%" PRIu64 "/%" PRIu64 "] %s -> %s\n",
                    i + 1, g.n_tensors, t->name, ds4q_type_name(t->new_type));
        }
        size_t padded = ds4q_pad(t->new_size, g.alignment);
        write_padding(fp, padded - t->new_size);
    }
    if (fclose(fp) != 0) die_errno("close output", out_path);
    fclose(in_fp);
    printf("wrote %s (%zu tensors, %zu changed)\n", out_path, (size_t)g.n_tensors, changed);

    if (verify) {
        /* Checksum every copied tensor against the input; re-encoded tensors
         * are checked for size only (their bytes are new by design). */
        printf("verifying copied tensors against input...\n");
        FILE *a = fopen(in_path, "rb");
        FILE *b = fopen(out_path, "rb");
        if (!a || !b) die_errno("open for verify", !a ? in_path : out_path);
        uint64_t mismatches = 0;
        for (uint64_t i = 0; i < g.n_tensors; i++) {
            const tensor_meta *t = &g.tensors[i];
            if (t->new_type != t->type) continue;
            if (fseeko(a, (off_t)(g.data_offset + t->old_offset), SEEK_SET) != 0 ||
                fseeko(b, (off_t)(data_offset + t->new_offset), SEEK_SET) != 0) {
                die_errno("seek for verify", t->name);
            }
            uint64_t ha = 0, hb = 0;
            size_t left = t->size;
            while (left) {
                size_t chunk = left > (64 << 20) ? (64 << 20) : left;
                if (fread(copy_buf, 1, chunk, a) != chunk) die_errno("verify read a", t->name);
                ha = fnv1a64_bytes(copy_buf, chunk) ^ (ha * 1099511628211ull);
                if (fread(copy_buf, 1, chunk, b) != chunk) die_errno("verify read b", t->name);
                hb = fnv1a64_bytes(copy_buf, chunk) ^ (hb * 1099511628211ull);
                left -= chunk;
            }
            if (ha != hb) {
                printf("MISMATCH: %s\n", t->name);
                mismatches++;
            }
        }
        fclose(a);
        fclose(b);
        if (mismatches) die("verification failed for %" PRIu64 " copied tensors", mismatches);
        printf("verify: all copied tensors byte-identical\n");
    }
    free(copy_buf);
    return 0;
}
