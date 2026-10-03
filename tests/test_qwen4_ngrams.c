#define DS4_NO_GPU
#include <sys/types.h>
#include <unistd.h>
static ssize_t test_pread(int, void *, size_t, off_t);
#define pread test_pread
#ifndef __APPLE__
#include <pthread.h>
static int test_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
#define pthread_create test_pthread_create
#endif
#include "../ds4.c"
#undef pread
/* Inject failures through the same pread loop used by both readers. Atomic
 * selection makes exactly one worker fail, without depending on its order. */
static volatile int read_fault, read_fault_calls;
static ssize_t test_pread(int fd, void *buf, size_t size, off_t off) {
    if (read_fault) {
        const int call = __sync_fetch_and_add(&read_fault_calls, 1);
        if (!call) {
            if (read_fault == 3) return 0;
            errno = read_fault == 1 ? EINTR : EIO;
            return -1;
        }
        if (read_fault == 1 && size > 37) size = 37;
    }
    return pread(fd, buf, size, off);
}
#ifndef __APPLE__
#undef pthread_create
static int thread_budget = -1;
static int test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                               void *(*start)(void *), void *arg) {
    if (!thread_budget) return EAGAIN;
    if (thread_budget > 0) thread_budget--;
    return pthread_create(thread,attr,start,arg);
}
#endif
#include <assert.h>
#include <sys/wait.h>

static void u32(FILE *f, uint32_t n) { assert(fwrite(&n, 4, 1, f) == 1); }
static void u64(FILE *f, uint64_t n) { assert(fwrite(&n, 8, 1, f) == 1); }
static void str(FILE *f, const char *s) { u64(f, strlen(s)); assert(fwrite(s, strlen(s), 1, f) == 1); }

static uint16_t value(size_t row, size_t col) {
    const uint16_t edge[] = {0, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f7f};
    return col < sizeof(edge)/sizeof(*edge) ? edge[col] : (uint16_t)(0x3000 + (row*17+col) % 4096);
}

static void fixture(const char *path, bool bad_alignment, uint32_t type) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite("GGUF", 4, 1, f) == 1);
    u32(f, 3); u64(f, 2); u64(f, 1);
    str(f, "general.architecture"); u32(f, 8); str(f, "qwen4exp");
    str(f, "token_embd.weight"); u32(f, 2); u64(f, 1); u64(f, 1); u32(f, 0); u64(f, 0);
    str(f, "per_layer_token_embd.weight"); u32(f, 2); u64(f, 160); u64(f, 1000); u32(f, type);
    uint64_t start = ((uint64_t)ftell(f) + 8 + 31) / 32 * 32;
    u64(f, 65536 - start + (bad_alignment ? 1 : 0));
    assert(!fseek(f, (long)start, SEEK_SET));
    u32(f, 0x3f800000);
    assert(!fseek(f, 65536, SEEK_SET));
    for (size_t r = 0; r < 1000; r++) {
        for (size_t c = 0; c < 160; c++) {
            uint16_t v = value(r, c);
            uint8_t bytes[2] = {v & 255, v >> 8};
            assert(fwrite(bytes, 2, 1, f) == 1);
        }
    }
    if (bad_alignment) fputc(0, f);
    assert(!fclose(f));
}

static void exact_read(const ds4_model *m, const uint32_t *rows, size_t n, float *out) {
    memset(out, 0xa5, (n * 160 + 16) * sizeof(*out));
    assert(qwen4_ngram_read(m, rows, n, out));
    for (size_t i = 0; i < n; i++) for (size_t c = 0; c < 160; c++) {
        uint32_t bits;
        memcpy(&bits, out + i * 160 + c, 4);
        assert(bits == (uint32_t)value(rows[i], c) << 16);
    }
    for (size_t i = 0; i < 16; i++) {
        uint32_t bits;
        memcpy(&bits, out + n * 160 + i, 4);
        assert(bits == 0xa5a5a5a5u);
    }
}

static void reader_recovery_cases(const ds4_model *m, uint32_t *rows, float *out) {
    /* Each sorted duplicate run straddles a nominal 256-row worker
     * boundary. An odd permutation checks original-output scattering. */
    for (size_t i = 0; i < 4096; i++) rows[i] = ((i * 173u % 4096) / 257) * 7;
    exact_read(m, rows, 4096, out);
    /* One run crosses all fifteen boundaries and the 4096-row batch. */
    for (size_t i = 0; i < 4097; i++) rows[i] = 317;
    exact_read(m, rows, 4097, out);
    /* Workers cover 62/63 consecutive 320-byte rows (>8192 bytes);
     * the final row of every partition must be read. */
    for (size_t i = 0; i < 1000; i++) rows[i] = (i * 173u) % 1000;
    exact_read(m, rows, 1000, out);
    /* Disjoint spans and duplicates must not cause gaps to be decoded. */
    for (size_t i = 0; i < 4096; i++) rows[i] = i % 3 == 0 ? 999 : (i * 173u) % 100 * 7;
    exact_read(m, rows, 4096, out);
    for (int fault = 1; fault <= 3; fault++) {
        read_fault_calls = 0; read_fault = fault;
        bool ok = qwen4_ngram_read(m, rows, 4096, out);
        const int error = errno;
        read_fault = 0;
        assert(read_fault_calls > 0);
        assert(fault == 1 ? ok : !ok && error == EIO);
        if (ok) for (size_t i = 0; i < 4096; i++) for (size_t c = 0; c < 160; c++) {
            uint32_t bits;
            memcpy(&bits, out + i * 160 + c, 4);
            assert(bits == (uint32_t)value(rows[i], c) << 16);
        }
        /* The successful retry must refill every destination exactly. */
        exact_read(m, rows, 4096, out);
    }
}

int main(void) {
    char path[] = "/tmp/ds4-qwen-ngrams-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    fixture(path, false, 30);
    ds4_model m;
    model_open(&m, path, false, false);
    assert(m.size == 65536 && m.file_size == 65536 + 320000);
    assert(m.ngram_fd >= 0 && m.ngram_tensor && m.max_tensor_bytes == 4);
    assert(*(const float *)tensor_data(&m, model_find_tensor(&m, "token_embd.weight")) == 1.0f);
    enum { N = 9001 };
    uint32_t *rows = malloc(N * sizeof(*rows));
    float *out = malloc(((size_t)N * 160 + 16) * sizeof(*out));
    assert(rows && out);
    for (size_t i = 0; i < N; i++) rows[i] = (i * 173u) % 1000;
    rows[1] = rows[0]; rows[N-1] = 999;
    const size_t sizes[] = {0,16,255,256,257,4095,4096,4097,N};
    for (size_t ni = 0; ni < sizeof(sizes)/sizeof(*sizes); ni++) {
        size_t n = sizes[ni];
        assert(qwen4_ngram_read(&m, rows, n, out));
        for (size_t i = 0; i < n; i++) {
            for (size_t c = 0; c < 160; c++) {
                uint32_t bits, expected = (uint32_t)value(rows[i], c) << 16;
                memcpy(&bits, out + i*160+c, 4);
                assert(bits == expected);
            }
        }
    }
    reader_recovery_cases(&m, rows, out);
    for (size_t i = 0; i < N; i++) rows[i] = (i * 173u) % 1000;
    rows[1] = rows[0]; rows[N-1] = 999;
#ifndef __APPLE__
    const int budgets[] = {0,1,7};
    for (size_t bi = 0; bi < sizeof(budgets)/sizeof(*budgets); bi++) {
        thread_budget = budgets[bi];
        memset(out,0xff,(size_t)N*160*sizeof(*out));
        assert(qwen4_ngram_read(&m,rows,N,out));
        for (size_t i = 0; i < N; i++) for (size_t c = 0; c < 160; c++) {
            uint32_t bits;
            memcpy(&bits,out+i*160+c,4);
            assert(bits == (uint32_t)value(rows[i],c)<<16);
        }
    }
    thread_budget = -1;
#endif
    uint32_t invalid = 1000;
    assert(!qwen4_ngram_read(&m, &invalid, 1, out) && errno == EINVAL);
    assert(!qwen4_ngram_read(&m, rows, SIZE_MAX, out) && errno == EINVAL);
    assert(!qwen4_ngram_read(&m, NULL, 1, out) && errno == EINVAL);
    fd = open(path, O_WRONLY);
    assert(fd >= 0);
    uint8_t nan[2] = {0xc0, 0x7f};
    assert(pwrite(fd, nan, 2, 65536) == 2);
    uint32_t first = 0;
    assert(!qwen4_ngram_read(&m, &first, 1, out) && errno == EDOM);
    assert(!qwen4_ngram_read(&m, rows, N, out) && errno == EDOM);
    uint16_t zero = 0;
    assert(pwrite(fd,&zero,2,65536) == 2);
    assert(qwen4_ngram_read(&m,rows,N,out));
    assert(!ftruncate(fd, 65536 + 320000 - 1));
    uint32_t last = 999;
    assert(!qwen4_ngram_read(&m, &last, 1, out) && errno == EIO);
    assert(!qwen4_ngram_read(&m,rows,N,out) && errno == EIO);
    /* Repair the missing byte without reopening the disk-backed model. */
    const uint8_t final_byte = (uint8_t)(value(999, 159) >> 8);
    assert(pwrite(fd, &final_byte, 1, 65536 + 320000 - 1) == 1);
    exact_read(&m, rows, N, out);
    close(fd);
    model_close(&m);
    assert(m.ngram_fd == -1 && !m.ngram_tensor);
    for (int bad = 0; bad < 4; bad++) {
        fixture(path, bad == 0, bad == 1 ? 1 : bad == 2 ? 3 : 30);
        if (bad == 3) assert(!truncate(path, 65536 + 320000 - 1));
        pid_t pid = fork();
        assert(pid >= 0);
        if (!pid) { model_open(&m, path, false, false); _exit(0); }
        int status;
        assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) != 0);
    }
    free(rows); free(out); unlink(path);
    puts("Qwen BF16 n-grams: disk-only mapping, exact reads, batches and errors OK");
    return 0;
}
