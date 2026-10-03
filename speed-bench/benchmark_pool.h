#ifndef DS4_BENCHMARK_POOL_H
#define DS4_BENCHMARK_POOL_H

#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

/* Harness-only Objective-C lifetime boundary. Compile benchmark_pool.m WITHOUT
 * ARC. Create/drain on the same thread and drain after completed GPU work and
 * session cleanup. Never drain a pool from inside a timed interval. */
void *ds4_bench_pool_create(void);
void ds4_bench_pool_drain(void *pool);

/* Untimed full-payload fingerprint for cross-build comparison. Supports inputs
 * larger than CommonCrypto's per-update 32-bit byte count. */
void ds4_bench_sha256(const void *data, size_t bytes, char hex[65]);

typedef struct {
    int process_valid;
    int swap_usage_valid;
    int vm_statistics_valid;
    uint64_t resident_bytes;
    uint64_t physical_footprint_bytes;
    uint64_t peak_physical_footprint_bytes;
    uint64_t disk_read_bytes;
    uint64_t disk_write_bytes;
    uint64_t swap_used_bytes;
    uint64_t swap_total_bytes;
    uint64_t swapins_pages;
    uint64_t swapouts_pages;
    uint64_t pageins_pages;
    uint64_t pageouts_pages;
    uint64_t page_size_bytes;
} ds4_bench_memory;

/* Actual process memory and disk-byte counters, plus GLOBAL swap counters.
 * This adds no GPU synchronization. Call outside timing. Invalid fields are
 * identified by the three validity flags, not inferred from zero values.
 * Disk bytes are NOT elapsed I/O time; use the n-gram diagnostic for that. */
void ds4_bench_memory_sample(pid_t pid, ds4_bench_memory *out);
#endif
