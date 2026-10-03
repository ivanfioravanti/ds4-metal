/* clang -O2 -Wall -Wextra -Werror -fno-objc-arc -c
 *     speed-bench/benchmark_pool.m -o speed-bench/benchmark_pool.o
 * Link the harness object with -framework Foundation (already used by Metal). */
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "benchmark_pool.h"
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysctl.h>

void *ds4_bench_pool_create(void) {
    return [[NSAutoreleasePool alloc] init];
}

void ds4_bench_pool_drain(void *pool) {
    [(NSAutoreleasePool *)pool drain];
}

void ds4_bench_sha256(const void *data, size_t bytes, char hex[65]) {
    const unsigned char *p = data;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    while (bytes) {
        CC_LONG take = bytes > UINT32_MAX ? UINT32_MAX : (CC_LONG)bytes;
        CC_SHA256_Update(&ctx, p, take);
        p += take;
        bytes -= take;
    }
    CC_SHA256_Final(digest, &ctx);
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(digest); i++) {
        hex[2*i] = digits[digest[i] >> 4];
        hex[2*i+1] = digits[digest[i] & 15u];
    }
    hex[64] = '\0';
}

void ds4_bench_memory_sample(pid_t pid, ds4_bench_memory *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    struct rusage_info_v4 info = {0};
    if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&info) == 0) {
        out->process_valid = 1;
        out->resident_bytes = info.ri_resident_size;
        out->physical_footprint_bytes = info.ri_phys_footprint;
        out->peak_physical_footprint_bytes = info.ri_lifetime_max_phys_footprint;
        out->disk_read_bytes = info.ri_diskio_bytesread;
        out->disk_write_bytes = info.ri_diskio_byteswritten;
    }
    struct xsw_usage swap = {0};
    size_t bytes = sizeof(swap);
    if (sysctlbyname("vm.swapusage", &swap, &bytes, NULL, 0) == 0) {
        out->swap_usage_valid = 1;
        out->swap_used_bytes = swap.xsu_used;
        out->swap_total_bytes = swap.xsu_total;
    }
    vm_statistics64_data_t vm = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count) == KERN_SUCCESS) {
        out->vm_statistics_valid = 1;
        out->swapins_pages = vm.swapins;
        out->swapouts_pages = vm.swapouts;
        out->pageins_pages = vm.pageins;
        out->pageouts_pages = vm.pageouts;
        vm_size_t page_size;
        if (host_page_size(host, &page_size) == KERN_SUCCESS) out->page_size_bytes = page_size;
    }
    mach_port_deallocate(mach_task_self(), host);
}
