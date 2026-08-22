/* C1 probe: CPU<->GPU rendezvous latency of a Shared-memory flag spin at
 * decode cadence (NOT MTLSharedEvent).  One GPU kernel spins on a
 * CPU-written flag, answers on a second flag, loops; the CPU measures the
 * full write->wake->answer->wake round trip per iteration. */

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <string.h>

static const char *kSource =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void c1_spin(device atomic_uint *flag_in,\n"
    "                    device atomic_uint *flag_out,\n"
    "                    device uint *spin_total,\n"
    "                    constant uint &iters) {\n"
    "    uint total = 0;\n"
    "    for (uint seq = 1; seq <= iters; seq++) {\n"
    "        uint spins = 0;\n"
    "        while (atomic_load_explicit(flag_in, memory_order_relaxed) != seq) {\n"
    "            if (++spins > (1u << 28)) break;\n"
    "        }\n"
    "        total += spins;\n"
    "        atomic_store_explicit(flag_out, seq, memory_order_relaxed);\n"
    "    }\n"
    "    *spin_total = total;\n"
    "}\n";

static int cmp_doubles(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static double now_us(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * (double)tb.numer / (double)tb.denom / 1000.0;
}

int main(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { fprintf(stderr, "no Metal device\n"); return 1; }

    NSError *error = nil;
    MTLCompileOptions *opts = [MTLCompileOptions new];
    id<MTLLibrary> library =
        [device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                             options:opts
                               error:&error];
    if (!library) {
        fprintf(stderr, "library: %s\n", [[error localizedDescription] UTF8String]);
        return 1;
    }
    id<MTLFunction> fn = [library newFunctionWithName:@"c1_spin"];
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
        fprintf(stderr, "pso: %s\n", [[error localizedDescription] UTF8String]);
        return 1;
    }

    const uint32_t iters = 2000;
    id<MTLBuffer> flag_in =
        [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> flag_out =
        [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> spins =
        [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    volatile uint32_t *fi = (volatile uint32_t *)[flag_in contents];
    volatile uint32_t *fo = (volatile uint32_t *)[flag_out contents];
    __atomic_store_n(fi, 0, __ATOMIC_RELEASE); __atomic_store_n(fo, 0, __ATOMIC_RELEASE);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:flag_in offset:0 atIndex:0];
    [enc setBuffer:flag_out offset:0 atIndex:1];
    [enc setBuffer:spins offset:0 atIndex:2];
    [enc setBytes:&iters length:sizeof(iters) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];

    /* Warm-up handshake, then timed iterations. */
    double warm0 = now_us();
    uint32_t seq = 0;
    for (uint32_t i = 0; i < 32; i++) {
        __atomic_store_n(fi, ++seq, __ATOMIC_RELEASE);
        while (__atomic_load_n(fo, __ATOMIC_ACQUIRE) != seq) {}
    }
    double warm1 = now_us();

    double best = 1e30, total = 0.0;
    double *samples = malloc(iters * sizeof(double));
    uint32_t n_timed = iters - 32;
    for (uint32_t i = 0; i < n_timed; i++) {
        const double t0 = now_us();
        __atomic_store_n(fi, ++seq, __ATOMIC_RELEASE);
        while (__atomic_load_n(fo, __ATOMIC_ACQUIRE) != seq) {}
        const double t1 = now_us();
        samples[i] = t1 - t0;
        total += samples[i];
        if (samples[i] < best) best = samples[i];
    }
    [cb waitUntilCompleted];

    qsort(samples, n_timed, sizeof(double), cmp_doubles);

    printf("c1 flag-spin round trip (CPU write -> GPU wake -> GPU answer -> CPU wake):\n");
    printf("  handshake warm-up (32 iters): %.1f us total (%.2f us each)\n",
           warm1 - warm0, (warm1 - warm0) / 32.0);
    printf("  min %.2f us  median %.2f us  p90 %.2f us  mean %.2f us  (n=%u)\n",
           best, samples[n_timed / 2], samples[(n_timed * 9) / 10],
           total / (double)n_timed, n_timed);
    printf("  GPU-side spins per iter: %.1f\n",
           (double)*(uint32_t *)[spins contents] / (double)iters);
    return 0;
}
