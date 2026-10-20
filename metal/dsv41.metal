// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_router {
    uint n_expert, top_k, has_bias;
    float scale;
};

// One-token router for up to 1024 experts in one dispatch: the softplus/sqrt
// probability transform, the biased top-k through the same bitonic network as
// kernel_argsort_f32_i32_desc (padding indices sort last), and the weights
// normalised as the generic chain does: gather, SIMD sum over the top_k lanes,
// clamp, divide, scale. Threadgroup memory holds nth ints and 2 * nth floats.
kernel void kernel_dsv41_router_one(
        constant ds4_metal_args_dsv41_router &args,
        device const float *logits,
        device const float *bias,
        device float *probs,
        device int32_t *selected,
        device float *weights,
        threadgroup int32_t *shmem_i32 [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint2 tpos [[thread_position_in_threadgroup]],
        uint2 tsize [[threads_per_threadgroup]]) {
    const uint col = tpos.x, ntg = tsize.x;
    const int width = (int)args.n_expert;
    logits += tg.y * args.n_expert;   /* one threadgroup per row */
    probs += tg.y * args.n_expert;
    selected += tg.y * args.top_k;
    weights += tg.y * args.top_k;
    threadgroup float *score = (threadgroup float *)(shmem_i32 + ntg);
    threadgroup float *prob = score + ntg;
    shmem_i32[col] = (int)col;
    if ((int)col < width) {
        const float x = logits[col];
        const float ex = exp(x);
        const float em = min(ex, 0.03125f);
        const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float p = sqrt(sp);
        probs[col] = p;
        prob[col] = p;
        score[col] = args.has_bias ? p + bias[col] : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 2; k <= (int)ntg; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = (int)col ^ j;
            if (ixj > (int)col) {
                const int a = shmem_i32[col], b = shmem_i32[ixj];
                const bool swap = ((int)col & k) == 0 ?
                    (a >= width || (b < width && score[a] < score[b])) :
                    (b >= width || (a < width && score[a] > score[b]));
                if (swap) { shmem_i32[col] = b; shmem_i32[ixj] = a; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (col < args.top_k) {
        const int32_t idx = shmem_i32[col];
        selected[col] = idx;
        const float w = prob[idx];
        const float total = clamp(simd_sum(w), 6.103515625e-5f, INFINITY);
        // two stores as in the generic chain: the quotient is rounded before the scale
        threadgroup volatile float *quotient = (threadgroup volatile float *)(prob + ntg) + col;
        *quotient = w / total;
        weights[col] = *quotient * args.scale;
    }
}

/* SIMD nominations for the unique finite top-six case; ties use the original
 * bitonic order. Keep six active lanes in the normalization reduction. */
template<bool COHERENT>
static inline float ds4_dsv41_router_logit(device const float *logits, uint i) {
    return COHERENT ? dsv41_coherent_load(logits + i) : logits[i];
}

/* kernel_dsv41_router_hier's body; COHERENT reads logits another threadgroup
 * of the same dispatch stored (kernel_dsv41_router_fused). */
template<bool COHERENT>
static inline void ds4_dsv41_router_hier_body(
        constant ds4_metal_args_dsv41_router & args,
        device const float *logits,
        device const float *bias,
        device       float *probs,
        device     int32_t *selected,
        device       float *weights,
        threadgroup   char *shmem,
        uint tid) {
    const uint lane = tid % 32u, sg = tid / 32u;
    threadgroup float *cscore = (threadgroup float *)shmem;
    threadgroup uint *cid = (threadgroup uint *)(shmem + 128u*4u);
    threadgroup uint *valids = (threadgroup uint *)(shmem + 256u*4u);
    float score = -INFINITY, prob = 0.f;
    bool valid = true;
    if (tid < 384u) {
        const float x = ds4_dsv41_router_logit<COHERENT>(logits, tid);
        const float ex = exp(x);
        const float em = min(ex, 0.03125f);
        const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        prob = sqrt(sp);
        probs[tid] = prob;
        score = args.has_bias ? prob + bias[tid] : prob;
        valid = isfinite(prob) && isfinite(score);
    }
    valid = simd_all(valid);
    if (lane == 0u) valids[sg] = valid;
    for (uint k=0; k<6u; ++k) {
        const float best = simd_max(score);
        const uint id = simd_min(score == best ? tid : 0xffffffffu);
        if (lane == 0u) {cscore[sg*6u+k] = best; cid[sg*6u+k] = id;}
        if (tid == id) score = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (sg == 0u) {
        float v[3]; uint ix[3];
        for (uint j=0;j<3u;++j) {v[j] = cscore[lane+32u*j];ix[j] = cid[lane+32u*j];}
        bool ok = args.n_expert == 384u && args.top_k == 6u && simd_all(lane >= 16u || valids[lane] != 0u);
        float selected_prob=0.f;
        for (uint k=0;k<6u;++k) {
            const float best=simd_max(max(v[0],max(v[1],v[2])));
            uint count=0,id=0xffffffffu;
            for (uint j=0;j<3u;++j) if(v[j]==best){++count;id=min(id,ix[j]);}
            count=simd_sum(count);id=simd_min(id);
            ok = ok && count == 1u && id < 384u;
            const uint safe=min(id,383u);
            if(lane==k){selected[k]=int(safe);selected_prob=probs[safe];}
            for(uint j=0;j<3u;++j) if(ix[j]==id)v[j]=-INFINITY;
        }
        if(lane<6u){
            const float total=simd_sum(selected_prob);
            const float clamped=clamp(total,6.103515625e-5f,INFINITY);
            if(lane<6u) ((threadgroup volatile float *)cscore)[lane]=selected_prob/clamped;
        }
        if(lane==0u)valids[0]=ok;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if(valids[0]){
        if(tid<6u) weights[tid]=((threadgroup volatile float *)cscore)[tid]*args.scale;
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int NT = 512;                 /* argsort threadgroup width */
    const int width = (int)args.n_expert;   /* 384 */

    threadgroup int32_t *sidx =
        (threadgroup int32_t *)shmem;
    threadgroup float *sscore =
        (threadgroup float *)(shmem + NT * sizeof(int32_t));
    threadgroup float *sprob =
        (threadgroup float *)(shmem + NT * sizeof(int32_t) + NT * sizeof(float));

    const int col = (int)tid;
    sidx[col] = col;
    if (col < width) {
        const float x = ds4_dsv41_router_logit<COHERENT>(logits, col);
        const float ex = exp(x);
        const float em = min(ex, 0.03125f);
        const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float p = sqrt(sp);
        probs[col] = p;
        sprob[col] = p;
        sscore[col] = args.has_bias ? p + bias[col] : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= NT; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (sidx[col] >= width ||
                        (sidx[ixj] < width &&
                         sscore[sidx[col]] < sscore[sidx[ixj]])) {
                        const int32_t t = sidx[col];
                        sidx[col] = sidx[ixj];
                        sidx[ixj] = t;
                    }
                } else {
                    if (sidx[ixj] >= width ||
                        (sidx[col] < width &&
                         sscore[sidx[col]] > sscore[sidx[ixj]])) {
                        const int32_t t = sidx[col];
                        sidx[col] = sidx[ixj];
                        sidx[ixj] = t;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int used = (int)args.top_k;
    if (col < used) {
        selected[col] = sidx[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* sum_rows stages element j in lane j of one simdgroup and reduces with
     * simd_sum; reproduce that lane layout rather than a serial sum. */
    /* sum_rows is dispatched with exactly n_expert_used threads (nth is clamped
     * to ne00 in ds4_gpu_encode_sum_rows_f32), so its simd_sum runs with only
     * those lanes active; reproduce that by calling simd_sum under the same
     * lane predicate rather than padding to a full simdgroup.  The divide and
     * the scale were two kernels with an f32 store between them, so the
     * intermediate is round-tripped through threadgroup memory to keep that
     * rounding boundary. */
    threadgroup volatile float *wtmp =
        (threadgroup volatile float *)(shmem + NT * sizeof(int32_t));
    if (col < used) {
        const float wv = sprob[sidx[col]];
        const float total = simd_sum(wv);
        const float clamped = clamp(total, 6.103515625e-5f, INFINITY);
        wtmp[col] = wv / clamped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (col < used) {
        weights[col] = wtmp[col] * args.scale;
    }
}

kernel void kernel_dsv41_router_hier(
        constant ds4_metal_args_dsv41_router & args,
        device const float *logits,
        device const float *bias,
        device       float *probs,
        device     int32_t *selected,
        device       float *weights,
        threadgroup   char *shmem [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    ds4_dsv41_router_hier_body<false>(args, logits, bias, probs, selected, weights, shmem, tid);
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
    uint round_all;
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong head = ((ulong)group.y * args.heads + group.x) * args.width;
    if (args.round_all) {
        for (uint k = lane; k + 64u < args.width; k += 32u) x[head + k] = dsv41_bf16(x[head + k]);
    }
    const ulong i = head + args.width - 64u + 2u * lane;
    float re = x[i], im = x[i + 1u];
    if (args.round_all) { re = dsv41_bf16(re); im = dsv41_bf16(im); }
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_rope_quantize {
    uint width, start, inverse, mode;
    float frequencies[32];
};

/* One row's rope, quantization and store into a cache row: the rotation of
 * the last 64 columns as kernel_dsv41_rope applies it, then the block
 * quantization of kernel_dsv41_quantize, written to dst. */
static inline void dsv41_rope_quantize_body(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group,
        uint lane) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? x[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float theta = float(args.start) * args.frequencies[(column - rotary) >> 1u];
        const float c = precise::cos(theta);
        const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        /* the contraction kernel_dsv41_rope compiles to */
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) dst[column] = dsv41_bf16(result);
}

kernel void kernel_dsv41_rope_quantize(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? x[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float theta = float(args.start) * args.frequencies[(column - rotary) >> 1u];
        const float c = precise::cos(theta);
        const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        // the contraction kernel_dsv41_rope compiles to
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) dst[column] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_norm_pair {
    uint n0, n1, ntg0, ntg1;
    float eps;
};

// Two weighted RMS norms in one dispatch, each row on its own thread count
// (ntg0 threads then ntg1), so every reduction matches the single-row kernel.
kernel void kernel_dsv41_norm_pair(
        constant ds4_metal_args_dsv41_norm_pair &args,
        device const float4 *x0,
        device const float4 *w0,
        device float4 *y0,
        device const float4 *x1,
        device const float4 *w1,
        device float4 *y1,
        threadgroup float *shmem [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort2 tpos [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort tid = tpos.x;
    const bool second = tid >= args.ntg0;
    const uint ntg = second ? args.ntg1 : args.ntg0;
    const uint t = second ? tid - args.ntg0 : tid;
    const uint sg = second ? sgitg - args.ntg0 / 32u : sgitg;
    const int n = int(second ? args.n1 : args.n0);
    const int ne00_t = n / 4;
    const uint row = tg.y;   /* grid row = activation row */
    device const float4 *x = (second ? x1 : x0) + (ulong)row * ne00_t;
    device const float4 *w = second ? w1 : w0;
    device float4 *y = (second ? y1 : y0) + (ulong)row * ne00_t;
    threadgroup float *sh = shmem + (second ? 32u : 0u);
    if (sg == 0) {
        sh[tiisg] = 0.0f;
    }
    float sumf = 0.0f;
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        sumf += dot(x[i], x[i]);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sh[sg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = sh[tiisg];
    sumf = simd_sum(sumf);
    const float mean  = sumf/n;
    const float scale = 1.0f/sqrt(mean + args.eps);
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        float4 v = x[i]*scale;
        v = v*w[i];
        y[i] = ds4_bf16_round(v);
    }
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    /* Issue eight iterations' loads before accumulating them in the original
     * order: one simdgroup per stream was waiting out a DRAM round trip per
     * step. Each lane's sums see the same operands in the same order. */
    uint i = lane;
    for (; i + 7u * 32u < args.width; i += 8u * 32u) {
        float h[8], k[8], w[8];
        for (uint u = 0; u < 8u; u++) {
            const uint j = i + u * 32u;
            const uint wi = group.y * args.width + j;
            h[u] = residual[offset + j];
            k[u] = dsv41_bf16(kv[key_offset + j]);
            w[u] = q_weight[wi] * k_weight[wi];
        }
        for (uint u = 0; u < 8u; u++) {
            h2 += h[u] * h[u];
            k2 += k[u] * k[u];
            dot += h[u] * w[u] * k[u];
        }
    }
    for (; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    uint o = lane;
    for (; o + 7u * 32u < args.width; o += 8u * 32u) {
        float r[8], v[8];
        for (uint u = 0; u < 8u; u++) { r[u] = residual[offset + o + u * 32u]; v[u] = kv[value_offset + o + u * 32u]; }
        for (uint u = 0; u < 8u; u++)
            residual[offset + o + u * 32u] = dsv41_bf16(r[u] + gate * dsv41_bf16(v[u]));
    }
    for (; o < args.width; o += 32u)
        residual[offset + o] = dsv41_bf16(residual[offset + o] +
            gate * dsv41_bf16(kv[value_offset + o]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_indexer_all {
    uint rows;
    uint start;
    uint ratio;
    uint top_k;
};

// Rows with at most top_k visible keys attend to all of them.
kernel void kernel_dsv41_indexer_all(
        constant ds4_metal_args_dsv41_indexer_all &args,
        device int *ids,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.top_k || index.y >= args.rows) return;
    const uint visible = (args.start + index.y + 1u) / args.ratio;
    if (index.x < min(visible, args.top_k))
        ids[(ulong)index.y * args.top_k + index.x] = (int)index.x;
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
    uint mask_stride;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * args.mask_stride + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

/* The 2048 candidate blocks of a 0/-inf mask in ascending order; one
 * threadgroup of 1024 threads, four mask entries per thread and step. */
kernel void kernel_dsv41_candidate_list(
        constant uint &blocks,
        device const float *mask,
        device uint *list,
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint part[33];
    uint base = 0u;
    for (uint b0 = 0u; b0 < blocks; b0 += 4096u) {
        const uint b = b0 + tid * 4u;
        bool keep[4];
        uint count = 0u;
        for (uint j = 0u; j < 4u; j++) {
            keep[j] = b + j < blocks && mask[b + j] == 0.0f;
            count += keep[j] ? 1u : 0u;
        }
        const uint before = simd_prefix_exclusive_sum(count);
        if (lane == 31u) part[simd] = before + count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            const uint c = part[lane], c_before = simd_prefix_exclusive_sum(c);
            part[lane] = c_before;
            if (lane == 31u) part[32] = c_before + c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint at = base + part[simd] + before;
        for (uint j = 0u; j < 4u; j++) {
            if (keep[j] && at < 2048u) list[at] = b + j;
            at += keep[j] ? 1u : 0u;
        }
        base += part[32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* Positions in the candidate list back to key ids, in place and in order. */
kernel void kernel_dsv41_candidate_ids(
        constant uint &count,
        device const uint *list,
        device int *selected,
        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint p = (uint)selected[i];
    selected[i] = (int)(list[p >> 3] * 8u + (p & 7u));
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

// DSpark: the draft head reads the mean over the copies of the stream entering
// each target layer, packed side by side: out[t][out_off + j].
struct ds4_metal_args_dsv41_hc_mean {
    uint dim;
    uint hc;
    uint out_stride;
    uint out_off;
};

kernel void kernel_dsv41_hc_mean(
        constant ds4_metal_args_dsv41_hc_mean &args,
        device const float *stream,   // [rows, hc, dim]
        device float *out,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.dim) return;
    device const float *row = stream + (ulong)index.y * args.hc * args.dim;
    float acc = 0.0f;
    for (uint c = 0; c < args.hc; c++) acc += row[c * args.dim + index.x];
    out[(ulong)index.y * args.out_stride + args.out_off + index.x] = acc / (float)args.hc;
}

// The Markov head: position `step`'s logits gain head[v] . embed[token[step]] and its
// draft is the argmax; the confidence is proj . [x[step], embed[token[step]]].  The
// chain runs on the device, one part/final pair per position, so no token comes back
// to the host in between.  Tables are [vocab, rank] rows, F16 or F32.
struct ds4_metal_args_dsv41_markov {
    uint vocab;
    uint rank;
    uint dim;
    uint step;
    uint n_parts;
    uint f16;
};

static inline float dsv41_markov_tab(device const void *p, ulong i, uint f16) {
    return f16 ? (float)((device const half *)p)[i] : ((device const float *)p)[i];
}

#define DSV41_MARKOV_MAX_PER_LANE 16u   // rank <= 512

// one simdgroup per row stride, lanes over the rank; ties go to the lower index, NaN loses
kernel void kernel_dsv41_markov_part(
        constant ds4_metal_args_dsv41_markov &a,
        device const float *logits,   // [block, vocab]
        device const void *embed,
        device const void *head,
        device const int *tokens,     // [block + 1]
        device float *part_val,       // [n_parts]
        device int *part_idx,
        threadgroup float *e [[threadgroup(0)]],  // [rank]
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup int si[32];
    const uint per = a.rank / 32u;
    if (per == 0u || per > DSV41_MARKOV_MAX_PER_LANE) return;
    const int prev = tokens[a.step];
    for (uint r = tid; r < a.rank; r += ntg) e[r] = dsv41_markov_tab(embed, (ulong)prev * a.rank + r, a.f16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // a lane owns `per` consecutive ranks, so a row is one contiguous load per lane; four
    // rows go per step so their loads overlap
    float ev[DSV41_MARKOV_MAX_PER_LANE];
    for (uint i = 0; i < DSV41_MARKOV_MAX_PER_LANE; i++) ev[i] = i < per ? e[lane * per + i] : 0.0f;
    device const float *lg = logits + (ulong)a.step * a.vocab;
    float best = -INFINITY;
    int bi = -1;
    const uint stride = a.n_parts * nsg;
    for (uint v0 = tgpig * nsg + sg; v0 < a.vocab; v0 += 4u * stride) {
        float p4[4];
        for (ushort j = 0; j < 4; j++) {
            const uint v = min(v0 + j * stride, a.vocab - 1u);
            const ulong base = (ulong)v * a.rank + lane * per;
            float p = 0.0f;
            if (a.f16 && per == 8u) {
                device const half4 *hp = (device const half4 *)((device const half *)head + base);
                const float4 h0 = float4(hp[0]), h1 = float4(hp[1]);
                p = dot(h0, float4(ev[0], ev[1], ev[2], ev[3])) + dot(h1, float4(ev[4], ev[5], ev[6], ev[7]));
            } else {
                for (uint i = 0; i < DSV41_MARKOV_MAX_PER_LANE; i++) {
                    if (i < per) p = fma(dsv41_markov_tab(head, base + i, a.f16), ev[i], p);
                }
            }
            p4[j] = p;
        }
        for (ushort j = 0; j < 4; j++) {
            const uint v = v0 + j * stride;
            if (v >= a.vocab) break;
            const float p = simd_sum(p4[j]) + lg[v];
            if (p > best || (p == best && (int)v < bi)) { best = p; bi = (int)v; }
        }
    }
    if (lane == 0) { sv[sg] = best; si[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint k = 1; k < nsg; k++) {
            if (si[k] >= 0 && (sv[k] > best || (sv[k] == best && si[k] < bi))) { best = sv[k]; bi = si[k]; }
        }
        part_val[tgpig] = best;
        part_idx[tgpig] = bi;
    }
}

kernel void kernel_dsv41_markov_final(
        constant ds4_metal_args_dsv41_markov &a,
        device const float *part_val,
        device const int *part_idx,
        device int *tokens,           // [block + 1]: tokens[step + 1] is written
        device const float *x,        // [block, dim], the head's input before its norm
        device const void *embed,
        device const float *conf_proj, // [dim + rank]
        device float *conf,           // [block]
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort nsg  [[simdgroups_per_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup int si[32];
    float best = -INFINITY;
    int bi = -1;
    for (uint p = tid; p < a.n_parts; p += ntg) {
        const float v = part_val[p];
        const int i = part_idx[p];
        if (i >= 0 && (v > best || (v == best && i < bi))) { best = v; bi = i; }
    }
    for (uint o = 16; o > 0; o >>= 1) {
        const float ov = simd_shuffle_down(best, o);
        const int oi = simd_shuffle_down(bi, o);
        if (oi >= 0 && (ov > best || (ov == best && oi < bi))) { best = ov; bi = oi; }
    }
    if (lane == 0) { sv[sg] = best; si[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint k = 1; k < nsg; k++) {
            if (si[k] >= 0 && (sv[k] > best || (sv[k] == best && si[k] < bi))) { best = sv[k]; bi = si[k]; }
        }
        tokens[a.step + 1u] = bi < 0 ? 0 : bi;
    }
    const int prev = tokens[a.step];
    float acc = 0.0f;
    for (uint d = tid; d < a.dim; d += ntg) acc = fma(conf_proj[d], x[(ulong)a.step * a.dim + d], acc);
    for (uint r = tid; r < a.rank; r += ntg) {
        acc = fma(conf_proj[a.dim + r], dsv41_markov_tab(embed, (ulong)prev * a.rank + r, a.f16), acc);
    }
    acc = simd_sum(acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sv[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float c = 0.0f;
        for (uint k = 0; k < nsg; k++) c += sv[k];
        conf[a.step] = c;
    }
}

// Independent residual collapse can run before the mixer. The mixer and its
// Sinkhorn tail then ride the larger query grid, retaining both reference trees.
// Keep the original block helper whole, including its small norm recomputation:
// removing that code changed the compiled Sinkhorn result in exactness tests. The
// first collapse supplies query inputs; the duplicate work is off that dependency.
struct ds4_deferred_hc_args {
    int32_t n_embd, n_hc, sinkhorn_iters;
    float hc_eps, norm_eps;
};
kernel void kernel_dsv41_deferred_collapse(
        constant ds4_deferred_hc_args &args,
        device const float *x, device const float *pre,
        device float *collapsed, device const float *norm_weight,
        device float *norm_dst, device atomic_uint *counter,
        threadgroup float *shared [[threadgroup(0)]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {

    const uint n_embd = (uint)args.n_embd;
    const uint n4 = n_embd >> 2;

    threadgroup float4 *row = (threadgroup float4 *)shared;
    threadgroup float  *sum_shmem = shared + n_embd;

    /* kernel_rms_norm_fuse_impl zeroes its 32 accumulator slots from
     * simdgroup 0 before the parallel sum. */
    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    /* --- Collapse with the preceding sublayer's pre weights, then its BF16
     * boundary, and the weighted RMS partial sum over the rounded values. --- */
    const float w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    device const float4 *x0 = (device const float4 *)(x + 0u * n_embd);
    device const float4 *x1 = (device const float4 *)(x + 1u * n_embd);
    device const float4 *x2 = (device const float4 *)(x + 2u * n_embd);
    device const float4 *x3 = (device const float4 *)(x + 3u * n_embd);
    device float4 *collapsed4 = (device float4 *)collapsed;

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        float4 acc = 0.0f;
        acc += x0[i] * w0;
        acc += x1[i] * w1;
        acc += x2[i] * w2;
        acc += x3[i] * w3;
        const float4 v = float4(dsv41_bf16(acc.x), dsv41_bf16(acc.y),
                                dsv41_bf16(acc.z), dsv41_bf16(acc.w));
        collapsed4[i] = v;
        row[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf / (float)args.n_embd;
    const float rscale = 1.0f / sqrt(mean + args.norm_eps);

    device const float4 *w = (device const float4 *)norm_weight;
    device float4 *y = (device float4 *)norm_dst;
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = (row[i] * rscale) * w[i];
        y[i] = float4(dsv41_bf16(v.x), dsv41_bf16(v.y),
                      dsv41_bf16(v.z), dsv41_bf16(v.w));
    }

    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);

}

kernel void kernel_dsv41_query_deferred_hc(
        constant ds4_metal_args_hc_norm_mix &hc [[buffer(0)]],
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm &split_args [[buffer(1)]],
        device const char *residual [[buffer(2)]], device const char *hc_weight [[buffer(3)]],
        device char *mix [[buffer(4)]], device const float *scale [[buffer(5)]],
        device const float *base [[buffer(6)]], device char *split [[buffer(7)]],
        device atomic_uint *completion [[buffer(8)]],
        constant ds4_metal_args_mul_mv &mv [[buffer(9)]],
        device const char *weight [[buffer(10)]], device const char *x [[buffer(11)]],
        device float *q [[buffer(12)]], constant ds4_metal_args_mv_rope &rope [[buffer(13)]],
        device char *collapsed [[buffer(14)]], device const char *norm_weight [[buffer(15)]],
        device char *norm [[buffer(16)]], device const float *pre [[buffer(17)]],
        constant ds4_metal_args_dsv41_rope_quantize &kv_args [[buffer(18)]],
        device const float *kv [[buffer(19)]], device float *cache [[buffer(20)]],
        constant uint &store_kv [[buffer(21)]],
        threadgroup char *shmem [[threadgroup(0)]], uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort simd [[simdgroup_index_in_threadgroup]]) {
    if (group.x < 16u) {
        dsv41_hc_block_input_body<2>(hc, split_args, residual, hc_weight, mix,
            scale, base, split, collapsed, norm_weight, norm, pre, completion, shmem,
            group, lane, simd);
        return;
    }
    if (store_kv && group.x < 18u) {
        dsv41_rope_quantize_body(kv_args, kv, cache, (group.x - 16u) * 8u + simd, lane);
        return;
    }
    const uint tile = group.x - 16u - (store_kv ? 2u : 0u);
    const float value = ds4_q8_short_value<40, false>(mv, weight, x, uint3(tile*2u,0,0), lane, simd);
    threadgroup float *pairs = (threadgroup float *)shmem;
    if (lane == 0) pairs[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && (simd & 1u) == 0) {
        const uint row = tile * 8u + simd, col = row % rope.width;
        const float re = pairs[simd], im = pairs[simd + 1u];
        if (col >= rope.width - 64u) {
            const float theta = float(rope.start) * rope.frequencies[(col - (rope.width - 64u)) >> 1u];
            const float c = precise::cos(theta);
            const float sn = precise::sin(theta);
            q[row] = ds4_bf16_round(fma(re, c, -(im * sn)));
            q[row + 1u] = ds4_bf16_round(fma(im, c, re * sn));
        } else { q[row] = re; q[row + 1u] = im; }
    }
}

// Two token rows share each lane's half4 weight staging. Each virtual
// row retains NSG, NR0, ib0/stride, four dot accumulation steps, then the exact
// existing two-level SIMD reduction. Inputs remain float4, never narrowed.
// Eligibility requires K divisible by 32 and output rows divisible by NR0.
template<short NR0>
static void dsv41_project_f16_rows2_impl(
        constant ds4_metal_args_mul_mv &args,
        device const char *weights, device const char *input, device char *output,
        threadgroup char *scratch, uint2 group, ushort lane, ushort sg) {
    constexpr short NW = 32, NB = 32, NF = 16, NF4 = 4;
    const short NSG = FC_mul_mv_nsg;
    const int r0 = group.x * NR0;
    const int token0 = group.y * 2;
    const int nb = args.ne00 / NB;
    const short ix = lane / (NW / NF), il = lane % (NW / NF);
    const int ib0 = sg * NF + ix;
    float sums[2][NR0] = {};
    for (int ib = ib0; ib < nb; ib += NSG * NF) {
        float4 y[2][NF4];
        for (short t = 0; t < 2; ++t) {
            if (token0 + t < args.ne1) {
                device const float4 *p = (device const float4 *)(input +
                    (ulong)(token0 + t) * args.nb11) + (ib * NB + il * NF) / 4;
                for (short i = 0; i < NF4; ++i) y[t][i] = p[i];
            }
        }
        for (short row = 0; row < NR0; ++row) {
            device const half4 *p = (device const half4 *)(weights +
                (ulong)(r0 + row) * args.nb01) + (ib * NB + il * NF) / 4;
            half4 staged[NF4];
            FOR_UNROLL (short i = 0; i < NF4; ++i) staged[i] = p[i];
            for (short t = 0; t < 2; ++t) {
                if (token0 + t < args.ne1) {
                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NF4; ++i)
                        sumq += dot(float4(staged[i]), float4(y[t][i]));
                    sums[t][row] += sumq;
                }
            }
        }
    }
    // Disjoint scratch for the two reductions: the original helper has no
    // trailing barrier, so reusing token0's scratch immediately would race.
    for (short t = 0; t < 2; ++t) {
        if (token0 + t < args.ne1) {
            device float *dst = (device float *)output + (ulong)(token0 + t) * args.ne0;
            helper_mv_reduce_and_write<NR0>(dst, sums[t], r0, args.ne01,
                lane, sg, scratch + t * NW * NR0 * sizeof(float));
        }
    }
}

kernel void kernel_dsv41_project_f16_rows2(
        constant ds4_metal_args_mul_mv &args,
        device const char *weights, device const char *input, device char *output,
        threadgroup char *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (args.nr0 == 2)
        dsv41_project_f16_rows2_impl<2>(args, weights, input, output, scratch, group, lane, sg);
    else if (args.nr0 == 4)
        dsv41_project_f16_rows2_impl<4>(args, weights, input, output, scratch, group, lane, sg);
}


// Copy bits from the unchanged F16 gather, repeat four HC streams, and seed
// each token's preceding mixer. No arithmetic or new quantization boundary.
kernel void kernel_dsv41_repeat_init4(
        constant uint2 &args,
        device const uint4 *rows,
        device uint4 *residual,
        device uint4 *pre,
        uint gid [[thread_position_in_grid]]) {
    const uint vectors = args.x / 4u;
    if (gid >= args.y * vectors) return;
    const uint token = gid / vectors, d = gid % vectors;
    const uint4 value = rows[gid];
    for (uint h = 0; h < 4u; ++h)
        residual[((ulong)token * 4u + h) * vectors + d] = value;
    if (d == 0u) pre[token] = uint4(0x3f800000u, 0u, 0u, 0u);
}


// One F32 componentwise add, then exactly the standalone BF16 bit mapping.
// In particular b is not independently rounded; preserve the parent's words.
kernel void kernel_dsv41_add_bf16_rows4(
        constant uint &vectors,
        device const float4 *a, device const float4 *b, device uint4 *out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= vectors) return;
    uint4 bits = as_type<uint4>(a[gid] + b[gid]);
    const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
    bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
    out[gid] = bits & 0xffff0000u;
}

/* V4.1 decode router in one dispatch: the BF16 router matvec exactly as
 * kernel_mul_mv_bf16_f32_4 (NSG 8, NR0 2: the same K walk and reduction),
 * its logits stored coherently; the last threadgroup to finish then runs
 * kernel_dsv41_router_hier's body on them. 512 threads: simdgroups 8-15 only
 * join the barriers of the matvec half. */
kernel void kernel_dsv41_router_fused(
        constant ds4_metal_args_mul_mv & margs,
        device const char * src0,
        device const char * src1,
        device       float * logits_out,
        constant ds4_metal_args_dsv41_router & args,
        device const float *logits,
        device const float *bias,
        device       float *probs,
        device     int32_t *selected,
        device       float *weights,
        device atomic_uint *completion,
        threadgroup   char *shmem [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        uint   ntgs [[threadgroups_per_grid]],
        uint   tid [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    {
        constexpr short NR0 = 2;
        const short NSG = 8;
        constexpr short NW = N_SIMDWIDTH;
        constexpr short NB  = 32;
        constexpr short NF  = 16;
        constexpr short NF4 = NF/4;
        const int nb = margs.ne00/NB;
        const int r0 = (int)tgpig*NR0;
        float sumf[NR0] = { 0.f };
        if (sgitg < NSG) {
            device const float4 * y4 = (device const float4 *) src1;
            device const bfloat4 * ax4[NR0];
            FOR_UNROLL (short row = 0; row < NR0; ++row)
                ax4[row] = (device const bfloat4 *) ((device char *) src0 + (uint64_t)(r0 + row)*margs.nb01);
            const short ix = tiisg/(NW/NF);
            const short il = tiisg%(NW/NF);
            const int ib0 = sgitg*NF + ix;
            float4 yl4[NF4];
            device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;
            for (int ib = ib0; ib < nb; ib += NSG*NF) {
                for (short i = 0; i < NF4; ++i) yl4[i] = yb4[i];
                for (short row = 0; row < NR0; row++) {
                    device const bfloat4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;
                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NF4; ++i) sumq += dot(float4(xb4[i]), float4(yl4[i]));
                    sumf[row] += sumq;
                }
                yb4 += NSG*NF*NW/4;
            }
        }
        threadgroup float * sh[NR0];
        for (short row = 0; row < NR0; ++row) {
            sh[row] = (threadgroup float *) shmem + NW*row;
            if (sgitg == 0) sh[row][tiisg] = 0.0f;
            sumf[row] = simd_sum(sumf[row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short row = 0; row < NR0; ++row)
            if (tiisg == 0 && sgitg < NSG) sh[row][sgitg] = sumf[row];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short row = 0; row < NR0 && r0 + row < margs.ne01; ++row) {
            const float tot = simd_sum(sh[row][tiisg]);
            if (tiisg == 0 && sgitg == 0) dsv41_coherent_store(logits_out + r0 + row, tot);
        }
        threadgroup float * flag = (threadgroup float *) shmem + 2*NW;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0) {
            const uint old = atomic_fetch_add_explicit(completion, 1u, memory_order_relaxed);
            flag[0] = old + 1u == ntgs ? 1.0f : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (flag[0] == 0.0f) return;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (tid == 0) atomic_store_explicit(completion, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ds4_dsv41_router_hier_body<true>(args, logits, bias, probs, selected, weights, shmem, tid);
}
