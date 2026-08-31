// Qwen3.8-Flash-Next native kernels. Quantized matrices use the ordinary
// GGML block payloads emitted by GGUF: Q8_0 for dense projections and Q4_K
// for routed experts. Inputs, outputs, and recurrent state remain FP32.

static inline float qwen4_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

static inline ushort qwen4_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (ushort)(bits >> 16u);
}

struct qwen4_q8_0_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

static inline float qwen4_q8_0_value(
        device const block_q8_0 *row,
        uint k) {
    device const block_q8_0 *block = row + k / QK8_0;
    return (float)block->d * (float)block->qs[k & (QK8_0 - 1u)];
}

/* One SIMDgroup owns one output row. Lanes walk 8-value slices so the block
 * delta is loaded once for each slice, matching the decode-friendly GGML
 * Q8_0 traversal without materializing dequantized weights. */
static inline float qwen4_q8_0_dot_f32(
        device const block_q8_0 *row,
        device const float *x,
        uint in_dim,
        ushort lane) {
    float sum = 0.0f;
    for (uint base = (uint)lane * 8u; base < in_dim;
         base += 32u * 8u) {
        float dotq = 0.0f;
        const uint block_index = base / QK8_0;
        const uint in_block = base & (QK8_0 - 1u);
        device const block_q8_0 *block = row + block_index;
        for (uint i = 0u; i < 8u; i++)
            dotq = fma((float)block->qs[in_block + i], x[base + i], dotq);
        sum = fma((float)block->d, dotq, sum);
    }
    return simd_sum(sum);
}

static inline float qwen4_q8_0_dot_bf16(
        device const block_q8_0 *row,
        device const ushort *x,
        uint in_dim,
        ushort lane) {
    float sum = 0.0f;
    for (uint base = (uint)lane * 8u; base < in_dim;
         base += 32u * 8u) {
        float dotq = 0.0f;
        const uint block_index = base / QK8_0;
        const uint in_block = base & (QK8_0 - 1u);
        device const block_q8_0 *block = row + block_index;
        for (uint i = 0u; i < 8u; i++) {
            dotq = fma((float)block->qs[in_block + i],
                       qwen4_bf16_to_f32(x[base + i]), dotq);
        }
        sum = fma((float)block->d, dotq, sum);
    }
    return simd_sum(sum);
}

kernel void kernel_qwen4_q8_0_f32(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;
    const uint row_blocks = args.in_dim / QK8_0;
    device const block_q8_0 *wr = weights + (ulong)out_row * row_blocks;
    device const float *xr = x + (ulong)token * args.in_dim;
    const float sum = qwen4_q8_0_dot_f32(wr, xr, args.in_dim, lane);
    if (lane == 0u)
        out[(ulong)token * args.out_dim + out_row] = sum;
}

kernel void kernel_qwen4_q8_0_f32_m1(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup * (uint)nsg + (uint)sg;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    if (lane == 0u) out[out_row] = sum;
}

/* Expert-independent wide-prefill specialization. Each SIMDgroup owns one
 * output row and advances eight token rows together, decoding each Q8_0
 * block slice once and retaining an independent FP32 reduction per token. */
kernel void kernel_qwen4_q8_0_f32_rows8(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token0 = tgpig.y * 8u;
    if (out_row >= args.out_dim || token0 >= args.n_rows) return;
    const uint row_blocks = args.in_dim / QK8_0;
    device const block_q8_0 *wr = weights + (ulong)out_row * row_blocks;
    float4 sum0 = 0.0f;
    float4 sum1 = 0.0f;
    for (uint base = (uint)lane * 8u; base < args.in_dim;
         base += 32u * 8u) {
        const uint block_index = base / QK8_0;
        const uint in_block = base & (QK8_0 - 1u);
        device const block_q8_0 *block = wr + block_index;
        const float delta = (float)block->d;
        for (uint i = 0u; i < 8u; i++) {
            const uint k = base + i;
            const float q = (float)block->qs[in_block + i] * delta;
            float4 xv0 = 0.0f;
            float4 xv1 = 0.0f;
            if (token0 + 0u < args.n_rows) xv0.x = x[((ulong)token0 + 0u) * args.in_dim + k];
            if (token0 + 1u < args.n_rows) xv0.y = x[((ulong)token0 + 1u) * args.in_dim + k];
            if (token0 + 2u < args.n_rows) xv0.z = x[((ulong)token0 + 2u) * args.in_dim + k];
            if (token0 + 3u < args.n_rows) xv0.w = x[((ulong)token0 + 3u) * args.in_dim + k];
            if (token0 + 4u < args.n_rows) xv1.x = x[((ulong)token0 + 4u) * args.in_dim + k];
            if (token0 + 5u < args.n_rows) xv1.y = x[((ulong)token0 + 5u) * args.in_dim + k];
            if (token0 + 6u < args.n_rows) xv1.z = x[((ulong)token0 + 6u) * args.in_dim + k];
            if (token0 + 7u < args.n_rows) xv1.w = x[((ulong)token0 + 7u) * args.in_dim + k];
            sum0 = fma(float4(q), xv0, sum0);
            sum1 = fma(float4(q), xv1, sum1);
        }
    }
    sum0.x = simd_sum(sum0.x); sum0.y = simd_sum(sum0.y);
    sum0.z = simd_sum(sum0.z); sum0.w = simd_sum(sum0.w);
    sum1.x = simd_sum(sum1.x); sum1.y = simd_sum(sum1.y);
    sum1.z = simd_sum(sum1.z); sum1.w = simd_sum(sum1.w);
    if (lane != 0u) return;
    if (token0 + 0u < args.n_rows) out[((ulong)token0 + 0u) * args.out_dim + out_row] = sum0.x;
    if (token0 + 1u < args.n_rows) out[((ulong)token0 + 1u) * args.out_dim + out_row] = sum0.y;
    if (token0 + 2u < args.n_rows) out[((ulong)token0 + 2u) * args.out_dim + out_row] = sum0.z;
    if (token0 + 3u < args.n_rows) out[((ulong)token0 + 3u) * args.out_dim + out_row] = sum0.w;
    if (token0 + 4u < args.n_rows) out[((ulong)token0 + 4u) * args.out_dim + out_row] = sum1.x;
    if (token0 + 5u < args.n_rows) out[((ulong)token0 + 5u) * args.out_dim + out_row] = sum1.y;
    if (token0 + 6u < args.n_rows) out[((ulong)token0 + 6u) * args.out_dim + out_row] = sum1.z;
    if (token0 + 7u < args.n_rows) out[((ulong)token0 + 7u) * args.out_dim + out_row] = sum1.w;
}

kernel void kernel_qwen4_q8_0_bf16(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const ushort      *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_bf16(
        weights + (ulong)out_row * row_blocks,
        x + (ulong)token * args.in_dim,
        args.in_dim, lane);
    if (lane == 0u)
        out[(ulong)token * args.out_dim + out_row] = sum;
}

kernel void kernel_qwen4_q8_0_f32_m1_silu(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup * (uint)nsg + (uint)sg;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float value = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    if (lane == 0u) out[out_row] = value / (1.0f + exp(-value));
}

kernel void kernel_qwen4_q8_0_f32_m1_swiglu(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0 *gate_weight [[buffer(1)]],
        device const block_q8_0 *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup * (uint)nsg + (uint)sg;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float gate = qwen4_q8_0_dot_f32(
        gate_weight + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    const float up = qwen4_q8_0_dot_f32(
        up_weight + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    if (lane == 0u) out[out_row] = (gate / (1.0f + exp(-gate))) * up;
}

kernel void kernel_qwen4_q8_0_f32_m1_pair(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0 *weight_a [[buffer(1)]],
        device const block_q8_0 *weight_b [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device float *out_a [[buffer(4)]],
        device float *out_b [[buffer(5)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup * (uint)nsg + (uint)sg;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float a = qwen4_q8_0_dot_f32(
        weight_a + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    const float b = qwen4_q8_0_dot_f32(
        weight_b + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    if (lane == 0u) { out_a[out_row] = a; out_b[out_row] = b; }
}

kernel void kernel_qwen4_q8_0_f32_m1_concat(
        constant qwen4_q8_0_args &args_a [[buffer(0)]],
        constant qwen4_q8_0_args &args_b [[buffer(1)]],
        device const block_q8_0 *weight_a [[buffer(2)]],
        device const block_q8_0 *weight_b [[buffer(3)]],
        device const float *x [[buffer(4)]],
        device float *out_a [[buffer(5)]],
        device float *out_b [[buffer(6)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint global_row = tgroup * (uint)nsg + (uint)sg;
    if (global_row >= args_a.out_dim + args_b.out_dim) return;
    const bool second = global_row >= args_a.out_dim;
    const uint out_row = second ? global_row - args_a.out_dim : global_row;
    const uint row_blocks = args_a.in_dim / QK8_0;
    device const block_q8_0 *weights = second ? weight_b : weight_a;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks, x, args_a.in_dim, lane);
    if (lane == 0u) {
        if (second) out_b[out_row] = sum;
        else out_a[out_row] = sum;
    }
}

kernel void kernel_qwen4_bf16_matmul_f32(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const ushort      *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;
    device const ushort *wr = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    for (uint k = lane; k < args.in_dim; k += 32u)
        sum = fma(qwen4_bf16_to_f32(wr[k]), xr[k], sum);
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * args.out_dim + out_row] = sum;
}

struct qwen4_embedding_args {
    uint hidden_dim;
    uint vocab_size;
    uint n_tokens;
};

kernel void kernel_qwen4_q8_0_embedding_f32(
        constant qwen4_embedding_args &args [[buffer(0)]],
        device const block_q8_0       *weights [[buffer(1)]],
        device const int              *token_ids [[buffer(2)]],
        device float                  *out [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token_row = gid.y;
    if (dim >= args.hidden_dim || token_row >= args.n_tokens) return;
    const int token = token_ids[token_row];
    if (token < 0 || (uint)token >= args.vocab_size) {
        out[(ulong)token_row * args.hidden_dim + dim] = 0.0f;
        return;
    }
    const uint row_blocks = args.hidden_dim / QK8_0;
    device const block_q8_0 *row = weights + (ulong)(uint)token * row_blocks;
    out[(ulong)token_row * args.hidden_dim + dim] = qwen4_q8_0_value(row, dim);
}

struct qwen4_rows_args {
    uint width;
    uint rows;
    uint weight_rows;
    float eps;
};

kernel void kernel_qwen4_rms_norm_bf16_f32(
        constant qwen4_rows_args &args [[buffer(0)]],
        device const float       *x [[buffer(1)]],
        device const ushort      *weight [[buffer(2)]],
        device float             *out [[buffer(3)]],
        threadgroup float        *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.rows) return;
    const ulong base = (ulong)row * args.width;
    float sum = 0.0f;
    for (uint col = tid; col < args.width; col += (uint)nsg * 32u) {
        const float value = x[base + col];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u) partial[0] = rsqrt(sum / (float)args.width + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const ulong weight_base =
        (ulong)(row % args.weight_rows) * args.width;
    for (uint col = tid; col < args.width; col += (uint)nsg * 32u) {
        out[base + col] = x[base + col] * scale *
            qwen4_bf16_to_f32(weight[weight_base + col]);
    }
}

struct qwen4_hc_norm_inject_args {
    uint n_tokens;
    uint hidden_dim;
    uint stream_count;
    float eps;
};

kernel void kernel_qwen4_hc_norm_inject_f32(
        constant qwen4_hc_norm_inject_args &args [[buffer(0)]],
        device const float *streams [[buffer(1)]],
        device const ushort *norm_weight [[buffer(2)]],
        device const ushort *inject_weight [[buffer(3)]],
        device float *normalized [[buffer(4)]],
        device float *inject_partials [[buffer(5)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint total_rows = args.n_tokens * args.stream_count;
    if (row >= total_rows) return;
    const uint token = row / args.stream_count;
    const uint stream = row % args.stream_count;
    const ulong base = (ulong)row * args.hidden_dim;
    float sum = 0.0f;
    for (uint col = tid; col < args.hidden_dim; col += (uint)nsg * 32u) {
        const float value = streams[base + col];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(sum / (float)args.hidden_dim + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const ulong norm_weight_base = (ulong)stream * args.hidden_dim;
    for (uint col = tid; col < args.hidden_dim; col += (uint)nsg * 32u) {
        normalized[base + col] = streams[base + col] * scale *
            qwen4_bf16_to_f32(norm_weight[norm_weight_base + col]);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (sg >= args.stream_count) return;
    const uint output = sg;
    const uint hc_dim = args.stream_count * args.hidden_dim;
    device const ushort *wr = inject_weight +
        (ulong)output * hc_dim + (ulong)stream * args.hidden_dim;
    device const float *xr = normalized + base;
    float dot = 0.0f;
    for (uint k = lane; k < args.hidden_dim; k += 32u)
        dot = fma(qwen4_bf16_to_f32(wr[k]), xr[k], dot);
    dot = simd_sum(dot);
    if (lane == 0u)
        inject_partials[((ulong)token * args.stream_count + stream) *
                        args.stream_count + output] = dot;
}

kernel void kernel_qwen4_hc_inject_reduce_f32(
        constant qwen4_hc_norm_inject_args &args [[buffer(0)]],
        device const float *inject_partials [[buffer(1)]],
        device float *inject [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint output = gid.x;
    const uint token = gid.y;
    if (output >= args.stream_count || token >= args.n_tokens) return;
    float sum = 0.0f;
    for (uint stream = 0; stream < args.stream_count; stream++)
        sum += inject_partials[((ulong)token * args.stream_count + stream) *
                               args.stream_count + output];
    inject[(ulong)token * args.stream_count + output] =
        2.0f / (1.0f + exp(-sum));
}

kernel void kernel_qwen4_silu_f32(
        device const float *x [[buffer(0)]],
        device float *out [[buffer(1)]],
        constant ulong &elements [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if ((ulong)gid >= elements) return;
    const float value = x[gid];
    out[gid] = value / (1.0f + exp(-value));
}

kernel void kernel_qwen4_swiglu_f32(
        device const float *gate [[buffer(0)]],
        device const float *up [[buffer(1)]],
        device float *out [[buffer(2)]],
        constant ulong &elements [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if ((ulong)gid >= elements) return;
    const float value = gate[gid];
    out[gid] = (value / (1.0f + exp(-value))) * up[gid];
}

struct qwen4_hc_args {
    uint n_tokens;
    uint hidden_dim;
    uint stream_count;
};

struct qwen4_hc_up_mix_args {
    uint low_dim;
    uint hidden_dim;
    uint stream_count;
};

kernel void kernel_qwen4_repeat_streams_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *hidden [[buffer(1)]],
        device float *streams [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token = gid.y;
    if (dim >= args.hidden_dim || token >= args.n_tokens) return;
    const float value = hidden[(ulong)token * args.hidden_dim + dim];
    for (uint stream = 0; stream < args.stream_count; stream++) {
        streams[((ulong)token * args.stream_count + stream) *
                args.hidden_dim + dim] = value;
    }
}

kernel void kernel_qwen4_hc_mix_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *normalized [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device float *mixed [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token = gid.y;
    if (dim >= args.hidden_dim || token >= args.n_tokens) return;
    float sum = 0.0f;
    for (uint stream = 0; stream < args.stream_count; stream++) {
        const ulong index = ((ulong)token * args.stream_count + stream) *
                            args.hidden_dim + dim;
        const float gate = 1.0f / (1.0f + exp(-raw_gate[index]));
        sum = fma(normalized[index], gate, sum);
    }
    mixed[(ulong)token * args.hidden_dim + dim] =
        sum / (float)args.stream_count;
}

kernel void kernel_qwen4_hc_up_mix_q8_0_m1_f32(
        constant qwen4_hc_up_mix_args &args [[buffer(0)]],
        device const block_q8_0 *weights [[buffer(1)]],
        device const float *low [[buffer(2)]],
        device const float *normalized [[buffer(3)]],
        device float *mixed [[buffer(4)]],
        threadgroup float *raw_gate [[threadgroup(0)]],
        uint dim [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (dim >= args.hidden_dim || sg >= args.stream_count) return;
    const uint output = (uint)sg * args.hidden_dim + dim;
    const uint row_blocks = args.low_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)output * row_blocks, low, args.low_dim, lane);
    if (lane == 0u) raw_gate[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float value = 0.0f;
        for (uint stream = 0; stream < args.stream_count; stream++) {
            const float gate = 1.0f / (1.0f + exp(-raw_gate[stream]));
            value = fma(normalized[(ulong)stream * args.hidden_dim + dim],
                        gate, value);
        }
        mixed[dim] = value / (float)args.stream_count;
    }
}

kernel void kernel_qwen4_hc_inject_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *raw_inject [[buffer(1)]],
        device float *inject [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong elements = (ulong)args.n_tokens * args.stream_count;
    if ((ulong)gid >= elements) return;
    inject[gid] = 2.0f / (1.0f + exp(-raw_inject[gid]));
}

kernel void kernel_qwen4_hc_write_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *block_output [[buffer(1)]],
        device const float *inject [[buffer(2)]],
        device float *streams [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token = gid.y;
    if (dim >= args.hidden_dim || token >= args.n_tokens) return;
    const float value = block_output[(ulong)token * args.hidden_dim + dim];
    for (uint stream = 0; stream < args.stream_count; stream++) {
        const ulong index = ((ulong)token * args.stream_count + stream) *
                            args.hidden_dim + dim;
        streams[index] = fma(value,
                             inject[(ulong)token * args.stream_count + stream],
                             streams[index]);
    }
}

kernel void kernel_qwen4_hc_write_partials_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *block_output [[buffer(1)]],
        device const float *inject_partials [[buffer(2)]],
        device float *streams [[buffer(3)]],
        threadgroup float *inject [[threadgroup(0)]],
        uint2 tgroup [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint token = tgroup.y;
    if (token >= args.n_tokens) return;
    if (tid < args.stream_count) {
        float sum = 0.0f;
        for (uint stream = 0; stream < args.stream_count; stream++)
            sum += inject_partials[
                ((ulong)token * args.stream_count + stream) *
                args.stream_count + tid];
        inject[tid] = 2.0f / (1.0f + exp(-sum));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint dim = tgroup.x * 256u + tid;
    if (dim >= args.hidden_dim) return;
    const float value = block_output[(ulong)token * args.hidden_dim + dim];
    for (uint stream = 0; stream < args.stream_count; stream++) {
        const ulong index = ((ulong)token * args.stream_count + stream) *
                            args.hidden_dim + dim;
        streams[index] = fma(value, inject[stream], streams[index]);
    }
}

/* Decode Gated DeltaNet output projection plus its hyper-connection write.
 * The Q8_0 reduction and injection-partial reduction keep the same orders
 * as their standalone kernels; only the intermediate block write/read and a
 * second dispatch are removed. */
kernel void kernel_qwen4_q8_0_f32_m1_hc_write(
        constant qwen4_q8_0_args &projection [[buffer(0)]],
        constant qwen4_hc_args &hc [[buffer(1)]],
        device const block_q8_0 *weights [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const float *inject_partials [[buffer(4)]],
        device float *streams [[buffer(5)]],
        threadgroup float *inject [[threadgroup(0)]],
        uint tgroup [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (tid < hc.stream_count) {
        float partial = 0.0f;
        for (uint stream = 0; stream < hc.stream_count; stream++)
            partial += inject_partials[
                (ulong)stream * hc.stream_count + tid];
        inject[tid] = 2.0f / (1.0f + exp(-partial));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint out_row = tgroup * (uint)nsg + sg;
    if (out_row >= projection.out_dim) return;
    const uint row_blocks = projection.in_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks,
        x, projection.in_dim, lane);
    if (lane == 0u) {
        for (uint stream = 0; stream < hc.stream_count; stream++) {
            const ulong at = (ulong)stream * hc.hidden_dim + out_row;
            streams[at] = fma(sum, inject[stream], streams[at]);
        }
    }
}

struct qwen4_ple_args {
    uint n_tokens;
    uint stream_count;
    uint hidden_dim;
    uint conv_width;
    uint dilation;
    uint state_len;
    uint has_mask;
    float eps;
};

kernel void kernel_qwen4_ple_gate_norm_f32(
        constant qwen4_ple_args &args [[buffer(0)]],
        device const float *hidden_streams [[buffer(1)]],
        device const float *key_raw [[buffer(2)]],
        device const float *value_raw [[buffer(3)]],
        device const ushort *key_weight [[buffer(4)]],
        device const ushort *query_weight [[buffer(5)]],
        device const ushort *conv_norm_weight [[buffer(6)]],
        device const uchar *mask [[buffer(7)]],
        device float *gated [[buffer(8)]],
        device float *gated_norm [[buffer(9)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint stream = group.y;
    if (token >= args.n_tokens || stream >= args.stream_count) return;
    const ulong stream_base =
        ((ulong)token * args.stream_count + stream) * args.hidden_dim;
    const ulong value_base = (ulong)token * args.hidden_dim;
    const ulong weight_base = (ulong)stream * args.hidden_dim;
    float key_sum = 0.0f;
    float query_sum = 0.0f;
    float value_sum = 0.0f;
    for (uint dim = tid; dim < args.hidden_dim; dim += (uint)nsg * 32u) {
        const float key = key_raw[stream_base + dim];
        const float query = hidden_streams[stream_base + dim];
        const float value = value_raw[value_base + dim];
        key_sum = fma(key, key, key_sum);
        query_sum = fma(query, query, query_sum);
        value_sum = fma(value, value, value_sum);
    }
    key_sum = simd_sum(key_sum);
    query_sum = simd_sum(query_sum);
    value_sum = simd_sum(value_sum);
    if (lane == 0u) {
        partial[3u * sg] = key_sum;
        partial[3u * sg + 1u] = query_sum;
        partial[3u * sg + 2u] = value_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        key_sum = lane < nsg ? partial[3u * lane] : 0.0f;
        query_sum = lane < nsg ? partial[3u * lane + 1u] : 0.0f;
        value_sum = lane < nsg ? partial[3u * lane + 2u] : 0.0f;
        key_sum = simd_sum(key_sum);
        query_sum = simd_sum(query_sum);
        value_sum = simd_sum(value_sum);
        if (lane == 0u) {
            partial[3u * nsg] = rsqrt(
                key_sum / (float)args.hidden_dim + args.eps);
            partial[3u * nsg + 1u] = rsqrt(
                query_sum / (float)args.hidden_dim + args.eps);
            partial[3u * nsg + 2u] = value_sum / (float)args.hidden_dim;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float key_scale = partial[3u * nsg];
    const float query_scale = partial[3u * nsg + 1u];
    float dot = 0.0f;
    for (uint dim = tid; dim < args.hidden_dim; dim += (uint)nsg * 32u) {
        const float key = key_raw[stream_base + dim] * key_scale *
            qwen4_bf16_to_f32(key_weight[weight_base + dim]);
        const float query = hidden_streams[stream_base + dim] * query_scale *
            qwen4_bf16_to_f32(query_weight[weight_base + dim]);
        dot = fma(key, query, dot);
    }
    dot = simd_sum(dot);
    if (lane == 0u) partial[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        dot = lane < nsg ? partial[lane] : 0.0f;
        dot = simd_sum(dot) * rsqrt((float)args.hidden_dim);
        if (lane == 0u) {
            const float sign = dot > 0.0f ? 1.0f :
                               (dot < 0.0f ? -1.0f : 0.0f);
            const float transformed = sign * sqrt(max(abs(dot), 1.0e-6f));
            partial[0] = 1.0f / (1.0f + exp(-transformed));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = partial[0];
    const float norm_scale = rsqrt(
        gate * gate * partial[3u * nsg + 2u] + args.eps);
    const bool active = args.has_mask == 0u || mask[token] != 0u;
    for (uint dim = tid; dim < args.hidden_dim; dim += (uint)nsg * 32u) {
        const float value = active ? gate * value_raw[value_base + dim] : 0.0f;
        gated[stream_base + dim] = value;
        gated_norm[stream_base + dim] = active
            ? value * norm_scale *
              qwen4_bf16_to_f32(conv_norm_weight[weight_base + dim])
            : 0.0f;
    }
}

kernel void kernel_qwen4_ple_dilated_conv_f32(
        constant qwen4_ple_args &args [[buffer(0)]],
        device const float *gated_norm [[buffer(1)]],
        device const ushort *conv_weight [[buffer(2)]],
        device float *conv_state [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint channel [[thread_position_in_grid]]) {
    const uint channels = args.stream_count * args.hidden_dim;
    if (channel >= channels || args.state_len != 9u ||
        args.conv_width != 4u || args.dilation != 3u) return;
    float history[9];
    for (uint i = 0; i < 9u; i++)
        history[i] = conv_state[(ulong)channel * 9u + i];
    for (uint token = 0; token < args.n_tokens; token++) {
        const ulong at = (ulong)token * channels + channel;
        const float current = gated_norm[at];
        float sum = current * qwen4_bf16_to_f32(
            conv_weight[(ulong)channel * 4u + 3u]);
        sum = fma(history[0], qwen4_bf16_to_f32(
                      conv_weight[(ulong)channel * 4u]), sum);
        sum = fma(history[3], qwen4_bf16_to_f32(
                      conv_weight[(ulong)channel * 4u + 1u]), sum);
        sum = fma(history[6], qwen4_bf16_to_f32(
                      conv_weight[(ulong)channel * 4u + 2u]), sum);
        for (uint i = 0; i < 8u; i++) history[i] = history[i + 1u];
        history[8] = current;
        const float activated = sum / (1.0f + exp(-sum));
        out[at] += activated;
    }
    for (uint i = 0; i < 9u; i++)
        conv_state[(ulong)channel * 9u + i] = history[i];
}

struct qwen4_moe_args {
    uint n_rows;
    uint n_experts;
    uint top_k;
    uint in_dim;
    uint expert_dim;
    uint out_dim;
    uint down_dim;
};

kernel void kernel_qwen4_moe_topk_f32(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const float *router_logits [[buffer(1)]],
        device int *selected [[buffer(2)]],
        device float *weights [[buffer(3)]],
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.n_rows || args.top_k > 16u ||
        args.n_experts > 512u || nsg > 8u) return;
    device const float *scores =
        router_logits + (ulong)row * args.n_experts;
    threadgroup float *partial_score = scratch;
    threadgroup int *partial_id = (threadgroup int *)(scratch + 8u);
    threadgroup float *chosen_score = scratch + 16u;
    threadgroup int *chosen_id = (threadgroup int *)(scratch + 32u);
    float local_score[2] = {-INFINITY, -INFINITY};
    int local_id[2] = {INT_MAX, INT_MAX};
    for (uint i = 0; i < 2u; i++) {
        const uint expert = tid + i * 256u;
        if (expert < args.n_experts) {
            local_score[i] = scores[expert];
            local_id[i] = (int)expert;
        }
    }

    for (uint slot = 0; slot < args.top_k; slot++) {
        float best = local_score[0];
        int id = local_id[0];
        if (local_score[1] > best ||
            (local_score[1] == best && local_id[1] < id)) {
            best = local_score[1];
            id = local_id[1];
        }
        for (ushort offset = 16u; offset != 0u; offset >>= 1u) {
            const float other = simd_shuffle_down(best, offset);
            const int other_id = simd_shuffle_down(id, offset);
            if (lane + offset < 32u &&
                (other > best || (other == best && other_id < id))) {
                best = other;
                id = other_id;
            }
        }
        if (lane == 0u) {
            partial_score[sg] = best;
            partial_id[sg] = id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            best = lane < nsg ? partial_score[lane] : -INFINITY;
            id = lane < nsg ? partial_id[lane] : INT_MAX;
            for (ushort offset = 16u; offset != 0u; offset >>= 1u) {
                const float other = simd_shuffle_down(best, offset);
                const int other_id = simd_shuffle_down(id, offset);
                if (lane + offset < 32u &&
                    (other > best || (other == best && other_id < id))) {
                    best = other;
                    id = other_id;
                }
            }
            if (lane == 0u) {
                chosen_score[slot] = best;
                chosen_id[slot] = id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int remove = chosen_id[slot];
        if (local_id[0] == remove) {
            local_score[0] = -INFINITY;
            local_id[0] = INT_MAX;
        }
        if (local_id[1] == remove) {
            local_score[1] = -INFINITY;
            local_id[1] = INT_MAX;
        }
    }
    if (tid == 0u) {
        const float max_score = chosen_score[0];
        float selected_sum = 0.0f;
        for (uint slot = 0; slot < args.top_k; slot++) {
            chosen_score[slot] = exp(
                chosen_score[slot] - max_score);
            selected_sum += chosen_score[slot];
        }
        const float inv = 1.0f / selected_sum;
        for (uint slot = 0; slot < args.top_k; slot++) {
            const ulong at = (ulong)row * args.top_k + slot;
            selected[at] = chosen_id[slot];
            weights[at] = chosen_score[slot] * inv;
        }
    }
}

/* Deterministic token-major to expert-major map for the exact rows8 path.
 * Each expert scans the same staged route order, so hids remains stable across
 * runs and every work item names up to eight original token/slot assignments. */
kernel void kernel_qwen4_moe_map_rows8(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const int *selected [[buffer(1)]],
        device uint *counts [[buffer(2)]],
        device uint *hids [[buffer(3)]],
        device char *work [[buffer(4)]],
        threadgroup int *staged [[threadgroup(0)]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    const uint expert = (uint)tid;
    uint count = 0u;
    device uint *expert_hids = hids + (ulong)expert * args.n_rows;

    for (uint row0 = 0u; row0 < args.n_rows; row0 += (uint)ntg) {
        const uint row = row0 + expert;
        if (row < args.n_rows) {
            for (uint slot = 0u; slot < args.top_k; slot++) {
                staged[(ulong)expert * args.top_k + slot] =
                    selected[(ulong)row * args.top_k + slot];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint rows = min((uint)ntg, args.n_rows - row0);
        for (uint local = 0u; local < rows; local++) {
            for (uint slot = 0u; slot < args.top_k; slot++) {
                if (staged[(ulong)local * args.top_k + slot] == (int)expert)
                    expert_hids[count++] = (row0 + local) * args.top_k + slot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    counts[expert] = count;
    threadgroup int *tile_counts = staged;
    tile_counts[expert] = (int)((count + 7u) / 8u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint work_base = 0u;
    for (uint i = 0u; i < expert; i++) work_base += tile_counts[i];
    device uint *work_count = (device uint *)work;
    device uint2 *work_items = (device uint2 *)(work + 8);
    for (uint tile = 0u; tile < (uint)tile_counts[expert]; tile++)
        work_items[work_base + tile] = uint2(expert, tile * 8u);
    if (expert + 1u == args.n_experts)
        work_count[0] = work_base + (uint)tile_counts[expert];
}

/* Q4_K expert rows use the same GGML decoder as the established GLM path.
 * Qwen differs in routing width (512/top-10), so the fused scheduling remains
 * local while the block interpretation is shared. */
static inline float qwen4_q4_k_dot(
        device const block_q4_K *row,
        device const float *x,
        uint logical_dim,
        ushort lane) {
    float sum = 0.0f;
    for (uint k = (uint)lane; k < logical_dim; k += 32u)
        sum = fma(ds4_glm_q4_K_value(row, k), x[k], sum);
    return simd_sum(sum);
}

kernel void kernel_qwen4_moe_q4_k_gate_up_rows8(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *gate_weight [[buffer(1)]],
        device const block_q4_K *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const uint *counts [[buffer(4)]],
        device const uint *hids [[buffer(5)]],
        device const char *work [[buffer(6)]],
        device float *mid [[buffer(7)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const uint *work_count = (device const uint *)work;
    if (tgpig.x >= work_count[0]) return;
    device const uint2 *work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[tgpig.x];
    const uint expert = item.x;
    const uint local0 = item.y;
    const uint output = tgpig.y * (uint)nsg + (uint)sg;
    if (expert >= args.n_experts || output >= args.expert_dim) return;

    const uint valid = min(8u, counts[expert] - local0);
    uint assignments[8];
    uint tokens[8];
    device const uint *expert_hids = hids + (ulong)expert * args.n_rows;
    for (uint r = 0u; r < valid; r++) {
        assignments[r] = expert_hids[local0 + r];
        tokens[r] = assignments[r] / args.top_k;
    }

    const uint row_blocks = args.in_dim / 256u;
    const ulong weight_row = (ulong)expert * args.expert_dim + output;
    device const block_q4_K *gwr = gate_weight + weight_row * row_blocks;
    device const block_q4_K *uwr = up_weight + weight_row * row_blocks;
    float4 gate0 = 0.0f, gate1 = 0.0f;
    float4 up0 = 0.0f, up1 = 0.0f;
    for (uint k = (uint)lane; k < args.in_dim; k += 32u) {
        const float gw = ds4_glm_q4_K_value(gwr, k);
        const float uw = ds4_glm_q4_K_value(uwr, k);
        for (uint r = 0u; r < valid; r++) {
            const float xv = x[(ulong)tokens[r] * args.in_dim + k];
            if (r < 4u) {
                gate0[r] = fma(gw, xv, gate0[r]);
                up0[r] = fma(uw, xv, up0[r]);
            } else {
                gate1[r - 4u] = fma(gw, xv, gate1[r - 4u]);
                up1[r - 4u] = fma(uw, xv, up1[r - 4u]);
            }
        }
    }
    for (uint r = 0u; r < valid; r++) {
        const float gate = simd_sum(r < 4u ? gate0[r] : gate1[r - 4u]);
        const float up = simd_sum(r < 4u ? up0[r] : up1[r - 4u]);
        if (lane == 0u)
            mid[(ulong)assignments[r] * args.expert_dim + output] =
                (gate / (1.0f + exp(-gate))) * up;
    }
}

kernel void kernel_qwen4_moe_q4_k_down_rows8(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const uint *counts [[buffer(3)]],
        device const uint *hids [[buffer(4)]],
        device const char *work [[buffer(5)]],
        device float *contributions [[buffer(6)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const uint *work_count = (device const uint *)work;
    if (tgpig.x >= work_count[0]) return;
    device const uint2 *work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[tgpig.x];
    const uint expert = item.x;
    const uint local0 = item.y;
    const uint output = tgpig.y * (uint)nsg + (uint)sg;
    if (expert >= args.n_experts || output >= args.out_dim) return;
    const uint valid = min(8u, counts[expert] - local0);
    uint assignments[8];
    device const uint *expert_hids = hids + (ulong)expert * args.n_rows;
    for (uint r = 0u; r < valid; r++) assignments[r] = expert_hids[local0 + r];

    const uint row_blocks = args.down_dim / 256u;
    const ulong weight_row = (ulong)expert * args.out_dim + output;
    device const block_q4_K *wr = down_weight + weight_row * row_blocks;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    /* k=[640,768) is the Q4_K alignment tail. Its activation is defined as
     * zero, so omitting those FMAs is exact and keeps mid physically 640-wide. */
    for (uint k = (uint)lane; k < args.expert_dim; k += 32u) {
        const float w = ds4_glm_q4_K_value(wr, k);
        for (uint r = 0u; r < valid; r++) {
            const float xv = mid[(ulong)assignments[r] * args.expert_dim + k];
            if (r < 4u) sum0[r] = fma(w, xv, sum0[r]);
            else sum1[r - 4u] = fma(w, xv, sum1[r - 4u]);
        }
    }
    for (uint r = 0u; r < valid; r++) {
        const float sum = simd_sum(r < 4u ? sum0[r] : sum1[r - 4u]);
        if (lane == 0u)
            contributions[(ulong)assignments[r] * args.out_dim + output] = sum;
    }
}

/* Keep the top-10 reduction token-major and strictly slot ordered.  The
 * rows8 kernels only change how expert contributions are produced; this
 * final accumulation must match decode and the scalar reference exactly. */
kernel void kernel_qwen4_moe_weighted_sum10_f32(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const int *selected [[buffer(1)]],
        device const float *selected_weights [[buffer(2)]],
        device const float *contributions [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint2 tpig [[thread_position_in_grid]]) {
    const uint output = tpig.x;
    const uint row = tpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    float total = 0.0f;
    for (uint slot = 0u; slot < 10u; slot++) {
        const ulong assignment = (ulong)row * args.top_k + slot;
        const int expert = selected[assignment];
        if (expert < 0 || (uint)expert >= args.n_experts) continue;
        total = fma(contributions[assignment * args.out_dim + output],
                    selected_weights[assignment], total);
    }
    out[(ulong)row * args.out_dim + output] = total;
}

kernel void kernel_qwen4_moe_q4_k_gate_up(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *gate_weight [[buffer(1)]],
        device const block_q4_K *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const int *selected [[buffer(4)]],
        device float *mid [[buffer(5)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint output = tgpig.x * (uint)nsg + (uint)sg;
    const uint row = tgpig.y;
    const uint slot = tgpig.z;
    if (output >= args.expert_dim || row >= args.n_rows || slot >= args.top_k) return;
    const ulong assignment = (ulong)row * args.top_k + slot;
    const int expert = selected[assignment];
    if (expert < 0 || (uint)expert >= args.n_experts) return;
    const uint row_blocks = args.in_dim / 256u;
    const ulong weight_row = (ulong)(uint)expert * args.expert_dim + output;
    const float gate = qwen4_q4_k_dot(
        gate_weight + weight_row * row_blocks,
        x + (ulong)row * args.in_dim, args.in_dim, lane);
    const float up = qwen4_q4_k_dot(
        up_weight + weight_row * row_blocks,
        x + (ulong)row * args.in_dim, args.in_dim, lane);
    if (lane == 0u)
        mid[assignment * args.expert_dim + output] =
            (gate / (1.0f + exp(-gate))) * up;
}

kernel void kernel_qwen4_moe_q4_k_down(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint output = tgpig.x * (uint)nsg + (uint)sg;
    const uint row = tgpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    const uint row_blocks = args.down_dim / 256u;
    float total = 0.0f;
    for (uint slot = 0u; slot < args.top_k; slot++) {
        const ulong assignment = (ulong)row * args.top_k + slot;
        const int expert = selected[assignment];
        if (expert < 0 || (uint)expert >= args.n_experts) continue;
        const ulong weight_row = (ulong)(uint)expert * args.out_dim + output;
        const float sum = qwen4_q4_k_dot(
            down_weight + weight_row * row_blocks,
            mid + assignment * args.expert_dim,
            args.expert_dim, lane);
        total = fma(sum, selected_weights[assignment], total);
    }
    if (lane == 0u) out[(ulong)row * args.out_dim + output] = total;
}

kernel void kernel_qwen4_moe_q4_k_down_split(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        threadgroup float *partials [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output = tgpig.x;
    const uint row = tgpig.y;
    if (sg < args.top_k) {
        const ulong assignment = (ulong)row * args.top_k + (uint)sg;
        const int expert = selected[assignment];
        float sum = 0.0f;
        if (output < args.out_dim && row < args.n_rows &&
            expert >= 0 && (uint)expert < args.n_experts) {
            const uint row_blocks = args.down_dim / 256u;
            const ulong weight_row = (ulong)(uint)expert * args.out_dim + output;
            sum = qwen4_q4_k_dot(
                down_weight + weight_row * row_blocks,
                mid + assignment * args.expert_dim,
                args.expert_dim, lane);
        }
        if (lane == 0u) partials[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u && output < args.out_dim && row < args.n_rows) {
        float total = 0.0f;
        for (uint slot = 0u; slot < args.top_k; slot++) {
            const ulong assignment = (ulong)row * args.top_k + slot;
            total = fma(partials[slot], selected_weights[assignment], total);
        }
        out[(ulong)row * args.out_dim + output] = total;
    }
}

kernel void kernel_qwen4_shared_expert_add_f32(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const float *routed [[buffer(1)]],
        device const float *shared [[buffer(2)]],
        device const float *raw_gate [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint row = gid.y;
    if (dim >= args.out_dim || row >= args.n_rows) return;
    const ulong at = (ulong)row * args.out_dim + dim;
    const float gate = 1.0f / (1.0f + exp(-raw_gate[row]));
    out[at] = fma(shared[at], gate, routed[at]);
}

kernel void kernel_qwen4_shared_expert_add_model_f32(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const ushort *router_weight [[buffer(1)]],
        device const float *routed [[buffer(2)]],
        device const float *shared [[buffer(3)]],
        device const float *hidden [[buffer(4)]],
        device float *out [[buffer(5)]],
        threadgroup float *gate_value [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    // The unfused BF16 router assigns its only valid output to SIMD group 0.
    // Keep that exact 32-lane FMA and reduction order, then let the full
    // threadgroup apply the resulting scalar gate across the hidden row.
    if (sg == 0u) {
        device const float *xr = hidden + (ulong)row * args.out_dim;
        float sum = 0.0f;
        for (uint dim = lane; dim < args.out_dim; dim += 32u)
            sum = fma(qwen4_bf16_to_f32(router_weight[dim]), xr[dim], sum);
        sum = simd_sum(sum);
        if (lane == 0u) gate_value[0] = 1.0f / (1.0f + exp(-sum));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = gate_value[0];
    const ulong base = (ulong)row * args.out_dim;
    for (uint dim = tid; dim < args.out_dim; dim += 256u) {
        const ulong at = base + dim;
        out[at] = fma(shared[at], gate, routed[at]);
    }
}

struct qwen4_gdn_args {
    uint n_tokens;
    uint key_heads;
    uint value_heads;
    uint head_dim;
    uint has_mask;
};

struct qwen4_gdn_prepare_args {
    uint n_tokens;
    uint key_heads;
    uint value_heads;
    uint head_dim;
    uint conv_width;
    uint has_mask;
    float l2_eps;
};

// One thread owns one depthwise-convolution channel and walks the outer
// chunk in token order.  This preserves the exact causal state transition
// while keeping only four FP32 values live for the supported Qwen geometry.
kernel void kernel_qwen4_gdn_conv_split_f32(
        constant qwen4_gdn_prepare_args &args [[buffer(0)]],
        device const float *mixed_qkv [[buffer(1)]],
        device const ushort *conv_weight [[buffer(2)]],
        device const uchar *mask [[buffer(3)]],
        device float *conv_state [[buffer(4)]],
        device float *q [[buffer(5)]],
        device float *k [[buffer(6)]],
        device float *v [[buffer(7)]],
        uint channel [[thread_position_in_grid]]) {
    const uint key_dim = args.key_heads * args.head_dim;
    const uint value_dim = args.value_heads * args.head_dim;
    const uint conv_dim = 2u * key_dim + value_dim;
    if (channel >= conv_dim || args.conv_width != 4u) return;

    float history[4];
    for (uint tap = 0; tap < 4u; tap++)
        history[tap] = conv_state[(ulong)channel * 4u + tap];

    for (uint token = 0; token < args.n_tokens; token++) {
        history[0] = history[1];
        history[1] = history[2];
        history[2] = history[3];
        const bool active = args.has_mask == 0u || mask[token] != 0u;
        history[3] = active
            ? mixed_qkv[(ulong)token * conv_dim + channel]
            : 0.0f;
        float sum = 0.0f;
        for (uint tap = 0; tap < 4u; tap++)
            sum = fma(history[tap],
                      qwen4_bf16_to_f32(
                          conv_weight[(ulong)channel * 4u + tap]),
                      sum);
        const float activated = sum / (1.0f + exp(-sum));
        if (channel < key_dim) {
            q[(ulong)token * key_dim + channel] = activated;
        } else if (channel < 2u * key_dim) {
            k[(ulong)token * key_dim + channel - key_dim] = activated;
        } else {
            v[(ulong)token * value_dim + channel - 2u * key_dim] =
                activated;
        }
    }
    for (uint tap = 0; tap < 4u; tap++)
        conv_state[(ulong)channel * 4u + tap] = history[tap];
}

kernel void kernel_qwen4_gdn_qk_l2norm_f32(
        constant qwen4_gdn_prepare_args &args [[buffer(0)]],
        device float *q [[buffer(1)]],
        device float *k [[buffer(2)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= args.n_tokens || head >= args.key_heads) return;
    const ulong base = ((ulong)token * args.key_heads + head) * args.head_dim;
    float qsum = 0.0f;
    float ksum = 0.0f;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float qv = q[base + dim];
        const float kv = k[base + dim];
        qsum = fma(qv, qv, qsum);
        ksum = fma(kv, kv, ksum);
    }
    qsum = simd_sum(qsum);
    ksum = simd_sum(ksum);
    if (lane == 0u) {
        partial[2u * sg] = qsum;
        partial[2u * sg + 1u] = ksum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        qsum = lane < nsg ? partial[2u * lane] : 0.0f;
        ksum = lane < nsg ? partial[2u * lane + 1u] : 0.0f;
        qsum = simd_sum(qsum);
        ksum = simd_sum(ksum);
        if (lane == 0u) {
            partial[0] = rsqrt(qsum + args.l2_eps) *
                         rsqrt((float)args.head_dim);
            partial[1] = rsqrt(ksum + args.l2_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        q[base + dim] *= partial[0];
        k[base + dim] *= partial[1];
    }
}

kernel void kernel_qwen4_gdn_gates_f32(
        constant qwen4_gdn_prepare_args &args [[buffer(0)]],
        device const float *raw_decay [[buffer(1)]],
        device const float *raw_beta [[buffer(2)]],
        device const ushort *a_log [[buffer(3)]],
        device const ushort *dt_bias [[buffer(4)]],
        device const uchar *mask [[buffer(5)]],
        device float *decay [[buffer(6)]],
        device float *beta [[buffer(7)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint head = gid.x;
    const uint token = gid.y;
    if (head >= args.value_heads || token >= args.n_tokens) return;
    const ulong at = (ulong)token * args.value_heads + head;
    const bool active = args.has_mask == 0u || mask[token] != 0u;
    const float a = active ? raw_decay[at] : 0.0f;
    const float b = active ? raw_beta[at] : 0.0f;
    const float shifted = a + qwen4_bf16_to_f32(dt_bias[head]);
    const float softplus = shifted > 20.0f
        ? shifted
        : log(1.0f + exp(shifted));
    const float rate = exp(qwen4_bf16_to_f32(a_log[head]));
    decay[at] = exp(-rate * softplus);
    beta[at] = 1.0f / (1.0f + exp(-b));
}

struct qwen4_gdn_output_args {
    uint rows;
    uint head_dim;
    float eps;
};

kernel void kernel_qwen4_gdn_output_norm_f32(
        constant qwen4_gdn_output_args &args [[buffer(0)]],
        device const float *core [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *weight [[buffer(3)]],
        device float *out [[buffer(4)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.rows) return;
    const ulong base = (ulong)row * args.head_dim;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float value = core[base + dim];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(sum / (float)args.head_dim + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = partial[0];
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float gate = 1.0f /
            (1.0f + exp(-raw_gate[base + dim]));
        out[base + dim] = core[base + dim] * inv_rms *
                          qwen4_bf16_to_f32(weight[dim]) * gate;
    }
}

template <uint R>
static inline void qwen4_gdn_rows(
        constant qwen4_gdn_args &args,
        device const float      *q,
        device const float      *k,
        device const float      *v,
        device const float      *decay,
        device const float      *beta,
        device const uchar      *mask,
        device float            *state,
        device float            *out,
        uint3 gid,
        ushort lane) {
    const uint row0 = gid.y * R;
    const uint hv = gid.z;
    if (lane >= 32u || row0 >= args.head_dim || hv >= args.value_heads) return;
    const uint hk = hv / (args.value_heads / args.key_heads);
    constexpr uint per_lane = 4u; // supported geometry has Dk=128.
    float local[R][per_lane];
    for (uint r = 0; r < R; r++) {
        for (uint i = 0; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            const ulong index = ((ulong)hv * args.head_dim + row0 + r) *
                                args.head_dim + dk;
            local[r][i] = state[index];
        }
    }
    for (uint t = 0; t < args.n_tokens; t++) {
        const bool active = args.has_mask == 0u || mask[t] != 0u;
        if (active) {
            const float d = decay[(ulong)t * args.value_heads + hv];
            const float b = beta[(ulong)t * args.value_heads + hv];
            for (uint r = 0; r < R; r++) {
                float kv = 0.0f;
                for (uint i = 0; i < per_lane; i++) {
                    const uint dk = lane * per_lane + i;
                    const float kk = k[((ulong)t * args.key_heads + hk) *
                                       args.head_dim + dk];
                    local[r][i] *= d;
                    kv = fma(local[r][i], kk, kv);
                }
                kv = simd_sum(kv);
                const float delta =
                    (v[((ulong)t * args.value_heads + hv) * args.head_dim +
                       row0 + r] - kv) * b;
                float y = 0.0f;
                for (uint i = 0; i < per_lane; i++) {
                    const uint dk = lane * per_lane + i;
                    const ulong qk_index =
                        ((ulong)t * args.key_heads + hk) * args.head_dim + dk;
                    local[r][i] = fma(k[qk_index], delta, local[r][i]);
                    y = fma(local[r][i], q[qk_index], y);
                }
                y = simd_sum(y);
                if (lane == 0u)
                    out[((ulong)t * args.value_heads + hv) * args.head_dim +
                        row0 + r] = y;
            }
        } else if (lane == 0u) {
            for (uint r = 0; r < R; r++)
                out[((ulong)t * args.value_heads + hv) * args.head_dim +
                    row0 + r] = 0.0f;
        }
    }
    for (uint r = 0; r < R; r++) {
        for (uint i = 0; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            const ulong index = ((ulong)hv * args.head_dim + row0 + r) *
                                args.head_dim + dk;
            state[index] = local[r][i];
        }
    }
}

#define QWEN4_GDN_KERNEL(NAME, R)                                               \
kernel void NAME(                                                              \
        constant qwen4_gdn_args &args [[buffer(0)]],                           \
        device const float *q [[buffer(1)]],                                   \
        device const float *k [[buffer(2)]],                                   \
        device const float *v [[buffer(3)]],                                   \
        device const float *decay [[buffer(4)]],                               \
        device const float *beta [[buffer(5)]],                                \
        device const uchar *mask [[buffer(6)]],                                \
        device float *state [[buffer(7)]],                                     \
        device float *out [[buffer(8)]],                                       \
        uint3 gid [[thread_position_in_grid]],                                 \
        ushort lane [[thread_index_in_simdgroup]]) {                           \
    qwen4_gdn_rows<R>(args, q, k, v, decay, beta, mask, state, out, gid, lane);\
}

QWEN4_GDN_KERNEL(kernel_qwen4_gdn_r4_f32, 4)
QWEN4_GDN_KERNEL(kernel_qwen4_gdn_r2_f32, 2)
QWEN4_GDN_KERNEL(kernel_qwen4_gdn_r1_f32, 1)

struct qwen4_qsa_prepare_args {
    uint cache_pos;
    uint n_tokens;
    uint cache_cap;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint index_heads;
    uint index_head_dim;
    uint ratio;
    uint rope_dim;
    uint pool_block_start;
    uint has_mrope;
    uint mrope_len;
    float rope_theta;
    float rms_eps;
};

static inline float qwen4_partial_rope(
        float current,
        float paired,
        uint dim,
        uint rope_dim,
        uint position,
        float theta) {
    if (dim >= rope_dim) return current;
    const uint half_dim = rope_dim / 2u;
    const uint freq = dim < half_dim ? dim : dim - half_dim;
    const float inv_freq = pow(theta,
        -2.0f * (float)freq / (float)rope_dim);
    const float angle = (float)position * inv_freq;
    const float c = cos(angle);
    const float s = sin(angle);
    return dim < half_dim ? fma(-paired, s, current * c)
                          : fma(paired, s, current * c);
}

static inline uint qwen4_rope_position(
        constant qwen4_qsa_prepare_args &args,
        device const int *mrope_positions,
        uint absolute,
        uint dim) {
    if (args.has_mrope == 0u) return absolute;
    const uint frequency = dim < args.rope_dim / 2u
        ? dim : dim - args.rope_dim / 2u;
    const uint axis = frequency % 3u;
    return (uint)mrope_positions[(ulong)axis * args.mrope_len + absolute];
}

kernel void kernel_qwen4_qsa_main_q_prepare_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const float *q_gate_raw [[buffer(1)]],
        device const ushort *norm_weight [[buffer(2)]],
        device float *q [[buffer(3)]],
        device float *gate [[buffer(4)]],
        device const int *mrope_positions [[buffer(5)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= args.n_tokens || head >= args.query_heads) return;
    const ulong raw_base =
        (ulong)token * args.query_heads * args.head_dim * 2u +
        (ulong)head * args.head_dim * 2u;
    const ulong out_base =
        ((ulong)token * args.query_heads + head) * args.head_dim;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float value = q_gate_raw[raw_base + dim];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(sum / (float)args.head_dim + args.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const uint half_dim = args.rope_dim / 2u;
    const uint absolute = args.cache_pos + token;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        float value = q_gate_raw[raw_base + dim] * scale *
            qwen4_bf16_to_f32(norm_weight[dim]);
        float paired = 0.0f;
        if (dim < args.rope_dim) {
            const uint pair = dim < half_dim
                ? dim + half_dim : dim - half_dim;
            paired = q_gate_raw[raw_base + pair] * scale *
                qwen4_bf16_to_f32(norm_weight[pair]);
        }
        q[out_base + dim] = qwen4_partial_rope(
            value, paired, dim, args.rope_dim,
            qwen4_rope_position(args, mrope_positions, absolute, dim),
            args.rope_theta);
        gate[out_base + dim] = q_gate_raw[raw_base + args.head_dim + dim];
    }
}

kernel void kernel_qwen4_qsa_main_kv_prepare_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const float *key_raw [[buffer(1)]],
        device const float *value_raw [[buffer(2)]],
        device const ushort *norm_weight [[buffer(3)]],
        device ushort *key_cache [[buffer(4)]],
        device ushort *value_cache [[buffer(5)]],
        device const int *mrope_positions [[buffer(6)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= args.n_tokens || head >= args.kv_heads) return;
    const ulong raw_base =
        ((ulong)token * args.kv_heads + head) * args.head_dim;
    const ulong cache_base =
        ((ulong)(args.cache_pos + token) * args.kv_heads + head) *
        args.head_dim;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float value = key_raw[raw_base + dim];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(sum / (float)args.head_dim + args.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const uint half_dim = args.rope_dim / 2u;
    const uint absolute = args.cache_pos + token;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        float value = key_raw[raw_base + dim] * scale *
            qwen4_bf16_to_f32(norm_weight[dim]);
        float paired = 0.0f;
        if (dim < args.rope_dim) {
            const uint pair = dim < half_dim
                ? dim + half_dim : dim - half_dim;
            paired = key_raw[raw_base + pair] * scale *
                qwen4_bf16_to_f32(norm_weight[pair]);
        }
        value = qwen4_partial_rope(
            value, paired, dim, args.rope_dim,
            qwen4_rope_position(args, mrope_positions, absolute, dim),
            args.rope_theta);
        key_cache[cache_base + dim] = qwen4_f32_to_bf16(value);
        value_cache[cache_base + dim] =
            qwen4_f32_to_bf16(value_raw[raw_base + dim]);
    }
}

kernel void kernel_qwen4_qsa_index_q_prepare_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const float *index_qk_raw [[buffer(1)]],
        device const ushort *norm_weight [[buffer(2)]],
        device float *index_q [[buffer(3)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= args.n_tokens || head >= args.index_heads) return;
    const ulong raw_base =
        (ulong)token * (args.index_heads + 1u) * args.index_head_dim +
        (ulong)head * args.index_head_dim;
    const ulong out_base =
        ((ulong)token * args.index_heads + head) * args.index_head_dim;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.index_head_dim;
         dim += (uint)nsg * 32u) {
        const float value = index_qk_raw[raw_base + dim];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(
                sum / (float)args.index_head_dim + args.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const uint half_dim = args.rope_dim / 2u;
    const uint position = args.cache_pos + token;
    for (uint dim = tid; dim < args.index_head_dim;
         dim += (uint)nsg * 32u) {
        float value = index_qk_raw[raw_base + dim] * scale *
            qwen4_bf16_to_f32(norm_weight[dim]);
        float paired = 0.0f;
        if (dim < args.rope_dim) {
            const uint pair = dim < half_dim
                ? dim + half_dim : dim - half_dim;
            paired = index_qk_raw[raw_base + pair] * scale *
                qwen4_bf16_to_f32(norm_weight[pair]);
        }
        index_q[out_base + dim] = qwen4_partial_rope(
            value, paired, dim, args.rope_dim, position, args.rope_theta);
    }
}

kernel void kernel_qwen4_qsa_index_key_store_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const float *index_qk_raw [[buffer(1)]],
        device ushort *raw_index_cache [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token = gid.y;
    if (dim >= args.index_head_dim || token >= args.n_tokens) return;
    const ulong raw_at =
        (ulong)token * (args.index_heads + 1u) * args.index_head_dim +
        (ulong)args.index_heads * args.index_head_dim + dim;
    raw_index_cache[(ulong)(args.cache_pos + token) *
                    args.index_head_dim + dim] =
        qwen4_f32_to_bf16(index_qk_raw[raw_at]);
}

static inline float qwen4_qsa_pooled_raw(
        device const ushort *raw_index_cache,
        uint block,
        uint ratio,
        uint head_dim,
        uint dim) {
    float sum = 0.0f;
    for (uint item = 0; item < ratio; item++)
        sum += qwen4_bf16_to_f32(
            raw_index_cache[((ulong)block * ratio + item) * head_dim + dim]);
    const float mean = sum / (float)ratio;
    return qwen4_bf16_to_f32(qwen4_f32_to_bf16(mean));
}

kernel void kernel_qwen4_qsa_index_pool_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const ushort *raw_index_cache [[buffer(1)]],
        device const ushort *norm_weight [[buffer(2)]],
        device ushort *pooled_index_cache [[buffer(3)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint local_block [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = args.pool_block_start + local_block;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.index_head_dim;
         dim += (uint)nsg * 32u) {
        const float value = qwen4_qsa_pooled_raw(
            raw_index_cache, block, args.ratio, args.index_head_dim, dim);
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(
                sum / (float)args.index_head_dim + args.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const uint half_dim = args.rope_dim / 2u;
    const uint position = block * args.ratio;
    for (uint dim = tid; dim < args.index_head_dim;
         dim += (uint)nsg * 32u) {
        float value = qwen4_qsa_pooled_raw(
            raw_index_cache, block, args.ratio, args.index_head_dim, dim) *
            scale * qwen4_bf16_to_f32(norm_weight[dim]);
        float paired = 0.0f;
        if (dim < args.rope_dim) {
            const uint pair = dim < half_dim
                ? dim + half_dim : dim - half_dim;
            paired = qwen4_qsa_pooled_raw(
                raw_index_cache, block, args.ratio,
                args.index_head_dim, pair) * scale *
                qwen4_bf16_to_f32(norm_weight[pair]);
        }
        value = qwen4_partial_rope(
            value, paired, dim, args.rope_dim, position, args.rope_theta);
        pooled_index_cache[(ulong)block * args.index_head_dim + dim] =
            qwen4_f32_to_bf16(value);
    }
}

struct qwen4_qsa_attention_args {
    uint queries;
    uint cache_cap;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint top_k;
    uint ratio;
    uint max_selected;
};

template <bool MEMOIZE_NUMERATORS>
static inline void qwen4_qsa_attention_bf16_f32(
        constant qwen4_qsa_attention_args &args,
        device const float *q,
        device const float *raw_gate,
        device const ushort *key_cache,
        device const ushort *value_cache,
        device const uint *selected_blocks,
        device const uint *selected_counts,
        device const uint *visible_tokens,
        device float *out,
        threadgroup float *scratch,
        uint2 group,
        uint tid,
        ushort lane,
        ushort sg,
        ushort nsg) {
    const uint query = group.x;
    const uint head = group.y;
    if (query >= args.queries || head >= args.query_heads) return;
    const uint visible = min(visible_tokens[query], args.cache_cap);
    const uint complete = visible / args.ratio;
    const uint block_count = min(
        min(selected_counts[query], args.top_k), complete);
    const uint tail = visible - complete * args.ratio;
    const uint selected = block_count * args.ratio + tail;
    const ulong qbase =
        ((ulong)query * args.query_heads + head) * args.head_dim;
    const uint kv_head = head / (args.query_heads / args.kv_heads);

    for (uint rank = sg; rank < selected; rank += (uint)nsg) {
        uint token;
        if (rank < block_count * args.ratio) {
            const uint block = selected_blocks[
                (ulong)query * args.top_k + rank / args.ratio];
            token = block * args.ratio + rank % args.ratio;
        } else {
            token = complete * args.ratio +
                    rank - block_count * args.ratio;
        }
        float dot = 0.0f;
        if (token < visible) {
            const ulong kbase =
                ((ulong)token * args.kv_heads + kv_head) * args.head_dim;
            for (uint dim = lane; dim < args.head_dim; dim += 32u)
                dot = fma(q[qbase + dim],
                          qwen4_bf16_to_f32(key_cache[kbase + dim]), dot);
            dot = simd_sum(dot);
            if (lane == 0u)
                scratch[rank] = dot * rsqrt((float)args.head_dim);
        } else if (lane == 0u) {
            scratch[rank] = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (selected == 0u) {
        if (tid < args.head_dim) out[qbase + tid] = 0.0f;
        return;
    }

    const uint aux = args.max_selected;
    float local_max = -INFINITY;
    for (uint rank = tid; rank < selected; rank += (uint)nsg * 32u)
        local_max = max(local_max, scratch[rank]);
    local_max = simd_max(local_max);
    if (lane == 0u) scratch[aux + sg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        local_max = lane < nsg ? scratch[aux + lane] : -INFINITY;
        local_max = simd_max(local_max);
        if (lane == 0u) scratch[aux] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    /* Keep the broadcast maximum outside the per-simdgroup partial range.
     * The next reduction immediately reuses scratch[aux + sg]; sharing slot
     * zero lets SIMD group 0 overwrite the maximum before another group has
     * loaded it. */
    const uint maximum_slot = aux + (uint)nsg;
    if (sg == 0u && lane == 0u) scratch[maximum_slot] = scratch[aux];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float maximum = scratch[maximum_slot];

    float local_sum = 0.0f;
    for (uint rank = tid; rank < selected; rank += (uint)nsg * 32u) {
        /* The maximum is fixed, so the memoized specialization can replace
         * each score with its unnormalized probability in the same slot. */
        const float numerator = exp(scratch[rank] - maximum);
        if (MEMOIZE_NUMERATORS) scratch[rank] = numerator;
        local_sum += numerator;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0u) scratch[aux + sg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        local_sum = lane < nsg ? scratch[aux + lane] : 0.0f;
        local_sum = simd_sum(local_sum);
        if (lane == 0u) scratch[aux] = 1.0f / local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = scratch[aux];

    if (tid < args.head_dim) {
        float value = 0.0f;
        for (uint rank = 0; rank < selected; rank++) {
            uint token;
            if (rank < block_count * args.ratio) {
                const uint block = selected_blocks[
                    (ulong)query * args.top_k + rank / args.ratio];
                token = block * args.ratio + rank % args.ratio;
            } else {
                token = complete * args.ratio +
                        rank - block_count * args.ratio;
            }
            if (token < visible) {
                const ulong vbase =
                    ((ulong)token * args.kv_heads + kv_head) * args.head_dim;
                const float numerator = MEMOIZE_NUMERATORS
                    ? scratch[rank] : exp(scratch[rank] - maximum);
                value = fma(numerator * inv_sum,
                            qwen4_bf16_to_f32(value_cache[vbase + tid]),
                            value);
            }
        }
        const float gate = 1.0f / (1.0f + exp(-raw_gate[qbase + tid]));
        out[qbase + tid] = value * gate;
    }
}

kernel void kernel_qwen4_qsa_attention_bf16_f32(
        constant qwen4_qsa_attention_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *key_cache [[buffer(3)]],
        device const ushort *value_cache [[buffer(4)]],
        device const uint *selected_blocks [[buffer(5)]],
        device const uint *selected_counts [[buffer(6)]],
        device const uint *visible_tokens [[buffer(7)]],
        device float *out [[buffer(8)]],
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    qwen4_qsa_attention_bf16_f32<true>(
        args, q, raw_gate, key_cache, value_cache, selected_blocks,
        selected_counts, visible_tokens, out, scratch, group, tid, lane, sg,
        nsg);
}

kernel void kernel_qwen4_qsa_attention_bf16_f32_legacy(
        constant qwen4_qsa_attention_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *key_cache [[buffer(3)]],
        device const ushort *value_cache [[buffer(4)]],
        device const uint *selected_blocks [[buffer(5)]],
        device const uint *selected_counts [[buffer(6)]],
        device const uint *visible_tokens [[buffer(7)]],
        device float *out [[buffer(8)]],
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    qwen4_qsa_attention_bf16_f32<false>(
        args, q, raw_gate, key_cache, value_cache, selected_blocks,
        selected_counts, visible_tokens, out, scratch, group, tid, lane, sg,
        nsg);
}

kernel void kernel_qwen4_qsa_dense_attention_bf16_f32(
        constant qwen4_qsa_attention_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *key_cache [[buffer(3)]],
        device const ushort *value_cache [[buffer(4)]],
        device const uint *visible_tokens [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint query = group.x;
    const uint head = group.y;
    if (query >= args.queries || head >= args.query_heads) return;
    const uint visible = min(visible_tokens[query], args.cache_cap);
    const uint kv_head = head / (args.query_heads / args.kv_heads);
    const ulong qbase =
        ((ulong)query * args.query_heads + head) * args.head_dim;
    float qv[8];
    float acc[8];
    for (uint i = 0; i < 8u; i++) {
        qv[i] = q[qbase + (uint)lane + i * 32u];
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;
    const float scale = rsqrt((float)args.head_dim);
    for (uint token = 0; token < visible; token++) {
        const ulong base =
            ((ulong)token * args.kv_heads + kv_head) * args.head_dim;
        float dot = 0.0f;
        for (uint i = 0; i < 8u; i++)
            dot = fma(qv[i], qwen4_bf16_to_f32(
                key_cache[base + (uint)lane + i * 32u]), dot);
        dot = simd_sum(dot) * scale;
        const float next = max(maximum, dot);
        const float old_scale = maximum == -INFINITY
            ? 0.0f : exp(maximum - next);
        const float new_scale = exp(dot - next);
        denominator = denominator * old_scale + new_scale;
        for (uint i = 0; i < 8u; i++) {
            const uint d = (uint)lane + i * 32u;
            acc[i] = acc[i] * old_scale + new_scale *
                qwen4_bf16_to_f32(value_cache[base + d]);
        }
        maximum = next;
    }
    for (uint i = 0; i < 8u; i++) {
        const uint d = (uint)lane + i * 32u;
        const float value = visible == 0u ? 0.0f : acc[i] / denominator;
        out[qbase + d] = value /
            (1.0f + exp(-raw_gate[qbase + d]));
    }
}

struct qwen4_qsa_score_args {
    uint blocks;
    uint heads;
    uint head_dim;
    uint valid_blocks;
};

// Exact single-query QSA block scoring: one simdgroup owns one pooled key,
// retaining the same 32-lane FP32 reduction tree for every context length.
kernel void kernel_qwen4_qsa_score_m1_f32(
        constant qwen4_qsa_score_args &args [[buffer(0)]],
        device const float            *q [[buffer(1)]],
        device const float            *pooled_k [[buffer(2)]],
        device float                  *scores [[buffer(3)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = tgp * (uint)nsg + sg;
    if (block >= args.blocks) return;
    float score = 0.0f;
    for (uint h = 0; h < args.heads; h++) {
        float dot_value = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            dot_value = fma(q[(ulong)h * args.head_dim + d],
                            pooled_k[(ulong)block * args.head_dim + d],
                            dot_value);
        }
        dot_value = simd_sum(dot_value);
        if (lane == 0u) score += max(dot_value, 0.0f);
    }
    if (lane == 0u) {
        scores[block] = block < args.valid_blocks
            ? score * rsqrt((float)args.head_dim)
            : -INFINITY;
    }
}

// Low-precision cache variants retain FP32 products and the identical
// lane/reduction order used by the F32 reference specialization.
kernel void kernel_qwen4_qsa_score_m1_f16(
        constant qwen4_qsa_score_args &args [[buffer(0)]],
        device const half             *q [[buffer(1)]],
        device const half             *pooled_k [[buffer(2)]],
        device float                  *scores [[buffer(3)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = tgp * (uint)nsg + sg;
    if (block >= args.blocks) return;
    float score = 0.0f;
    for (uint h = 0; h < args.heads; h++) {
        float dot_value = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            dot_value = fma((float)q[(ulong)h * args.head_dim + d],
                            (float)pooled_k[(ulong)block * args.head_dim + d],
                            dot_value);
        }
        dot_value = simd_sum(dot_value);
        if (lane == 0u) score += max(dot_value, 0.0f);
    }
    if (lane == 0u) {
        scores[block] = block < args.valid_blocks
            ? score * rsqrt((float)args.head_dim)
            : -INFINITY;
    }
}

kernel void kernel_qwen4_qsa_score_m1_bf16(
        constant qwen4_qsa_score_args &args [[buffer(0)]],
        device const ushort           *q [[buffer(1)]],
        device const ushort           *pooled_k [[buffer(2)]],
        device float                  *scores [[buffer(3)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = tgp * (uint)nsg + sg;
    if (block >= args.blocks) return;
    float score = 0.0f;
    for (uint h = 0; h < args.heads; h++) {
        float dot_value = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            dot_value = fma(qwen4_bf16_to_f32(
                                q[(ulong)h * args.head_dim + d]),
                            qwen4_bf16_to_f32(
                                pooled_k[(ulong)block * args.head_dim + d]),
                            dot_value);
        }
        dot_value = simd_sum(dot_value);
        if (lane == 0u) score += max(dot_value, 0.0f);
    }
    if (lane == 0u) {
        scores[block] = block < args.valid_blocks
            ? score * rsqrt((float)args.head_dim)
            : -INFINITY;
    }
}

kernel void kernel_qwen4_qsa_score_m1_f32_bf16(
        constant qwen4_qsa_score_args &args [[buffer(0)]],
        device const float            *q [[buffer(1)]],
        device const ushort           *pooled_k [[buffer(2)]],
        device float                  *scores [[buffer(3)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = tgp * (uint)nsg + sg;
    if (block >= args.blocks) return;
    float score = 0.0f;
    for (uint h = 0; h < args.heads; h++) {
        float dot_value = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u)
            dot_value = fma(q[(ulong)h * args.head_dim + d],
                            qwen4_bf16_to_f32(
                                pooled_k[(ulong)block * args.head_dim + d]),
                            dot_value);
        dot_value = simd_sum(dot_value);
        if (lane == 0u) score += max(dot_value, 0.0f);
    }
    if (lane == 0u) {
        scores[block] = block < args.valid_blocks
            ? score * rsqrt((float)args.head_dim)
            : -INFINITY;
    }
}

struct qwen4_qsa_tile_args {
    uint queries;
    uint blocks;
    uint block_start;
    uint tile_blocks;
    uint heads;
    uint head_dim;
    uint top_k;
};

// Score one pooled-key tile.  The controller folds it into a persistent
// per-query heap immediately, bounding scratch to queries * tile_blocks.
// visible_blocks carries causal lengths as runtime data, replacing a full
// [queries, context] boolean mask.
kernel void kernel_qwen4_qsa_score_tile_f32(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device const float           *q [[buffer(1)]],
        device const float           *pooled_k [[buffer(2)]],
        device const uint            *visible_blocks [[buffer(3)]],
        device float                 *tile_scores [[buffer(4)]],
        uint2 tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint local_block = tgp.x * (uint)nsg + sg;
    const uint query = tgp.y;
    if (query >= args.queries || local_block >= args.tile_blocks) return;
    const uint block = args.block_start + local_block;
    float score = 0.0f;
    if (block < args.blocks && block < visible_blocks[query]) {
        for (uint h = 0; h < args.heads; ++h) {
            float dot = 0.0f;
            const ulong qbase = ((ulong)query * args.heads + h) * args.head_dim;
            const ulong kbase = (ulong)block * args.head_dim;
            for (uint d = lane; d < args.head_dim; d += 32u)
                dot = fma(q[qbase + d], pooled_k[kbase + d], dot);
            dot = simd_sum(dot);
            if (lane == 0u) score += max(dot, 0.0f);
        }
        if (lane == 0u) score *= rsqrt((float)args.head_dim);
    } else if (lane == 0u) {
        score = -INFINITY;
    }
    if (lane == 0u)
        tile_scores[(ulong)query * args.tile_blocks + local_block] = score;
}

kernel void kernel_qwen4_qsa_score_tile_bf16(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device const float           *q [[buffer(1)]],
        device const ushort          *pooled_k [[buffer(2)]],
        device const uint            *visible_blocks [[buffer(3)]],
        device float                 *tile_scores [[buffer(4)]],
        uint2 tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint local_block = tgp.x * (uint)nsg + sg;
    const uint query = tgp.y;
    if (query >= args.queries || local_block >= args.tile_blocks) return;
    const uint block = args.block_start + local_block;
    float score = 0.0f;
    if (block < args.blocks && block < visible_blocks[query]) {
        for (uint h = 0; h < args.heads; ++h) {
            float dot = 0.0f;
            const ulong qbase = ((ulong)query * args.heads + h) * args.head_dim;
            const ulong kbase = (ulong)block * args.head_dim;
            for (uint d = lane; d < args.head_dim; d += 32u)
                dot = fma(q[qbase + d],
                          qwen4_bf16_to_f32(pooled_k[kbase + d]), dot);
            dot = simd_sum(dot);
            if (lane == 0u) score += max(dot, 0.0f);
        }
        if (lane == 0u) score *= rsqrt((float)args.head_dim);
    } else if (lane == 0u) {
        score = -INFINITY;
    }
    if (lane == 0u)
        tile_scores[(ulong)query * args.tile_blocks + local_block] = score;
}

static inline bool qwen4_qsa_worse(float as, uint ai, float bs, uint bi) {
    return as < bs || (as == bs && ai > bi);
}

static inline void qwen4_qsa_heap_down(
        device float *scores,
        device uint *indices,
        uint count,
        uint root) {
    while (true) {
        const uint left = root * 2u + 1u;
        if (left >= count) return;
        uint worse = left;
        const uint right = left + 1u;
        if (right < count &&
            qwen4_qsa_worse(scores[right], indices[right],
                            scores[left], indices[left])) worse = right;
        if (!qwen4_qsa_worse(scores[worse], indices[worse],
                             scores[root], indices[root])) return;
        const float ts = scores[root];
        const uint ti = indices[root];
        scores[root] = scores[worse];
        indices[root] = indices[worse];
        scores[worse] = ts;
        indices[worse] = ti;
        root = worse;
    }
}

kernel void kernel_qwen4_qsa_heap_init_f32(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device float                 *heap_scores [[buffer(1)]],
        device uint                  *heap_indices [[buffer(2)]],
        device uint                  *heap_counts [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong total = (ulong)args.queries * args.top_k;
    if ((ulong)gid < total) {
        heap_scores[gid] = -INFINITY;
        heap_indices[gid] = 0xffffffffu;
    }
    if (gid < args.queries) heap_counts[gid] = 0u;
}

// Dot products above are parallel; only the compact top-k structure is
// serialized per query.  A min-heap makes each candidate O(log(top_k)).
kernel void kernel_qwen4_qsa_heap_merge_f32(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device const float           *tile_scores [[buffer(1)]],
        device float                 *heap_scores [[buffer(2)]],
        device uint                  *heap_indices [[buffer(3)]],
        device uint                  *heap_counts [[buffer(4)]],
        uint query [[thread_position_in_grid]]) {
    if (query >= args.queries) return;
    device float *hs = heap_scores + (ulong)query * args.top_k;
    device uint *hi = heap_indices + (ulong)query * args.top_k;
    uint count = heap_counts[query];
    for (uint i = 0; i < args.tile_blocks; ++i) {
        const float score = tile_scores[(ulong)query * args.tile_blocks + i];
        if (!isfinite(score)) continue;
        const uint index = args.block_start + i;
        if (count < args.top_k) {
            uint child = count++;
            hs[child] = score;
            hi[child] = index;
            while (child != 0u) {
                const uint parent = (child - 1u) / 2u;
                if (!qwen4_qsa_worse(hs[child], hi[child], hs[parent], hi[parent])) break;
                const float ts = hs[parent];
                const uint ti = hi[parent];
                hs[parent] = hs[child]; hi[parent] = hi[child];
                hs[child] = ts; hi[child] = ti;
                child = parent;
            }
        } else if (!qwen4_qsa_worse(score, index, hs[0], hi[0]) &&
                   !(score == hs[0] && index == hi[0])) {
            hs[0] = score;
            hi[0] = index;
            qwen4_qsa_heap_down(hs, hi, count, 0u);
        }
    }
    heap_counts[query] = count;
}

// In-place heap sort leaves score/index pairs in descending score order with
// ascending block index as the deterministic tie break.
kernel void kernel_qwen4_qsa_heap_sort_f32(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device float                 *heap_scores [[buffer(1)]],
        device uint                  *heap_indices [[buffer(2)]],
        device const uint            *heap_counts [[buffer(3)]],
        uint query [[thread_position_in_grid]]) {
    if (query >= args.queries) return;
    device float *hs = heap_scores + (ulong)query * args.top_k;
    device uint *hi = heap_indices + (ulong)query * args.top_k;
    const uint count = heap_counts[query];
    for (uint end = count; end > 1u; --end) {
        const float ts = hs[0];
        const uint ti = hi[0];
        hs[0] = hs[end - 1u]; hi[0] = hi[end - 1u];
        hs[end - 1u] = ts; hi[end - 1u] = ti;
        qwen4_qsa_heap_down(hs, hi, end - 1u, 0u);
    }
}
