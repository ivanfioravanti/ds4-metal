struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    uint32_t causal_start;
    uint32_t causal_ratio;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
    uint32_t causal_start;
    uint32_t causal_ratio;
    uint32_t block_width;
    uint32_t block_top_k;
};

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order, bool causal = false, bool shuffle = false>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];
    const int width = causal ? min(args.ne00,
        int((args.causal_start + uint(i01) + 1u) / args.causal_ratio)) : args.ne00;
    if (i00 >= width) return;
    const int work_width = causal ?
        ((width - 1) / ntg.x) * args.top_k + min((width - 1) % ntg.x + 1, args.top_k) : args.ne0;

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < width) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int reg_idx = i00 + col;
    float reg_value = reg_idx < width ? shmem_f32[col] : 0.0f;
    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (shuffle && j < 32) {
                if (k > 32 && j == 16) {
                    reg_idx = shmem_i32[col];
                    reg_value = reg_idx < width ? shmem_f32[reg_idx - i00] : 0.0f;
                }
                const int other = simd_shuffle_xor(reg_idx, j);
                const float value = simd_shuffle_xor(reg_value, j);
                const bool first = ((col & k) == 0) == ((col & j) == 0);
                const bool exchange = first ?
                    (reg_idx >= width || (other < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value > value : reg_value < value))) :
                    (other >= width || (reg_idx < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value < value : reg_value > value)));
                if (exchange) { reg_idx = other; reg_value = value; }
                continue;
            }
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                       (shmem_i32[ixj] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                       (shmem_i32[col] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shuffle && k >= 32) {
            shmem_i32[col] = reg_idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < work_width && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shuffle ? reg_idx : shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_f32_i32_desc_shuffle")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, false, true>;
template [[host_name("kernel_argsort_f32_i32_desc_causal")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_f32_i32_desc_causal_shuffle")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order, bool causal = false, bool prefix = false>
kernel void kernel_argsort_merge_f32_i32(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);
    const uint width = causal ? min(uint(args.ne00),
        (args.causal_start + uint(i01) + 1u) / args.causal_ratio) : 0u;
    const int work_width = causal ? int(((width - 1u) / args.block_width) * args.block_top_k +
        min((width - 1u) % args.block_width + 1u, args.block_top_k)) : args.ne0;

    // A merged run can contribute at most 512 entries to the final result.
    // Keep the original run offsets so the comparison and tie order agree.
    const int read_limit = prefix ? min(args.len, 512) : args.len;
    const int len0 = MIN(read_limit, MAX(0, work_width - start));
    const int len1 = MIN(read_limit, MAX(0, work_width - (start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int limit = prefix ? min(args.top_k, 512) : args.top_k;
    const int k1 = MIN(MIN(k0 + chunk, total), limit);

    if (k0 >= limit) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal_prefix")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

// ---- Whole-GPU top-k for wide score rows -----------------------------------
//
// kernel_argsort_f32_i32 and its merge ladder finish a 131072-wide index row
// with two or three threadgroups doing the last merges, so one owner's
// selection costs several hundred microseconds at long context.  This path
// keeps every core busy: a radix select over the order-preserving key bits
// finds the k-th largest score in three histogram passes, a deterministic
// compaction writes the winners by ascending id, and one threadgroup sorts
// them by (score descending, id ascending).  Values by rank equal the merge
// path's; only exactly tied keys can change places.  All kernels run with
// 1024 threads; the host falls back to the merge ladder otherwise.

struct ds4_metal_args_topk_select {
    uint n;        // scores per row
    uint rows;
    uint top_k;    // at most 2048
    uint pass;     // histogram pass 0, 1, 2 over 11 + 11 + 10 key bits
    uint parity;   // pass-0 histogram slot this call fills; the other is cleared
    uint n_tg;     // element threadgroups per row
};

#define DS4_TOPK_BINS       2048u
#define DS4_TOPK_PER_TG     2048u
#define DS4_TOPK_HIST_WORDS (4u * DS4_TOPK_BINS)   // hist0[2], hist1, hist2

// Unsigned key with the float's order: larger score, larger key.
static inline uint ds4_topk_key(float f) {
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The bin holding the k-th largest key and the count of keys above that bin.
static inline uint2 ds4_topk_scan(device atomic_uint *h, uint bins, uint k,
                                  threadgroup uint *scratch,
                                  uint tid, uint lane, uint simd) {
    const uint per = bins / 1024u;
    uint c[2] = {0u, 0u}, local = 0u;
    for (uint q = 0; q < per; q++) {
        c[q] = atomic_load_explicit(h + bins - 1u - (tid * per + q), memory_order_relaxed);
        local += c[q];
    }
    const uint excl = simd_prefix_exclusive_sum(local);
    if (lane == 31u) scratch[simd] = excl + local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint run = 0u;
        for (uint s = 0; s < 32u; s++) { const uint t = scratch[s]; scratch[s] = run; run += t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = scratch[simd] + excl;
    for (uint q = 0; q < per; q++) {
        if (before < k && before + c[q] >= k) {
            scratch[32] = bins - 1u - (tid * per + q);
            scratch[33] = before;
        }
        before += c[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint2 r = uint2(scratch[32], scratch[33]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// Walk the finished passes: (key prefix, count still wanted inside it, keys
// above it).  After three passes that is (threshold, wanted ties, greater).
static inline uint3 ds4_topk_prefix(device atomic_uint *hist, uint parity, uint passes,
                                    uint top_k, threadgroup uint *scratch,
                                    uint tid, uint lane, uint simd) {
    uint prefix = 0u, k = top_k, above = 0u;
    if (passes > 0u) {
        const uint2 r = ds4_topk_scan(hist + parity * DS4_TOPK_BINS, 2048u, k, scratch, tid, lane, simd);
        prefix = r.x; k -= r.y; above += r.y;
    }
    if (passes > 1u) {
        const uint2 r = ds4_topk_scan(hist + 2u * DS4_TOPK_BINS, 2048u, k, scratch, tid, lane, simd);
        prefix = (prefix << 11) | r.x; k -= r.y; above += r.y;
    }
    if (passes > 2u) {
        const uint2 r = ds4_topk_scan(hist + 3u * DS4_TOPK_BINS, 1024u, k, scratch, tid, lane, simd);
        prefix = (prefix << 10) | r.x; k -= r.y; above += r.y;
    }
    return uint3(prefix, k, above);
}

kernel void kernel_topk_select_zero(
        device uint * words,
        constant uint & count,
        uint gid [[thread_position_in_grid]]) {
    if (gid < count) words[gid] = 0u;
}

// One histogram pass.  Pass 0 also clears the other pass-0 slot and hist1 for
// the next call and the next pass; pass 1 clears hist2.
kernel void kernel_topk_select_hist(
        constant ds4_metal_args_topk_select & args,
        device const float * scores,
        device atomic_uint * hist,
        threadgroup atomic_uint * bins [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint scratch[34];
    const uint row = tgpig.y;
    device atomic_uint * h = hist + row * DS4_TOPK_HIST_WORDS;
    const uint3 sel = ds4_topk_prefix(h, args.parity, args.pass, args.top_k, scratch, tid, lane, simd);
    if (args.pass == 0u) {
        for (uint i = tgpig.x * 1024u + tid; i < 2u * DS4_TOPK_BINS; i += args.n_tg * 1024u) {
            atomic_store_explicit(h + (i < DS4_TOPK_BINS ? (args.parity ^ 1u) * DS4_TOPK_BINS + i :
                                       DS4_TOPK_BINS + i), 0u, memory_order_relaxed);
        }
    } else if (args.pass == 1u) {
        for (uint i = tgpig.x * 1024u + tid; i < DS4_TOPK_BINS; i += args.n_tg * 1024u)
            atomic_store_explicit(h + 3u * DS4_TOPK_BINS + i, 0u, memory_order_relaxed);
    }
    for (uint i = tid; i < DS4_TOPK_BINS; i += 1024u) atomic_store_explicit(bins + i, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float * s = scores + (ulong)row * args.n;
    for (uint j = 0; j < DS4_TOPK_PER_TG / 1024u; j++) {
        const uint i = tgpig.x * DS4_TOPK_PER_TG + j * 1024u + tid;
        if (i >= args.n) break;
        const uint key = ds4_topk_key(s[i]);
        uint bin;
        if (args.pass == 0u) bin = key >> 21;
        else if (args.pass == 1u) { if ((key >> 21) != sel.x) continue; bin = (key >> 10) & 0x7ffu; }
        else { if ((key >> 10) != sel.x) continue; bin = key & 0x3ffu; }
        atomic_fetch_add_explicit(bins + bin, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device atomic_uint * out = h + (args.pass == 0u ? args.parity : args.pass + 1u) * DS4_TOPK_BINS;
    for (uint i = tid; i < DS4_TOPK_BINS; i += 1024u) {
        const uint c = atomic_load_explicit(bins + i, memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(out + i, c, memory_order_relaxed);
    }
}

// Threshold, greater count and wanted ties for the row, plus this
// threadgroup's greater/equal counts for the compaction offsets.
kernel void kernel_topk_select_count(
        constant ds4_metal_args_topk_select & args,
        device const float * scores,
        device atomic_uint * hist,
        device uint * state,
        device uint * counts,
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint scratch[34], part[64];
    const uint row = tgpig.y;
    const uint3 sel = ds4_topk_prefix(hist + row * DS4_TOPK_HIST_WORDS, args.parity, 3u, args.top_k,
                                      scratch, tid, lane, simd);
    if (tgpig.x == 0u && tid == 0u) {
        state[row * 4u] = sel.x;
        state[row * 4u + 1u] = sel.z;
        state[row * 4u + 2u] = sel.y;
    }
    device const float * s = scores + (ulong)row * args.n;
    uint gt = 0u, eq = 0u;
    for (uint j = 0; j < DS4_TOPK_PER_TG / 1024u; j++) {
        const uint i = tgpig.x * DS4_TOPK_PER_TG + j * 1024u + tid;
        if (i >= args.n) break;
        const uint key = ds4_topk_key(s[i]);
        gt += key > sel.x;
        eq += key == sel.x;
    }
    gt = simd_sum(gt); eq = simd_sum(eq);
    if (lane == 0u) { part[simd] = gt; part[32u + simd] = eq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint tg_gt = 0u, tg_eq = 0u;
        for (uint q = 0; q < 32u; q++) { tg_gt += part[q]; tg_eq += part[32u + q]; }
        counts[(row * args.n_tg + tgpig.x) * 2u] = tg_gt;
        counts[(row * args.n_tg + tgpig.x) * 2u + 1u] = tg_eq;
    }
}

// Winners by ascending id: keys above the threshold first, then the wanted
// number of keys equal to it.
kernel void kernel_topk_select_write(
        constant ds4_metal_args_topk_select & args,
        device const float * scores,
        device const uint * state,
        device const uint * counts,
        device int * selected,
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint part[66];
    const uint row = tgpig.y;
    const uint threshold = state[row * 4u], above = state[row * 4u + 1u], wanted = state[row * 4u + 2u];
    uint gt_before = 0u, eq_before = 0u;
    for (uint t = 0; t < tgpig.x; t++) {
        gt_before += counts[(row * args.n_tg + t) * 2u];
        eq_before += counts[(row * args.n_tg + t) * 2u + 1u];
    }
    device const float * s = scores + (ulong)row * args.n;
    device int * out = selected + (ulong)row * args.top_k;
    for (uint j = 0; j < DS4_TOPK_PER_TG / 1024u; j++) {
        const uint i = tgpig.x * DS4_TOPK_PER_TG + j * 1024u + tid;
        const bool live = i < args.n;
        const uint key = live ? ds4_topk_key(s[i]) : 0u;
        const uint gt = live && key > threshold, eq = live && key == threshold;
        const uint gt_x = simd_prefix_exclusive_sum(gt), eq_x = simd_prefix_exclusive_sum(eq);
        if (lane == 31u) { part[simd] = gt_x + gt; part[32u + simd] = eq_x + eq; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint run_gt = 0u, run_eq = 0u;
            for (uint q = 0; q < 32u; q++) {
                const uint a = part[q], b = part[32u + q];
                part[q] = run_gt; part[32u + q] = run_eq;
                run_gt += a; run_eq += b;
            }
            part[64] = run_gt; part[65] = run_eq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (gt) out[gt_before + part[simd] + gt_x] = (int)i;
        if (eq) {
            const uint r = eq_before + part[32u + simd] + eq_x;
            if (r < wanted) out[above + r] = (int)i;
        }
        gt_before += part[64]; eq_before += part[65];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Sort the row's winners in place: score descending, ties by ascending id.
kernel void kernel_topk_select_sort(
        constant ds4_metal_args_topk_select & args,
        device const float * scores,
        device int * selected,
        threadgroup uint * keys [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint * ids = keys + 2048u;
    device const float * s = scores + (ulong)row * args.n;
    device int * out = selected + (ulong)row * args.top_k;
    for (uint i = tid; i < 2048u; i += 1024u) {
        if (i < args.top_k) {
            const uint id = (uint)out[i];
            keys[i] = ds4_topk_key(s[id]);
            ids[i] = id;
        } else {
            keys[i] = 0u;
            ids[i] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2u; k <= 2048u; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            for (uint i = tid; i < 2048u; i += 1024u) {
                const uint o = i ^ j;
                if (o <= i) continue;
                const bool first = keys[i] > keys[o] || (keys[i] == keys[o] && ids[i] < ids[o]);
                if (((i & k) == 0u) != first) {
                    const uint tk = keys[i], ti = ids[i];
                    keys[i] = keys[o]; ids[i] = ids[o];
                    keys[o] = tk; ids[o] = ti;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = tid; i < args.top_k; i += 1024u) out[i] = (int)ids[i];
}
