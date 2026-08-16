#pragma once

// ninfer::ops - split-KV GQA small-T attention, FP6 (E3M2) KV-cache partial kernel.
// Derived from the BF16 partial kernel (gqa_attention_decode_bf16.cuh): the full
// split/window/softmax/QK/PV bf16-MMA structure is preserved, with two surgical
// replacements for the packed U8 code plane:
//
//   (a) Fused append: the owning split's current rows are encoded+packed into the
//       packed plane (one warp per (token, group), 64 dims -> FP16 scale + 48
//       packed bytes) instead of raw bf16 stores. Current-token attention still
//       reads those rows from the input buffer (from_new), never from this write.
//   (b) Key-loop staging: cp.async 16-byte chunks of the packed plane and the
//       per-row FP16 group scales into dynamic-smem arenas, then a dequant pass
//       (unpack + decode * scale, byte-wise) writes the swizzled bf16 tile. Fresh
//       (from_new) rows are staged as bf16 directly from input, exactly like the
//       BF16 kernel; everything else is read from the packed cache.
//
// The dynamic arenas (2 * Bc * 192 packed bytes + 2 * Bc * 4 group scales) push
// the Wc=4 per-block total to 50816 B, over the 49152 B (48 KiB) default opt-in
// threshold (Wc=2 is 48320 B, under it), so the launcher opts in to the
// sm_120/sm_120a dynamic-smem cap of 101376 B via cudaFuncSetAttribute.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_fp6.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaDecodeFp6KeyBlock = 32;
inline constexpr std::size_t kGqaDecodeFp6DynamicSmemBytes =
    2 * kGqaDecodeFp6KeyBlock * kGqaKvFp6LeadingExtent +
    2 * kGqaDecodeFp6KeyBlock * kGqaKvQuantGroups * sizeof(__half);

// Dequant one [Bc, D] K or V tile from the packed/scale smem arenas into the
// swizzled bf16 tile. from_new rows are skipped (they were staged as bf16 from
// input by the cp.async wave); rows outside [split_start, split_end) are zeroed
// so the padded cache tail never feeds NaNs into the tensor cores (mirrors the
// BF16 kernel's out-of-range zero-fill).
template <typename Geometry>
__device__ __forceinline__ void gqa_decode_stage_fp6_kv(__nv_bfloat16* dst,
                                                        const std::uint8_t* packed,
                                                        const __half* scale, int k0,
                                                        int split_start, int split_end,
                                                        bool writes_cache, int first_pos,
                                                        int valid_tokens, int tid, int threads) {
    constexpr int D       = kGqaHeadDim;
    constexpr int Bc      = kGqaDecodeFp6KeyBlock;
    constexpr int DBlocks = D / kGqaKvFp6BlockDims;
    constexpr int Items   = Bc * DBlocks;
    for (int i = tid; i < Items; i += threads) {
        const int key_l  = i / DBlocks;
        const int dblock = i - key_l * DBlocks;
        const int key    = k0 + key_l;
        __nv_bfloat16* p =
            &dst[key_l * D + gqa_small_t_tc_swz(key_l, dblock * kGqaKvFp6BlockDims)];
        if (key >= split_start && key < split_end) {
            bool from_new = false;
            if (writes_cache) {
                const int new_token = key - first_pos;
                from_new = new_token >= 0 && new_token < valid_tokens && key >= first_pos;
            }
            if (from_new) { continue; }
            const int grp = dblock / kGqaKvFp6BlockDims;
            store_vec(p, gqa_kv_fp6_dequant_dblock8(
                             packed + key_l * kGqaKvFp6LeadingExtent +
                                 dblock * kGqaKvFp6BlockBytes,
                             scale[key_l * kGqaKvQuantGroups + grp]));
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void gqa_attention_decode_fp6_tiled_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::uint8_t* cache_k,
    std::uint8_t* cache_v, __half* cache_k_scale, __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = kGqaDecodeFp6KeyBlock;
    constexpr int D       = kGqaHeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt    = Bc / 8;
    constexpr int QKKs    = D / 16;
    constexpr int PVNt    = D / 8;
    constexpr int PVKs    = Bc / 16;
    constexpr int Groups  = kGqaKvQuantGroups;
    // The GQA Op's 262144-key maximum envelope spans at most 49 pages in one 27B split.
    constexpr int PageIds       = 64;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];
    // Per-warp (token, group) encode/pack staging for the fused append (K + V).
    __shared__ std::uint8_t append_codes_s[Wc * 2 * kGqaKvQuantGroup];
    __shared__ __align__(16) std::uint8_t append_packed_s[Wc * 2 * 48];
    // Packed code + group-scale arenas for the key loop, staged via cp.async.
    extern __shared__ __align__(16) std::uint8_t fp6_dynamic_s[];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;
    std::uint8_t* packed_k_s = fp6_dynamic_s;
    std::uint8_t* packed_v_s = fp6_dynamic_s + Bc * kGqaKvFp6LeadingExtent;
    __half* scale_k_s = reinterpret_cast<__half*>(fp6_dynamic_s + 2 * Bc * kGqaKvFp6LeadingExtent);
    __half* scale_v_s = scale_k_s + Bc * kGqaKvQuantGroups;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // Fused append: encode+pack the owning split's current rows into the packed
        // code plane. One warp per (token, group) unit encodes 64 dims (absmax ->
        // FP16-rounded scale -> encode), stages the codes, packs to 48 bytes, and
        // stores the plane + scale. Mirrors the FP6 fill kernels' encode+pack+scale.
        const int units      = valid_tokens * Groups;
        const int unit_iters = div_up(units, Wc);
        for (int uiter = 0; uiter < unit_iters; ++uiter) {
            const int  unit     = uiter * Wc + warp;
            const bool active   = unit < units;
            const int  token    = active ? unit / Groups : 0;
            const int  grp      = active ? unit - token * Groups : 0;
            const int  position = active ? pos[token] : 0;
            const bool in_split = active && position >= split_start && position < split_end &&
                                  position >= 0 && position < logical_capacity;
            const int d0 = grp * kGqaKvQuantGroup + lane;
            const int d1 = d0 + 32;
            float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f;
            if (in_split) {
                const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
                const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
                k0                      = __bfloat162float(input.k[src0]);
                k1                      = __bfloat162float(input.k[src1]);
                v0                      = __bfloat162float(input.v[src0]);
                v1                      = __bfloat162float(input.v[src1]);
            }
            const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
            const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
            const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / kGqaKvFp6MaxFinite : 0.0f);
            const float k_inv = ksh == __half(0.0f) ? 0.0f : 1.0f / static_cast<float>(ksh);
            const __half vsh = __float2half_rn(v_abs > 0.0f ? v_abs / kGqaKvFp6MaxFinite : 0.0f);
            const float v_inv = vsh == __half(0.0f) ? 0.0f : 1.0f / static_cast<float>(vsh);
            std::uint8_t* codes = &append_codes_s[warp * 2 * kGqaKvQuantGroup];
            codes[lane]         = static_cast<std::uint8_t>(gqa_kv_fp6_encode(k0, k_inv) & 0x3Fu);
            codes[lane + 32]    = static_cast<std::uint8_t>(gqa_kv_fp6_encode(k1, k_inv) & 0x3Fu);
            codes[kGqaKvQuantGroup + lane] =
                static_cast<std::uint8_t>(gqa_kv_fp6_encode(v0, v_inv) & 0x3Fu);
            codes[kGqaKvQuantGroup + lane + 32] =
                static_cast<std::uint8_t>(gqa_kv_fp6_encode(v1, v_inv) & 0x3Fu);
            __syncthreads();
            std::uint8_t* packed = &append_packed_s[warp * 2 * 48];
            if (active && lane < 8) {
                std::uint8_t block[8];
                for (int j = 0; j < 8; ++j) { block[j] = codes[lane * 8 + j]; }
                gqa_kv_fp6_pack8(block, packed + lane * 6);
                for (int j = 0; j < 8; ++j) { block[j] = codes[kGqaKvQuantGroup + lane * 8 + j]; }
                gqa_kv_fp6_pack8(block, packed + 48 + lane * 6);
            }
            __syncthreads();
            if (in_split) {
                int physical_page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
                physical_page     = __shfl_sync(FullMask, physical_page, 0);
                const int page_off = position & kPagedKVPageMask;
                const std::int64_t token_base =
                    paged_kv_element_offset<kGqaKvFp6LeadingExtent, Geometry::KVHeads>(
                        physical_page, kv_head, page_off, 0) +
                    grp * 48;
                if (lane < 3) {
                    reinterpret_cast<int4*>(cache_k + token_base)[lane] =
                        reinterpret_cast<const int4*>(packed)[lane];
                    reinterpret_cast<int4*>(cache_v + token_base)[lane] =
                        reinterpret_cast<const int4*>(packed + 48)[lane];
                }
                if (lane == 0) {
                    const std::int64_t so =
                        gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_off);
                    cache_k_scale[so] = ksh;
                    cache_v_scale[so] = vsh;
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();
    int physical_page = physical_pages_s[0];
    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        // Stage the packed K/V tile. cp.async 16-byte chunks of the packed plane
        // and 8-byte group scales into the dynamic arenas; fresh rows are staged
        // as bf16 straight from the input buffer (they are not read back from the
        // cache write above).
        {
            const std::int64_t packed_plane = paged_kv_element_offset<kGqaKvFp6LeadingExtent,
                                                                      Geometry::KVHeads>(
                physical_page, kv_head, k0 & kPagedKVPageMask, 0);
            constexpr int PackedChunks = kGqaKvFp6LeadingExtent / 16;
#pragma unroll 1
            for (int chunk = tid; chunk < Bc * PackedChunks; chunk += Threads) {
                const int key_l = chunk / PackedChunks;
                const int c     = chunk - key_l * PackedChunks;
                const int key   = k0 + key_l;
                bool from_new   = false;
                if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - first_pos;
                    from_new = new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                }
                if (from_new) { continue; }
                if (key >= split_start && key < split_end) {
                    const std::int64_t off =
                        packed_plane + static_cast<std::int64_t>(key_l) * kGqaKvFp6LeadingExtent +
                        c * 16;
                    ninfer::ops::cp_async<16>(
                        &packed_k_s[key_l * kGqaKvFp6LeadingExtent + c * 16], &cache_k[off]);
                    ninfer::ops::cp_async<16>(
                        &packed_v_s[key_l * kGqaKvFp6LeadingExtent + c * 16], &cache_v[off]);
                }
            }
#pragma unroll 1
            for (int key_l = tid; key_l < Bc; key_l += Threads) {
                const int key = k0 + key_l;
                bool from_new = false;
                if constexpr (CacheInput::writes_cache) {
                    const int new_token = key - first_pos;
                    from_new = new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                }
                if (from_new) { continue; }
                if (key >= split_start && key < split_end) {
                    const std::int64_t off = gqa_kv_quant_scale_index<Geometry>(
                        physical_page, kv_head, 0, key & kPagedKVPageMask);
                    ninfer::ops::cp_async<8>(&scale_k_s[key_l * Groups], &cache_k_scale[off]);
                    ninfer::ops::cp_async<8>(&scale_v_s[key_l * Groups], &cache_v_scale[off]);
                }
            }
            if constexpr (CacheInput::writes_cache) {
#pragma unroll 1
                for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
                    const int key_l = chunk / (D / 8);
                    const int d     = (chunk - key_l * (D / 8)) * 8;
                    const int key   = k0 + key_l;
                    if (key >= split_start && key < split_end) {
                        const int new_token = key - first_pos;
                        const bool from_new =
                            new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                        if (from_new) {
                            const std::int64_t off =
                                gqa_kv_new_index<Geometry>(kv_head, d, key - first_pos);
                            ninfer::ops::cp_async<16>(
                                &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)], &input.k[off]);
                            ninfer::ops::cp_async<16>(
                                &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)], &input.v[off]);
                        }
                    }
                }
            }
        }
        ninfer::ops::cp_commit();
        ninfer::ops::cp_wait<0>();
        __syncthreads();
        // Dequant the packed arenas into the swizzled bf16 tile; from_new rows were
        // already staged as bf16 above and are skipped, out-of-range rows zeroed.
        gqa_decode_stage_fp6_kv<Geometry>(k_s, packed_k_s, scale_k_s, k0, split_start, split_end,
                                          CacheInput::writes_cache, first_pos, valid_tokens, tid,
                                          Threads);
        gqa_decode_stage_fp6_kv<Geometry>(v_s, packed_v_s, scale_v_s, k0, split_start, split_end,
                                          CacheInput::writes_cache, first_pos, valid_tokens, tid,
                                          Threads);
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            score[nt][0] =
                (row0 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                    ? score[nt][0] * scale
                    : -CUDART_INF_F;
            score[nt][1] =
                (row0 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                    ? score[nt][1] * scale
                    : -CUDART_INF_F;
            score[nt][2] =
                (row1 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                    ? score[nt][2] * scale
                    : -CUDART_INF_F;
            score[nt][3] =
                (row1 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                    ? score[nt][3] * scale
                    : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0  = nt * 8 + 2 * lid;
            const int col1  = col0 + 1;
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }

    // MMA fragments hold each row in four-lane groups. Stage the final split-local
    // accumulator through shared memory so partial_acc is written as contiguous d-vector stores.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < row_count) {
            qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    __syncthreads();

    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
}

} // namespace ninfer::ops
