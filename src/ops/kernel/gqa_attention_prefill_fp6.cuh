#pragma once

// FP6-native GQA prompt KV append: quantize the freshly projected K/V into the packed
// FP6 E3M2 U8 code plane (192 bytes per 256-dim token). The schedule mirrors the INT8
// fill kernels: block 256 (8 warps), one warp owns one (token, kv_head, 64-d group)
// unit with two dimensions per lane (d0 = group*64 + lane, d1 = d0 + 32). The codec
// stage is the difference: each warp encodes its 64 values, stages the six-bit codes
// in its private shared slice, cooperatively packs eight 8-code blocks to six bytes
// each, and writes the 48 packed bytes as three int4 stores. token_base is provably
// 16-aligned (256-byte plane alignment, 192-byte token stride, and 48-byte group
// stride are all divisible by 16) and each warp's packed slice is 16-aligned, so the
// typed stores are safe on device. Staging is per-warp because all eight warps reach
// the same __syncthreads barriers (the active/valid guard keeps warp divergence out
// of the barrier region).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/gqa_attention_kv_fp6.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cstdint>

namespace ninfer::ops {

// One warp per (token, kv_head, 64-d group) unit; two dims per lane, per-warp packed
// FP6 staging. Scale is the FP16-rounded amax / kGqaKvFp6MaxFinite; encode inverts the
// rounded scale, and zero groups store scale 0 with all-zero codes (gqa_kv_fp6_encode
// yields code 0 when the inverse scale is zero, so the store path needs no zero guard).
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void gqa_attention_prefill_fill_fp6_kernel(const __nv_bfloat16* __restrict__ k,
                                               const __nv_bfloat16* __restrict__ v,
                                               const std::int32_t* __restrict__ positions,
                                               Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                               std::uint8_t* __restrict__ cache_v,
                                               __half* __restrict__ scale_k,
                                               __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kGqaKvQuantGroups;
    const bool active           = unit < units;

    __shared__ std::uint8_t sh_codes[Warps * 64];
    __shared__ __align__(16) std::uint8_t sh_packed[Warps * 48];

    const int group   = active ? unit % kGqaKvQuantGroups : 0;
    const int tmp     = active ? unit / kGqaKvQuantGroups : 0;
    const int kv_head = active ? tmp % Geometry::KVHeads : 0;
    const int token   = active ? tmp / Geometry::KVHeads : 0;
    const int d0      = group * kGqaKvQuantGroup + lane;
    const int d1      = d0 + 32;

    float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f;
    if (active) {
        const std::int64_t src0 = gqa_kv_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1 = gqa_kv_quant_src_index<Geometry>(kv_head, d1, token);
        k0                      = __bfloat162float(k[src0]);
        k1                      = __bfloat162float(k[src1]);
        v0                      = __bfloat162float(v[src0]);
        v1                      = __bfloat162float(v[src1]);
    }
    const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
    const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);

    const int position              = active ? positions[0] + token : 0;
    const std::int32_t* block_table = metadata.block_table();
    int page = lane == 0 && active ? paged_kv_physical_page(block_table, position) : 0;
    page                            = __shfl_sync(FullMask, page, 0);
    const int page_off              = active ? position & kPagedKVPageMask : 0;

    const std::int64_t token_base =
        paged_kv_element_offset<kGqaKvFp6LeadingExtent, Geometry::KVHeads>(page, kv_head,
                                                                           page_off, 0) +
        group * 48;

    std::uint8_t* codes  = &sh_codes[warp * 64];
    std::uint8_t* packed = &sh_packed[warp * 48];

    const __half ksh   = __float2half_rn(k_abs > 0.0f ? k_abs / kGqaKvFp6MaxFinite : 0.0f);
    const float  ksinv = ksh == __half(0.0f) ? 0.0f : 1.0f / static_cast<float>(ksh);
    if (active) {
        codes[lane]      = static_cast<std::uint8_t>(gqa_kv_fp6_encode(k0, ksinv) & 0x3Fu);
        codes[lane + 32] = static_cast<std::uint8_t>(gqa_kv_fp6_encode(k1, ksinv) & 0x3Fu);
    }
    __syncthreads();
    if (active && lane < 8) {
        std::uint8_t block[8];
        for (int j = 0; j < 8; ++j) { block[j] = codes[lane * 8 + j]; }
        gqa_kv_fp6_pack8(block, packed + lane * 6);
    }
    __syncthreads();
    if (active && lane < 3) {
        reinterpret_cast<int4*>(cache_k + token_base)[lane] =
            reinterpret_cast<const int4*>(packed)[lane];
    }
    if (active && lane == 0) {
        scale_k[gqa_kv_quant_scale_index<Geometry>(page, kv_head, group, page_off)] = ksh;
    }
    __syncthreads();

    const __half vsh   = __float2half_rn(v_abs > 0.0f ? v_abs / kGqaKvFp6MaxFinite : 0.0f);
    const float  vsinv = vsh == __half(0.0f) ? 0.0f : 1.0f / static_cast<float>(vsh);
    if (active) {
        codes[lane]      = static_cast<std::uint8_t>(gqa_kv_fp6_encode(v0, vsinv) & 0x3Fu);
        codes[lane + 32] = static_cast<std::uint8_t>(gqa_kv_fp6_encode(v1, vsinv) & 0x3Fu);
    }
    __syncthreads();
    if (active && lane < 8) {
        std::uint8_t block[8];
        for (int j = 0; j < 8; ++j) { block[j] = codes[lane * 8 + j]; }
        gqa_kv_fp6_pack8(block, packed + lane * 6);
    }
    __syncthreads();
    if (active && lane < 3) {
        reinterpret_cast<int4*>(cache_v + token_base)[lane] =
            reinterpret_cast<const int4*>(packed)[lane];
    }
    if (active && lane == 0) {
        scale_v[gqa_kv_quant_scale_index<Geometry>(page, kv_head, group, page_off)] = vsh;
    }
}

// Large appends are scheduled in absolute eight-token tiles, exactly like the INT8 page
// fill. Eight divides P=64, so each CTA is page-local; per-warp packed FP6 staging is
// unchanged, with warp == token within the tile. Only instantiated for KVHeads == 2.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void gqa_attention_prefill_fill_fp6_page_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int group             = static_cast<int>(blockIdx.z);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) { return; }

    __shared__ std::uint8_t sh_codes[8 * 64];
    __shared__ __align__(16) std::uint8_t sh_packed[8 * 48];

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;

    const int token  = token_begin + warp;
    const bool valid = token < token_end;
    const int d0     = group * kGqaKvQuantGroup + lane;
    const int d1     = d0 + 32;
    float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f;
    if (valid) {
        const std::int64_t src0 = gqa_kv_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1 = gqa_kv_quant_src_index<Geometry>(kv_head, d1, token);
        k0                      = __bfloat162float(k[src0]);
        k1                      = __bfloat162float(k[src1]);
        v0                      = __bfloat162float(v[src0]);
        v1                      = __bfloat162float(v[src1]);
    }
    const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
    const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
    physical_page     = __shfl_sync(FullMask, physical_page, 0);

    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;
    const std::int64_t token_base =
        paged_kv_page_head_offset<kGqaKvFp6LeadingExtent, Geometry::KVHeads>(physical_page,
                                                                             kv_head) +
        static_cast<std::int64_t>(page_off) * kGqaKvFp6LeadingExtent + group * 48;

    std::uint8_t* codes  = &sh_codes[warp * 64];
    std::uint8_t* packed = &sh_packed[warp * 48];

    const __half ksh   = __float2half_rn(k_abs > 0.0f ? k_abs / kGqaKvFp6MaxFinite : 0.0f);
    const float  ksinv = ksh == __half(0.0f) ? 0.0f : 1.0f / static_cast<float>(ksh);
    if (valid) {
        codes[lane]      = static_cast<std::uint8_t>(gqa_kv_fp6_encode(k0, ksinv) & 0x3Fu);
        codes[lane + 32] = static_cast<std::uint8_t>(gqa_kv_fp6_encode(k1, ksinv) & 0x3Fu);
    }
    __syncthreads();
    if (valid && lane < 8) {
        std::uint8_t block[8];
        for (int j = 0; j < 8; ++j) { block[j] = codes[lane * 8 + j]; }
        gqa_kv_fp6_pack8(block, packed + lane * 6);
    }
    __syncthreads();
    if (valid && lane < 3) {
        reinterpret_cast<int4*>(cache_k + token_base)[lane] =
            reinterpret_cast<const int4*>(packed)[lane];
    }
    if (valid && lane == 0) {
        const std::int64_t scale_offset =
            paged_kv_page_head_offset<kGqaKvQuantGroups, Geometry::KVHeads>(physical_page,
                                                                            kv_head) +
            static_cast<std::int64_t>(page_off) * kGqaKvQuantGroups + group;
        scale_k[scale_offset] = ksh;
    }
    __syncthreads();

    const __half vsh   = __float2half_rn(v_abs > 0.0f ? v_abs / kGqaKvFp6MaxFinite : 0.0f);
    const float  vsinv = vsh == __half(0.0f) ? 0.0f : 1.0f / static_cast<float>(vsh);
    if (valid) {
        codes[lane]      = static_cast<std::uint8_t>(gqa_kv_fp6_encode(v0, vsinv) & 0x3Fu);
        codes[lane + 32] = static_cast<std::uint8_t>(gqa_kv_fp6_encode(v1, vsinv) & 0x3Fu);
    }
    __syncthreads();
    if (valid && lane < 8) {
        std::uint8_t block[8];
        for (int j = 0; j < 8; ++j) { block[j] = codes[lane * 8 + j]; }
        gqa_kv_fp6_pack8(block, packed + lane * 6);
    }
    __syncthreads();
    if (valid && lane < 3) {
        reinterpret_cast<int4*>(cache_v + token_base)[lane] =
            reinterpret_cast<const int4*>(packed)[lane];
    }
    if (valid && lane == 0) {
        const std::int64_t scale_offset =
            paged_kv_page_head_offset<kGqaKvQuantGroups, Geometry::KVHeads>(physical_page,
                                                                            kv_head) +
            static_cast<std::int64_t>(page_off) * kGqaKvQuantGroups + group;
        scale_v[scale_offset] = vsh;
    }
}

// FP6 prefill attention: the exact BF16 FA2 schedule (D=256, Br=Bc=64, 128
// threads, Q/K/V in 96 KiB of base dynamic smem) over the packed U8 cache. K and
// V tiles are staged by dequantizing the code plane + group scales directly from
// global memory instead of cp.async from a bf16 cache. The BF16 kernel's barrier
// placement is preserved verbatim, but the dequant staging is currently fully
// synchronous: only the prologue's Q cp.async overlaps with the K(0) staging,
// with no loop overlap. A later arena variant could overlap the loop stagings
// with the MMAs. The empty cp.async commit groups are kept purely for structural
// fidelity to the BF16 kernel; the prologue's Q staging drains through the
// iteration-0 cp_wait<0>() regardless.

// Dequantize one 8-dim FP6 d-block into 8 bf16 packed as an int4, preserving dim
// order for the ldmatrix consumers: unpack the six code bytes (byte-wise, safe at
// any alignment), decode each code, and scale by the group's stored FP16 scale.
__device__ __forceinline__ int4 gqa_kv_fp6_dequant_dblock8(const std::uint8_t* packed,
                                                           __half scale) {
    std::uint8_t codes[8];
    gqa_kv_fp6_unpack8(packed, codes);
    const float s = __half2float(scale);
    unsigned out[4];
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const float x0 = gqa_kv_fp6_decode(codes[j]) * s;
        const float x1 = gqa_kv_fp6_decode(codes[j + 1]) * s;
        out[j >> 1]    = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(out[0]), static_cast<int>(out[1]), static_cast<int>(out[2]),
                     static_cast<int>(out[3]));
}

// Stage one [Bc, D] K or V tile into the swizzled bf16 smem buffer by dequantizing
// the packed FP6 plane straight from the cache. 64 rows x 32 d-blocks = 2048 items
// over 128 threads; each item decodes one 8-dim block (six code bytes) with its
// group's FP16 scale (group = d-block / 8). Keys beyond max_query_abs, which the
// causal mask always drops, are zeroed so the padded cache tail never feeds NaNs
// into the tensor cores (mirrors the BF16 kernel's OOB zero-fill).
template <typename Geometry>
__device__ __forceinline__ void gqa_prefill_stage_fp6_kv(__nv_bfloat16* dst,
                                                         const std::uint8_t* cache,
                                                         const __half* scale_cache, int kv_head,
                                                         int k0, int max_query_abs,
                                                         int physical_page, int tid) {
    constexpr int D       = kGqaPrefillHeadDim;
    constexpr int Bc      = kGqaPrefillBc;
    constexpr int Threads = kGqaPrefillThreads;
    constexpr int DBlocks = D / kGqaKvFp6BlockDims; // 32
    constexpr int Items   = Bc * DBlocks;           // 2048
    for (int i = tid; i < Items; i += Threads) {
        const int key_l    = i / DBlocks;
        const int dblock   = i % DBlocks;
        const int key      = k0 + key_l;
        const int page_off = key & kPagedKVPageMask;
        __nv_bfloat16* p   = &dst[key_l * D + gqa_prefill_swz(key_l, dblock * kGqaKvFp6BlockDims)];
        if (key <= max_query_abs) {
            // The page/head plane base and group scale are recomputed per d-block
            // (the plane base repeats across the row's 32 d-blocks, each group
            // scale across 8) rather than hoisted; the decode ALU dominates the
            // item, so this is fine, and a row-major inner loop could hoist them.
            const std::int64_t plane =
                paged_kv_element_offset<kGqaKvFp6LeadingExtent, Geometry::KVHeads>(physical_page,
                                                                                    kv_head,
                                                                                    page_off, 0);
            const std::int64_t scale_off = gqa_kv_quant_scale_index<Geometry>(
                physical_page, kv_head, dblock / kGqaKvFp6BlockDims, page_off);
            store_vec(p, gqa_kv_fp6_dequant_dblock8(cache + plane +
                                                        static_cast<std::int64_t>(dblock) *
                                                            kGqaKvFp6BlockBytes,
                                                    scale_cache[scale_off]));
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

// FlashAttention-2 forward, one CTA per (query 64-row block, query head). Grid is
// (ceil(tokens/64), q_heads). seqlen_q = tokens, seqlen_k = base_pos + tokens, with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
template <typename Geometry, typename Metadata>
__launch_bounds__(kGqaPrefillThreads, 1) __global__
    void gqa_attention_prefill_fp6_kernel(const __nv_bfloat16* __restrict__ q,
                                          const std::uint8_t* __restrict__ cache_k,
                                          const std::uint8_t* __restrict__ cache_v,
                                          const __half* __restrict__ cache_k_scale,
                                          const __half* __restrict__ cache_v_scale,
                                          Metadata metadata,
                                          const std::int32_t* __restrict__ positions, float scale,
                                          __nv_bfloat16* __restrict__ out, std::int32_t width) {
    constexpr int D             = kGqaPrefillHeadDim; // 256
    constexpr int Br            = kGqaPrefillBr;      // 64 query rows
    constexpr int Bc            = kGqaPrefillBc;      // 64 key cols
    constexpr int Threads       = kGqaPrefillThreads; // 128
    constexpr int QKNt          = Bc / 8;             // 8  QK score n-tiles
    constexpr int QKKs          = D / 16;             // 16 QK contraction steps over head_dim
    constexpr int PVNt          = D / 8;              // 32 PV output n-tiles
    constexpr int PVKs          = Bc / 16;            // 4  PV contraction steps over keys
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Threads == 128);

    extern __shared__ __align__(16) __nv_bfloat16 gqa_smem[];
    __nv_bfloat16* q_s = gqa_smem;     // [Br, D] swizzled
    __nv_bfloat16* k_s = q_s + Br * D; // [Bc, D] swizzled
    __nv_bfloat16* v_s = k_s + Bc * D; // [Bc, D] swizzled

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16; // this warp owns rows [warp_row0, warp_row0+16)

    // Per-lane precomputed swizzled ldmatrix base addresses (see gqa_prefill_swz_addr).
    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    // Q A-fragment: row = warp_row0 + a_rowoff, col = k*16 + a_coloff.
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    // K B-fragment via ldmatrix.x4 (2 n-tiles/instr): lanes 16-31 fetch the +8-key
    // half (extra 4096 bytes), lanes with bit3 set fetch the +8 d-contract half.
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    // V B-fragment via ldmatrix.x4.trans (2 n-tiles/instr): row = k*16 + (bit3)*8 + b_rin,
    // col = n*8 + (lane>>4)*8.
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    // Stage Q into smem once via cp.async (overlaps with the K(0) prologue load
    // below); it stays resident for the whole key loop. Global Q rows are 256 bf16
    // contiguous, with a token stride of 256*QHeads.
    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads; // global stride between tokens
        const __nv_bfloat16* q_block = q + gqa_prefill_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int n_block_max   = (max_query_abs / Bc) + 1; // n_block_min == 0

    // Fold softmax_scale into the exp2 (FA-style): scores stay raw, so the
    // per-element "* scale" multiply drops out of the QK epilogue entirely.
    const float scale_l2 = scale * Log2E;
    int physical_page    = block_table[0];

    // Prologue: commit Q, then dequant-stage K(0). The loop's wait<0> below drains both.
    ninfer::ops::cp_commit();
    gqa_prefill_stage_fp6_kv<Geometry>(k_s, cache_k, cache_k_scale, kv_head, 0, max_query_abs,
                                       physical_page, tid);
    ninfer::ops::cp_commit();

    for (int kb = 0; kb < n_block_max; ++kb) {
        const int k0                 = kb * Bc;
        const int next_physical_page = (kb + 1 < n_block_max) ? block_table[kb + 1] : physical_page;

        ninfer::ops::cp_wait<0>(); // q_s landed; K(kb) dequant published by the sync below
        __syncthreads();

        // Stage V(kb) directly from the packed plane; consumed by the PV MMA after
        // the softmax barrier (single-buffered like the BF16 kernel's V slot).
        gqa_prefill_stage_fp6_kv<Geometry>(v_s, cache_v, cache_v_scale, kv_head, k0,
                                           max_query_abs, physical_page, tid);
        ninfer::ops::cp_commit();

        // S = Q Kᵀ for this warp's 16 rows over all Bc keys, in registers.
        // Software-pipelined like cute's gemm: issue the ldmatrix for contraction
        // step k+1 while the m16n8k16 MMAs for step k run, so the LSU (ldmatrix)
        // and tensor pipes overlap instead of stalling on each other.
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        // Swizzled ldmatrix addresses via precomputed per-lane bases + immediates.
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        {
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096),
                                                 0u, k_as, k_r));
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                             k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        const int row0             = warp_row0 + gid;
        const int row1             = warp_row0 + gid + 8;
        const int qrow0            = q0 + row0;
        const int qrow1            = q0 + row1;
        const int qabs0            = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1            = (qrow1 < tokens) ? base_pos + qrow1 : -1;
        const bool full_score_tile = (q0 + Br <= tokens) && ((k0 + Bc - 1) <= (base_pos + q0));

        // block row-max on raw (unscaled) scores; scale is folded into exp2 below
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = (qrow0 < tokens && key0 <= qabs0) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = (qrow0 < tokens && key1 <= qabs0) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = (qrow1 < tokens && key0 <= qabs1) ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = (qrow1 < tokens && key1 <= qabs1) ? score[nt][3] : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        const float alpha0     = exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1     = exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        // P = exp2(S - m), repacked into the PV A-fragment layout, plus local block row-sum.
        // The row-sum allreduce is deferred to the epilogue; only row max must be reduced per tile.
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = (score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = (score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = (score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = (score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        __syncthreads(); // V(kb) dequant visible to all threads before PV; QK done with k_s

        // Stage K(kb+1) into the (now-free) K buffer; synchronous, so it completes
        // before this iteration's PV MMA, and is consumed by the next loop
        // iteration's QK MMA after the loop-top barrier.
        if (kb + 1 < n_block_max) {
            physical_page = next_physical_page;
            gqa_prefill_stage_fp6_kv<Geometry>(k_s, cache_k, cache_k_scale, kv_head, (kb + 1) * Bc,
                                               max_query_abs, physical_page, tid);
            ninfer::ops::cp_commit();
        }

        // O += P V, contracting over the Bc keys. The (k, n) iteration space is
        // flattened and software-pipelined: the transposed ldmatrix for the next
        // V fragment is issued while the current MMA runs.
        // Each x4.trans load covers 2 output n-tiles (16 dims); pipeline the next
        // load against the current pair of MMAs.
        constexpr int PVHalf  = PVNt / 2;      // 16 n-tile pairs
        constexpr int PVLoads = PVKs * PVHalf; // 64 x4.trans loads
        // Swizzled V x4.trans addresses via precomputed per-lane base + immediates.
        unsigned vf[2][4];
        {
            ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                          gqa_prefill_swz_addr(v_lane_base, 0u, v_as, v_r));
        }
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k   = li / PVHalf;
            const int n2  = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2       = (li + 1) / PVHalf;
                const int n2b      = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              gqa_prefill_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 8192),
                                                   ckv, v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);

    // Normalize once per row via reciprocal-multiply instead of 128 IEEE divides.
    const float inv_l0 = (l0 > 0.0f) ? __frcp_rn(l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (qrow1 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid, Threads);
}

} // namespace ninfer::ops
