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

} // namespace ninfer::ops
