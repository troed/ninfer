#pragma once

// ninfer::ops - group-64 packed FP6 (E3M2) KV cache codec, shared by the FP6
// GQA attention kernels and the host oracle test. Every helper is __host__
// __device__ so the pure-host unit test exercises the exact same bit math as
// the device kernels.
//
// FP6 E3M2 format: 1 sign + 3 exp + 2 mantissa bits, exponent bias 3.
//   exp 0 (subnormal): magnitude = m * 2^-4, m in 1..3 -> 0.0625, 0.125, 0.1875
//   exp 1..6 (normal): magnitude = 2^(e-3) * (1 + m/4)
//   exp 7: inf/nan, clamped to max finite = 14.0 (e=6, m=3)
// Code bits: sign<<5 | exp<<2 | mant. +0 = 0x00, -0 = 0x20.
// Scale: per 64-dim group. scale = FP16_RNE(amax/14.0f); the decoded value is
//   fp6_decode(code) * scale. Zero group -> scale 0, all codes 0.
// Packing: LSB-first bitstream per token. Code i (dim) sits at bit [6i, 6i+6):
//   byte offset (3i)>>2, bit (6i)&7. A full 256-dim token code plane is
//   256*6/8 = 192 bytes (kGqaKvFp6LeadingExtent). A d-block of 8 consecutive
//   dims (d%8==0) occupies 48 bits at byte offset (d/8)*6. Pack8/unpack8 are
//   byte-wise (six scalar byte loads/stores at any alignment), so d-block
//   offsets 2-mod-4 are always access-safe on device.
//
// kGqaKvFp6LeadingExtent must stay in sync with
// src/ops/wrapper/gqa_attention.cpp::code_leading_extent, which derives the
// code-plane extent as (kHeadDim * 6) / 8.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaKvFp6LeadingExtent = 192;  // bytes/token code plane
inline constexpr int kGqaKvFp6Bits          = 6;
inline constexpr int kGqaKvFp6BlockDims     = 8;
inline constexpr int kGqaKvFp6BlockBytes    = 6;
inline constexpr float kGqaKvFp6MaxFinite   = 14.0f;

// 16-byte-aligned token and group strides keep the FP6 prefill fill kernels' int4
// cache stores aligned on device; a 64-dim group packs to (64 * 6) / 8 = 48 bytes.
static_assert(kGqaKvFp6LeadingExtent % 16 == 0,
              "fp6 token code plane must stay 16-byte aligned for int4 cache stores");
static_assert(((64 * kGqaKvFp6Bits) / 8) % 16 == 0,
              "fp6 group code stride (48 bytes) must stay 16-byte aligned for int4 cache stores");

// Byte offset of the 8-dim block containing dim d within one token's code plane.
__host__ __device__ __forceinline__ int gqa_kv_fp6_block_offset(int d) {
    return (d / kGqaKvFp6BlockDims) * kGqaKvFp6BlockBytes;
}

// Decode one 6-bit code (sign bit included) to its exact FP6 value. All 28
// magnitude decodes are exact binary floats (dyadic rationals with denominator
// at most 16), so this is an exact value, not a rounded approximation.
__host__ __device__ __forceinline__ float gqa_kv_fp6_decode(std::uint32_t code) {
    const std::uint32_t exp  = (code >> 2) & 7u;
    const std::uint32_t mant = code & 3u;
    float               mag;
    if (exp == 0) {
        mag = static_cast<float>(mant) * 0.0625f;  // m * 2^-4
    } else if (exp == 7) {
        mag = kGqaKvFp6MaxFinite;  // inf/nan clamped to max finite
    } else {
        mag = ldexpf(1.0f + static_cast<float>(mant) * 0.25f,
                     static_cast<int>(exp) - 3);  // exact (1 + m/4) * 2^(e-3)
    }
    return (code & 0x20u) ? -mag : mag;
}

// Nearest magnitude code for v in [0, 14), ties broken toward even mantissa.
// Scans all 28 magnitude codes (exp 0..6 x mant 0..3); the reference loop keeps
// the first minimum so a tie between two odd-mantissa codes resolves to the
// lower value, and a tie involving an even code always prefers the even one.
__host__ __device__ __forceinline__ std::uint32_t gqa_kv_fp6_nearest_magnitude(float v) {
    std::uint32_t best     = 0;
    float         best_err = __FLT_MAX__;
    for (int e = 0; e <= 6; ++e) {
        for (int m = 0; m < 4; ++m) {
            const std::uint32_t cand = static_cast<std::uint32_t>((e << 2) | m);
            const float         val  = gqa_kv_fp6_decode(cand);
            const float         err  = fabsf(v - val);
            if (err < best_err || (err == best_err && (cand & 1) == 0 && (best & 1) == 1)) {
                best     = cand;
                best_err = err;
            }
        }
    }
    return best;
}

// Encode one value to its 6-bit code (sign bit included): x * inv_scale with
// round-to-nearest-even, clamped so any magnitude >= 14.0 maps to the max-finite
// code (exp 6, mant 3). Bit-identical to the host oracle, which scans the 28
// magnitude codes for the minimum-error code with ties toward even mantissa.
// -0.0 and NaN encode to code 0; the -0 code 0x20 is never emitted.
__host__ __device__ __forceinline__ std::uint32_t gqa_kv_fp6_encode(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return 0; }
    const float   v   = x * inv_scale;
    const bool    neg = v < 0.0f;
    const float   mag = fabsf(v);
    std::uint32_t code;
    if (mag >= kGqaKvFp6MaxFinite) {
        code = 0x1Bu;  // exp 6, mant 3 = max finite (14.0)
    } else {
        code = gqa_kv_fp6_nearest_magnitude(mag);
    }
    return neg ? (code | 0x20u) : code;
}

// Pack the 8 codes for dims [d, d+8) (d%8==0) into 6 bytes at token-plane byte
// offset gqa_kv_fp6_block_offset(d). Byte-wise store: six scalar byte writes at
// any alignment are always safe on device, unlike a typed store at a d-block
// offset 2-mod-4. Out-of-range inputs are masked to 6 bits so they cannot
// corrupt neighbor codes.
__host__ __device__ __forceinline__ void gqa_kv_fp6_pack8(const std::uint8_t codes[8],
                                                          std::uint8_t* out) {
    const std::uint32_t mask = (1u << kGqaKvFp6Bits) - 1u;
    std::uint64_t       raw  = 0;
    raw |= static_cast<std::uint64_t>(codes[0] & mask) << 0;
    raw |= static_cast<std::uint64_t>(codes[1] & mask) << 6;
    raw |= static_cast<std::uint64_t>(codes[2] & mask) << 12;
    raw |= static_cast<std::uint64_t>(codes[3] & mask) << 18;
    raw |= static_cast<std::uint64_t>(codes[4] & mask) << 24;
    raw |= static_cast<std::uint64_t>(codes[5] & mask) << 30;
    raw |= static_cast<std::uint64_t>(codes[6] & mask) << 36;
    raw |= static_cast<std::uint64_t>(codes[7] & mask) << 42;
    out[0] = static_cast<std::uint8_t>(raw >> 0);
    out[1] = static_cast<std::uint8_t>(raw >> 8);
    out[2] = static_cast<std::uint8_t>(raw >> 16);
    out[3] = static_cast<std::uint8_t>(raw >> 24);
    out[4] = static_cast<std::uint8_t>(raw >> 32);
    out[5] = static_cast<std::uint8_t>(raw >> 40);
}

// Unpack the 6 bytes at token-plane byte offset gqa_kv_fp6_block_offset(d) into
// the 8 codes for dims [d, d+8) (d%8==0). Byte-wise load: six scalar byte reads
// at any alignment are always safe on device, unlike a typed load at a d-block
// offset 2-mod-4.
__host__ __device__ __forceinline__ void gqa_kv_fp6_unpack8(const std::uint8_t* in,
                                                            std::uint8_t codes[8]) {
    const std::uint64_t raw = static_cast<std::uint64_t>(in[0]) |
                              (static_cast<std::uint64_t>(in[1]) << 8) |
                              (static_cast<std::uint64_t>(in[2]) << 16) |
                              (static_cast<std::uint64_t>(in[3]) << 24) |
                              (static_cast<std::uint64_t>(in[4]) << 32) |
                              (static_cast<std::uint64_t>(in[5]) << 40);
    codes[0] = static_cast<std::uint8_t>((raw >> 0) & 0x3Fu);
    codes[1] = static_cast<std::uint8_t>((raw >> 6) & 0x3Fu);
    codes[2] = static_cast<std::uint8_t>((raw >> 12) & 0x3Fu);
    codes[3] = static_cast<std::uint8_t>((raw >> 18) & 0x3Fu);
    codes[4] = static_cast<std::uint8_t>((raw >> 24) & 0x3Fu);
    codes[5] = static_cast<std::uint8_t>((raw >> 30) & 0x3Fu);
    codes[6] = static_cast<std::uint8_t>((raw >> 36) & 0x3Fu);
    codes[7] = static_cast<std::uint8_t>((raw >> 42) & 0x3Fu);
}

} // namespace ninfer::ops
