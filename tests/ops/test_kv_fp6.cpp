// Pure-host bit-level oracle for the packed FP6 (E3M2) KV codec
// (src/ops/kernel/gqa_attention_kv_fp6.cuh). Every expected value in this file
// is derived from the raw E3M2 bit semantics in reference_value(), never from
// the codec's decode/encode helpers. The encode-parity sweep therefore
// qualifies the codec against an independent oracle per the numerical
// correctness contract. No CUDA runtime is touched at test time; the codec
// header is self-contained for a host compiler.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "ops/kernel/gqa_attention_kv_fp6.cuh"

using namespace ninfer::ops;

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                           \
    do {                                                           \
        if (!(cond)) {                                             \
            ++g_failures;                                          \
            std::printf("  fail: " __VA_ARGS__);                   \
            std::printf("\n");                                     \
        }                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// Independent E3M2 oracle, in double, from raw bit semantics.
// ---------------------------------------------------------------------------

// Decoded value of (sign, exp, mant) with the E3M2 rules: exp 0 -> m * 2^-4,
// exp 7 -> inf/nan clamped to 14.0, else (1 + m/4) * 2^(e-3).
double reference_value(int sign, int exp, int mant) {
    if (exp == 0) { return sign ? -mant * 0.0625 : mant * 0.0625; }
    if (exp == 7) { return sign ? -14.0 : 14.0; }
    return sign ? -std::ldexp(1.0 + mant * 0.25, exp - 3)
                : std::ldexp(1.0 + mant * 0.25, exp - 3);
}

double reference_decode(std::uint32_t code) {
    return reference_value(static_cast<int>((code >> 5) & 1u),
                           static_cast<int>((code >> 2) & 7u),
                           static_cast<int>(code & 3u));
}

// Nearest of the 28 magnitude codes (exp 0..6 x mant 0..3) to v in [0, 14),
// ties broken toward even mantissa. Scans only reference_value(); it does NOT
// call gqa_kv_fp6_nearest_magnitude or gqa_kv_fp6_decode.
std::uint32_t oracle_nearest_magnitude(double v) {
    std::uint32_t best     = 0;
    double        best_err = std::numeric_limits<double>::infinity();
    for (int e = 0; e <= 6; ++e) {
        for (int m = 0; m < 4; ++m) {
            const std::uint32_t cand = static_cast<std::uint32_t>((e << 2) | m);
            const double        err  = std::fabs(reference_value(0, e, m) - v);
            if (err < best_err || (err == best_err && (cand & 1) == 0 && (best & 1) == 1)) {
                best     = cand;
                best_err = err;
            }
        }
    }
    return best;
}

bool close(double got, double expected, double rel) {
    return std::fabs(got - expected) <= rel * (std::fabs(expected) + 1.0);
}

// ---------------------------------------------------------------------------
// 1. decode_all_codes
// ---------------------------------------------------------------------------

void test_decode_all_codes() {
    for (std::uint32_t code = 0; code < 64; ++code) {
        const double expected = reference_decode(code);
        const double got      = static_cast<double>(gqa_kv_fp6_decode(code));
        CHECK(close(got, expected, 1e-6), "decode(0x%02X) = %.17g, oracle %.17g", code, got,
              expected);
    }

    const float pz = gqa_kv_fp6_decode(0x00u);
    const float nz = gqa_kv_fp6_decode(0x20u);
    CHECK(pz == 0.0f && !std::signbit(pz), "decode(0x00) should be +0.0");
    CHECK(nz == 0.0f && std::signbit(nz), "decode(0x20) should be -0.0");
    CHECK(gqa_kv_fp6_decode(0x04u) == 0.25f, "decode(0x04) should be 0.25");
    CHECK(gqa_kv_fp6_decode(0x1Bu) == 14.0f, "decode(0x1B) should be 14.0");
    CHECK(gqa_kv_fp6_decode(0x03u) == 0.1875f, "decode(0x03) should be 0.1875");
}

// ---------------------------------------------------------------------------
// 2. encode_oracle_parity (identity scale; the critical contract)
// ---------------------------------------------------------------------------

void test_encode_oracle_parity() {
    // Dense 1/256 grid across [0, 14).
    for (int k = 0; k < 14 * 256; ++k) {
        const float        v        = static_cast<float>(k) / 256.0f;
        const std::uint32_t expected = oracle_nearest_magnitude(static_cast<double>(v));
        const std::uint32_t got      = gqa_kv_fp6_encode(v, 1.0f);
        CHECK(got == expected, "encode(%g, 1.0f) = 0x%02X, oracle 0x%02X", static_cast<double>(v),
              got, expected);
    }

    // All 28 exact code magnitudes.
    for (int e = 0; e <= 6; ++e) {
        for (int m = 0; m < 4; ++m) {
            const double        mag = reference_value(0, e, m);
            const std::uint32_t expected = static_cast<std::uint32_t>((e << 2) | m);
            const std::uint32_t got      = gqa_kv_fp6_encode(static_cast<float>(mag), 1.0f);
            CHECK(got == expected, "encode(%g, 1.0f) should hit exact code 0x%02X", mag, expected);
        }
    }

    // The 27 exact midpoints between adjacent magnitudes (ties to even mantissa).
    // The 28 magnitude codes are enumerated e*4+m in monotone order, so code
    // indices c and c+1 are adjacent magnitudes.
    for (int c = 0; c < 27; ++c) {
        const int    e1  = c >> 2, m1 = c & 3;
        const int    e2  = (c + 1) >> 2, m2 = (c + 1) & 3;
        const double lo  = reference_value(0, e1, m1);
        const double hi  = reference_value(0, e2, m2);
        const double mid = (lo + hi) / 2.0;
        const std::uint32_t expected = oracle_nearest_magnitude(mid);
        const std::uint32_t got      = gqa_kv_fp6_encode(static_cast<float>(mid), 1.0f);
        CHECK(got == expected, "encode(%g, 1.0f) at midpoint = 0x%02X, oracle 0x%02X", mid, got,
              expected);
    }

    // A few values just below 14: nearest code is max finite 0x1B.
    const double near_top[] = {13.5, 13.75, 13.875, 13.9375, 13.99, 13.999, 13.9999, 13.99999};
    for (double v : near_top) {
        CHECK(gqa_kv_fp6_encode(static_cast<float>(v), 1.0f) == 0x1Bu,
              "encode(%g, 1.0f) should be max finite 0x1B", v);
    }

    CHECK(gqa_kv_fp6_encode(0.0f, 1.0f) == 0, "encode(+0, 1.0f) should be code 0");
}

// ---------------------------------------------------------------------------
// 3. encode_scale_path (randomized scale-multiply path)
// ---------------------------------------------------------------------------

void test_encode_scale_path() {
    std::mt19937 rng(0x6B6Bu);
    std::uniform_real_distribution<float> draw_x(-14.0f, 14.0f);
    std::uniform_real_distribution<float> draw_inv(0.01f, 10.0f);

    for (int i = 0; i < 20000; ++i) {
        const float  x         = draw_x(rng);
        const float  inv_scale = draw_inv(rng);
        const double w         = static_cast<double>(x) * static_cast<double>(inv_scale);
        const std::uint32_t code    = gqa_kv_fp6_encode(x, inv_scale);
        const double        decoded = static_cast<double>(gqa_kv_fp6_decode(code));

        CHECK(gqa_kv_fp6_encode(x, 0.0f) == 0, "encode(%g, 0.0f) should be code 0",
              static_cast<double>(x));

        if (std::fabs(w) >= 14.0) {
            // Clamp region: the code must be the signed max-finite code and the
            // decoded magnitude exactly 14.0. The quantization error relative
            // to x is dominated by the clamp here, not by the lattice, so the
            // one-code-step tolerance does not apply.
            const std::uint32_t expected = w < 0.0 ? 0x3Bu : 0x1Bu;
            CHECK(code == expected, "encode(%g, %g) clamp = 0x%02X, expected 0x%02X",
                  static_cast<double>(x), static_cast<double>(inv_scale), code, expected);
            CHECK(decoded == (w < 0.0 ? -14.0 : 14.0), "clamped decode should be +/-14.0");
        } else {
            const double mag = std::fabs(w);
            // The FP6 lattice is logarithmic, not uniform: 1.5*(14/56) (= 0.375
            // scaled units) covers only the fine band (|w| <= 4.0, where the
            // half-step is at most 0.25). In the coarse band a code step is up
            // to 2.0 wide, so the error there is bounded by the largest local
            // half-step (1.0) in the linear region.
            if (mag <= 4.0) {
                const double limit = 1.5 * (14.0 / 56.0) / static_cast<double>(inv_scale) + 1e-6;
                const double err   = std::fabs(decoded / static_cast<double>(inv_scale) - x);
                CHECK(err <= limit, "encode(%g, %g): scaled round-trip err %.9g > limit %.9g",
                      static_cast<double>(x), static_cast<double>(inv_scale), err, limit);
            } else {
                const double err = std::fabs(decoded - w);  // signed like w
                CHECK(err <= 1.0, "encode(%g, %g): coarse err %.9g > 1.0",
                      static_cast<double>(x), static_cast<double>(inv_scale), err);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. clamp_boundary
// ---------------------------------------------------------------------------

void test_clamp_boundary() {
    CHECK(gqa_kv_fp6_encode(14.0f, 1.0f) == 0x1Bu, "encode(14, 1.0f) should be max finite");
    CHECK(gqa_kv_fp6_encode(-14.0f, 1.0f) == 0x3Bu, "encode(-14, 1.0f) should be -max finite");
    CHECK(gqa_kv_fp6_encode(100.0f, 1.0f) == 0x1Bu, "encode(100, 1.0f) should be max finite");
    CHECK(gqa_kv_fp6_encode(-100.0f, 1.0f) == 0x3Bu, "encode(-100, 1.0f) should be -max finite");
    // Non-identity scale: the product still saturates when it exceeds 14.
    CHECK(gqa_kv_fp6_encode(30.0f, 0.5f) == 0x1Bu, "encode(30, 0.5f) should be max finite");
    CHECK(gqa_kv_fp6_encode(-30.0f, 0.5f) == 0x3Bu, "encode(-30, 0.5f) should be -max finite");
    // 14 * 0.5 = 7.0 is exactly representable, so it does NOT clamp; it encodes
    // to the exact 7.0 magnitude code 0x17 (0x37 with the sign bit).
    CHECK(gqa_kv_fp6_encode(14.0f, 0.5f) == 0x17u, "encode(14, 0.5f) should be exact 7.0 code");
    CHECK(gqa_kv_fp6_encode(-14.0f, 0.5f) == 0x37u, "encode(-14, 0.5f) should be exact -7.0 code");
}

// ---------------------------------------------------------------------------
// 5. pack_unpack_round_trip
// ---------------------------------------------------------------------------

void test_pack_unpack_round_trip() {
    std::mt19937 rng(0xA11CEu);
    std::uniform_int_distribution<int> draw(0, 63);
    std::uint8_t codes[8] = {};
    std::uint8_t rec[8]   = {};
    std::uint8_t out[6]   = {};
    for (int i = 0; i < 2000; ++i) {
        for (int j = 0; j < 8; ++j) { codes[j] = static_cast<std::uint8_t>(draw(rng)); }
        gqa_kv_fp6_pack8(codes, out);
        gqa_kv_fp6_unpack8(out, rec);
        bool ok = true;
        for (int j = 0; j < 8; ++j) { ok = ok && codes[j] == rec[j]; }
        CHECK(ok, "pack8/unpack8 round trip %d", i);
    }
}

// ---------------------------------------------------------------------------
// 6. pack_bit_layout
// ---------------------------------------------------------------------------

void test_pack_bit_layout() {
    // LSB-first: code j sits at bit [6j, 6j+6) of the assembled 48-bit block.
    // For codes 0..7 the 6 bytes are 0x40 0x20 0x0c 0x44 0x61 0x1c: byte 0 holds
    // code 0 (zero) plus code 1's bit 0 at bit 6 (0x40), etc.
    const std::uint8_t codes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::uint8_t       out[6]   = {};
    gqa_kv_fp6_pack8(codes, out);
    const std::uint8_t expected[6] = {0x40, 0x20, 0x0C, 0x44, 0x61, 0x1C};
    for (int i = 0; i < 6; ++i) {
        CHECK(out[i] == expected[i], "pack({0..7}) byte %d = 0x%02X, expected 0x%02X", i,
              static_cast<unsigned>(out[i]), static_cast<unsigned>(expected[i]));
    }

    // Pseudo-random sets: reconstruct the block as a little-endian uint64 and
    // check each code lands exactly at bit 6*j.
    std::mt19937 rng(0xDEADBEEFu);
    std::uniform_int_distribution<int> draw(0, 63);
    for (int i = 0; i < 200; ++i) {
        std::uint8_t rc[8] = {};
        for (int j = 0; j < 8; ++j) { rc[j] = static_cast<std::uint8_t>(draw(rng)); }
        gqa_kv_fp6_pack8(rc, out);
        std::uint64_t raw = 0;
        for (int b = 0; b < 6; ++b) { raw |= static_cast<std::uint64_t>(out[b]) << (8 * b); }
        for (int j = 0; j < 8; ++j) {
            const std::uint64_t got = (raw >> (6 * j)) & 0x3Fu;
            CHECK(got == static_cast<std::uint64_t>(rc[j]), "set %d: code %d not at bit %d", i, j,
                  6 * j);
        }
    }
}

// ---------------------------------------------------------------------------
// 7. plane_layout
// ---------------------------------------------------------------------------

void test_plane_layout() {
    static_assert(kGqaKvFp6LeadingExtent == 192, "a 256-dim token code plane must be 192 bytes");
    std::uint8_t plane[kGqaKvFp6LeadingExtent] = {};
    std::uint8_t codes[8]                      = {};
    for (int d = 0; d < 256; d += 8) {
        for (int j = 0; j < 8; ++j) { codes[j] = static_cast<std::uint8_t>((d + j) & 0x3F); }
        gqa_kv_fp6_pack8(codes, plane + (d / 8) * 6);
    }
    // Code at dim d lives at byte (d/8)*6, bit 6*(d%8) of the block.
    for (int d = 0; d < 256; ++d) {
        std::uint64_t raw = 0;
        for (int b = 0; b < 6; ++b) {
            raw |= static_cast<std::uint64_t>(plane[(d / 8) * 6 + b]) << (8 * b);
        }
        const std::uint32_t got = static_cast<std::uint32_t>((raw >> (6 * (d % 8))) & 0x3Fu);
        CHECK(got == static_cast<std::uint32_t>(d & 0x3F), "plane dim %d code = 0x%02X", d, got);
    }
}

// ---------------------------------------------------------------------------
// 8. block_offset
// ---------------------------------------------------------------------------

void test_block_offset() {
    for (int d = 0; d < 256; d += 8) {
        const int off = gqa_kv_fp6_block_offset(d);
        CHECK(off == (d / 8) * 6, "block_offset(%d) = %d, expected %d", d, off, (d / 8) * 6);
        CHECK(off >= 0 && off < kGqaKvFp6LeadingExtent, "block_offset(%d) = %d out of range", d,
              off);
    }
}

}  // namespace

int main() {
    test_decode_all_codes();
    test_encode_oracle_parity();
    test_encode_scale_path();
    test_clamp_boundary();
    test_pack_unpack_round_trip();
    test_pack_bit_layout();
    test_plane_layout();
    test_block_offset();

    if (g_failures == 0) {
        std::printf("ninfer_kv_fp6_test: ok\n");
        return 0;
    }
    std::printf("ninfer_kv_fp6_test: %d failure(s)\n", g_failures);
    return 1;
}
