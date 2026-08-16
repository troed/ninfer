# FP6 Packed KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a group-64 FP6 (E3M2) KV cache format to NInfer — packed 6-bit codes (192 bytes/token/head, vs 512 bf16 / 256 int8), FP16 per-group scale plane — plumbed through the public CLI/serve surface, target planning, wrapper validation, prefill/decode kernels, and the GQA attention test harness, with DFlash explicitly out of scope.

**Architecture:** FP6 is a scalar codec inserted into the existing dtype+scale-plane seam. The code plane becomes `DType::U8` holding LSB-first packed 6-bit codes with leading extent 192 (bytes/token) instead of 256 (elements/token); the scale plane is unchanged FP16, one value per 64-dim group. Attention kernels keep bf16 MMA for QK/PV and add a dequant (unpack+decode+scale) staging pass from the packed plane; fused append encodes+packa. No codec framework: the FP6 path is three kernel shapes derived from the bf16 family, plus a small shared `__host__ __device__` codec header.

**Tech Stack:** C++/CUDA 13.1, sm_120a (RTX 5090), `ninfer_ops` static lib (nvcc-compiled `.cu` launchers + CXX-compiled `.cpp` wrappers), `ctest` with `ninfer_add_op_test` (SKIP_RETURN_CODE 77, `-fno-fast-math`, `-ffp-contract=off`), CUDA Graphs.

---

## Key existing facts (sources of truth for this plan)

- **FP6 format (locked):** E3M2 (1 sign + 3 exp + 2 mantissa), bias 3. Max finite = 14.0 (exp 6, mant 3); exp 7 = inf/nan → clamp to max finite. Subnormals (exp 0): `m * 2^-4` for m=1..3. Normal min = 0.25. Code bits: `sign<<5 | exp<<2 | mant`. `+0` = code 0, `-0` = code 32.
- **Scale:** per group of 64 dims (same plane as INT8): `scale = FP16_RNE(amax/14.0f)`; `decode = fp6_decode(code) * scale`. Zero group → scale 0, all codes 0.
- **Packing (locked):** LSB-first bitstream per token. Code `i` (dim) at bit `[6i, 6i+6)`; byte offset `(3i)>>2`, bit `(6i)&7`. Token code block = `256*6/8 = 192` bytes. A d-block of 8 consecutive dims (`d%8==0`) is 48 bits at byte offset `(d/8)*6`. **Alignment rule (learned during Task 5):** 6-byte-block offsets are 2-mod-4 for half of all blocks, and typed 32/64-bit accesses at those offsets ABORT on device (misaligned address) in both global and shared memory — pack8/unpack8 must use scalar byte-wise assembly/disassembly (build a `std::uint64_t` in a local, emit/load 6 individual `uint8_t` accesses). Host x86 masks this bug; device tests must exercise 2-mod-4 offsets explicitly. A 64-group = 48 bytes.
- **Enum/parse surface:** `include/ninfer/types.h:21` `enum class KvCacheStorage : uint8_t { BFloat16, Int8Group64 }`; `apps/cli/options.cpp:55-59` `parse_kv_cache` + usage text line 86 `[--kv-dtype bf16|int8]`; `src/serve/serve_options.cpp:48-53` `parse_kv_dtype`.
- **Family mapping:** `src/targets/qwen3_6/impl/runtime/layouts_impl.h:698-699` `kv_dtype = options.kv_cache==BFloat16 ? BF16 : I8`, `kv_quant_group = ... ? 0 : kKvQuantGroup`. `src/targets/qwen3_6/impl/runtime/program_impl.h:2212` maps back for `MemorySummary`.
- **Target planning:** `src/targets/qwen3_6/impl/state/decoder_state.cpp` `plan_cache` (line 14) — line 23 `quantized = dtype == DType::I8`; lines 40-45 per-layer planes `{dtype, head_dim, kv_heads, 256}` + 2 FP16 scale planes `{FP16, head_dim/quant_group, kv_heads, 256}`. `layer_view`/`batch_layer_view` use `quantized = dtype_ == DType::I8` and stride 4 vs 2.
- **Core:** `src/core/paged_kv_cache.h` `PagedKVPlaneSpec{dtype, leading_extent, head_extent, alignment=256}`; `plan_paged_kv_pool` builds `{leading_extent, 64, head_extent, physical_pages}` per plane. No core change needed.
- **Wrapper:** `src/ops/wrapper/gqa_attention.cpp` — `validate_cache` (line 54), `validate_batch_cache` (line 110) enforce dtype ∈ {BF16,I8}; `code_dtype = cache.dtype==I8 ? I8 : BF16` (lines 74/132); k_pages/v_pages shape leading extent `kHeadDim`=256 (lines 78-81/136-139); scale shape `{groups=4, 64, kv_heads, pages}`. `gqa_attention_workspace_capacity_bytes` line 353 gates dtype ∈ {BF16,I8}.
- **Decode launcher:** `src/ops/launcher/gqa_attention_decode.cu` — `gqa_attention_split_capacity` line 220 gates dtype ∈ {BF16,I8}; dispatch macro `NINFER_GQA_SMALL_T_DISPATCH` line 249 branches `cache.dtype == DType::I8` → `launch_tc_partial_i8` else `launch_tc_partial_bf16`; reduce dispatch line 339 `if (cache.dtype == DType::I8) launch_for_dtype<true>() else launch_for_dtype<false>()`.
- **Decode bf16 kernel:** `src/ops/kernel/gqa_attention_decode_bf16.cuh` — `gqa_attention_small_t_tc_partial_bf16_kernel<Geometry,TokenTile,WarpsPerCta,MultiBatch,Masked,CacheInput>`, `__launch_bounds__(128,2)`. Fused append at lines 148-168; key-loop K/V staging cp_async at lines ~222-257 (reads bf16 directly from cache into swizzled bf16 tile).
- **Prefill launcher:** `src/ops/launcher/gqa_attention_prefill.cu` — `gqa_attention_prompt_attention_launch_for` dispatch line 33 (`cache.dtype == DType::I8` → `gqa_attention_prefill_i8_kernel`, else bf16); `gqa_kv_append_launch_for` dispatch (`I8` → `fill_i8_page_kernel` (tokens>=128 && KVHeads==2) or `fill_i8_kernel`, else `fill_bf16_kernel`).
- **Prefill fill i8 kernel:** `src/ops/kernel/gqa_attention_prefill_i8.cuh` — `gqa_attention_prefill_fill_i8_kernel` (line 84, 8 warps, warp per (token,kv_head,group), 2 dims/lane, `absmax/127`→`__float2half_rn`, `gqa_kv_quant_code`, writes codes + lane0 scale); `gqa_attention_prefill_fill_i8_page_kernel` (line 148, 8-token tiles, KVHeads==2).
- **Codec index helpers:** `src/ops/kernel/gqa_attention_kv_quant.cuh` — `gqa_kv_quant_code_index`, `gqa_kv_quant_scale_index`, `gqa_kv_quant_src_index` (extent-parameterized via `paged_kv_element_offset`), `gqa_kv_quant_code`, `gqa_kv_dequant_i8x8_from`. Address math `paged_kv_element_offset<LeadingExtent,HeadExtent>` is extent-parameterized → FP6 uses extent 192.
- **Tests:** `tests/ops/test_gqa_attention.cpp` (1366 lines) — kHeadDim=256, kQuantGroup=64, kQuantGroups=4, kAttentionScale=0.0625f; criteria bf16 `{2.8e-3, 1.0e-3, 2.7e-3}` / int8 `{3.15e-3, 1.1e-3, 2.2e-3}`; `HostCache` (line 316), `encode_group` (line 329), `make_cache` (line 352), `append_cache` (line 386), `cache_value` (line 417), `ideal_attention` (line 432), `DeviceCache` (line 489), `BatchDeviceCache` (line 635), `run_geometry` (line 1274, dtype loop at 1276), `verify_workspace_capacity_contract` (line 1319). Register in `tests/CMakeLists.txt` near line 218.
- **DFlash:** `src/ops/launcher/kv_cache_append_prefix.cu` + `src/ops/kernel/kv_cache_append_prefix.cuh` are BF16-only (head_dim 128, cyclic window 4096). **OUT OF SCOPE** — stays BF16; document as scope boundary in docs task.

---

## File map

| File | Change |
|---|---|
| `include/ninfer/types.h` | add `Fp6Group64` enum value |
| `apps/cli/options.cpp` | `parse_kv_cache` `fp6` arm + usage text |
| `src/serve/serve_options.cpp` | `parse_kv_dtype` `fp6` arm |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h` | kv_dtype/kv_quant_group mapping for Fp6Group64 |
| `src/targets/qwen3_6/impl/runtime/program_impl.h` | MemorySummary mapping for U8 |
| `src/targets/qwen3_6/impl/state/decoder_state.cpp` | generalize plan_cache + layer_view to U8/192 |
| `src/ops/kernel/gqa_attention_kv_fp6.cuh` | NEW FP6 codec header (`__host__ __device__`) |
| `src/ops/wrapper/gqa_attention.cpp` | validate_cache/batch_cache + workspace gate accept U8/FP6 |
| `src/ops/kernel/gqa_attention_prefill_fp6.cuh` | NEW fill + attention kernels |
| `src/ops/launcher/gqa_attention_prefill.cu` | dispatch U8 → fp6 kernels |
| `src/ops/kernel/gqa_attention_decode_fp6.cuh` | NEW decode small-T kernel |
| `src/ops/launcher/gqa_attention_decode.cu` | `launch_tc_partial_fp6` + dispatch arms |
| `tests/ops/test_kv_fp6.cpp` | NEW host codec unit test |
| `tests/ops/test_gqa_attention.cpp` | FP6 test arms |
| `tests/CMakeLists.txt` | register new test |
| docs (cli.md, serving.md, performance.md, maintainer/paged-kv-cache.md) | document fp6 |

---

## Phase 1: Plumbing (enum, parsers, mapping, planning)

### Task 1: Public type + CLI/serve parsers

**Files:**
- Modify: `include/ninfer/types.h:21`
- Modify: `apps/cli/options.cpp:55-59` and `:86`
- Modify: `src/serve/serve_options.cpp:48-53`

- [ ] **Step 1: Add the enum value**

In `include/ninfer/types.h`, change the enum to:

```cpp
enum class KvCacheStorage : uint8_t { BFloat16, Int8Group64, Fp6Group64 };
```

- [ ] **Step 2: Add the CLI parser arm**

In `apps/cli/options.cpp`, `parse_kv_cache` currently ends with two `if` arms and a throw. Add the third arm before the throw:

```cpp
    if (text == "fp6") { return KvCacheStorage::Fp6Group64; }
```

And in the usage text (line 86) change:

```cpp
"[--kv-dtype bf16|int8|fp6]"
```

- [ ] **Step 3: Add the serve parser arm**

In `src/serve/serve_options.cpp`, `parse_kv_dtype(const char* value)` add before the throw:

```cpp
    if (std::strcmp(value, "fp6") == 0) { return KvCacheStorage::Fp6Group64; }
```

(verify the existing arms use `std::strcmp` and mirror them)

- [ ] **Step 4: Commit**

```bash
git add include/ninfer/types.h apps/cli/options.cpp src/serve/serve_options.cpp
git commit -m "feat(engine): expose fp6 kv-dtype on the public surface"
```

### Task 2: Family mapping + memory summary

**Files:**
- Modify: `src/targets/qwen3_6/impl/runtime/layouts_impl.h:698-699`
- Modify: `src/targets/qwen3_6/impl/runtime/program_impl.h:2212`

- [ ] **Step 1: Extend the kv_dtype mapping**

In `src/targets/qwen3_6/impl/runtime/layouts_impl.h`, the two lines currently read (line 698-699):

```cpp
        kv_dtype      = options.kv_cache == KvCacheStorage::BFloat16 ? DType::BF16 : DType::I8;
        kv_quant_group = options.kv_cache == KvCacheStorage::BFloat16 ? 0 : qwen3_6::kKvQuantGroup;
```

Replace with a chain that selects by the enum. Use a small helper or an if/else so it stays readable:

```cpp
        DType kv_dtype;
        int32_t kv_quant_group;
        if (options.kv_cache == KvCacheStorage::BFloat16) {
            kv_dtype       = DType::BF16;
            kv_quant_group = 0;
        } else if (options.kv_cache == KvCacheStorage::Fp6Group64) {
            kv_dtype       = DType::U8;
            kv_quant_group = qwen3_6::kKvQuantGroup;
        } else {
            kv_dtype       = DType::I8;
            kv_quant_group = qwen3_6::kKvQuantGroup;
        }
```

Match the surrounding declaration style (these lines sit inside an initializer list in the actual file — if so, keep it a ternary chain instead of statements):

```cpp
        kv_dtype = options.kv_cache == KvCacheStorage::BFloat16
                       ? DType::BF16
                       : (options.kv_cache == KvCacheStorage::Fp6Group64 ? DType::U8 : DType::I8),
        kv_quant_group = options.kv_cache == KvCacheStorage::BFloat16 ? 0 : qwen3_6::kKvQuantGroup,
```

Open the file first and confirm whether lines 698-699 are inside an aggregate initializer; use the ternary form if so.

- [ ] **Step 2: Extend the MemorySummary mapping**

In `src/targets/qwen3_6/impl/runtime/program_impl.h:2212`, change:

```cpp
        out.kv_cache = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16 : KvCacheStorage::Int8Group64;
```

to:

```cpp
        out.kv_cache = kv_dtype == DType::BF16   ? KvCacheStorage::BFloat16
                       : kv_dtype == DType::U8 ? KvCacheStorage::Fp6Group64
                                                : KvCacheStorage::Int8Group64;
```

- [ ] **Step 3: Build to verify**

Run: `cmake --build build -j 8`
Expected: compiles clean (types.h enum is forward-compatible; no exhaustive switch on KvCacheStorage should break, but if the build fails on an unhandled switch, add the Fp6Group64 arm there too).

- [ ] **Step 4: Commit**

```bash
git add src/targets/qwen3_6/impl/runtime/layouts_impl.h src/targets/qwen3_6/impl/runtime/program_impl.h
git commit -m "feat(engine): map fp6 kv storage to packed u8 cache planes"
```

### Task 3: Generalize target KV planning

**Files:**
- Modify: `src/targets/qwen3_6/impl/state/decoder_state.cpp` (plan_cache line 14, layer_view line 98, batch_layer_view line 116)

- [ ] **Step 1: Generalize `plan_cache`**

In `src/targets/qwen3_6/impl/state/decoder_state.cpp`, change the quantized flag and validity check (lines 23-27) to treat both I8 and U8 (FP6) as quantized codecs:

```cpp
    const bool quantized = dtype == DType::I8 || dtype == DType::U8;
    const bool fp6       = dtype == DType::U8;
    if ((!quantized && (dtype != DType::BF16 || quant_group != 0)) ||
        (quantized && (quant_group != kKvQuantGroup || head_dim % quant_group != 0))) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }
```

Then compute the code-plane leading extent. For BF16/I8 the plane holds `head_dim` elements per token; for U8 (FP6) the 256 packed codes occupy 192 bytes:

```cpp
    const std::int32_t code_leading_extent =
        fp6 ? (head_dim * 6) / 8 : head_dim; // 256 -> 192 packed bytes
```

Replace the per-layer plane push (lines 40-45) to use `code_leading_extent` for the code planes:

```cpp
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        pool_spec.planes.push_back({dtype, code_leading_extent, kv_heads, 256});
        pool_spec.planes.push_back({dtype, code_leading_extent, kv_heads, 256});
        if (quantized) {
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
        }
    }
```

- [ ] **Step 2: Generalize `layer_view` and `batch_layer_view`**

In both `PagedKVCache::layer_view` (line 98) and `PagedKVCache::batch_layer_view` (line 116), change:

```cpp
    const bool quantized     = dtype_ == DType::I8;
```

to:

```cpp
    const bool quantized     = dtype_ == DType::I8 || dtype_ == DType::U8;
```

The stride (`quantized ? 4ULL : 2ULL`) and scale-plane selection are then already correct.

- [ ] **Step 3: Build to verify**

Run: `cmake --build build -j 8`
Expected: compiles clean. (No runtime effect yet — the mapping path is not exercised until FP6 kernels dispatch.)

- [ ] **Step 4: Commit**

```bash
git add src/targets/qwen3_6/impl/state/decoder_state.cpp
git commit -m "feat(engine): plan packed 192-byte fp6 code planes per kv layer"
```

### Task 4: Wrapper validation + workspace gate

**Files:**
- Modify: `src/ops/wrapper/gqa_attention.cpp` (validate_cache line 54, validate_batch_cache line 110, workspace gate line 353)

- [ ] **Step 1: Add the extent helper**

Add a file-local helper near the top (after the `require_contiguous_nonnull` function):

```cpp
namespace {

// Leading extent (elements/token) of the KV cache code plane. BF16/I8 hold one
// element per head_dim; FP6 packs 6-bit codes into 192 bytes per 256-dim token.
// Must stay in sync with gqa_attention_kv_fp6.cuh::kGqaKvFp6LeadingExtent.
constexpr std::int32_t code_leading_extent(DType dtype) {
    return dtype == DType::U8 ? (kHeadDim * 6) / 8 : kHeadDim;
}

} // namespace
```

Place it inside the existing anonymous namespace so it is visible to both validate functions.

- [ ] **Step 2: Update `validate_cache`**

Change the dtype/quant-group checks (lines 55-64):

```cpp
    if ((cache.dtype != DType::BF16 && cache.dtype != DType::I8 && cache.dtype != DType::U8) ||
        cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    if (cache.dtype == DType::BF16 && cache.quant_group != 0) {
        throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have quant_group");
    }
    if (cache.dtype != DType::BF16 && cache.quant_group != kQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": quantized KV cache must use quant_group 64");
    }
```

Change the `code_dtype` mapping (line 74):

```cpp
    const DType code_dtype = cache.dtype == DType::I8   ? DType::I8
                             : cache.dtype == DType::U8 ? DType::U8
                                                        : DType::BF16;
```

Change the k/v pages shape (lines 78-81) to use the extent helper:

```cpp
    const std::int32_t extent = code_leading_extent(cache.dtype);
    require_shape(cache.k_pages, extent, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, extent, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v pages");
```

- [ ] **Step 3: Update `validate_batch_cache`**

Apply the identical three edits to lines 112-139: dtype check, `code_dtype` (line 132), and k/v pages shapes (lines 136-139) using the same helper.

- [ ] **Step 4: Update the workspace gate**

In `gqa_attention_workspace_capacity_bytes` (line 353), extend the dtype condition to accept U8:

```cpp
    if ((cache_dtype != DType::BF16 && cache_dtype != DType::I8 && cache_dtype != DType::U8) || ...
```

- [ ] **Step 5: Build to verify**

Run: `cmake --build build -j 8`
Expected: compiles clean.

- [ ] **Step 6: Run existing gqa tests to confirm no regression**

Run: `ctest --test-dir build -R ninfer_gqa_attention_test`
Expected: all pass (BF16 and I8 validation paths unchanged in behavior).

- [ ] **Step 7: Commit**

```bash
git add src/ops/wrapper/gqa_attention.cpp
git commit -m "feat(ops): accept packed fp6 cache planes in gqa validation"
```

---

## Phase 2: FP6 codec + host oracle test

### Task 5: Codec header

**Files:**
- Create: `src/ops/kernel/gqa_attention_kv_fp6.cuh`

- [ ] **Step 1: Write the codec header**

Create `src/ops/kernel/gqa_attention_kv_fp6.cuh`:

```cpp
#pragma once

// ninfer::ops - group-64 packed FP6 (E3M2) KV cache codec, shared by the FP6
// GQA attention kernels and the host oracle test. Every helper is __host__
// __device__ so the pure-host unit test exercises the exact same bit math as
// the device kernels.
//
// FP6 E3M2: 1 sign + 3 exp + 2 mantissa, bias 3.
//   exp 0 (subnormal): value = m * 2^-4, m in 1..3   (0.0625 .. 0.1875)
//   exp 1..6 (normal): value = (-1)^s * 2^(e-3) * (1 + m/4)
//   exp 7:             inf/nan, clamped to max finite 14.0 (e=6, m=3)
// Code bits: sign<<5 | exp<<2 | mant. +0 = 0x00, -0 = 0x20.
//
// Packing is LSB-first per token: code i lives at bit [6i, 6i+6); byte offset
// (3*i)>>2, bit (6*i)&7. A token block is 192 bytes (256 * 6 / 8). A d-block of
// 8 consecutive dims (d%8==0) is 48 bits at byte offset (d/8)*6.

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaKvFp6LeadingExtent = 192; // bytes/token code plane
inline constexpr int kGqaKvFp6Bits          = 6;
inline constexpr int kGqaKvFp6BlockDims     = 8;
inline constexpr int kGqaKvFp6BlockBytes    = 6;   // 8 codes * 6 bits = 48 bits

// Byte offset within the token code block of the d-block that starts at `d`
// (requires d % 8 == 0).
__host__ __device__ __forceinline__ int gqa_kv_fp6_block_offset(int d) {
    return (d / kGqaKvFp6BlockDims) * kGqaKvFp6BlockBytes;
}

// Decode one 6-bit code to its FP6 value.
__host__ __device__ __forceinline__ float gqa_kv_fp6_decode(std::uint32_t code) {
    const int sign = (code >> 5) & 1;
    const int exp  = (code >> 2) & 7;
    const int mant = code & 3;
    float mag;
    if (exp == 0) {
        mag = static_cast<float>(mant) * 0.0625f; // m * 2^-4
    } else if (exp == 7) {
        mag = 14.0f; // inf/nan clamped to max finite
    } else {
        mag = __ldexp(1.0f + static_cast<float>(mant) * 0.25f, exp - 3);
    }
    return sign ? -mag : mag;
}

// Round a non-negative magnitude to the nearest representable FP6 magnitude,
// round-half-to-even, clamped to max finite. Used by encode; returns the
// magnitude code (exp+mant bits, sign handled by caller).
__host__ __device__ __forceinline__ std::uint32_t gqa_kv_fp6_quantize_mag(float x) {
    if (x <= 0.0f) { return 0; }
    const float scaled  = x * 8.0f; // x * 2^3 to work in integer units of 2^-3
    const std::uint32_t f = static_cast<std::uint32_t>(scaled + 12582912.0f); // 1.5*2^23 magic
    ...
}
```

**Implement the rounding carefully.** The canonical approach: represent each FP6 value as `v = code_value / 8` in a half-integer lattice where the round-half-to-even boundary is exact. Use the standard round-to-nearest-even integer conversion trick applied to `x * 16` (each FP6 step is 1/16 of a unit in the subnormal range and 1/8 in the normal range; nearest-FP6 is found by computing `q = llround_even(x * 16.0f)` then mapping `q` to the nearest representable code via a lookup, or via direct construction:

```cpp
__host__ __device__ __forceinline__ std::uint32_t gqa_kv_fp6_encode(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return 0; }
    const float v   = x * inv_scale;         // normalized to [-14, 14] approx
    const bool neg  = v < 0.0f;
    const float mag = __fabsf(v);
    if (mag >= 14.0f) { return neg ? 0x3Fu : 0x1Fu; } // clamp max finite (exp6, mant3)
    // Quantize to the nearest E3M2 magnitude, round-half-to-even.
    const float mag16 = mag * 16.0f;         // lattice of half-units of the code
    const std::uint32_t q = static_cast<std::uint32_t>(mag16 + 0.5f); // RNE via truncation trick
    // Map lattice q (0..224) to {code, exponent} by scanning candidate codes
    // (56 distinct magnitudes). A tiny lookup table indexed by q>>2 gives the
    // code; tie handling for odd q requires checking both neighbors.
    std::uint32_t code = gqa_kv_fp6_nearest_code(q);
    return neg ? (code | 0x20u) : code;
}
```

Rather than hand-roll the tie logic inline, provide a helper that, given a candidate code `c` and the target magnitude, picks the code minimizing |decode(code) - mag| with ties to even mantissa. The simplest correct implementation that matches the oracle exactly:

```cpp
// Nearest code to `v` in (0,14), ties resolved to the code whose mantissa is
// even. Valid for v strictly less than 14.
__host__ __device__ __forceinline__ std::uint32_t gqa_kv_fp6_nearest_magnitude(float v) {
    std::uint32_t best = 0;
    float best_err = __FLT_MAX__;
    for (int e = 0; e <= 6; ++e) {
        for (int m = 0; m < 4; ++m) {
            const float val = gqa_kv_fp6_decode((e << 2) | m);
            const float err = __fabsf(v - val);
            if (err < best_err ||
                (err == best_err && ((e << 2) | m) % 2 == 0 && best % 2 == 1)) {
                best = (e << 2) | m;
                best_err = err;
            }
        }
    }
    return best;
}
```

This is the independent oracle-quality reference (56 candidates, deterministic). The plan's correctness criterion is that the **device kernel encode** path and the **host oracle** produce bit-identical codes; if a faster closed-form mapping is later substituted in the kernel, it must match this reference for all 56 distinct magnitudes and all RNE tie cases.

Finalize the header with:

```cpp
// Pack 8 codes (dim block) into 6 bytes at out[0..6).
__host__ __device__ __forceinline__ void gqa_kv_fp6_pack8(const std::uint8_t codes[8], std::uint8_t* out) {
    std::uint64_t raw = 0;
    for (int j = 0; j < 8; ++j) { raw |= static_cast<std::uint64_t>(codes[j] & 0x3Fu) << (6 * j); }
    // store 48 bits: bytes [0,4) as uint, bytes [4,6) as ushort
    *reinterpret_cast<std::uint32_t*>(out) = static_cast<std::uint32_t>(raw);
    *reinterpret_cast<std::uint16_t*>(out + 4) = static_cast<std::uint16_t>(raw >> 32);
}

// Unpack 8 codes from the 6 bytes at in[0..6).
__host__ __device__ __forceinline__ void gqa_kv_fp6_unpack8(const std::uint8_t* in, std::uint8_t codes[8]) {
    const std::uint64_t raw = *reinterpret_cast<const std::uint64_t*>(in);
    for (int j = 0; j < 8; ++j) { codes[j] = static_cast<std::uint8_t>((raw >> (6 * j)) & 0x3Fu); }
}
```

Note: the device kernels will do unaligned 8-byte loads from the packed plane; reinterpret_cast to uint64 of a 6-byte-aligned pointer is safe on CUDA (x86- and Blackwell-aligned loads tolerate it; the extra 2 bytes read beyond the block are within the token's 192-byte block boundary only if the block is not the last in the token — for the last block (d=248) the 8-byte load reads 2 bytes into the next token's data. **Constraint:** pad each token block to 192 bytes and ensure physical page stride is 192*64 = 12288 bytes (page boundary), so a 6-byte-aligned 8-byte load at byte offset 186 of a token reads bytes 186..194 — bytes 192..194 belong to the same page (next token). That is within the page allocation, so the load is always in-bounds. The packing math only uses bits 0..47.)

- [ ] **Step 2: Commit**

```bash
git add src/ops/kernel/gqa_attention_kv_fp6.cuh
git commit -m "feat(ops): add packed e3m2 fp6 kv codec header"
```

### Task 6: Host codec unit test

**Files:**
- Create: `tests/ops/test_kv_fp6.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the test**

Create `tests/ops/test_kv_fp6.cpp`:

```cpp
// Host-only unit test for the packed FP6 (E3M2) KV codec. Exercises decode,
// encode/RNE tie handling, pack/unpack round-trips, and per-group scale math
// against an independent bit-level reference.

#include <ninfer/ops/...> // no public header needed; include the codec directly

#include "ops/kernel/gqa_attention_kv_fp6.cuh"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);  \
            ++failures;                                                           \
        }                                                                         \
    } while (0)

float code_value(std::uint32_t code) {
    return ninfer::ops::gqa_kv_fp6_decode(code);
}

// Independent reference: every FP6 code's exact value from raw bits.
std::uint32_t reference_code(int sign, int exp, int mant) {
    return static_cast<std::uint32_t>((sign << 5) | (exp << 2) | mant);
}

double reference_value(int sign, int exp, int mant) {
    if (exp == 0) { return sign ? -mant * 0.0625 : mant * 0.0625; }
    if (exp == 7) { return sign ? -14.0 : 14.0; }
    const double mag = std::ldexp(1.0 + mant * 0.25, exp - 3);
    return sign ? -mag : mag;
}

void test_decode_all_codes() {
    for (int s = 0; s < 2; ++s) {
        for (int e = 0; e < 8; ++e) {
            for (int m = 0; m < 4; ++m) {
                const std::uint32_t code = reference_code(s, e, m);
                const double expected = reference_value(s, e, m);
                const double actual = code_value(code);
                CHECK(std::fabs(actual - expected) < 1e-6,
                      "fp6 decode mismatch");
            }
        }
    }
}

void test_scale_round_trip() {
    // A group with all zero values -> scale 0, all codes 0.
    // A group with amax = 7.0 -> scale fp16(7.0/14)=0.5, decode(14)*0.5 = 7.
    const std::uint32_t code = ninfer::ops::gqa_kv_fp6_encode(7.0f, 2.0f); // 7 * (1/0.5) = 14 -> max
    CHECK(ninfer::ops::gqa_kv_fp6_decode(code) == 14.0f, "amax encode clamps to 14");
}

void test_pack_unpack_round_trip() {
    for (int iter = 0; iter < 100; ++iter) {
        std::uint8_t codes[8];
        std::uint8_t packed[kGqaKvFp6BlockBytes];
        std::uint8_t unpacked[8];
        for (int j = 0; j < 8; ++j) { codes[j] = static_cast<std::uint8_t>((iter * 7 + j * 13) & 0x3Fu); }
        ninfer::ops::gqa_kv_fp6_pack8(codes, packed);
        ninfer::ops::gqa_kv_fp6_unpack8(packed, unpacked);
        for (int j = 0; j < 8; ++j) { CHECK(unpacked[j] == codes[j], "pack/unpack round-trip"); }
    }
}

void test_full_plane_layout() {
    // A full 256-dim token: 32 d-blocks * 6 bytes = 192 bytes. Pack 256 codes
    // and verify each code's byte offset = (3*d)>>2 and bit = (6*d)&7.
    std::uint8_t codes[256];
    std::uint8_t plane[192];
    for (int d = 0; d < 256; ++d) { codes[d] = static_cast<std::uint8_t>(d & 0x3Fu); }
    for (int b = 0; b < 32; ++b) {
        ninfer::ops::gqa_kv_fp6_pack8(codes + 8 * b, plane + 6 * b);
    }
    for (int d = 0; d < 256; ++d) {
        const std::uint64_t raw = *reinterpret_cast<const std::uint64_t*>(plane + (3 * d) / 8 * 6);
        // careful: (3*d)/8 * 6 is the block byte offset only when d%8==0; for
        // arbitrary d the code crosses blocks. Use block offset (d/8)*6 and bit
        // (6*(d%8)).
        const int byte_off = (d / 8) * 6;
        const int bit_off  = 6 * (d % 8);
        const std::uint64_t raw2 = *reinterpret_cast<const std::uint64_t*>(plane + byte_off);
        const std::uint32_t got = static_cast<std::uint32_t>((raw2 >> bit_off) & 0x3Fu);
        CHECK(got == (codes[d] & 0x3Fu), "full-plane bit layout");
    }
}

void test_encode_rne_ties() {
    // Round-half-to-even: value exactly between two codes selects even mantissa.
    // 0.25 (code 0x04, mant 0, exp 1) and 0.3125 (code 0x05, mant 1) are
    // adjacent; 0.28125 is the midpoint. Even mantissa code = 0x04.
    const std::uint32_t c = ninfer::ops::gqa_kv_fp6_encode(0.28125f, 1.0f) & 0x1Fu;
    CHECK(c == 0x04u, "RNE tie to even mantissa");
}

} // namespace

int main() {
    test_decode_all_codes();
    test_scale_round_trip();
    test_pack_unpack_round_trip();
    test_full_plane_layout();
    test_encode_rne_ties();
    if (failures != 0) {
        std::fprintf(stderr, "ninfer_kv_fp6_test: %d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stdout, "ninfer_kv_fp6_test: ok\n");
    return 0;
}
```

Adjust the test to match the final codec API (the exact helper names must match Task 5). Key point: this test is the **independent oracle** for the codec — it derives expected values from raw bit semantics, not from the codec implementation. If the plan's `gqa_kv_fp6_encode` name differs, rename in both places.

- [ ] **Step 2: Register the test**

In `tests/CMakeLists.txt`, after the existing gqa attention test registration (line 218-220), add:

```cmake
ninfer_add_op_test(ninfer_kv_fp6_test SOURCES ops/test_kv_fp6.cpp LIBRARIES ninfer_ops)
```

The test target needs the source dir on the include path to reach `ops/kernel/gqa_attention_kv_fp6.cuh` — check whether `ninfer_add_op_test` already adds `ops/kernel` to target_include_directories (the gqa test includes `ops/op_tester.h` similarly); if not, add `target_include_directories(ninfer_kv_fp6_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)` or the equivalent used by sibling tests.

- [ ] **Step 3: Configure and run**

Run:
```bash
cmake -S . -B build >/dev/null && cmake --build build -j 8 && ctest --test-dir build -R ninfer_kv_fp6_test
```
Expected: PASS (`ninfer_kv_fp6_test: ok`).

- [ ] **Step 4: Commit**

```bash
git add tests/ops/test_kv_fp6.cpp tests/CMakeLists.txt
git commit -m "test(ops): host oracle for packed fp6 kv codec"
```

---

## Phase 3: Prefill fill (encode+pack) kernels

### Task 7: FP6 fill kernels

**Files:**
- Create: `src/ops/kernel/gqa_attention_prefill_fp6.cuh`

- [ ] **Step 1: Write the fill kernel**

Create `src/ops/kernel/gqa_attention_prefill_fp6.cuh` deriving from `gqa_attention_prefill_fill_i8_kernel` (line 84 of the i8 file). Same structure: 8 warps (block 256), one warp per (token, kv_head, group) unit, 2 dims per lane. The difference is the codec stage.

```cpp
#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/gqa_attention_kv_fp6.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh" // src index helpers
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/paged_kv_address.cuh"

namespace ninfer::ops {

// FP6 fill: warp per (token, kv_head, group) unit; lanes 0..31 cover 2 dims
// (d0 = group*64+lane, d1 = d0+32) like the i8 fill. After computing absmax and
// the per-group FP16 scale, each lane encodes its 2 values, then the warp packs
// its 64 codes (8 d-blocks x 8) into 48 bytes at byte offset group*48 of the
// token plane.
template <typename Geometry, typename CacheView, typename Metadata>
__global__ void gqa_attention_prefill_fill_fp6_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, std::int32_t tokens) {
    const std::int32_t unit  = blockIdx.x;
    const std::int32_t token = unit / (Geometry::KVHeads * kGqaKvQuantGroups);
    const std::int32_t head  = (unit / kGqaKvQuantGroups) % Geometry::KVHeads;
    const std::int32_t group = unit % kGqaKvQuantGroups;
    if (token >= tokens) { return; }

    const int lane   = threadIdx.x & 31;
    const int d0     = group * kGqaKvQuantGroup + lane;
    const int d1     = d0 + 32;
    const std::int32_t position = positions[token];
    const std::int32_t page_off = position & kPagedKVPageMask;
    const std::int32_t page     = paged_kv_physical_page(metadata.block_table(), position);

    // Stage the 64 dims, compute absmax.
    float v0 = static_cast<float>(k[gqa_kv_quant_src_index<Geometry>(head, d0, token)]);
    float v1 = static_cast<float>(k[gqa_kv_quant_src_index<Geometry>(head, d1, token)]);
    float amax = warp_max<32>(max(__fabsf(v0), __fabsf(v1)));
    __shared__ float sh[32]; // per-lane pack staging reused for k and v
    const float scale = amax / 14.0f;
    const float inv_scale = amax == 0.0f ? 0.0f : 1.0f / scale;
    __shared__ __half sh_scale;
    if (lane == 0) { sh_scale = __float2half_rn(scale); }
    __syncthreads();
    const __half stored = sh_scale;
    const float sinv = stored == __half(0.0f) ? 0.0f : 1.0f / static_cast<float>(stored);

    const std::int64_t code_base =
        paged_kv_element_offset<kGqaKvFp6LeadingExtent, Geometry::KVHeads>(page, head, page_off, 0);
    const std::int64_t token_base = code_base + group * 48; // group*64 dims -> group*48 bytes

    if (sinv != 0.0f) {
        // Encode the two dims per lane into a shared staging slot, then pack.
        __shared__ std::uint8_t sh_codes[64];
        __shared__ std::uint8_t sh_packed[48];
        sh_codes[lane]      = static_cast<std::uint8_t>(gqa_kv_fp6_encode(v0, sinv) & 0x3Fu);
        sh_codes[lane + 32] = static_cast<std::uint8_t>(gqa_kv_fp6_encode(v1, sinv) & 0x3Fu);
        __syncthreads();
        // 8 d-blocks: lanes 0..7 each pack one block of 8 codes into 6 bytes.
        if (lane < 8) {
            std::uint8_t block[8];
            for (int j = 0; j < 8; ++j) { block[j] = sh_codes[lane * 8 + j]; }
            gqa_kv_fp6_pack8(block, sh_packed + lane * 6);
        }
        __syncthreads();
        // Copy 48 bytes to the cache plane (12 int4 stores).
        const int4* src4 = reinterpret_cast<const int4*>(sh_packed);
        int4* dst4 = reinterpret_cast<int4*>(cache_k + token_base);
        if (lane < 12) { dst4[lane] = src4[lane]; }
    }
    if (lane == 0) { scale_k[gqa_kv_quant_scale_index<Geometry>(page, head, group, page_off)] = stored; }

    // Same for V.
    float w0 = static_cast<float>(v[gqa_kv_quant_src_index<Geometry>(head, d0, token)]);
    float w1 = static_cast<float>(v[gqa_kv_quant_src_index<Geometry>(head, d1, token)]);
    float vamax = warp_max<32>(max(__fabsf(w0), __fabsf(w1)));
    const float vscale = vamax / 14.0f;
    const float vinv = vamax == 0.0f ? 0.0f : 1.0f / vscale;
    __syncthreads();
    if (lane == 0) { sh_scale = __float2half_rn(vscale); }
    __syncthreads();
    const __half vstored = sh_scale;
    const float vsinv = vstored == __half(0.0f) ? 0.0f : 1.0f / static_cast<float>(vstored);
    if (vsinv != 0.0f) {
        sh_codes[lane]      = static_cast<std::uint8_t>(gqa_kv_fp6_encode(w0, vsinv) & 0x3Fu);
        sh_codes[lane + 32] = static_cast<std::uint8_t>(gqa_kv_fp6_encode(w1, vsinv) & 0x3Fu);
        __syncthreads();
        if (lane < 8) {
            std::uint8_t block[8];
            for (int j = 0; j < 8; ++j) { block[j] = sh_codes[lane * 8 + j]; }
            gqa_kv_fp6_pack8(block, sh_packed + lane * 6);
        }
        __syncthreads();
        int4* dstv4 = reinterpret_cast<int4*>(cache_v + token_base);
        const int4* srcv4 = reinterpret_cast<const int4*>(sh_packed);
        if (lane < 12) { dstv4[lane] = srcv4[lane]; }
    }
    if (lane == 0) { scale_v[gqa_kv_quant_scale_index<Geometry>(page, head, group, page_off)] = vstored; }
}

} // namespace ninfer::ops
```

Verify the helpers actually used exist with those exact names/signatures before finalizing: `warp_max<N>`, `paged_kv_physical_page`, `paged_kv_element_offset`, `gqa_kv_quant_src_index`, `gqa_kv_quant_scale_index`, `__nv_bfloat16`/`__half` includes (add `cuda_bf16.h`/`cuda_fp16.h` if not pulled transitively). The `__syncthreads` between the k and v sections must be checked against the i8 fill's actual reuse pattern — the i8 fill computes k and v in the same warp without a sync because it only writes final global results; the FP6 version writes intermediate packed bytes to `sh_packed`, so the extra syncs shown are required. If a single-warp-each-unit schedule is used (like i8), only one unit's `sh_scale`/`sh_codes`/`sh_packed` arrays are live per CTA, which is safe with `__shared__` declarations + the syncs above.

- [ ] **Step 2: Add the page-local variant**

Add `gqa_attention_prefill_fill_fp6_page_kernel` mirroring `gqa_attention_prefill_fill_i8_page_kernel` (line 148 of the i8 file): 8-token tiles, `KVHeads==2`, block 256, warp = token in tile, page-local code base via `paged_kv_page_head_offset<kGqaKvFp6LeadingExtent, Geometry::KVHeads>(page, kv_head) + page_off*192`. Same pack staging per warp.

- [ ] **Step 3: Wire into the prefill launcher**

In `src/ops/launcher/gqa_attention_prefill.cu`, extend `gqa_kv_append_launch_for` so the dispatch handles `cache.dtype == DType::U8`:

```cpp
    if (cache.dtype == DType::I8) {
        // existing i8 path
    } else if (cache.dtype == DType::U8) {
        if (tokens >= 128 && KVHeads == 2) {
            // gqa_attention_prefill_fill_fp6_page_kernel
        } else {
            // gqa_attention_prefill_fill_fp6_kernel
        }
    } else {
        // existing bf16 path
    }
```

Mirror the exact grid/block/args of the i8 arms, swapping kernel names and pointer types (cache_k/v become `std::uint8_t*`).

- [ ] **Step 4: Build**

Run: `cmake --build build -j 8`
Expected: compiles clean.

- [ ] **Step 5: Commit**

```bash
git add src/ops/kernel/gqa_attention_prefill_fp6.cuh src/ops/launcher/gqa_attention_prefill.cu
git commit -m "feat(ops): fp6 prefill fill kernels with cooperative pack"
```

---

## Phase 4: Prefill attention kernel

### Task 8: FP6 prefill attention kernel

**Files:**
- Create: `src/ops/kernel/gqa_attention_prefill_fp6.cuh` (append to the file from Task 7)

- [ ] **Step 1: Write the kernel**

Add `gqa_attention_prefill_fp6_kernel<Geometry, Metadata>` to the same header, derived from `gqa_attention_prefill_bf16_kernel` (line 103 of `gqa_attention_prefill_bf16.cuh`). Keep the whole structure identical (D=256, Br=64, Bc=64, Threads=128, QKNt=8, QKKs=16, PVNt=32, PVKs=4, q_s/k_s/v_s bf16 smem `[64*256]`, Q staged once, bf16 QK/PV MMA, online softmax exp2 with `scale_l2`). Replace only the K/V staging:

**Replacement staging block** (replaces `gqa_prefill_stage_kv<Geometry>` at line 64 of the bf16 file):

```cpp
// Stage one Bc x 256 K (or V) tile from the packed FP6 cache plane into the
// swizzled bf16 smem tile. Two passes with a barrier between them:
//   1) cp.async the packed bytes (Bc*192 per tile) into a packed smem arena;
//   2) dequant: unpack each d-block, decode, multiply by the group scale, and
//      write the swizzled bf16 tile.
// The scale for each of the Bc rows is cp.async'd separately (4 groups * 2 bytes).
```

Concretely, add a `packed_kv_s` smem arena sized `Bc*192` bytes, plus a small `scale_s[64][4]` `__half` arena. In the key loop per tile: cp_async 16-byte chunks of `cache_k + paged_kv_element_offset<192,KVHeads>(page, head, key&63, 0) + key_tile*192` into `packed_kv_s`; cp_async the 8 scale bytes (`scale_k + gqa_kv_quant_scale_index<Geometry>(page, head, group, key&63)`) into `scale_s`. Then one dequant pass over `(row, dblock)` pairs: each thread handles one (row, dblock) = 8 dims → loads 6 packed bytes from `packed_kv_s + row*192 + dblock*6`, `gqa_kv_fp6_unpack8`, decodes `gqa_kv_fp6_decode` × scale, and stores 4 `__half` pairs into `k_s` at the swizzled bf16 addresses (the existing `gqa_prefill_swz_addr`/`gqa_prefill_swz` mapping). `__syncthreads()` after the dequant pass before the ldmatrix that consumes the tile.

The dequant pass must be sized to the thread count: 64 rows × 32 dblocks = 2048 work items; with 128 threads that's 16 items/thread. Grid the loop as `for (int i = threadIdx.x; i < 64*32; i += 128)` with `row = i / 32; dblock = i % 32`.

Smem budget: existing `kGqaPrefillSmemBytes` = `(64+128)*256*2` = 98304 bytes. Adding `packed_kv_s` (2 buffers? K and V each need their own arena if both are live across the pipeline — the bf16 kernel keeps both k_s and v_s live; for FP6 the packed arenas can be smaller if dequant happens immediately after the cp_async of each buffer, so a single `packed_s` of `Bc*192` = 12288 bytes reused for K then V is sufficient because V is staged after K is consumed). Compute the new total and set `cudaFuncSetAttribute` for the fp6 kernel's `MaxDynamicSharedMemorySize` in the launcher (Task 8 Step 2).

**EXECUTION AMENDMENT (learned when Task 8 was implemented): the arena design is NON-LAUNCHABLE on sm_120.** Consumer Blackwell (sm_120/sm_120a, RTX 50 series) caps opt-in dynamic shared memory at **101376 B** (measured `sharedMemPerBlockOptin` on sm_120; the 227 KB figure is datacenter Hopper/Blackwell sm_90/sm_100). The bf16 base kernel already uses 98304 B, leaving only **3072 B** headroom — no packed arena of any size fits while keeping q_s/k_s/v_s. Task 8 was therefore implemented with **direct-global dequant staging** (no packed smem arena): the staging function reads 6 packed bytes + the group scale directly from global per (row, d-block) item (byte-wise per the alignment rule), decodes, and writes the swizzled bf16 tile; smem stays at 98304 B. Cost: no cp.async K/V loop overlap (prologue Q cp.async still overlaps K(0)); the halved-bandwidth read win is retained. A later arena/overlap variant would require reducing q_s/k_s/v_s (e.g. Br=32) — out of scope. Note also: `cudaFuncSetAttribute` is REQUIRED for the 98304 B launch (48 KiB default opt-in threshold), exactly as the bf16 arm does it.

- [ ] **Step 2: Wire into the prefill launcher**

In `gqa_attention_prompt_attention_launch_for` (src/ops/launcher/gqa_attention_prefill.cu, dispatch line 33), add a U8 arm between the i8 and bf16 paths:

```cpp
    } else if (cache.dtype == DType::U8) {
        // cudaFuncSetAttribute(gqa_attention_prefill_fp6_kernel<Geometry, Metadata>,
        //                     MaxDynamicSharedMemorySize, kGqaPrefillFp6SmemBytes);
        // gqa_attention_prefill_fp6_kernel<Geometry, Metadata><<<
        //     dim3(div_up(tokens, kGqaPrefillBr), q_heads), kGqaPrefillThreads,
        //     kGqaPrefillFp6SmemBytes>>>(q, cache_k, cache_v, cache_k_scale, cache_v_scale,
        //     metadata, positions, scale, out, tokens);
    }
```

Define `kGqaPrefillFp6SmemBytes` in the fp6 header (base smem + 12288 packed + 512 scale).

- [ ] **Step 3: Build**

Run: `cmake --build build -j 8`
Expected: compiles clean.

- [ ] **Step 4: Commit**

```bash
git add src/ops/kernel/gqa_attention_prefill_fp6.cuh src/ops/launcher/gqa_attention_prefill.cu
git commit -m "feat(ops): fp6 prefill attention kernel with packed dequant staging"
```

---

## Phase 5: Decode small-T kernel

### Task 9: FP6 decode kernel

**Files:**
- Create: `src/ops/kernel/gqa_attention_decode_fp6.cuh`

- [ ] **Step 1: Write the kernel**

Create `gqa_attention_decode_fp6_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch, Masked, CacheInput>` derived from `gqa_attention_small_t_tc_partial_bf16_kernel` (`gqa_attention_decode_bf16.cuh`). Keep the full split/window/softmax/QK/PV bf16 MMA structure. Two surgical replacements:

**(a) Fused append (replaces bf16 lines 148-168).** For `CacheInput::writes_cache`, after staging the current rows (which the bf16 kernel reads from `input.k/v`), encode+pack each row into the packed plane instead of raw bf16 stores. Reuse a `qkv_s`-like staging region: since the bf16 kernel already has `qkv_s[64*256]` bf16 for the current rows, after a `__syncthreads()` the warp(s) can read the bf16 staged rows, compute per-(row,group) absmax, and pack. Simpler and matching the fill-kernel pattern: one warp per (token, group) encodes 64 dims from the staged bf16 rows, packs to 48 bytes, writes to `cache_k + gqa_kv_fp6_block_offset(...)` and the FP16 scale to `scale_k`. With TokenTile ≤ 6 and up to 4 warps/CTA this is at most 6 rows × 4 groups = 24 encode units — schedule them across the CTA's warps with a loop.

**(b) Key-loop staging (replaces bf16 lines ~222-257).** Where the bf16 kernel does `cp_async<16>` of bf16 into the swizzled bf16 tile, the FP6 kernel:
1. cp_async 16-byte chunks of packed bytes from `cache_k + paged_kv_element_offset<192,KVHeads>(physical_page, kv_head, key&63, 0) + (key_tile)*192` into a `packed_s` smem arena (`Bc*192` = 32*192 = 6144 bytes per buffer; K and V staged with separate arenas or a single reused arena with ordering like the prefill kernel);
2. cp_async the row's 4 FP16 group scales into `scale_s`;
3. dequant pass (same as Task 8) writing the swizzled bf16 `k_s`/`v_s` tiles;
4. `__syncthreads()` before the ldmatrix for QK/PV.

The from_new path (reading `input.k/v` for fresh tokens) can keep the bf16 cp_async path — those rows are also encoded into the cache by the fused append, so the kernel must read fresh rows from the bf16 input buffer, not the packed cache. Dispatch per-tile: `if (from_new) cp_async bf16 (existing path) else fp6 packed staging`.

Smem: the bf16 kernel uses static smem (qkv_s + p_s). The FP6 kernel adds `packed_s` (2 × 6144) + `scale_s`, so it needs dynamic smem — set `cudaFuncSetAttribute` in the launcher like the i8 kernel.

**EXECUTION AMENDMENT (Task 9):** implemented with TWO separate packed arenas (K+V both live, 2×32×192 = 12288 B + 512 B scale = 12800 B dynamic), matching the bf16 kernel's barrier structure — a single reused arena would save 6144 B but buy zero occupancy (still 2 blocks/SM) while adding two barriers per tile on the latency path. Total 50816 B (Wc=4) fits the 101376 B sm_120 cap with `__launch_bounds__(128,2)` occupancy intact (2×50816 = 101632 ≤ 102400). cudaFuncSetAttribute(12800) is required for Wc=4 (50816 > 49152 default), Wc=2 (48320 B) is under it. Fused append encodes directly from `input.k/v` (not staged rows). The shared d-block dequant helper lives in the codec header (`gqa_kv_fp6_dequant_dblock8`, using `__floats2bfloat162_rn` instead of device-only `pack_bf16x2` so the host oracle test can compile it — bit-identical on sm_80+).

- [ ] **Step 2: Wire into the decode launcher**

In `src/ops/launcher/gqa_attention_decode.cu`:
1. Add `launch_tc_partial_fp6<Geometry, TokenTile, ...>` mirroring `launch_tc_partial_bf16` (line 87), with the fp6 kernel, dynamic smem via `cudaFuncSetAttribute`, and packed pointer types (`std::uint8_t*` for cache_k/v, `__half*` for scales). Choose warp configs like the bf16 small-T path (WarpsPerCta 1-4 by TokenTile) — do **not** reuse the i8 special tier configs.
2. In `NINFER_GQA_SMALL_T_DISPATCH` (line 249), add the U8 arm:

```cpp
    if (cache.dtype == DType::I8) {
        launch_tc_partial_i8<...>(...);
    } else if (cache.dtype == DType::U8) {
        launch_tc_partial_fp6<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(...);
    } else {
        launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(...);
    }
```

3. In `gqa_attention_split_capacity` (line 220), add `cache_dtype != DType::U8` to the gate, and route the U8 capacity through the bf16 split counts (`gqa_small_t_split_upper_bound`) — extend `gqa_small_t_split_count<Geometry>(window, tokens, kv_dtype)` to return the bf16 default for U8.
4. The reduce kernel dispatch (line 339) already falls to `launch_for_dtype<false>()` for non-I8 → U8 uses the bf16-style reduce. No change.

- [ ] **Step 3: Build**

Run: `cmake --build build -j 8`
Expected: compiles clean.

- [ ] **Step 4: Commit**

```bash
git add src/ops/kernel/gqa_attention_decode_fp6.cuh src/ops/launcher/gqa_attention_decode.cu
git commit -m "feat(ops): fp6 decode kernel with packed dequant + encode append"
```

---

## Phase 6: Test extension + integration

### Task 10: Extend the GQA attention test harness

**Files:**
- Modify: `tests/ops/test_gqa_attention.cpp`

- [ ] **Step 1: Add FP6 criterion and host-side encode**

Add the FP6 attention criterion next to the existing ones:

```cpp
const ReductionCriterion kAttentionFp6Criterion{
    .relative_l2                = 2.9e-3f,
    .gross_absolute             = 1.05e-3f,
    .gross_relative_to_max_ref  = 2.4e-3f,
};
```

(These bounds are tighter than the INT8 criterion because FP6 E3M2 has finer near-zero resolution; tune after the first run if the kernel is noisier.)

Add an FP6 `encode_group` analog: same absmax flow as `encode_group` (line 329) but scale `unrounded = absmax/14.0f`, and encode+pack per d-block. Add a helper `fp6_encode_pack` on the host:

```cpp
void fp6_encode_group(const std::vector<std::uint16_t>& source, std::size_t base,
                      std::vector<std::uint8_t>& codes, std::size_t plane_base,
                      std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float amax = 0.0f;
    for (int i = 0; i < 64; ++i) {
        amax = std::max(amax, std::fabs(bf16_to_f32(source[base + i])));
    }
    const std::uint16_t stored = f32_to_f16_bits(amax / 14.0f);
    scales[scale_offset] = stored;
    const float s = f16_bits_to_f32(stored);
    const float inv = s == 0.0f ? 0.0f : 1.0f / s;
    for (int b = 0; b < 8; ++b) {
        std::uint8_t block[8];
        for (int j = 0; j < 8; ++j) {
            const int d = b * 8 + j;
            block[j] = static_cast<std::uint8_t>(ninfer::ops::gqa_kv_fp6_encode(
                bf16_to_f32(source[base + d]), inv) & 0x3Fu);
        }
        ninfer::ops::gqa_kv_fp6_pack8(block, codes.data() + plane_base + b * 6);
    }
}
```

(best is to reuse the codec's encode/decode/pack helpers from `gqa_attention_kv_fp6.cuh`, which the test target must include — the test already includes `ops/kernel/gqa_attention_kv_quant.cuh`-style headers; mirror that include.)

- [ ] **Step 2: Extend `HostCache`**

Add packed storage to `HostCache` (line 316):

```cpp
    std::vector<std::uint8_t> k_fp6; // 192 bytes/token/head logical plane
    std::vector<std::uint8_t> v_fp6;
```

`make_cache` (line 352) and `append_cache` (line 386): for `dtype == DType::U8`, size the fp6 planes to `192 * padded_context * kv_heads`, and fill via `fp6_encode_group` per (position, head, group). `cache_value` (line 417): for U8, decode the packed code:

```cpp
    } else if (cache.dtype == DType::U8) {
        const std::uint64_t raw = *reinterpret_cast<const std::uint64_t*>(
            cache.k_fp6.data() + (d / 8) * 6 + 192 * (position + padded * head));
        const std::uint32_t code = static_cast<std::uint32_t>((raw >> (6 * (d % 8))) & 0x3Fu);
        return ninfer::ops::gqa_kv_fp6_decode(code) * f16_bits_to_f32(cache.k_scale[scale_offset]);
    }
```

- [ ] **Step 3: Extend `DeviceCache` and `BatchDeviceCache`**

Both must handle the packed plane:
- `code_leading_extent`: 192 for U8, 256 otherwise.
- `k_`/`v_` sized `extent * 64 * kv_heads * physical_pages` bytes for U8 (no element type multiplier).
- `k_scale_` present for both I8 and U8.
- upload/download via `scatter_paged`/`gather_paged` with the U8 extent.
- `view()`/`batch_view()` (lines 539/675): `k_pages`/`v_pages` Tensor `{192, 64, kv_heads, physical_pages}` dtype `DType::U8`, scales `{4, 64, kv_heads, pages}` FP16, `quant_group` 64.

- [ ] **Step 4: Extend dtype loops and naming**

- `cache_name(dtype)` (line 849): add U8 → `"fp6-g64"`.
- `attention_criterion(dtype)` (line 851): U8 → `kAttentionFp6Criterion`.
- `run_geometry` (line 1274): extend the dtype loop at line 1276 to `{DType::BF16, DType::I8, DType::U8}`.
- `run_batch_cases()` (line 1251): add one FP6 batch case (e.g. `run_batch_case(kGeometries[0], DType::U8, {...})` mirroring an existing width/context case).
- `verify_workspace_capacity_contract` (line 1319): add U8 to the dtype set it exercises.

- [ ] **Step 5: Build and run the attention test**

Run:
```bash
cmake -S . -B build >/dev/null && cmake --build build -j 8 && ctest --test-dir build -R ninfer_gqa_attention_test
```
Expected: all BF16/I8 cases pass as before; FP6 cases pass. If FP6 cases fail on the criterion, tune `kAttentionFp6Criterion` bounds based on the reported max relative-L2 error, but only if the error is tight (same order as the bf16 kernel) and consistent.

- [ ] **Step 6: Run the codec unit test + full gqa set**

Run: `ctest --test-dir build -R 'ninfer_(kv_fp6|gqa_attention)_test'`
Expected: both pass.

- [ ] **Step 7: Commit**

```bash
git add tests/ops/test_gqa_attention.cpp
git commit -m "test(ops): fp6 attention + append + batch coverage in gqa harness"
```

### Task 11: End-to-end verification

- [ ] **Step 1: Full build + full test sweep**

Run:
```bash
cmake --build build -j 8
ctest --test-dir build
```
Expected: all tests pass (including the existing bf16/int8 gqa and kv_cache_append_prefix suites, confirming no regression).

- [ ] **Step 2: Real-artifact run with fp6**

Run:
```bash
./build/bin/ninfer out/qwen3_6_27b.ninfer --prompt "..." --kv-dtype fp6 --max-context 4096
```
Expected: engine runs; `MemorySummary.kv_cache` reports `Fp6Group64`. Confirm parity on a short prompt by comparing generation to the same run with `--kv-dtype bf16` (identical sampling parameters; tokens should match exactly, since fp6 decode is deterministic).

- [ ] **Step 3: Measure the memory win**

Run a fixed-context measurement at 96000 tokens with `--kv-dtype fp6` and record the KV payload from the memory summary. Expected: FP6 KV ≈ `3.21 GiB * (192+8)/264` ≈ 2.43 GiB (vs 3.21 GiB int8, ~6.4 GiB bf16). Record the actual number in the performance doc update (Task 12).

---

## Phase 7: Documentation

### Task 12: Update docs

**Files:**
- Modify: `docs/cli.md`
- Modify: `docs/serving.md`
- Modify: `docs/performance.md`
- Modify: `docs/maintainer/paged-kv-cache.md`

- [ ] **Step 1: CLI docs**

In `docs/cli.md`, extend the `--kv-dtype` option text: add `fp6` (group-64 FP6 E3M2, packed 6-bit codes, 192 bytes/token) to the accepted values and note it halves INT8 KV size with finer near-zero resolution at slightly higher compute cost.

- [ ] **Step 2: Serving docs**

In `docs/serving.md`, extend the serving kv-dtype option to accept `fp6`.

- [ ] **Step 3: Performance docs**

In `docs/performance.md`, add the FP6 KV footprint number measured in Task 11 Step 3 and, if the benchmark methodology table lists kv formats, add FP6. State the scope boundary: DFlash KV stays BF16.

- [ ] **Step 4: Maintainer paged-kv-cache doc**

In `docs/maintainer/paged-kv-cache.md`, add a section describing the packed sub-byte code plane: FP6 uses `DType::U8` with leading extent 192 (bytes/token), LSB-first 6-bit packing, FP16 scale plane unchanged (group-64), and the codec/reference contract pointing at `gqa_attention_kv_fp6.cuh`.

- [ ] **Step 5: Commit**

```bash
git add docs/cli.md docs/serving.md docs/performance.md docs/maintainer/paged-kv-cache.md
git commit -m "docs: document packed fp6 kv storage"
```

---

## Future work (out of scope for this package)

- **Rotation (TurboQuant-style).** llama.cpp applies a random rotation to KV vectors before quantization; it is the same basic trick TurboQuant uses and closes most of the low-bit quality gap. NInfer's INT8 (and this FP6) do **not** rotate, so quality sits below a rotated baseline — at 8 bits the gap is small (uniform grid is fine-grained enough), at 6 bits modest. Rotation is orthogonal to the codec/plane architecture built here: it would be added as a fixed rotation applied on the Q side (folded into the q-projection weights offline) and on the K/V side (rotated before encode, inverse-rotated after decode at attention time), with no change to the packed-plane or scale-plane design. Note the counter-tradeoff measured in llama.cpp on Qwen3.6-27B (head_dim=256): the rotated-K path inflates the K cache footprint (higher-precision storage for deferred rotation at attention time) — NInfer's design deliberately avoids that cost, so a rotation follow-up must weigh quality gain vs the extra memory/bandwidth. Candidate follow-up work package.
- **KVarN (variance-normalized KV quant), beellama.cpp fork** (`https://github.com/Anbeeld/beellama.cpp`). A variance-normalized KV-cache codec from the llama.cpp ecosystem with native rotated-domain attention, published as the community's recommended KV-quant approach in the r/LocalLLaMA TurboQuant thread. Includes a Qwen3.6-27B-specific benchmark set (75 cache-type pairs, q8/q6/q5/q4, KVarN, Turbo/TCQ) that is directly relevant calibration data for this target family. KVarN sits in the 3-5 bit band; if a later work package targets that band, KVarN's normalization scheme and its llama.cpp FlashAttention integration are worth a dedicated research pass before choosing INT/FP/KVarN there.

---

## Self-review notes

- **Spec coverage:** the plan covers enum/parse surface (Task 1), mapping + memory summary (Task 2), planning (Task 3), wrapper validation (Task 4), codec + oracle (Tasks 5-6), fill kernels (Task 7), prefill attention (Task 8), decode (Task 9), tests (Task 10), e2e verification (Task 11), docs (Task 12). DFlash is explicitly scoped out per the user decision and documented in Task 12 Step 3.
- **Placeholder check:** kernel tasks reference exact source lines and provide the changed blocks; they intentionally do not reproduce the 400-600 line surrounding kernels. The one open item is the exact rounding math in `gqa_kv_fp6_encode` (Task 5) — the plan pins the required **behavior** (RNE, bit-identical to the 56-code reference) and provides the reference implementation; the closed-form kernel mapping may be substituted if it matches. (Resolution: the reference scan in `gqa_kv_fp6_nearest_magnitude` was shipped as the encode path; a closed-form swap is a later perf-only change. Clamp is 0x1B = exp6/mant3 max finite, not 0x1F.)
- **Hardware fact (execution-learned):** sm_120 consumer Blackwell dynamic-smem opt-in cap is 101376 B, not 227 KB; the prefill fp6 kernel must not exceed 98304 B (bf16 base) and needs cudaFuncSetAttribute at that size.
- **Type consistency:** the codec helper names used across Tasks 5, 6, 7, 9, 10 (`gqa_kv_fp6_decode`, `gqa_kv_fp6_encode`, `gqa_kv_fp6_pack8`, `gqa_kv_fp6_unpack8`, `gqa_kv_fp6_block_offset`, `kGqaKvFp6LeadingExtent`) are consistent; `code_leading_extent`/192 coupling between the wrapper and the codec is flagged in both places.
