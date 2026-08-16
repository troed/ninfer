# Qwen3.6-27B Staging Arena and Slot Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Surface 4 of the weight-offload work: allocate a fixed-address device staging arena, give every host-placed (offloaded) 27B FFN weight a fixed slot address, re-point the model-view `Weight` device pointers (`payload`/`qdata`/`qhigh`/`scales`) at those slots (which also re-points NVFP4 TMA maps, since TMA descriptors are encoded per-launch from `Weight.qdata`/`.scales`), and publish an offloaded-tensor → slot map that surface 5's graph interleave will consume.

**Architecture:** Kernel weight pointers are baked into captured graphs at capture time, so slot addresses must be fixed before `create_program`/`prepare_graphs`. The re-point seam is therefore the 27B `LoadedModelData` constructor (`bindings.cpp:448`), which already runs before `create_program` in `construct_registered` and is directly exercised by the residency test. The family `ModelView` (shared, `model_view.h`) gains the `staged_weights` slot map and a `staging_arena` pointer so the family `Program` (surface 5) can reach them through the `const ModelView&` it already holds. Slot layout follows the design's double-buffer scheme: each text layer is one offload group `g`, layer `g` streams into buffer `g % 2`, arena capacity = 2 × largest streaming unit (one layer's FFN bytes).

**Tech Stack:** C++20, CUDA 13.3, CMake/Ninja (`sm_120a`), RTX 5060 Ti 16GB dev machine. All builds must use `cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=120a` and `/usr/bin/ctest` (the `~/.local/bin/ctest` shim is broken).

---

## Key existing types (verified against master `fbf8eed`)

- `src/core/tensor.h` — `struct Tensor { void* data; const void* host; DType dtype; int32 ne[4]; int64 nb[4]; }`; `struct Weight { const void* payload; const void* host; std::uint64_t payload_bytes, high_plane_bytes; QType qtype; std::uint32_t group_size; int32 shape[4], padded_shape[4]; std::uint32_t ndim; const void* qdata, qhigh, scales; int32 n, k, group; QuantLayout layout; DType scale_dtype; int32 scale_ne[4]; int64 scale_nb[4]; float weight_scale_divisor, input_scale_divisor; }`.
- `src/core/arena.h` — `class DeviceArena { explicit DeviceArena(std::size_t capacity_bytes); explicit DeviceArena(DeviceSpan); ...; DeviceSpan alloc_bytes(bytes, align=256); void* base() const noexcept; std::size_t capacity() const noexcept; std::size_t used() const noexcept; ... }` (move-only, owns a single `cudaMalloc` on the current context; `arena.cu:134-148`). `using WorkspaceArena = DeviceArena;`.
- `src/artifact/materializer.h` — `MaterializedArtifact` public API: `void* device_data(ObjectHandle) const; void* host_data(ObjectHandle) const; void* device_data_or_null(ObjectHandle) const noexcept; void* host_data_or_null(ObjectHandle) const noexcept; const MaterializationStats& stats() const noexcept; DeviceArena& device_arena();`. `ObjectHandle` is `struct { std::size_t index = 0; }` (no `operator==`).
- `src/artifact/reader.h` — `RowSplitGeometry { rows, columns, padded_columns, group_size, groups_per_row, low_bytes_per_group, high_bytes_per_group, low_plane_bytes, high_plane_offset, high_plane_bytes, scale_plane_offset, scale_plane_bytes, encoded_bytes }`; `RowSplitGeometry row_split_geometry(NumericFormat, span<const uint64> shape)`; `BlockScaleGeometry { rows, columns, groups_per_row, k_tiles, code_plane_bytes, scale_plane_offset, scale_plane_bytes, weight_divisor_offset, encoded_bytes }`; `BlockScaleGeometry block_scale_geometry(NumericFormat, span)`.
- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h` — forward-declares `class DeviceArena;` (line 14); `ModelView<...>` has `DeviceArena* weights_arena = nullptr; Weight token_embedding; std::array<FullLayer, FullAttentionLayers> full_layers; std::array<GdnLayer, GdnLayers> gdn_layers; Tensor final_norm; Weight output_head; StartupFeatures features; optional optimized_proposal/mtp/dflash/vision`. 27B `RuntimeModelView = ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload, MtpAttentionPayload, DensePostMixerPayload, DFlashWeights<6>, 16, 48>` (bindings.h:169-172).
- `src/targets/qwen3_6_27b/impl/load/bindings.h` — `kTextLayers=64`, `kFullAttentionLayers=16`, `kGdnLayers=48`; `WeightPlan{object, format, weight_scale_divisor_bits, input_scale_divisor_bits}`; `MlpPlan{gate_up, down}`; `TextLayerPlan{input_norm, attention, gdn, is_full_attention, post_attention_norm, mlp}`; `BindingPlan{frontend, features, token_embedding, text_layers[64], final_norm, output_head, draft_head, draft_head_token_ids, mtp, vision_*}`; `LoadedModelData{backing, frontend, runtime}` ctor `(BindingPlan, artifact::MaterializedArtifact)`, deleted copy/move.
- `src/targets/qwen3_6_27b/impl/load/bindings.cpp` — anonymous-namespace helpers `is_full_layer`, `is_early_attention_input`, `is_bf16_attention_output`, `is_bf16_gdn_output`, `endpoint_format`, `ffn_placement` (46-49), `read_u32_le` (51-61), `require_positive_finite` (63-68). `bind_weight` (70-78), `bind_nvfp4_weight` (80-108), `materialized_weight` (110-144, host-aware via `device_data_or_null`/`host_data_or_null`; non-NVFP4 delegates to `artifact::materialized_weight`, NVFP4 branch builds `Weight` with `qdata=bytes`, `scales=bytes==nullptr?nullptr:bytes+scale_plane_offset`, `host=host_data_or_null`), `row_view` (146-160), `load_mlp` (173-178: `gate_up = materialized_weight(materialized, plan.gate_up, 34816, 5120)`, `down = materialized_weight(materialized, plan.down, 5120, 17408)`).
- `LoadedModelData` ctor (bindings.cpp:448-553): `backing(std::move(materialized))`; `runtime.weights_arena = &backing.device_arena()`; loop layer 0..63; full branch sets `target.post_mixer = load_mlp(source.mlp, backing)` (line 477); GDN branch same (line 498); `token_embedding` (460), `output_head` (506), optimized_proposal, mtp (post_mixer line 539), vision.
- Residency test `tests/targets/qwen3_6_27b/test_residency.cpp` (148 lines) — `verify_residency` currently asserts (lines 98-111) that FfnOffload MLP weights are host-only (`host != nullptr && payload == nullptr`); this is the surface-3 contract that surface 4 supersedes. Test constructs `ninfer::DeviceContext device(0)` then `materialize` then `LoadedModelData data(std::move(ffn_plan.bindings), std::move(materialized));`.

**Important: the residency test's `host != nullptr && payload == nullptr` assertions (lines 98-117) must be REPLACED in Task 1.** After surface 4, host-placed MLP weights carry BOTH a valid `host` (copy source, object-granularity `backing.host_data(handle)`) AND a non-null `payload`/`qdata`/`qhigh`/`scales` pointing into the staging arena.

---

## Task 1: Failing staging-arena and slot-map test (red)

**Files:**
- Modify: `tests/targets/qwen3_6_27b/test_residency.cpp`

This task rewrites the `LoadedModelData` section of `verify_residency` to assert the surface-4 contract. It will **fail to compile** because `ModelView` has no `staging_arena`/`staged_weights` members yet (Task 2).

- [x] **Step 1: Write the failing test**

Replace the include block header at `tests/targets/qwen3_6_27b/test_residency.cpp:9-14` with:

```cpp
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
```

Replace the entire block at `test_residency.cpp:94-117` (from `ninfer::DeviceContext device(0);` through the vocabulary-endpoint check `return 1;` before `return 0;`) with:

```cpp
    ninfer::DeviceContext device(0);
    auto materialized =
        ninfer::artifact::materialize(reader, ffn_plan.materialization, device, nullptr);
    std::array<std::pair<ninfer::artifact::ObjectHandle, ninfer::artifact::ObjectHandle>,
               kTextLayers>
        mlp_handles;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        mlp_handles[layer] = {ffn_plan.bindings.text_layers[layer].mlp.gate_up.object,
                              ffn_plan.bindings.text_layers[layer].mlp.down.object};
    }
    LoadedModelData data(std::move(ffn_plan.bindings), std::move(materialized));
    if (data.runtime.staging_arena == nullptr) {
        std::cerr << "FfnOffload did not allocate a staging arena\n";
        return 1;
    }
    if (data.runtime.staged_weights.size() != 2 * kTextLayers) {
        std::cerr << "slot map size mismatch: got " << data.runtime.staged_weights.size()
                  << " expected " << 2 * kTextLayers << '\n';
        return 1;
    }
    const auto* arena_begin = static_cast<const std::uint8_t*>(data.runtime.staging_arena->base());
    const std::uintptr_t arena_lo = reinterpret_cast<std::uintptr_t>(arena_begin);
    const std::uintptr_t arena_hi = arena_lo + data.runtime.staging_arena->capacity();
    const std::size_t unit_bytes  = data.runtime.staging_arena->capacity() / 2;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const auto& gate_up = data.runtime.staged_weights[2 * layer];
        const auto& down    = data.runtime.staged_weights[2 * layer + 1];
        if (gate_up.host_source != data.backing.host_data(mlp_handles[layer].first) ||
            down.host_source != data.backing.host_data(mlp_handles[layer].second)) {
            std::cerr << "a slot map entry does not source from the host store object\n";
            return 1;
        }
        const std::uintptr_t expected_gate =
            arena_lo + static_cast<std::uintptr_t>((layer % 2) * unit_bytes);
        const std::uintptr_t expected_down =
            expected_gate + ((gate_up.bytes + 255U) & ~std::uint64_t{255});
        if (reinterpret_cast<std::uintptr_t>(gate_up.slot) != expected_gate ||
            reinterpret_cast<std::uintptr_t>(down.slot) != expected_down) {
            std::cerr << "a slot address does not follow the double-buffer layer layout\n";
            return 1;
        }
    }
    for (const FullAttentionWeights& full : data.runtime.full_layers) {
        if (full.post_mixer.gate_up.host == nullptr || full.post_mixer.gate_up.payload == nullptr ||
            full.post_mixer.down.host == nullptr || full.post_mixer.down.payload == nullptr) {
            std::cerr << "a streamed MLP weight does not carry host and slot addresses\n";
            return 1;
        }
        const std::uintptr_t gu = reinterpret_cast<std::uintptr_t>(full.post_mixer.gate_up.payload);
        const std::uintptr_t dn = reinterpret_cast<std::uintptr_t>(full.post_mixer.down.payload);
        if (gu < arena_lo || gu >= arena_hi || dn < arena_lo || dn >= arena_hi) {
            std::cerr << "a streamed MLP weight does not point into the staging arena\n";
            return 1;
        }
    }
    for (const GdnWeights& gdn : data.runtime.gdn_layers) {
        if (gdn.post_mixer.gate_up.host == nullptr || gdn.post_mixer.gate_up.payload == nullptr ||
            gdn.post_mixer.down.host == nullptr || gdn.post_mixer.down.payload == nullptr) {
            std::cerr << "a streamed MLP weight does not carry host and slot addresses\n";
            return 1;
        }
        const std::uintptr_t gu = reinterpret_cast<std::uintptr_t>(gdn.post_mixer.gate_up.payload);
        const std::uintptr_t dn = reinterpret_cast<std::uintptr_t>(gdn.post_mixer.down.payload);
        if (gu < arena_lo || gu >= arena_hi || dn < arena_lo || dn >= arena_hi) {
            std::cerr << "a streamed MLP weight does not point into the staging arena\n";
            return 1;
        }
    }
    if (data.runtime.token_embedding.payload == nullptr ||
        data.runtime.token_embedding.host != nullptr || data.runtime.output_head.payload == nullptr ||
        data.runtime.output_head.host != nullptr) {
        std::cerr << "a resident vocabulary weight does not carry device-only addresses\n";
        return 1;
    }
```

- [x] **Step 2: Build the test target to verify the red state**

```bash
cmake --build build --target ninfer_qwen3_6_27b_residency_test
```

Expected: **compile failure** with `'class ninfer::targets::qwen3_6::ModelView<...>' has no member named 'staging_arena'` and `'staged_weights'` (the `struct StagedWeight` type also does not exist yet). Nothing else should fail — every other symbol in the block (`materialize`, `LoadedModelData`, `FullAttentionWeights`, `GdnWeights`, `data.backing.host_data`, `kTextLayers`) already exists.

- [x] **Step 3: Commit**

```bash
git add tests/targets/qwen3_6_27b/test_residency.cpp
git commit -m "test(targets): add 27B staging arena and slot-map scenarios"
```

---

## Task 2: Family ModelView staging surface (compiles, still red)

**Files:**
- Modify: `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h`

Add the slot-map value type and the two `ModelView` members. This is family-owned shared surface: 35B-A3B's `ModelView` instantiation simply keeps them default (null pointer, empty vector), which is correct — 35B only ever binds `AllResident`.

- [x] **Step 1: Add the `StagedWeight` struct and the members**

In `model_view.h`, change the include block (`:8-10`) to:

```cpp
#include <array>
#include <cstddef>
#include <optional>
#include <vector>
```

Immediately after the `DFlashWeights` template (`:73-79`), before `template <class FullProjectionPayload...>` (`:81`), insert:

```cpp
// One offloaded weight's fixed staging slot. host_source is the object-granularity
// host-store base (MaterializedArtifact::host_data(handle)); slot is the fixed device
// address the weight's device planes are re-pointed at.
struct StagedWeight {
    const void* host_source = nullptr;
    void* slot              = nullptr;
    std::size_t bytes       = 0;
};
```

In `ModelView` (`:84-101`), immediately after `DeviceArena* weights_arena = nullptr;` (`:90`), insert:

```cpp
    DeviceArena* staging_arena = nullptr;
    std::vector<StagedWeight> staged_weights;
```

- [x] **Step 2: Build and confirm the test now compiles but still fails**

```bash
cmake --build build --target ninfer_qwen3_6_27b_residency_test
```

Expected: **compiles**. Then run:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_residency_test
```

Expected: **fails** with `FfnOffload did not allocate a staging arena` (exit 1, not 77). This is the logical red: the members exist, but the 27B constructor does not populate them yet.

- [x] **Step 3: Commit**

```bash
git add src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h
git commit -m "feat(targets): add staging arena and slot map to the model view"
```

---

## Task 3: 27B staging arena allocation and slot binding (green)

**Files:**
- Modify: `src/targets/qwen3_6_27b/impl/load/bindings.h`
- Modify: `src/targets/qwen3_6_27b/impl/load/bindings.cpp`

Allocate a staging `DeviceArena` sized `2 × unit_bytes` when the plan has host-placed weights, re-point each offloaded MLP weight's device planes at its fixed slot, and publish the slot map into `runtime.staged_weights`.

### Layout constants (all 64 text layers are uniform)

- gate_up: Q4G64_F16S (groupwise) or NVFP4 (nvfp4 profile), shape `{34816, 5120}`.
- down: Q5G64_F16S (groupwise) or NVFP4, shape `{5120, 17408}`.
- `unit_bytes = align_up_256(encoded_bytes(gate_up)) + align_up_256(encoded_bytes(down))`.
- arena capacity = `2 * unit_bytes`. Layer `g` uses buffer `g % 2`; gate_up at `buffer`, down at `buffer + align_up_256(gate_up.encoded)`.

- [x] **Step 1: Add the LoadedModelData staging-arena member**

In `bindings.h`, change the include block (`:1-15`) to add the two new includes in the existing alphabetical std/core block:

```cpp
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
```

and add `#include "core/arena.h"` after `#include "core/tensor.h"` (bindings.h:17).

In `class LoadedModelData` (`:177-189`), add a private section after the three public members (`backing`, `frontend`, `runtime`) so the class becomes:

```cpp
class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;

private:
    std::optional<DeviceArena> staging_arena_;
};
```

(`std::optional<DeviceArena>` needs the complete `DeviceArena` type from `core/arena.h`; the optional stays empty for `AllResident`, so no allocation happens there.)

- [x] **Step 2: Add the geometry helpers to bindings.cpp**

In the anonymous namespace of `bindings.cpp` (after `require_positive_finite`, `:63-68`), insert:

```cpp
std::uint64_t align_up_256(std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - 255U) {
        throw std::overflow_error("staging slot size overflows u64");
    }
    return (value + 255U) & ~std::uint64_t{255};
}

std::uint64_t encoded_bytes(const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    if (plan.format == NumericFormat::NVFP4) {
        return artifact::block_scale_geometry(NumericFormat::NVFP4, shape).encoded_bytes;
    }
    return artifact::row_split_geometry(plan.format, shape).encoded_bytes;
}

// Re-point a host-placed weight's device planes at a fixed slot address. host is
// preserved: it remains the object-granularity copy source for staging.
void point_at_slot(Weight& weight, const void* slot, NumericFormat format, std::int32_t rows,
                   std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const auto* base                        = static_cast<const std::byte*>(slot);
    weight.payload                          = base;
    weight.qdata                            = base;
    if (format == NumericFormat::NVFP4) {
        const artifact::BlockScaleGeometry geometry =
            artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
        weight.qhigh  = nullptr;
        weight.scales = base + geometry.scale_plane_offset;
    } else {
        const artifact::RowSplitGeometry geometry = artifact::row_split_geometry(format, shape);
        weight.qhigh  = geometry.high_plane_bytes == 0 ? nullptr : base + geometry.high_plane_offset;
        weight.scales = base + geometry.scale_plane_offset;
    }
}
```

Note: `<numeric_limits>` is already available via `<limits>` (bindings.cpp:15); `<array>` and `<cstddef>` are already included.

- [x] **Step 3: Allocate the arena and bind slots in the LoadedModelData ctor**

In the `LoadedModelData` ctor (`bindings.cpp:448-553`), immediately after `runtime.features      = plan.features;` (`:452`), insert:

```cpp
    const bool offload =
        backing.host_data_or_null(plan.text_layers[0].mlp.gate_up.object) != nullptr;
    std::uint64_t staging_unit_bytes = 0;
    if (offload) {
        const std::uint64_t gate_bytes =
            align_up_256(encoded_bytes(plan.text_layers[0].mlp.gate_up, 34816, 5120));
        const std::uint64_t down_bytes =
            align_up_256(encoded_bytes(plan.text_layers[0].mlp.down, 5120, 17408));
        staging_unit_bytes = gate_bytes + down_bytes;
        staging_arena_.emplace(2 * staging_unit_bytes);
        runtime.staging_arena = &*staging_arena_;
    }
```

Then, before the layer loop (`for (std::size_t layer = 0; layer < kTextLayers; ++layer)` at `:463`), insert a staging lambda:

```cpp
    const auto stage_mlp = [&](DensePostMixerPayload& post_mixer, const MlpPlan& mlp,
                               std::size_t layer) {
        if (!offload) { return; }
        const auto* arena_base = static_cast<const std::byte*>(runtime.staging_arena->base());
        const auto* buffer =
            arena_base + static_cast<std::size_t>(layer % 2) * staging_unit_bytes;
        const std::uint64_t gate_bytes = align_up_256(encoded_bytes(mlp.gate_up, 34816, 5120));
        point_at_slot(post_mixer.gate_up, buffer, mlp.gate_up.format, 34816, 5120);
        point_at_slot(post_mixer.down, buffer + gate_bytes, mlp.down.format, 5120, 17408);
        runtime.staged_weights.push_back(
            {backing.host_data(mlp.gate_up.object),
             const_cast<void*>(post_mixer.gate_up.payload),
             static_cast<std::size_t>(post_mixer.gate_up.payload_bytes)});
        runtime.staged_weights.push_back(
            {backing.host_data(mlp.down.object), const_cast<void*>(post_mixer.down.payload),
             static_cast<std::size_t>(post_mixer.down.payload_bytes)});
    };
```

Note: `StagedWeight::slot` is `void*` (the surface-5 memcpy destination), but `Weight::payload` is `const void*`; the `const_cast<void*>` at this population site is the single, deliberate const boundary (confirmed by the Task-2 code-quality review).

Then update BOTH post_mixer assignment sites to call the lambda. In the full-attention branch, change `:477`:

```cpp
            target.post_mixer = load_mlp(source.mlp, backing);
```

to:

```cpp
            target.post_mixer = load_mlp(source.mlp, backing);
            stage_mlp(target.post_mixer, source.mlp, layer);
```

In the GDN branch, change `:498` the same way:

```cpp
            target.post_mixer = load_mlp(source.mlp, backing);
            stage_mlp(target.post_mixer, source.mlp, layer);
```

Do NOT touch the MTP `post_mixer` (`:539`) — MTP MLP weights are device-placed under `FfnOffload` (`bind_mtp` uses `Device` when `features.mtp()`), so `offload` routing must not stage them; the `stage_mlp` early-return keeps `host == nullptr` weights untouched anyway, but leave that call site unchanged for clarity.

- [x] **Step 4: Build and run the residency test**

```bash
cmake --build build --target ninfer_qwen3_6_27b_residency_test
NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_residency_test
```

Expected: **PASS** (exit 0). The materialize + `LoadedModelData` step fits in 16GB VRAM (FfnOffload device ~7GB + ~306MB staging arena).

- [x] **Step 5: Full tree build + full filtered gate**

```bash
cmake --build build
NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test|ninfer_qwen3_6_27b_residency_test'
```

Expected: all listed tests PASS (residency + artifact reader/materialization + arena + tensor). The pre-existing GPU failures (`ninfer_gdn_gating_proj_test` cudaErrorCooperativeLaunchTooLarge, `ninfer_swa_test`, `ninfer_qwen3_6_frontend_test`) are hardware/environment issues and are NOT in this filter.

- [x] **Step 6: Commit**

```bash
git add src/targets/qwen3_6_27b/impl/load/bindings.h src/targets/qwen3_6_27b/impl/load/bindings.cpp
git commit -m "feat(targets): bind 27B offloaded weights to fixed staging slots"
```

---

## Task 4: Full verification and design-doc status marker

**Files:**
- Modify: `docs/maintainer/weight-offload.md`
- Modify: `docs/superpowers/plans/2026-08-15-qwen3_6-27b-staging-arena.md`

- [x] **Step 1: Full verification**

```bash
cmake --build build
NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test|ninfer_qwen3_6_27b_residency_test'
git diff --check
```

Expected: full build clean; listed tests PASS; `git diff --check` silent.

- [x] **Step 2: Update the design doc status**

In `docs/maintainer/weight-offload.md`, section 5 Status paragraph (currently `:123-131`), replace the whole paragraph with:

```markdown
Status: surfaces 1-4 are implemented — binder host placement
(`TensorPlacement::Host`, host spans in `MaterializationPlan`), materializer host store
(`MaterializedArtifact::host_data(handle)` + host-bytes stats), `bind_tensor` host dispatch,
`Tensor`/`Weight` host addresses with view propagation, the 27B `ResidencyProfile`
(`AllResident` default, `FfnOffload` binds the per-layer FFN/SwiGLU gate/up + down matrices
host-only), and the fixed staging arena with offloaded-tensor -> slot binding: a
`2 x largest-streaming-unit` device `DeviceArena` whose slot addresses never change, the
host-placed `Weight` device planes (`payload`/`qdata`/`qhigh`/`scales`, and by extension the
NVFP4 TMA maps) re-pointed at those slots during `LoadedModelData` construction before
graph capture, and a `ModelView::staged_weights` slot map (`host_source`, `slot`, `bytes`)
for the graph-capture interleave. The host store is plain host memory; pinning
(`cudaHostRegister`) is deferred to the staging-copy phase (surface 5). Surface 4 does not
yet run decode: the graph interleave of H2D memcpy nodes is surface 5.
```

- [x] **Step 3: Mark this plan's checkboxes complete**

Change every `- [x]` in `docs/superpowers/plans/2026-08-15-qwen3_6-27b-staging-arena.md` to `- [x]`. Do not alter any prose.

- [x] **Step 4: Verify and commit**

```bash
git diff --check
git add docs/maintainer/weight-offload.md docs/superpowers/plans/2026-08-15-qwen3_6-27b-staging-arena.md
git commit -m "docs(offload): mark the 27B staging arena and slot binding as implemented"
```

---

## Follow-on requirements this plan records (for the surface-5 plan)

1. **Slot map consumption**: surface 5's `prepare_graphs()` interleave reads `ModelView::staged_weights` (and `ModelView::staging_arena`) to emit H2D `cudaMemcpyAsync` graph nodes from `host_source` → `slot` before each offload group's kernels, alternating buffers `g % 2`. Copy for group `g` is address-independent of compute for group `g-1` (different buffer), enabling overlap inside one graph.
2. **Object-granularity invariant**: `staged_weights[i].host_source` is always `MaterializedArtifact::host_data(handle)` (the object base), never a derived view's `host`; this test pins that invariant. `staged_weights[i].slot` is the const_cast'ed device address from the weight's re-pointed `payload`.
3. **Surface 4 does not run decode yet.** `FfnOffload` is not reachable via `Package::plan_load` (default `AllResident`), and nothing consumes `staged_weights` until surface 5. The residency test verifies addresses only.
4. **35B-A3B** is untouched: its `ModelView` instantiation keeps the new members default (null/empty).
