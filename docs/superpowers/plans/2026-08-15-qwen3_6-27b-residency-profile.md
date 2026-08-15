# Qwen3.6-27B Residency Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface 3 of `docs/maintainer/weight-offload.md`: give the 27B package a per-tensor device-vs-host residency decision so the FFN/SwiGLU matrices of every text layer bind host-only while attention/GDN projections, norms, vocabulary endpoints, and the MTP/draft/vision extras stay device-resident. `AllResident` stays the default, so engine behavior is unchanged until surfaces 4-6 land.

**Architecture:** Add `ResidencyProfile` (detail, alongside `WeightsProfile`), thread it through `bind_artifact`, and route the text-layer MLP gate_up/down binds through a placement-aware path. Surfaces 1-2 already make host-placed weights flow to `MaterializedArtifact::host_data` + `Tensor/Weight::host`; the NVFP4 wrapper's local `materialized_weight` must switch to the `_or_null` accessors so host-placed NVFP4 objects do not throw.

**Tech Stack:** C++17, CUDA 13.3 (sm_120a), CMake/Ninja, `/usr/bin/ctest`.

**Not in this plan (follow-on):** surface 4 staging arena + slot binding, surface 5 graph-capture interleave, surface 6 engine option + capacity planning + CLI. `Package::plan_load` is intentionally unchanged here.

---

### Task 1: Failing residency test (red)

**Files:**
- Create: `tests/targets/qwen3_6_27b/test_residency.cpp`
- Modify: `tests/CMakeLists.txt` (after the `ninfer_qwen3_6_27b_load_plan_test` block, ~line 116)

- [ ] **Step 1: Create the test file**

`tests/targets/qwen3_6_27b/test_residency.cpp`:

```cpp
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

namespace {

using namespace ninfer::targets::qwen3_6_27b::detail;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    const std::filesystem::path candidate =
        std::filesystem::path(NINFER_SOURCE_DIR) / "out" / filename;
    return std::filesystem::is_regular_file(candidate) ? candidate : std::filesystem::path{};
}

bool is_host_placed(const ninfer::artifact::MaterializationPlan& plan,
                    ninfer::artifact::ObjectHandle handle) {
    for (const auto& entry : plan.host_tensor_objects) {
        if (entry.object == handle) { return true; }
    }
    return false;
}

int verify_residency(const std::filesystem::path& path, WeightsProfile profile) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder all_binder(reader);
    const ArtifactLoadPlan all_plan =
        bind_artifact(all_binder, profile, {}, ResidencyProfile::AllResident);
    if (!all_plan.materialization.host_tensor_objects.empty()) {
        std::cerr << "AllResident plan contains host tensor objects\n";
        return 1;
    }

    ninfer::artifact::Binder ffn_binder(reader);
    ArtifactLoadPlan ffn_plan =
        bind_artifact(ffn_binder, profile, {}, ResidencyProfile::FfnOffload);
    if (ffn_plan.materialization.host_tensor_objects.size() != 2 * kTextLayers) {
        std::cerr << "FfnOffload host tensor count mismatch: got "
                  << ffn_plan.materialization.host_tensor_objects.size() << " expected "
                  << 2 * kTextLayers << '\n';
        return 1;
    }
    if (ffn_plan.materialization.host_capacity_bytes == 0 ||
        ffn_plan.materialization.device_capacity_bytes >=
            all_plan.materialization.device_capacity_bytes) {
        std::cerr << "FfnOffload did not move weight bytes off the device arena\n";
        return 1;
    }
    for (const TextLayerPlan& layer : ffn_plan.bindings.text_layers) {
        if (!is_host_placed(ffn_plan.materialization, layer.mlp.gate_up.object) ||
            !is_host_placed(ffn_plan.materialization, layer.mlp.down.object)) {
            std::cerr << "an MLP weight was not host-placed under FfnOffload\n";
            return 1;
        }
    }
    if (is_host_placed(ffn_plan.materialization, ffn_plan.bindings.token_embedding.object) ||
        is_host_placed(ffn_plan.materialization, ffn_plan.bindings.output_head.object)) {
        std::cerr << "vocabulary endpoints were host-placed under FfnOffload\n";
        return 1;
    }
    const TextLayerPlan& layer = ffn_plan.bindings.text_layers.front();
    if (layer.is_full_attention) {
        const auto* split = std::get_if<SplitAttentionProjectionPlan>(&layer.attention.projection);
        if (split != nullptr &&
            (is_host_placed(ffn_plan.materialization, split->query_key.object) ||
             is_host_placed(ffn_plan.materialization, split->gate_value.object))) {
            std::cerr << "attention projection was host-placed under FfnOffload\n";
            return 1;
        }
        const auto* fused = std::get_if<FusedAttentionProjectionPlan>(&layer.attention.projection);
        if (fused != nullptr &&
            is_host_placed(ffn_plan.materialization, fused->query_key_gate_value.object)) {
            std::cerr << "attention projection was host-placed under FfnOffload\n";
            return 1;
        }
    } else if (is_host_placed(ffn_plan.materialization, layer.gdn.output.object)) {
        std::cerr << "GDN projection was host-placed under FfnOffload\n";
        return 1;
    }

    ninfer::DeviceContext device(0);
    auto materialized =
        ninfer::artifact::materialize(reader, ffn_plan.materialization, device, nullptr);
    LoadedModelData data(std::move(ffn_plan.bindings), std::move(materialized));
    for (const FullAttentionWeights& full : data.runtime.full_layers) {
        if (full.post_mixer.gate_up.host == nullptr || full.post_mixer.gate_up.payload != nullptr ||
            full.post_mixer.down.host == nullptr || full.post_mixer.down.payload != nullptr) {
            std::cerr << "a streamed MLP weight does not carry host-only addresses\n";
            return 1;
        }
    }
    for (const GdnWeights& gdn : data.runtime.gdn_layers) {
        if (gdn.post_mixer.gate_up.host == nullptr || gdn.post_mixer.gate_up.payload != nullptr ||
            gdn.post_mixer.down.host == nullptr || gdn.post_mixer.down.payload != nullptr) {
            std::cerr << "a streamed MLP weight does not carry host-only addresses\n";
            return 1;
        }
    }
    if (data.runtime.token_embedding.data == nullptr ||
        data.runtime.token_embedding.host != nullptr || data.runtime.output_head.data == nullptr ||
        data.runtime.output_head.host != nullptr) {
        std::cerr << "a resident vocabulary weight does not carry device-only addresses\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const std::filesystem::path groupwise =
        artifact_path("NINFER_QWEN3_8_27B_WEIGHTS", "qwen3_8_27b.ninfer");
    const std::filesystem::path legacy =
        artifact_path("NINFER_QWEN3_6_27B_WEIGHTS", "qwen3_6_27b.ninfer");
    const std::filesystem::path nvfp4 =
        artifact_path("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS", "qwen3_6_27b_nvfp4.ninfer");
    if (!std::filesystem::is_regular_file(groupwise) &&
        !std::filesystem::is_regular_file(legacy) && !std::filesystem::is_regular_file(nvfp4)) {
        std::cerr << "skip: a real 27B-family artifact is required\n";
        return 77;
    }
    if (std::filesystem::is_regular_file(groupwise) &&
        verify_residency(groupwise, WeightsProfile::GroupwiseIntW8Endpoints) != 0) {
        return 1;
    }
    if (std::filesystem::is_regular_file(legacy) &&
        verify_residency(legacy, WeightsProfile::GroupwiseInt) != 0) {
        return 1;
    }
    if (std::filesystem::is_regular_file(nvfp4) &&
        verify_residency(nvfp4, WeightsProfile::Nvfp4) != 0) {
        return 1;
    }
    return 0;
}
```

Notes:
- The test binds with `StartupFeatures{}` (all features off), so MTP/draft/vision are `ValidateOnly` and only text layers + vocabulary endpoints are materialized; the `verify_residency` profile argument is passed straight to `bind_artifact`.
- `std::get_if` needs `<variant>`; `bindings.h` includes `<variant>` transitively. The file relies on `NINFER_SOURCE_DIR` (needs `NEEDS_SOURCE_DIR`) and CUDA (link `CUDA::cudart`), mirroring `ninfer_qwen3_6_27b_load_plan_test`.
- `verify_residency` is profile-agnostic: it reads `plan.bindings.text_layers`/`token_embedding`/`output_head` and `data.runtime.*` for whichever artifact is present. For NVFP4, MLP is bound via `bind_nvfp4_weight`, so the assertion count `2 * kTextLayers` still holds and `data.runtime.full_layers[].post_mixer.*` are NVFP4 `Weight`s (payload==nullptr, host!=nullptr under FfnOffload).

- [ ] **Step 2: Register the test**

Insert immediately after the `ninfer_qwen3_6_27b_load_plan_test` block in `tests/CMakeLists.txt`:

```cmake
ninfer_add_test(ninfer_qwen3_6_27b_residency_test
  SOURCES targets/qwen3_6_27b/test_residency.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_27b_residency_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ninfer_qwen3_6_27b_residency_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_27b_residency_test PROPERTIES SKIP_RETURN_CODE 77)
```

- [ ] **Step 3: Run the build to verify it fails (red)**

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --target ninfer_qwen3_6_27b_residency_test
```

Expected: FAIL. The target does not compile: `ResidencyProfile` is not a member of the 27B detail namespace, and `bind_artifact` has no 4th parameter.

- [ ] **Step 4: Commit**

```bash
git add tests/targets/qwen3_6_27b/test_residency.cpp tests/CMakeLists.txt
git commit -m "test(targets): add 27B residency profile scenarios"
```

---

### Task 2: ResidencyProfile type + bind_artifact threading

**Files:**
- Modify: `src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h`
- Modify: `src/targets/qwen3_6_27b/impl/load/bindings.h`
- Modify: `src/targets/qwen3_6_27b/impl/load/bindings.cpp`
- Modify: `src/targets/qwen3_6_27b/impl/package.cpp`

- [ ] **Step 1: Declare the enum and Package alias**

In `src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h`, inside `namespace detail` immediately after the `WeightsProfile` enum (which ends with `};` at line 30):

```cpp
enum class ResidencyProfile : std::uint8_t {
    AllResident,
    FfnOffload,
};
```

In the `struct Package` public alias block (near `using WeightsProfile = detail::WeightsProfile;`):

```cpp
    using ResidencyProfile = detail::ResidencyProfile;
```

- [ ] **Step 2: Thread through bind_artifact declaration**

In `src/targets/qwen3_6_27b/impl/load/bindings.h`, change the `bind_artifact` declaration (line 120-121) to:

```cpp
ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features,
                               ResidencyProfile residency = ResidencyProfile::AllResident);
```

`ResidencyProfile` resolves via the `detail` namespace; `bindings.h` already includes `package.h`.

- [ ] **Step 3: Update the definition and forward residency into the layer binders**

In `src/targets/qwen3_6_27b/impl/load/bindings.cpp`:

(a) `bind_groupwise_text_layers(artifact::Binder& binder, BindingPlan& out)` (line 199) gains a trailing `ResidencyProfile residency` parameter.

(b) `bind_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out)` (line 250) gains a trailing `ResidencyProfile residency` parameter.

(c) `bind_artifact` (line 339) signature becomes:

```cpp
ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features,
                               ResidencyProfile residency) {
```

and its two dispatch calls become `bind_groupwise_text_layers(binder, out, residency);` and `bind_nvfp4_text_layers(binder, out, residency);`.

- [ ] **Step 4: Update package.cpp plan_load call**

In `src/targets/qwen3_6_27b/impl/package.cpp:99-104`, the `plan_load` body stays exactly as-is: `bind_artifact(binder, weights_profile, qwen3_6::startup_features(options))` now uses the default `AllResident`. No change needed there; verify it still compiles.

- [ ] **Step 5: Build and run the residency test (still logically red on placement)**

```bash
cmake --build build --target ninfer_qwen3_6_27b_residency_test
NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_residency_test
```

Expected: the target COMPILES, and the test FAILS with `FfnOffload host tensor count mismatch: got 0 expected 128` (bindings still bind everything device).

- [ ] **Step 6: Commit**

```bash
git add src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h src/targets/qwen3_6_27b/impl/load/bindings.h src/targets/qwen3_6_27b/impl/load/bindings.cpp
git commit -m "feat(targets): thread the 27B residency profile through binding"
```

---

### Task 3: Placement-aware MLP binds (green)

**Files:**
- Modify: `src/targets/qwen3_6_27b/impl/load/bindings.cpp`

- [ ] **Step 1: Add a placement helper in the anonymous namespace**

In `src/targets/qwen3_6_27b/impl/load/bindings.cpp`, inside `namespace {` after `endpoint_format` (line 44):

```cpp
artifact::TensorPlacement ffn_placement(ResidencyProfile residency) noexcept {
    return residency == ResidencyProfile::FfnOffload ? artifact::TensorPlacement::Host
                                                     : artifact::TensorPlacement::Device;
}
```

- [ ] **Step 2: Groupwise MLP binds honor residency**

In `bind_groupwise_text_layers`, change the MLP binds (lines 243-246) to:

```cpp
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S, {34816, 5120},
                        ffn_placement(residency));
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {5120, 17408},
                        ffn_placement(residency));
```

`bind_weight` needs a trailing `artifact::TensorPlacement placement` parameter. Change its declaration/definition (lines 65-72) to:

```cpp
WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape,
                       artifact::TensorPlacement placement) {
    if (format == NumericFormat::NVFP4) {
        throw std::logic_error("NVFP4 weight requires a paired input divisor");
    }
    return WeightPlan{.object = artifact::bind_tensor(binder, name, format, shape, placement),
                      .format = format};
}
```

All other `bind_weight` call sites in the file (attention/GDN projections, output, endpoints) must now pass `artifact::TensorPlacement::Device` explicitly. These are: lines 208-211, 217-218, 231-234, 238-239, 260-261, 274-275, 300-301, 347-348, 362. (Do not change bindings for `bind_device_tensor`-style small tensors; those stay device.)

- [ ] **Step 3: NVFP4 MLP binds honor residency**

`bind_nvfp4_weight` (lines 74-97) gains a trailing `artifact::TensorPlacement placement` parameter and routes it:

```cpp
WeightPlan bind_nvfp4_weight(artifact::Binder& binder, std::string_view name, std::int32_t rows,
                             std::int32_t columns, std::string_view input_divisor_name,
                             artifact::TensorPlacement placement) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::ObjectHandle parent      = binder.require_tensor(
        name, NumericFormat::NVFP4, artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    if (placement == artifact::TensorPlacement::Host) {
        binder.materialize_tensor_on_host(parent);
    } else {
        binder.materialize_on_device(parent);
    }
    // ... remainder unchanged (input_divisor bind, geometry, read_u32_le, require_positive_finite)
```

In `bind_nvfp4_text_layers`, the MLP binds (lines 310-314) pass `ffn_placement(residency)` as the new trailing argument. All other `bind_nvfp4_weight` call sites (attention/GDN projections, output) pass `artifact::TensorPlacement::Device`.

- [ ] **Step 4: Make the NVFP4 wrapper host-safe**

The local `materialized_weight` (lines 99-131) must not call the throwing `materialized.device_data(plan.object)` when the object is host-placed. Replace the base-pointer line (line 109) and populate `out.host`:

```cpp
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data_or_null(plan.object));
    ...
    Weight out{};
    out.payload              = bytes;
    out.payload_bytes        = geometry.encoded_bytes;
    out.qtype                = QType::NVFP4;
    out.group_size           = 16;
    out.ndim                 = 2;
    out.qdata                = bytes;
    out.host                 = materialized.host_data_or_null(plan.object);
    out.scales               = bytes == nullptr ? nullptr : bytes + geometry.scale_plane_offset;
    // ... remainder unchanged
```

The `else` branch already delegates to `artifact::materialized_weight` which is host-aware.

- [ ] **Step 5: Build and run the residency test (green)**

```bash
cmake --build build --target ninfer_qwen3_6_27b_residency_test
NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_residency_test
```

Expected: PASS (exit 0). Also run the full filtered gate:

```bash
/usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test|ninfer_qwen3_6_27b_load_plan_test|ninfer_qwen3_6_27b_residency_test'
```

- [ ] **Step 6: Commit**

```bash
git add src/targets/qwen3_6_27b/impl/load/bindings.cpp
git commit -m "feat(targets): bind 27B FFN matrices host-resident under FfnOffload"
```

---

### Task 4: Full verification + design-doc status marker

**Files:**
- Modify: `docs/maintainer/weight-offload.md`

- [ ] **Step 1: Full build and verification**

```bash
cmake --build build
git diff --check
```

Expected: full tree builds (all targets), `git diff --check` silent.

- [ ] **Step 2: Extend the section-5 status paragraph**

In `docs/maintainer/weight-offload.md`, extend the Status paragraph (currently lines 123-127) to add surface 3:

```markdown
Status: surfaces 1-3 are implemented — binder host placement
(`TensorPlacement::Host`, host spans in `MaterializationPlan`), materializer host store
(`MaterializedArtifact::host_data(handle)` + host-bytes stats), `bind_tensor` host dispatch,
Tensor/Weight host addresses with view propagation, and the 27B `ResidencyProfile` (`AllResident`
default, `FfnOffload` binds the per-layer FFN/SwiGLU gate/up + down matrices host-only while
attention/GDN projections, norms, vocabulary endpoints, and the MTP/draft/vision extras stay
device-resident). The host store is plain host memory; pinning (`cudaHostRegister`) is deferred
to the staging-arena phase (surface 4). Surface 3 does not yet run decode: staged copies and
graph interleave are surfaces 4-5.
```

- [ ] **Step 3: Mark all plan checkboxes `- [x]`**

In `docs/superpowers/plans/2026-08-15-qwen3_6-27b-residency-profile.md`, change every `- [ ]` to `- [x]`.

- [ ] **Step 4: Commit**

```bash
git add docs/maintainer/weight-offload.md docs/superpowers/plans/2026-08-15-qwen3_6-27b-residency-profile.md
git commit -m "docs(offload): mark the 27B residency profile as implemented"
```

