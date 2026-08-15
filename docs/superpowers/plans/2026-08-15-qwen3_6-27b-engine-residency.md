# 27B Engine Residency Option (Surface 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make the FfnOffload residency profile selectable through the public `ninfer::Engine` so a 27B dense identity can be loaded and decoded on a 16 GB-VRAM `sm_120a` GPU (the offload deliverable's user-facing surface).

**Architecture:** `EngineOptions` gains a public `WeightResidency` field (default `AllResident`). The 27B `Package::plan_load` maps that field onto its existing `detail::ResidencyProfile` for `bind_artifact`; the 35B-A3B package rejects the offload value. `LoadSummary` reports the pinned host store and `MemorySummary` reports the staging arena plus host-store bytes. CLI gains `--weight-residency all|ffn`. No `kv_capacity.cpp` change is needed: the second `resolve_kv_capacity` call measures `current_free_device_bytes()` after `construct_loaded_model` has already allocated the staging arena, and the plan's `device_capacity_bytes` already excludes host-placed weights, so the existing preflight and resolution are residency-correct by construction.

**Tech Stack:** C++20, CUDA 13.3, CMake/Ninja, `sm_120a`.

---

## Key existing facts (verified on master `3f124ef`)

- `include/ninfer/types.h`: `KvCacheStorage` (lines 21-24) and `KvCapacityMode` (lines 26-29) enums; `EngineOptions` (lines 70-84, field `KvCacheStorage kv_cache` at line 79); `MemorySummary` (lines 373-396, `ArenaMemorySummary weights/sequence/workspace/request_transient`); `LoadSummary` (lines 414-425, `std::uint64_t peak_staging_bytes` at line 421).
- `include/ninfer/engine.h` (99 lines): `Engine(options)`, `options()`, `load_summary()`, `memory_summary()`.
- `src/targets/registry.cpp`: `validate_options` (lines 21-56, `std::invalid_argument` messages, kv_capacity mode switch with `default: throw`); `construct_registered` (lines 82-132) fills `LoadSummary summary` at lines 118-128 from `stats` (`artifact::MaterializationStats stats` at line 101); `Target::plan_load(binder, options, weights_profile)` at line 91.
- `src/targets/qwen3_6_27b/impl/package.cpp`: `Package::plan_load` (lines 99-104) calls `detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(options))` — **omits the 4th `residency` arg (defaults `AllResident`)**. Package is in namespace `ninfer::targets::qwen3_6_27b`, so `ninfer::WeightResidency` resolves unqualified; `ResidencyProfile` resolves to `Package::ResidencyProfile` inside the static member.
- `src/targets/qwen3_6_35b_a3b/impl/package.cpp`: `Package::plan_load` (lines 75-79) calls `detail::bind_artifact(binder, qwen3_6::startup_features(options))`. No `ResidencyProfile` exists under the 35B package.
- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h`: `ModelView` members `DeviceArena* staging_arena = nullptr;` and `std::vector<StagedWeight> staged_weights;` (lines 101-102).
- `src/targets/qwen3_6/impl/runtime/program_impl.h`: `ProgramImplCore::memory_summary() const noexcept` (lines 2163-2179) fills `out.weights` from `*model.weights_arena`. `model` is `const LoadedModelData&` = `Variant::ModelView`.
- `src/targets/qwen3_6_27b/impl/load/bindings.cpp`: `LoadedModelData` ctor sets `runtime.staging_arena = &*staging_arena_;` (line 501). `backing.stats().host_capacity_bytes` is the pinned store capacity; `stats().host_bytes` is the sum of host-placed object payloads.
- `apps/cli/options.h` (`struct Options`, lines 13-42), `apps/cli/options.cpp` (usage_text lines 74-92; parse loop lines ~110-192; `parse_kv_cache(std::string_view)` at lines 55-59), `apps/cli/main.cpp` (`print_load_summary` lines 141-152; memory block in `print_generation_summary` lines 180-200; `engine_options` build lines 259-269).
- `tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp` — the real-artifact 27B engine test pattern (Engine + `prepare_tokens` + `generate`, skip 77). Registered as `ninfer_qwen3_6_27b_prefix_real_test` (tests/CMakeLists.txt lines 106-109).
- Staging arena sizes: groupwise profiles (GroupwiseInt, GroupwiseIntW8Endpoints) = **306,380,800 B**; NVFP4 = **300,811,264 B**.
- Bench call sites use the 3-arg `plan_load` — unchanged since residency comes from `options`.
- Pre-existing GPU test failures on this hardware (not regressions): `ninfer_qwen3_6_frontend_test`, `ninfer_gdn_gating_proj_test`, `ninfer_swa_test`.

---

### Task 1: failing engine offload test (RED)

**Files:**
- Create: `tests/targets/qwen3_6_27b/test_engine_offload_real.cpp`
- Modify: `tests/CMakeLists.txt`

- [x] **Step 1: Write the failing test**

Create `tests/targets/qwen3_6_27b/test_engine_offload_real.cpp`:

```cpp
#include "ninfer/engine.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

ninfer::EngineOptions offload_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path    = artifact;
    options.max_context      = 4096;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk    = 1024;
    options.weight_residency = ninfer::WeightResidency::FfnOffload;
    return options;
}

int verify_offloaded_product(const ninfer::Engine& engine, std::size_t expected_staging_bytes) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.host_bytes == 0 || load.host_capacity_bytes == 0) {
        std::cerr << "offload load summary does not report the pinned host store\n";
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.staging.capacity_bytes != expected_staging_bytes) {
        std::cerr << "staging arena capacity is " << memory.staging.capacity_bytes
                  << ", expected " << expected_staging_bytes << '\n';
        return 1;
    }
    if (memory.host_store_bytes == 0 || memory.weights.capacity_bytes == 0 ||
        memory.weights.used_bytes == 0 || memory.kv_capacity == 0) {
        std::cerr << "offload Engine construction has incomplete memory backing\n";
        return 1;
    }
    return 0;
}

int exercise_offloaded_generation(ninfer::Engine& engine) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 4;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    const std::vector<ninfer::TokenId> prompt{198, 198};
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), options);
    if (result.generated_token_ids.size() != 4 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "offloaded decode did not complete through the staged graphs\n";
        return 1;
    }
    return 0;
}

int exercise_artifact(const char* artifact, std::size_t expected_staging_bytes) {
    ninfer::Engine engine(offload_options(artifact));
    if (const int result = verify_offloaded_product(engine, expected_staging_bytes);
        result != 0) {
        return result;
    }
    return exercise_offloaded_generation(engine);
}

} // namespace

int main() {
    const char* groupwise = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    const char* legacy    = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* nvfp4     = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    constexpr std::size_t kGroupwiseStagingBytes = 306380800;
    constexpr std::size_t kNvfp4StagingBytes     = 300811264;
    if ((groupwise == nullptr || *groupwise == '\0') && (legacy == nullptr || *legacy == '\0') &&
        (nvfp4 == nullptr || *nvfp4 == '\0')) {
        std::cout << "skip: no 27B weight artifact is configured\n";
        return 77;
    }
    if (groupwise != nullptr && *groupwise != '\0') {
        if (const int result = exercise_artifact(groupwise, kGroupwiseStagingBytes); result != 0) {
            return result;
        }
    }
    if (legacy != nullptr && *legacy != '\0') {
        if (const int result = exercise_artifact(legacy, kGroupwiseStagingBytes); result != 0) {
            return result;
        }
    }
    if (nvfp4 != nullptr && *nvfp4 != '\0') {
        if (const int result = exercise_artifact(nvfp4, kNvfp4StagingBytes); result != 0) {
            return result;
        }
    }
    std::cout << "ok\n";
    return 0;
}
```

Register in `tests/CMakeLists.txt` immediately after the `ninfer_qwen3_6_27b_prefix_real_test` block (lines 106-109):

```cmake
ninfer_add_test(ninfer_qwen3_6_27b_engine_offload_real_test
  SOURCES targets/qwen3_6_27b/test_engine_offload_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_engine_offload_real_test PROPERTIES SKIP_RETURN_CODE 77)
```

- [x] **Step 2: Build to verify it fails**

Run: `cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=120a && cmake --build build --target ninfer_qwen3_6_27b_engine_offload_real_test`

Expected: FAIL to compile. Errors reference members that do not exist yet: `ninfer::EngineOptions` has no member `weight_residency`, `ninfer::LoadSummary` has no member `host_bytes`/`host_capacity_bytes`, `ninfer::MemorySummary` has no member `staging`/`host_store_bytes`, and `ninfer::WeightResidency` is not declared. No other error kinds.

- [x] **Step 3: Commit**

```bash
git add tests/targets/qwen3_6_27b/test_engine_offload_real.cpp tests/CMakeLists.txt
git commit -m "test(targets): add 27B engine offload residency scenarios"
```

---

### Task 2: public WeightResidency type and summary fields

**Files:**
- Modify: `include/ninfer/types.h`

- [x] **Step 1: Add the public enum and fields**

In `include/ninfer/types.h`:

1. Immediately after the `KvCacheStorage` enum (closing brace at line 24), insert:

```cpp
enum class WeightResidency : std::uint8_t {
    AllResident,
    FfnOffload,
};
```

2. In `EngineOptions`, immediately after `KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;` (line 79), insert (aligning `=` to the struct's existing column):

```cpp
    WeightResidency weight_residency     = WeightResidency::AllResident;
```

3. In `MemorySummary`, immediately after `ArenaMemorySummary workspace;` (line 378), insert:

```cpp
    ArenaMemorySummary staging;
```

and immediately after that insert:

```cpp
    std::size_t host_store_bytes = 0;
```

4. In `LoadSummary`, immediately after `std::uint64_t peak_staging_bytes   = 0;` (line 421), insert:

```cpp
    std::uint64_t host_bytes           = 0;
    std::uint64_t host_capacity_bytes  = 0;
```

No other types change. Do not touch `engine.h` — the fields flow through existing `load_summary()`/`memory_summary()` pass-throughs.

- [x] **Step 2: Build**

Run: `cmake --build build`

Expected: full tree compiles (the Task-1 test now compiles; nothing else references the new fields yet).

- [x] **Step 3: Verify runtime RED**

Run: `NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_engine_offload_real_test`

Expected: FAIL. `Package::plan_load` still binds `AllResident` (residency not yet threaded), so on 16 GB the load throws `std::invalid_argument` "model weights require 17093490688 bytes of device memory ..." — the test exits 1. (If run where AllResident fits, the failure is instead "offload load summary does not report the pinned host store"; either is an acceptable RED.)

- [x] **Step 4: Commit**

```bash
git add include/ninfer/types.h
git commit -m "feat(engine): add the public weight residency option and summary fields"
```

---

### Task 3: thread residency through the load path and report the host store

**Files:**
- Modify: `src/targets/qwen3_6_27b/impl/package.cpp`
- Modify: `src/targets/qwen3_6_35b_a3b/impl/package.cpp`
- Modify: `src/targets/registry.cpp`

- [x] **Step 1: 27B plan_load maps the public option**

In `src/targets/qwen3_6_27b/impl/package.cpp`, replace the body of `Package::plan_load` (lines 99-104):

```cpp
Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    const ResidencyProfile residency =
        options.weight_residency == WeightResidency::FfnOffload ? ResidencyProfile::FfnOffload
                                                                : ResidencyProfile::AllResident;
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, detail::bind_artifact(binder, weights_profile,
                                               qwen3_6::startup_features(options), residency)));
}
```

`WeightResidency` resolves to `ninfer::WeightResidency` via the enclosing `ninfer` namespace; `ResidencyProfile` resolves to `Package::ResidencyProfile` inside the static member. `bindings.h` already declares the 4-arg `bind_artifact` with the residency default.

- [x] **Step 2: 35B plan_load rejects the offload value**

In `src/targets/qwen3_6_35b_a3b/impl/package.cpp`, at the top of the `Package::plan_load` body (line 76, before the existing `return`), insert:

```cpp
    if (options.weight_residency != WeightResidency::AllResident) {
        throw std::invalid_argument("target '" + std::string(target_key) +
                                    "' does not support weight residency offload");
    }
```

`<stdexcept>` is already transitively available in this file (it throws `std::runtime_error` below). No residency support is added to the 35B package.

- [x] **Step 3: validate_options guards the enum and construct_registered fills the host store**

In `src/targets/registry.cpp` `validate_options` (after the `kv_capacity` switch, before the `max_concurrency` check), insert:

```cpp
    switch (options.weight_residency) {
    case WeightResidency::AllResident:
    case WeightResidency::FfnOffload:
        break;
    default:
        throw std::invalid_argument("Engine weight_residency value is invalid");
    }
```

In `construct_registered`, in the `LoadSummary summary;` fill block (after `summary.peak_staging_bytes   = stats.peak_staging_bytes;`), insert:

```cpp
    summary.host_bytes          = stats.host_bytes;
    summary.host_capacity_bytes = stats.host_capacity_bytes;
```

- [x] **Step 4: Build and run the offload test**

Run: `cmake --build build`

Run: `NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_engine_offload_real_test`

Expected: engine now constructs under FfnOffload (device plan ~7 GB fits 16 GB). `load_summary().host_bytes/host_capacity_bytes` are populated. The test still FAILS because `memory_summary().staging.capacity_bytes` is 0 and `host_store_bytes` is 0 (Task 4), and `generate` has not been reached. Output includes `staging arena capacity is 0, expected 306380800`.

- [x] **Step 5: Commit**

```bash
git add src/targets/qwen3_6_27b/impl/package.cpp src/targets/qwen3_6_35b_a3b/impl/package.cpp src/targets/registry.cpp
git commit -m "feat(engine): thread weight residency through the target load path"
```

---

### Task 4: report the staging arena and host store in memory summaries

**Files:**
- Modify: `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h`
- Modify: `src/targets/qwen3_6_27b/impl/load/bindings.cpp`
- Modify: `src/targets/qwen3_6/impl/runtime/program_impl.h`

- [x] **Step 1: ModelView carries the host-store size**

In `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h`, immediately after `std::vector<StagedWeight> staged_weights;` (line 102), insert:

```cpp
    std::size_t host_store_bytes = 0;
```

No new include is needed (`<cstddef>` is already included; `std::size_t` is in scope). The 35B instantiation leaves the member defaulted to 0.

- [x] **Step 2: 27B LoadedModelData populates it**

In `src/targets/qwen3_6_27b/impl/load/bindings.cpp`, inside the `LoadedModelData` ctor immediately after `runtime.staging_arena = &*staging_arena_;` (line 501), insert:

```cpp
    runtime.host_store_bytes = static_cast<std::size_t>(backing.stats().host_capacity_bytes);
```

This is unconditional: for AllResident `host_capacity_bytes` is 0, matching the default. `backing.stats()` returns `const artifact::MaterializationStats&`.

- [x] **Step 3: Program memory_summary reports staging and host store**

In `src/targets/qwen3_6/impl/runtime/program_impl.h`, inside `ProgramImplCore::memory_summary() const noexcept` (lines 2163-2179), after the `out.workspace = ...` line and before `out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;`, insert:

```cpp
    if (model.staging_arena != nullptr) {
        DeviceArena& staging = *model.staging_arena;
        out.staging = ArenaMemorySummary{staging.capacity(), staging.used(), staging.peak_used()};
    }
    out.host_store_bytes = model.host_store_bytes;
```

`model.staging_arena` is null under AllResident and for the 35B target, leaving `out.staging` default-constructed (all zeros) — byte-identical reporting for existing paths.

- [x] **Step 4: Build and run the full gate**

Run: `cmake --build build`

Run: `NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_qwen3_6_27b_engine_offload_real_test|ninfer_qwen3_6_27b_residency_test|ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test'`

Expected: **5/5 PASS** (artifact_reader, artifact_materialization, tensor, arena, residency) plus **PASS** for `ninfer_qwen3_6_27b_engine_offload_real_test`. The engine test loads FfnOffload, verifies `staging.capacity_bytes == 306380800`, `host_store_bytes > 0`, and runs a real 4-token greedy `generate` through `prepare_tokens` — this executes the captured graph with the 128 staging H2D memcpy nodes end-to-end for the first time.

> **Note for the implementer:** `exercise_offloaded_generation` runs the full graph-capture + replay decode path with the staging interleave. If this fails, it is a genuine surface-5 capture-path bug surfaced by this integration test — diagnose and fix it in this branch (it is a realistic regression introduced by the change), then re-run. Expected first run ~30-90 s (load + capture).

- [x] **Step 5: Commit**

```bash
git add src/targets/qwen3_6/export/ninfer/targets/qwen3_6/model_view.h src/targets/qwen3_6_27b/impl/load/bindings.cpp src/targets/qwen3_6/impl/runtime/program_impl.h
git commit -m "feat(engine): report the staging arena and host store in memory summaries"
```

---

### Task 5: CLI flag and summary printing

**Files:**
- Modify: `apps/cli/options.h`
- Modify: `apps/cli/options.cpp`
- Modify: `apps/cli/main.cpp`

- [x] **Step 1: Options struct field**

In `apps/cli/options.h`, immediately after `KvCacheStorage kv_cache = KvCacheStorage::BFloat16;` (line 32), insert:

```cpp
    ninfer::WeightResidency weight_residency = ninfer::WeightResidency::AllResident;
```

- [x] **Step 2: parse helper and flag**

In `apps/cli/options.cpp`, immediately after the `parse_kv_cache` function (ends line 59), insert:

```cpp
ninfer::WeightResidency parse_weight_residency(std::string_view text) {
    if (text == "all") { return ninfer::WeightResidency::AllResident; }
    if (text == "ffn") { return ninfer::WeightResidency::FfnOffload; }
    throw std::invalid_argument("invalid weight-residency: " + std::string(text));
}
```

In the parse loop, immediately after the `--kv-dtype` branch (`options.kv_cache = parse_kv_cache(value(arg));`), insert:

```cpp
        } else if (arg == "--weight-residency") {
            options.weight_residency = parse_weight_residency(value(arg));
        }
```

In `usage_text`, extend the `[--kv-dtype bf16|int8] [--spec mtp|dflash --draft-tokens N]` line (line 80) to:

```cpp
           "       [--kv-dtype bf16|int8] [--weight-residency all|ffn] [--spec mtp|dflash --draft-tokens N]\n"
```

- [x] **Step 3: main.cpp mapping and printing**

In `apps/cli/main.cpp`, in the `ninfer::EngineOptions engine_options;` build block (after `engine_options.kv_cache       = cli.kv_cache;`), insert:

```cpp
        engine_options.weight_residency = cli.weight_residency;
```

In `print_load_summary`, after `print_metric("pinned staging peak", format_bytes(load.peak_staging_bytes));` (line 149), insert:

```cpp
    print_metric("host store", format_bytes(load.host_capacity_bytes));
    print_metric("host offloaded", format_bytes(load.host_bytes));
```

In `print_generation_summary` memory block, after `print_metric("gpu weights used", format_arena_used(memory.weights));` (line 188), insert:

```cpp
    print_metric("gpu staging arena", format_bytes(memory.staging.capacity_bytes));
    print_metric("host store", format_bytes(memory.host_store_bytes));
```

- [x] **Step 4: Build and smoke-test the CLI**

Run: `cmake --build build`

Run: `./build/apps/ninfer $HOME/llm-models/qwen3_8_27b.ninfer --weight-residency ffn --prompt "Hello" --max-new 4 --greedy --max-context 4096 2>&1 | grep -E "host store|gpu staging arena"`

Expected: engine loads under FfnOffload on 16 GB and both printed metrics are nonzero (`host store 9,364,992 B`-order capacity, `gpu staging arena 306,380,800 B`). The generation produces 4 tokens.

Also run `./build/apps/ninfer 2>&1 | head -5` to confirm the usage line renders.

- [x] **Step 5: Commit**

```bash
git add apps/cli/options.h apps/cli/options.cpp apps/cli/main.cpp
git commit -m "feat(cli): add the weight residency option and host-store reporting"
```

---

### Task 6: docs

**Files:**
- Modify: `docs/maintainer/weight-offload.md`
- Modify: `docs/cli.md`

- [x] **Step 1: weight-offload.md status**

In `docs/maintainer/weight-offload.md`:

1. Replace the top-of-file Status block (lines 3-8) with:

```text
Status: implementation in progress. Surfaces 1-6 of the change-surface table are implemented
(see section 5): artifact host placement, tensor host addresses, the 27B residency profile, the
staging arena/slot binding, the graph-capture staging interleave, and the public engine residency
option with load/memory-summary reporting. The remaining test/documentation surfaces (7-8) are not
yet implemented. Scope target is the Qwen3.6-27B dense identities (`groupwise-int` and `nvfp4`);
the 35B-A3B MoE target is a bonus item with extra work.
```

2. Replace the section-5 Status paragraph (surfaces 1-5 ...) with the same text plus a new surface-6 sentence. Use:

```text
Status: surfaces 1-6 are implemented — binder host placement
(`TensorPlacement::Host`, host spans in `MaterializationPlan`), materializer host store
(`MaterializedArtifact::host_data(handle)` + host-bytes stats, pinned via `cudaMallocHost`),
`bind_tensor` host dispatch, `Tensor`/`Weight` host addresses with view propagation, the 27B
`ResidencyProfile` (`AllResident` default, `FfnOffload` binds the per-layer FFN/SwiGLU gate/up +
down matrices host-only), the fixed staging arena with offloaded-tensor -> slot binding (a
`2 x largest-streaming-unit` device `DeviceArena` whose slot addresses never change, the
host-placed `Weight` device planes (`payload`/`qdata`/`qhigh`/`scales`, and by extension the
NVFP4 TMA maps) re-pointed at those slots during `LoadedModelData` construction before graph
capture, and a `ModelView::staged_weights` slot map), the graph-capture staging interleave
(the shared 27B `post_mixer` leaf issues a `cudaMemcpyAsync` host->slot copy for each staged
gate/up and down before its MLP kernels, gated on `weight.host != nullptr`, so the copies become
in-graph H2D memcpy nodes during capture and run eagerly for prefill and non-graph decode), and
the public engine residency option (`EngineOptions::weight_residency`, default `AllResident`;
`FfnOffload` is selectable through the CLI `--weight-residency ffn`), with `LoadSummary` reporting
the pinned host store (`host_bytes`, `host_capacity_bytes`) and `MemorySummary` reporting the
staging arena and `host_store_bytes`. FfnOffload decode runs correctly through the public Engine;
the 35B-A3B target rejects the offload option.
```

3. Verify no stale statements remain: no mention that surface 6 is "not selectable" and no `1 GiB arena` / `prepare_graphs() interleaves` / `pinning deferred` text.

- [x] **Step 2: cli.md option row**

In `docs/cli.md`, immediately after the `--kv-dtype bf16\|int8` row (line 140), insert:

```md
| `--weight-residency all\|ffn` | offload the per-layer FFN/SwiGLU matrices to pinned host memory and stream them through the fixed staging arena during decode; required to load 27B dense identities on GPUs with less VRAM than the resident weights | `all` |
```

- [x] **Step 3: plan checkboxes**

Mark every `- [x]` task checkbox in this file as `- [x]`.

- [x] **Step 4: verify**

Run: `git diff --check` (silent expected). Confirm only the two docs files changed.

- [x] **Step 5: Commit**

```bash
git add docs/maintainer/weight-offload.md docs/cli.md docs/superpowers/plans/2026-08-15-qwen3_6-27b-engine-residency.md
git commit -m "docs(offload): mark the engine residency option as implemented"
```

---

## Out of scope (recorded for follow-on)

- `src/runtime/engine/kv_capacity.cpp` — no change needed; existing two-phase resolution is residency-correct (see Architecture).
- Serving surface (`src/serve` ServeOptions) — the Engine option exists for any consumer; a `serve` flag is a later follow-on, not this plan.
- 35B-A3B residency — rejected in `plan_load` (above).
- `docs/performance.md` and model-card measurement updates — surface 8.
