# Qwen3.6-27B staging interleave (surface 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FfnOffload decode actually run by interleaving H2D staging copies of the offloaded FFN weight planes into the decode/prefill path, so the host-resident `Weight` objects land in their fixed staging slots before the MLP kernels that consume them.

**Architecture:** Inject the staging copies inside the 27B `Variant::post_mixer` leaf rather than restructuring `prepare_graphs()`. `post_mixer` is called once per text layer by the shared `run_layers` body, which is the SAME closure used for eager decode, graph capture, and prefill. Each staged `Weight` carries `host` (pinned host-store source) + `payload` (fixed slot destination) + `payload_bytes`; a `cudaMemcpyAsync(host -> slot)` gated on `weight.host != nullptr` becomes an in-graph H2D memcpy node during capture and runs eagerly otherwise. The host store is pinned (`cudaMallocHost`) so the copies are true async DMA and capturable.

**Tech Stack:** C++20, CUDA 13.3, sm_120a, CMake/Ninja, ctest. Real-artifact test on RTX 5060 Ti 16 GB via `NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer`.

---

## Background (required context)

- `Variant::post_mixer` (`src/targets/qwen3_6_27b/impl/variant.cpp:237-245`) runs the SwiGLU MLP: `ops::linear_swiglu(hidden, weights.gate_up, activation, ...)` then `ops::linear_add(activation, weights.down, residual, ...)`. It receives `const PostMixerWeights&` (= `DensePostMixerPayload{gate_up, down}`), `WorkspaceArena&`, and `cudaStream_t`.
- `run_layers` (`src/targets/qwen3_6/impl/runtime/text_context_impl.h:968-1015`) calls `mlp_tail` (→ `Variant::post_mixer`) for BOTH full-attention and GDN layers, for BOTH decode (`TextPhase::Verify`) and prefill (`TextPhase::Prefill`). Prefill is not graph-captured; decode capture wraps the same body (`decode_impl.h:12-47`, `capture_graph` in `graph_impl.h:20-24`, `DecodeGraphDefinition::capture` = `cudaStreamBeginCapture -> body -> EndCapture` in `decode_graph.cpp:58-79`).
- After surface 4, a staged `Weight` has: `host` = object base in the host store (== `MaterializedArtifact::host_data(handle)`), `payload`/`qdata`/`qhigh`/`scales` = fixed addresses in the staging arena, `payload_bytes` = the object's encoded byte size. `point_at_slot` re-derived the plane offsets so `qdata`/`qhigh`/`scales` sit at the same offsets within the slot as in the original object. Copying `payload_bytes` bytes from `host` to `payload` therefore reproduces the exact object byte layout at the slot.
- AllResident weights have `host == nullptr`; staging gated on `host != nullptr` makes the AllResident path a no-op (byte-identical graphs).
- `cudaMemcpyAsync` inside a captured body becomes a graph memcpy node — already proven by the ingress copy (`decode_impl.h:20-22`). Async H2D from pageable host memory is NOT properly capturable, so pinning the host store is required.
- `cudaMallocHost`/`cudaFreeHost` precedent: `PinnedHostBuffer` (`src/core/arena.cu:237-248`). `CUDA_CHECK` macro from `src/core/device.h:11`.
- FfnOffload is still NOT selectable through the public engine (`Package::plan_load` uses the `AllResident` default); the residency profile is only reachable via the 4-arg `detail::bind_artifact` used by tests. This plan does NOT add the engine option (that is surface 6).

---

## File structure

- Modify `tests/targets/qwen3_6_27b/test_residency.cpp` — add a staged-MLP round scenario (RED, Task 1).
- Modify `src/targets/qwen3_6_27b/impl/variant.cpp` — `stage_weight` helper + calls in `post_mixer` (GREEN, Task 2).
- Modify `src/artifact/materializer.{h,cpp}` — pin the host store via `cudaMallocHost` + custom deleter (Task 3).
- Modify `tests/test_artifact_materialization.cpp` — assert the host store is pinned (Task 3, RED then GREEN).
- Modify `docs/maintainer/weight-offload.md` + this plan's checkboxes (Task 4).

---

### Task 1: RED — staged MLP round scenario

**Files:**
- Modify: `tests/targets/qwen3_6_27b/test_residency.cpp`
- Test: same file (the existing `ninfer_qwen3_6_27b_residency_test` target)

- [ ] **Step 1: Add includes**

Edit `tests/targets/qwen3_6_27b/test_residency.cpp` (current lines 1-17):

```
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"
```

(Add `#include "core/arena.h"` between `artifact/reader.h` and `targets/...`.) And in the std block add `<cstring>` after `<cstdlib>` and `<vector>` after `<variant>` so the block reads:

```cpp
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
```

- [ ] **Step 2: Add the round helper**

Insert this function in the anonymous namespace immediately after `is_host_placed` (currently ends at line 38):

```cpp
int staged_mlp_round_matches_host(const ninfer::DeviceContext& device, WeightsProfile profile,
                                  const DensePostMixerPayload& post_mixer) {
    ninfer::DeviceArena io(1 << 20);
    const ninfer::Tensor hidden = io.alloc(ninfer::DType::BF16, {TextConfig::hidden, 1});
    ninfer::Tensor residual     = io.alloc(ninfer::DType::BF16, {TextConfig::hidden, 1});
    ninfer::WorkspaceArena work(Variant::post_mixer_workspace_capacity_bytes(
        profile, ninfer::targets::qwen3_6::TextPhase::Verify, 1, 1));
    work.reset();
    Variant::post_mixer(hidden, post_mixer, residual, ninfer::targets::qwen3_6::TextPhase::Verify,
                        work, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    for (const ninfer::Weight& weight : {post_mixer.gate_up, post_mixer.down}) {
        std::vector<std::byte> slot(static_cast<std::size_t>(weight.payload_bytes));
        CUDA_CHECK(cudaMemcpy(slot.data(), weight.payload, slot.size(), cudaMemcpyDeviceToHost));
        if (std::memcmp(slot.data(), weight.host, slot.size()) != 0) {
            std::cerr << "a staged MLP slot does not reproduce its host payload\n";
            return 1;
        }
    }
    return 0;
}
```

Notes: `DensePostMixerPayload`, `TextConfig`, `Variant`, `WeightsProfile` all resolve via the existing `using namespace ninfer::targets::qwen3_6_27b::detail;`. `CUDA_CHECK` is available transitively through `artifact/materializer.h` → `core/device.h`. The eager `post_mixer` round stages weights (after Task 2) and runs the real `linear_swiglu`/`linear_add` kernels on this GPU; before Task 2 the slots hold uninitialized `cudaMalloc` data, so the `memcmp` fails.

- [ ] **Step 3: Call the helper in `verify_residency`**

Insert between the vocabulary-endpoint check (currently lines 165-170) and the existing `return 0;` (line 171):

```cpp
    if (staged_mlp_round_matches_host(device, profile,
                                      data.runtime.full_layers[0].post_mixer) != 0 ||
        staged_mlp_round_matches_host(device, profile,
                                      data.runtime.gdn_layers[0].post_mixer) != 0) {
        return 1;
    }
    return 0;
```

- [ ] **Step 4: Build and run to verify RED**

Run: `cmake --build build --target ninfer_qwen3_6_27b_residency_test`
Expected: compiles cleanly.

Run: `NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_residency_test`
Expected: FAIL with `a staged MLP slot does not reproduce its host payload` (exit 1) — the staging slots hold uninitialized memory because `post_mixer` does not copy yet. This is the genuine RED.

- [ ] **Step 5: Commit**

```bash
git add tests/targets/qwen3_6_27b/test_residency.cpp
git commit -m "test(targets): add staged MLP round scenarios"
```

---

### Task 2: GREEN — stage offloaded FFN weights inside `post_mixer`

**Files:**
- Modify: `src/targets/qwen3_6_27b/impl/variant.cpp`
- Test: `ninfer_qwen3_6_27b_residency_test`

- [ ] **Step 1: Add the `core/device.h` include**

In `src/targets/qwen3_6_27b/impl/variant.cpp`, after `#include "targets/qwen3_6_27b/impl/variant.h"` (line 1), add:

```cpp
#include "core/device.h"
```

- [ ] **Step 2: Add the `stage_weight` helper**

In the anonymous namespace, immediately after `text_policy` (currently ends at line 50), add:

```cpp
void stage_weight(const Weight& weight, cudaStream_t stream) {
    if (weight.host == nullptr) { return; }
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.payload), weight.host,
                               static_cast<std::size_t>(weight.payload_bytes),
                               cudaMemcpyHostToDevice, stream));
}
```

`Weight` resolves as in the existing `text_policy` signature (`const Weight&`). `payload` is `const void*` (the slot address, `const_cast`ed at the single population site in surface 4); the host pointer is the pinned host-store object base.

- [ ] **Step 3: Call `stage_weight` at the top of `post_mixer`**

Replace the body of `Variant::post_mixer` (currently `variant.cpp:237-245`):

```cpp
void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    stage_weight(weights.gate_up, stream);
    stage_weight(weights.down, stream);
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, text_policy(weights.gate_up), workspace,
                       stream);
    ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace,
                    stream);
}
```

Do NOT touch `mtp_post_mixer` (MTP weights stay device-resident under FfnOffload; `mtp_post_mixer` is only called when the MTP feature is active, and no MTP weights are staged).

- [ ] **Step 4: Build and run to verify GREEN**

Run: `cmake --build build`
Expected: full tree compiles clean (both 27B and 35B-A3B variants compile against unchanged headers).

Run: `NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_residency_test`
Expected: PASS — the eager `post_mixer` round now stages both planes, and the slots reproduce the host payload byte-for-byte.

Run: `/usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test|ninfer_qwen3_6_27b_residency_test'`
Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/targets/qwen3_6_27b/impl/variant.cpp
git commit -m "feat(targets): stage offloaded FFN weights before MLP kernels"
```

---

### Task 3: pin the host store with `cudaMallocHost`

**Files:**
- Modify: `src/artifact/materializer.h`, `src/artifact/materializer.cpp`
- Test: `tests/test_artifact_materialization.cpp`

- [ ] **Step 1: RED — add a pinning assertion to the artifact materialization test**

In `tests/test_artifact_materialization.cpp`, immediately after the host-stats `require` (currently lines 233-235), add:

```cpp
            cudaPointerAttributes attributes{};
            CUDA_CHECK(cudaPointerGetAttributes(&attributes,
                                                host_materialized.host_data(host_tensor)));
            require(attributes.type == cudaMemoryTypeHost,
                    "host tensor payload is not pinned host memory");
```

Run: `cmake --build build --target ninfer_artifact_materialization_test`
Expected: compiles. Then:
Run: `/usr/bin/ctest --test-dir build --output-on-failure -R ninfer_artifact_materialization_test`
Expected: FAIL — `cudaPointerGetAttributes` on the current plain `std::make_unique<std::byte[]>` store returns `cudaErrorInvalidValue`, `CUDA_CHECK` throws, and the test exits 1. This is the RED for Task 3.

- [ ] **Step 2: Add the deleter to `materializer.h`**

In `src/artifact/materializer.h`, in namespace `ninfer::artifact` immediately before `class MaterializedArtifact` (line 34), add:

```cpp
struct PinnedHostStoreDeleter {
    void operator()(std::byte* pointer) const noexcept {
        if (pointer != nullptr) { (void)cudaFreeHost(pointer); }
    }
};
```

`cudaFreeHost` is available via `#include <cuda_runtime.h>` (transitively through `core/device.h`).

Change the private member (line 65) from:

```cpp
    std::unique_ptr<std::byte[]> host_store_;
```

to:

```cpp
    std::unique_ptr<std::byte[], PinnedHostStoreDeleter> host_store_;
```

The defaulted move ctor/assign still work: a `unique_ptr` with a stateless deleter is move-constructible and the deleter travels with it.

- [ ] **Step 3: Allocate the store with `cudaMallocHost` in `materializer.cpp`**

In `src/artifact/materializer.cpp`, in the `if (plan.host_capacity_bytes > 0)` block, replace the allocation (currently lines 156-162):

```cpp
        auto store = std::make_unique<std::byte[]>(store_bytes + pad);
        const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(store.get());
        const std::size_t base_pad = static_cast<std::size_t>(
            (store_alignment - (addr % store_alignment)) % store_alignment);
        out.host_store_       = std::move(store);
        out.host_store_base_  = out.host_store_.get() + base_pad;
        out.host_store_bytes_ = store_bytes + base_pad;
```

with:

```cpp
        void* store_ptr       = nullptr;
        CUDA_CHECK(cudaMallocHost(&store_ptr, store_bytes + pad));
        auto store = std::unique_ptr<std::byte[], PinnedHostStoreDeleter>(
            static_cast<std::byte*>(store_ptr));
        const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(store_ptr);
        const std::size_t base_pad = static_cast<std::size_t>(
            (store_alignment - (addr % store_alignment)) % store_alignment);
        out.host_store_       = std::move(store);
        out.host_store_base_  = out.host_store_.get() + base_pad;
        out.host_store_bytes_ = store_bytes + base_pad;
```

`cudaMallocHost` returns page-aligned pinned memory, so `base_pad` computes to 0 and `host_store_base_ == host_store_.get()`. Keep the alignment computation as-is for robustness. The existing `store_bytes > max - pad` overflow guard stays. `<cuda_runtime.h>` is already included (line 3).

NOTE (implementer deviation, verified at execution): `cudaMallocHost` packs allocations at 512-byte granularity, not 4096, so a host-store request of `store_bytes + pad` (e.g. 271 bytes in the 16-byte host-fixture test) shifted the subsequent `Slot` pinned buffers off 4096-byte alignment, breaking `read_direct`'s `direct_io_alignment` contract (`reader.cpp:230` throws "direct artifact read is not 4096-byte aligned"). The committed code therefore rounds the request up to `Reader::direct_io_alignment` (4096) via `align_up` so the host-store footprint is a 4096-multiple and subsequent pinned allocations land on 4096 boundaries:

- [ ] **Step 4: Build and run to verify GREEN**

Run: `cmake --build build`
Expected: full tree compiles clean.

Run: `/usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test'`
Expected: 4/4 PASS (reader, materialization, tensor, arena).

Run: `NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_residency_test`
Expected: PASS — the residency test exercises the pinned host store end-to-end (host source = pinned store; staging copy now a true async DMA from pinned memory).

- [ ] **Step 5: Commit**

```bash
git add src/artifact/materializer.h src/artifact/materializer.cpp tests/test_artifact_materialization.cpp
git commit -m "feat(artifact): pin the host store for staged weight copies"
```

---

### Task 4: docs — mark the staging interleave as implemented

**Files:**
- Modify: `docs/maintainer/weight-offload.md`
- Modify: `docs/superpowers/plans/2026-08-15-qwen3_6-27b-staging-interleave.md`

- [ ] **Step 1: Update the top-of-file Status line**

Replace `docs/maintainer/weight-offload.md` lines 3-4:

```
Status: investigation/design proposal. Not implemented. Scope target is the Qwen3.6-27B dense
identities (`groupwise-int` and `nvfp4`); the 35B-A3B MoE target is a bonus item with extra work.
```

with:

```
Status: implementation in progress. Surfaces 1-5 of the change-surface table are implemented
(see section 5): artifact host placement, tensor host addresses, the 27B residency profile, the
staging arena/slot binding, and the graph-capture staging interleave. The public engine residency
option, capacity planning, and CLI surface (surface 6) and the remaining tests/docs surfaces are
not yet implemented. Scope target is the Qwen3.6-27B dense identities (`groupwise-int` and
`nvfp4`); the 35B-A3B MoE target is a bonus item with extra work.
```

- [ ] **Step 2: Correct the change-surface table file columns**

In `docs/maintainer/weight-offload.md`:
- Row 4 (`Staging + slot binding`) — replace the Files column `src/targets/qwen3_6/impl/runtime/` (Program, `workspace_recipe.h`) with `` family `model_view.h`, `src/targets/qwen3_6_27b/impl/load/bindings.{h,cpp}` ``.
- Row 5 (`Graph capture`) — replace the Files column `` `program_impl.h` `prepare_graphs()`, schedule capture helpers `` with `` `src/targets/qwen3_6_27b/impl/variant.cpp`, `src/artifact/materializer.{h,cpp}` `` and change the Change column to `interleave H2D memcpy nodes per offload group inside the shared post_mixer leaf; pinned host store`.

- [ ] **Step 3: Replace the section-5 Status paragraph**

Replace the entire Status paragraph (currently lines 123-135) with:

```
Status: surfaces 1-5 are implemented — binder host placement
(`TensorPlacement::Host`, host spans in `MaterializationPlan`), materializer host store
(`MaterializedArtifact::host_data(handle)` + host-bytes stats, pinned via `cudaMallocHost`),
`bind_tensor` host dispatch, `Tensor`/`Weight` host addresses with view propagation, the 27B
`ResidencyProfile` (`AllResident` default, `FfnOffload` binds the per-layer FFN/SwiGLU gate/up +
down matrices host-only), the fixed staging arena with offloaded-tensor -> slot binding (a
`2 x largest-streaming-unit` device `DeviceArena` whose slot addresses never change, the
host-placed `Weight` device planes (`payload`/`qdata`/`qhigh`/`scales`, and by extension the
NVFP4 TMA maps) re-pointed at those slots during `LoadedModelData` construction before graph
capture, and a `ModelView::staged_weights` slot map), and the graph-capture staging interleave:
the shared 27B `post_mixer` leaf issues a `cudaMemcpyAsync` host->slot copy for each staged
gate/up and down before its MLP kernels, gated on `weight.host != nullptr`, so the copies become
in-graph H2D memcpy nodes during capture and run eagerly for prefill and non-graph decode.
FfnOffload decode now runs correctly; the residency profile is still not selectable through the
public engine option (surface 6).
```

- [ ] **Step 4: Mark this plan's checkboxes**

Change every `- [ ]` in this file to `- [x]`.

- [ ] **Step 5: Verify and commit**

Run: `git diff --check`
Expected: clean.

Run: `git diff --stat`
Expected: the two docs files only.

```bash
git add docs/maintainer/weight-offload.md docs/superpowers/plans/2026-08-15-qwen3_6-27b-staging-interleave.md
git commit -m "docs(offload): mark the staging interleave as implemented"
```

---

## Self-review

**Spec coverage:** surface-5 row of the design doc (interleave H2D memcpy nodes per offload group) is delivered by the `post_mixer`-leaf injection: each offload group's two planes are staged immediately before the kernels that read them, inside the shared body so eager/capture/prefill all get copies; the pinned host store makes the copies true DMA and capturable. `staged_weights`/`staging_arena` (surface 4) remain the load-time contract record; the leaf uses the `Weight`'s own `host`/`payload`/`payload_bytes`. Follow-on requirements honored: copy sources at object granularity (`weight.host == host_data(handle)`), NVFP4 auto-re-points TMA (descriptors encoded per launch from `qdata`/`scales` at capture), no changes to `prepare_graphs`/`program_impl.h`, MTP and 35B untouched, `AllResident` is a structural no-op.

**Placeholder scan:** every step has exact code and commands; no TBD/TODO.

**Type consistency:** `stage_weight` uses `Weight::host`/`payload`/`payload_bytes` matching surface-4 field names; `PinnedHostStoreDeleter` matches the `unique_ptr<std::byte[], D>` member; the test helper uses the same `post_mixer`/`workspace_capacity_bytes` signatures as `variant.h`.

## Execution handoff

After committing the plan doc to master, create worktree `.worktrees/qwen3_6-27b-staging-interleave` (branch `qwen3_6-27b-staging-interleave`) from master, run the baseline build + filtered ctest, then execute tasks 1-4 with subagent-driven development, then merge to master and clean up.
