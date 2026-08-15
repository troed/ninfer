# Artifact Host-Tensor Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the artifact layer the ability to keep weight tensors host-resident instead of device-resident, as the foundation for CPU/system-RAM weight offload.

**Architecture:** Extend the existing Binder/materializer split (per [weight-offload.md](../../maintainer/weight-offload.md) section 4.1). `Binder` gains a host placement that carves offsets out of a separate host capacity budget; `materialize()` writes host-placed tensor payloads into a `MaterializedArtifact` host store (plain host memory; pinning is a later offload-phase concern); `MaterializedArtifact::host_data(handle)` exposes the host pointer. Kernels and graph capture are untouched in this plan — this is purely the load-path capability, tested with the existing fixture artifact framework.

**Tech Stack:** C++20, CMake + Ninja, CUDA 13.1, `sm_120a`. Tests: `tests/test_artifact_materialization.cpp` (already registered as `ninfer_artifact_materialization_test`, links `ninfer_artifact`, device tests skip with return code 77).

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/artifact/binder.h` | placement plan types + `Binder` API | add `TensorPlacement::Host`, `HostTensorMaterialization`, host plan fields, `materialize_tensor_on_host()` |
| `src/artifact/binder.cpp` | plan construction | implement `materialize_tensor_on_host()` |
| `src/artifact/materializer.h` | `MaterializedArtifact` + stats | add `host_data()`, host store members, host stats |
| `src/artifact/materializer.cpp` | payload copy | host store allocation + host tensor copy; allow device-only/host-only/mixed plans |
| `src/artifact/typed_binding.cpp` | convenience binding | dispatch `TensorPlacement::Host` |
| `tests/test_artifact_materialization.cpp` | artifact contract test | host placement scenarios |

Context the implementer needs: `Binder::materialize_on_device` (binder.cpp:82-100) is the template for the host placement; `materialize()` (materializer.cpp:98-256) currently allocates one `DeviceArena` sized `plan.device_capacity_bytes` and throws when `capacity == 0` or when no device tensors exist; `Reader::payload()` returns a `PayloadSpan` whose `.data` span covers the object bytes in the mmap (copyable straight to host memory); fixture helper `tests/artifact_fixture.h::write_fixture` fills each object's payload bytes with a per-object marker (`marker++` per object in array order).

---

### Task 1: Failing host-placement test

**Files:**
- Modify: `tests/test_artifact_materialization.cpp`

- [ ] **Step 1: Add host-placement constants, fixture, and scenario**

Add `#include <cstring>` after the existing includes block (needed for `std::memcmp`). Add these constants next to the existing ones (after line 32):

```cpp
constexpr std::array<std::byte, 16> kHostTensor = {
    std::byte{1}, std::byte{1}, std::byte{1}, std::byte{1},
    std::byte{1}, std::byte{1}, std::byte{1}, std::byte{1},
    std::byte{1}, std::byte{1}, std::byte{1}, std::byte{1},
    std::byte{1}, std::byte{1}, std::byte{1}, std::byte{1},
};
constexpr std::array<std::byte, 4> kHostFixtureDeviceTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
```

Add this fixture function after `write_fixture()` (after line 62). The first object in the array is the host tensor and therefore receives marker byte `1`; the second is the device tensor and receives marker byte `2`:

```cpp
ninfer::test::artifact_fixture::TemporaryArtifact write_host_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "weights/host"},
                             {"kind", "tensor"},
                             {"shape", {8}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 16}},
                            {{"name", "weights/device"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 4}},
                        })},
        },
        "host-materialization");
}
```

Insert the host-placement scenario inside `main()` immediately before the final `return 0;` (after line 161):

```cpp
        {
            auto host_fixture = write_host_fixture();
            ninfer::artifact::Reader host_reader(host_fixture.path);
            ninfer::artifact::Binder host_binder(host_reader);
            const auto host_tensor = ninfer::artifact::bind_tensor(
                host_binder, "weights/host", ninfer::artifact::NumericFormat::BF16, {8},
                ninfer::artifact::TensorPlacement::Host);
            constexpr std::array<std::uint64_t, 1> host_device_shape = {2};
            const auto host_device_tensor = host_binder.require_tensor(
                "weights/device", ninfer::artifact::NumericFormat::BF16,
                ninfer::artifact::StorageLayout::ContiguousLeV1, host_device_shape);
            host_binder.materialize_on_device(host_device_tensor);
            const auto host_plan = host_binder.finish();
            require(host_plan.object_count == 2 && host_plan.host_tensor_objects.size() == 1 &&
                        host_plan.host_capacity_bytes == kHostTensor.size() &&
                        host_plan.device_objects.size() == 1 &&
                        host_plan.device_capacity_bytes == kHostFixtureDeviceTensor.size(),
                    "host binder produced the wrong materialization plan");

            ninfer::DeviceContext host_device(0);
            auto host_materialized =
                ninfer::artifact::materialize(host_reader, host_plan, host_device);
            require(std::memcmp(host_materialized.host_data(host_tensor), kHostTensor.data(),
                                kHostTensor.size()) == 0,
                    "host tensor payload differs from the artifact");
            std::array<std::byte, kHostFixtureDeviceTensor.size()> host_device_copied{};
            CUDA_CHECK(cudaMemcpy(host_device_copied.data(),
                                  host_materialized.device_data(host_device_tensor),
                                  host_device_copied.size(), cudaMemcpyDeviceToHost));
            require(host_device_copied == kHostFixtureDeviceTensor,
                    "host-fixture device tensor payload differs from the artifact");
            require(host_materialized.stats().host_bytes == kHostTensor.size() &&
                        host_materialized.stats().host_capacity_bytes == kHostTensor.size(),
                    "host materialization statistics are incomplete");
        }
```

- [ ] **Step 2: Configure and build to verify the test fails to compile**

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --target ninfer_artifact_materialization_test 2>&1 | tail -20
```

Expected: FAIL — `error: 'materialize_tensor_on_host' is not a member`, `'Host' is not a member of 'ninfer::artifact::TensorPlacement'`, `'host_data' is not a member`, and missing plan fields.

- [ ] **Step 3: Commit**

```bash
git add tests/test_artifact_materialization.cpp
git commit -m "test(artifact): add host tensor materialization scenarios"
```

---

### Task 2: Host tensor placement in the Binder

**Files:**
- Modify: `src/artifact/binder.h`, `src/artifact/binder.cpp`

- [ ] **Step 1: Extend the plan types and Binder API**

In `binder.h`, change the `TensorPlacement` enum (line 13) to:

```cpp
enum class TensorPlacement : std::uint8_t {
    Device,
    Host,
    ValidateOnly,
};
```

After `HostMaterialization` (line 31), add:

```cpp
struct HostTensorMaterialization {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
};
```

Change `MaterializationPlan` (lines 33-38) to:

```cpp
struct MaterializationPlan {
    std::size_t object_count             = 0;
    std::uint64_t device_capacity_bytes  = 0;
    std::vector<DeviceMaterialization> device_objects;
    std::vector<HostMaterialization> host_objects;
    std::uint64_t host_capacity_bytes    = 0;
    std::vector<HostTensorMaterialization> host_tensor_objects;
};
```

Add to the `Binder` public interface (after `materialize_on_device`, line 50):

```cpp
    void materialize_tensor_on_host(ObjectHandle handle);
```

- [ ] **Step 2: Implement `materialize_tensor_on_host`**

In `binder.cpp`, add this function immediately after `materialize_on_device` (after line 100), mirroring its validation and offset math against the host budget:

```cpp
void Binder::materialize_tensor_on_host(ObjectHandle handle) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("resource cannot be materialized as a host tensor");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(tensor->name));
    }
    const std::uint64_t alignment = tensor_alignment(tensor->layout);
    const std::uint64_t offset    = align_up(materialization_.host_capacity_bytes, alignment);
    if (tensor->bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
        throw ArtifactError("host materialization plan size overflows u64");
    }
    materialization_.host_tensor_objects.push_back(
        HostTensorMaterialization{handle, offset, tensor->bytes, alignment});
    materialization_.host_capacity_bytes = offset + tensor->bytes;
    planned_[handle.index]               = true;
}
```

(`align_up` is the file-local helper at binder.cpp:13; `tensor_alignment` comes from `artifact/reader.h`.)

- [ ] **Step 3: Build the test target and confirm it now compiles but the scenario is still red**

```bash
cmake --build build --target ninfer_artifact_materialization_test 2>&1 | tail -20
./build/tests/ninfer_artifact_materialization_test
```

Expected: compiles; FAILS at runtime with `"host tensor payload differs from the artifact"` (or throws `"object handle does not name a materialized tensor"` from `host_data`), because the materializer does not yet populate a host store.

- [ ] **Step 4: Commit**

```bash
git add src/artifact/binder.h src/artifact/binder.cpp
git commit -m "feat(artifact): add host tensor placement to the binder"
```

---

### Task 3: Host store in the materializer

**Files:**
- Modify: `src/artifact/materializer.h`, `src/artifact/materializer.cpp`

- [ ] **Step 1: Extend stats, storage, and accessors**

In `materializer.h`, add to `MaterializationStats` (after `device_capacity_bytes`, line 24):

```cpp
    std::uint64_t host_capacity_bytes   = 0;
    std::uint64_t host_bytes            = 0;
```

Change `ObjectStorage` (lines 53-56) to:

```cpp
    struct ObjectStorage {
        void* device = nullptr;
        void* host   = nullptr;
        std::vector<std::byte> resource;
    };
```

Add to the public interface of `MaterializedArtifact` (after `device_data`, line 41):

```cpp
    void* host_data(ObjectHandle handle) const;
```

Add members after `device_arena_` (line 58):

```cpp
    std::unique_ptr<std::byte[]> host_store_;
    std::size_t host_store_bytes_ = 0;
```

- [ ] **Step 2: Implement `host_data`**

In `materializer.cpp`, add after `device_data` (after line 75):

```cpp
void* MaterializedArtifact::host_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].host == nullptr) {
        throw ArtifactError("object handle does not name a host-materialized tensor");
    }
    return objects_[handle.index].host;
}
```

- [ ] **Step 3: Rework `materialize()` for mixed/host-only plans and copy host tensors**

In `materialize()` (materializer.cpp:98), make these changes:

3a. Replace the capacity guard (lines 102-107) so a host-only plan (no device arena) is legal, and count total tensors:

```cpp
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity > 0) {
        if (capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
            throw ArtifactError("artifact tensor backing size is invalid");
        }
        out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
        out.stats_.device_capacity_bytes = capacity;
    }
    out.stats_.tensor_count =
        plan.device_objects.size() + plan.host_tensor_objects.size();
    out.stats_.resource_count = plan.host_objects.size();
```

3b. After the existing resource-retention loop (after line 118), add the host tensor store. The host store is plain host memory; alignment is respected so `host_data` offsets are usable as future pinned-DMA destinations:

```cpp
    if (plan.host_capacity_bytes > 0) {
        if (plan.host_capacity_bytes > static_cast<std::uint64_t>(SIZE_MAX)) {
            throw ArtifactError("host tensor backing size is invalid");
        }
        out.host_store_ =
            std::make_unique<std::byte[]>(static_cast<std::size_t>(plan.host_capacity_bytes));
        out.host_store_bytes_ = static_cast<std::size_t>(plan.host_capacity_bytes);
        out.stats_.host_capacity_bytes = plan.host_capacity_bytes;
    }
    for (const HostTensorMaterialization& placement : plan.host_tensor_objects) {
        ObjectStorage& storage        = out.objects_.at(placement.object.index);
        const PayloadSpan payload     = reader.payload(reader.objects().at(placement.object.index));
        if (payload.data.size() != placement.bytes) {
            throw ArtifactError("host materialization plan does not match artifact payload");
        }
        std::memcpy(out.host_store_.get() + placement.offset, payload.data.data(),
                    payload.data.size());
        storage.host = out.host_store_.get() + placement.offset;
        out.stats_.host_bytes =
            checked_add(out.stats_.host_bytes, placement.bytes,
                        "artifact host tensor byte count overflows u64");
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, placement.bytes, "artifact read bytes overflow u64");
    }
```

3c. Replace the empty-tensors guard (currently line 145, `if (ranges.empty()) throw ArtifactError("materialization plan has no device tensors");`) with a check before the direct-I/O block that allows a device-free plan:

```cpp
    if (plan.device_objects.empty() && plan.host_tensor_objects.empty()) {
        throw ArtifactError("materialization plan has no tensors");
    }
```

3d. Guard the pinned-slot streaming block against zero aligned read bytes (host-only plans). Replace the block from `const std::size_t slot_bytes =` (line 174) through `for (const auto& slot : slots) { slot->wait(); }` (line 246) with the same block wrapped in `if (aligned_read_bytes > 0) { ... }`:

```cpp
    if (aligned_read_bytes > 0) {
        const std::size_t slot_bytes =
            static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
        const std::size_t slot_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_read_bytes - 1) / slot_bytes));
        std::vector<std::unique_ptr<Slot>> slots;
        slots.reserve(slot_count);
        for (std::size_t i = 0; i < slot_count; ++i) {
            slots.push_back(std::make_unique<Slot>(slot_bytes));
        }
        out.stats_.peak_staging_bytes = static_cast<std::uint64_t>(slot_bytes) * slot_count;

        std::size_t next_slot  = 0;
        std::size_t next_range = 0;
        const auto start       = std::chrono::steady_clock::now();
        if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
        for (const ReadSpan& span : read_spans) {
            for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
                Slot& slot = *slots[next_slot++ % slot_count];
                slot.wait();

                const std::uint64_t remaining = span.end - source;
                const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                    slot_bytes,
                    align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
                auto destination =
                    std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
                const std::size_t bytes_read = reader.read_direct(source, destination);
                const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
                if (bytes_read < required) {
                    throw ArtifactError("direct artifact read ended before the planned tensor range");
                }
                out.stats_.file_bytes =
                    checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
                const std::uint64_t chunk_end =
                    checked_add(source, bytes_read, "artifact direct I/O result overflows u64");

                while (next_range < ranges.size() && ranges[next_range].source_end <= source) {
                    ++next_range;
                }
                std::size_t range_index = next_range;
                while (range_index < ranges.size() && ranges[range_index].source_begin < chunk_end) {
                    const CopyRange& range         = ranges[range_index];
                    const std::uint64_t copy_begin = std::max(source, range.source_begin);
                    const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                    if (copy_begin < copy_end) {
                        const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                        CUDA_CHECK(cudaMemcpyAsync(
                            range.destination +
                                static_cast<std::size_t>(copy_begin - range.source_begin),
                            static_cast<std::byte*>(slot.buffer.data()) +
                                static_cast<std::size_t>(copy_begin - source),
                            amount, cudaMemcpyHostToDevice, device.load_stream));
                        copied =
                            checked_add(copied, amount, "artifact copied byte count overflows u64");
                    }
                    if (range.source_end <= chunk_end) {
                        ++range_index;
                    } else {
                        break;
                    }
                }
                next_range = range_index;
                CUDA_CHECK(cudaEventRecord(slot.event, device.load_stream));
                slot.pending = true;

                if (progress != nullptr && progress->callback && copied != last_published &&
                    copied < total) {
                    last_published = copied;
                    progress->callback("weights", copied, total);
                }
            }
        }
        for (const auto& slot : slots) { slot->wait(); }
        out.stats_.upload_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
```

Note the `upload_seconds` line moves inside the guard; the existing line at materializer.cpp:252 (`out.stats_.upload_seconds = ...`) must be removed since it now lives in the guarded block. The subsequent tail checks (`copied != total || next_range != ranges.size()`) pass trivially for the host-only case (`total == 0`, `next_range == 0`, `ranges.empty()`).

- [ ] **Step 4: Build and run the artifact tests**

```bash
cmake --build build --target ninfer_artifact_materialization_test ninfer_artifact_reader_test 2>&1 | tail -20
ctest --test-dir build -R 'ninfer_artifact_(materialization|reader)_test' --output-on-failure
```

Expected: both PASS (the materialization test now includes the host-placement scenario).

- [ ] **Step 5: Commit**

```bash
git add src/artifact/materializer.h src/artifact/materializer.cpp
git commit -m "feat(artifact): materialize host-placed tensors into a host store"
```

---

### Task 4: Convenience binding dispatch

**Files:**
- Modify: `src/artifact/typed_binding.cpp`

- [ ] **Step 1: Dispatch `TensorPlacement::Host`**

In `bind_tensor` (typed_binding.cpp:114-125), change the placement branch:

```cpp
    if (placement == TensorPlacement::Device) {
        binder.materialize_on_device(handle);
    } else if (placement == TensorPlacement::Host) {
        binder.materialize_tensor_on_host(handle);
    } else {
        binder.validate_only(handle);
    }
```

- [ ] **Step 2: Build and run the artifact tests (the host scenario already exercises `bind_tensor`)**

```bash
cmake --build build --target ninfer_artifact_materialization_test 2>&1 | tail -20
ctest --test-dir build -R 'ninfer_artifact_materialization_test' --output-on-failure
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/artifact/typed_binding.cpp
git commit -m "feat(artifact): dispatch host tensor placement in bind_tensor"
```

---

### Task 5: Full verification

- [ ] **Step 1: Build the whole tree with tests and run the artifact/runtime contract tests**

```bash
cmake --build build 2>&1 | tail -5
ctest --test-dir build --output-on-failure -R 'ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test'
git diff --check
```

Expected: all listed tests PASS, `git diff --check` silent. If a CUDA-less environment, note `ninfer_artifact_materialization_test` returns 77 (SKIP) as designed.

- [ ] **Step 2: Update the change-surface note in the offload design doc**

Append to `docs/maintainer/weight-offload.md` section 5, under row 1, a status marker: host tensor placement (binder/materializer/typed_binding) is now implemented as the first slice; the host store is plain host memory and pinning (`cudaHostRegister`) is deferred to the staging-arena phase.

- [ ] **Step 3: Commit**

```bash
git add docs/maintainer/weight-offload.md
git commit -m "docs(offload): mark artifact host placement as implemented"
```

---

## Follow-on plans (out of scope here)

Each is a separate plan and depends on this one:

1. **Tensor dual-address** — `Tensor`/`Weight` carry an optional host address beside device `data` (`src/core/tensor.h`, `tests/test_tensor.cpp`).
2. **27B residency profile** — `LoadedModelData` binds offloaded tensors to the host store (`src/targets/qwen3_6_27b/impl/load/bindings.cpp`).
3. **Staging arena + graph capture interleave** — fixed-slot staging arena and H2D memcpy nodes inside the captured decode graphs (`program_impl.h`, schedule capture).
4. **Engine sizing + CLI** — offload option, resident-bytes budget, memory/load summary host reporting.
5. **35B-A3B MoE expert streaming** (bonus) — per-layer router read-back and active-expert copy.

## Self-review

- **Spec coverage:** design doc §5 surface 1 (binder/plan), surface 1 (materializer host store + stats) and the typed-binding dispatch are fully covered by Tasks 1-4; §4.1 host store matches (plain memory in this phase, pinning deferred per plan Task 5 step 2). All other surfaces are explicitly listed as follow-on plans.
- **Placeholder scan:** every code step contains concrete code; no TBD/TODO/“handle edge cases”.
- **Type consistency:** `TensorPlacement::Host`, `materialize_tensor_on_host`, `HostTensorMaterialization`, `host_capacity_bytes`, `host_data`, `host_bytes`, `host_capacity_bytes` (stats) are named identically across all tasks and the test. `checked_add`, `align_up`, `tensor_alignment`, `PayloadSpan`, `ObjectHandle` all reference existing definitions.
