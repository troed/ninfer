# Tensor Dual-Address Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `Tensor` and `Weight` an optional host address beside their device `data`/`payload` pointers, so the artifact layer can produce views that describe host-resident (offloaded) tensors, and propagate that address through the tensor view operations.

**Architecture:** This is surface 2 of `docs/maintainer/weight-offload.md` ("Tensor type | `src/core/tensor.h` | `Tensor`/`Weight` carry an optional host address beside device `data`"). Surface 1 (already merged) made the binder/materializer able to keep tensor payloads in a host store and exposed `MaterializedArtifact::host_data(handle)`. This slice adds the type-level capability: `Tensor::host` mirrors `Tensor::data` through the constructor and `view`/`reshape`/`slice`/`permute` (slice advances both by the same byte offset; view/reshape/permute preserve the base); `Weight::host` mirrors the payload base. `materialized_tensor`/`materialized_weight` populate the host side from `MaterializedArtifact::host_data_or_null`, and device-side plane pointers stay null for host-only objects so a host-backed `Weight` is a valid, copyable descriptor. Resident tensors keep `host == nullptr`. The residency profile (surface 3) and staging/slot binding (surface 4) build on this; kernels are not touched.

**Tech Stack:** C++20, CUDA 13.1 (`sm_120a`), CMake/Ninja, existing `ninfer_tensor_test` (CPU-only) and `ninfer_artifact_materialization_test` (CUDA-gated, SKIP return 77).

**Repo context:** Work on `master` at commit `1de9948` (plan 1 merged). Build dir `build/` exists and is current. `ninfer_tensor_test` links no CUDA; `ninfer_artifact_materialization_test` is a device test. Device tests unrelated to this slice that fail on the local RTX 5060 Ti 16 GB hardware (pre-existing, not regressions): `ninfer_qwen3_6_frontend_test`, `ninfer_gdn_gating_proj_test`, `ninfer_swa_test`. Verification gate: `ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test`.

---

### Task 1: Failing test for host-address propagation on `Tensor`

**Files:**
- Modify: `tests/test_tensor.cpp`
- Test: `tests/test_tensor.cpp`

- [ ] **Step 1: Add the failing scenarios**

Add a second storage buffer and dual-address scenarios inside `main()` in `tests/test_tensor.cpp`. Insert the host storage declaration immediately after `auto* base = storage;` (line 63):

```cpp
    alignas(16) unsigned char host_storage[512] = {};
    auto* host_base                            = host_storage;
```

Insert the dual-address block immediately after the `reshaped.numel` check (line 121, after `failures += expect_i64(reshaped.numel(), 24, "reshaped.numel");`) and before the first `expect_invalid` call:

```cpp
    ninfer::Tensor dual(base, host_base, ninfer::DType::BF16, {2, 3, 4});
    if (dual.host != host_base) {
        ++failures;
        std::cerr << "dual constructor did not record the host address\n";
    }

    ninfer::Tensor dual_viewed = dual.view({4, 6});
    if (dual_viewed.host != host_base) {
        ++failures;
        std::cerr << "view dropped the host address\n";
    }

    ninfer::Tensor dual_sliced = dual.slice(1, 1, 2);
    if (dual_sliced.host != host_base + 4) {
        ++failures;
        std::cerr << "slice did not advance the host address by dim-1 stride\n";
    }

    ninfer::Tensor dual_permuted = dual.permute({2, 1, 0, 3});
    if (dual_permuted.host != host_base) {
        ++failures;
        std::cerr << "permute dropped the host address\n";
    }

    ninfer::Tensor dual_reshaped = dual.reshape({6, 4});
    if (dual_reshaped.host != host_base) {
        ++failures;
        std::cerr << "reshape dropped the host address\n";
    }
```

Note: `slice(1, 1, 2)` on a `{2,3,4}` BF16 tensor has `nb[1] == 4`, so the offset is `1 * 4 == 4` bytes; both `data` and `host` must advance by exactly 4 bytes.

- [ ] **Step 2: Run the test to verify it fails to compile**

Run: `cmake --build build --target ninfer_tensor_test`
Expected: compile failure — `'host' is not a member of 'ninfer::Tensor'` and no matching constructor for `Tensor(void*, void*, DType, ...)`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_tensor.cpp
git commit -m "test(core): add host-address propagation scenarios for Tensor"
```

---

### Task 2: Host address on `Tensor` and `Weight` with view propagation

**Files:**
- Modify: `src/core/tensor.h`
- Modify: `src/core/tensor.cpp`
- Test: `tests/test_tensor.cpp`

- [ ] **Step 1: Add the `host` member and constructor to `Tensor`**

In `src/core/tensor.h`, add the `host` member right after `data` and a second constructor:

```cpp
struct Tensor {
    void* data         = nullptr;
    const void* host   = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);
    Tensor(void* data, const void* host, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};
```

Add `const void* host = nullptr;` to the `Weight` struct, immediately after `payload`:

```cpp
struct Weight {
    const void* payload            = nullptr;
    const void* host               = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    ...
```

- [ ] **Step 2: Implement the constructor and propagation**

In `src/core/tensor.cpp`, replace the existing constructor with a delegating pair, and update `view` and `slice`. The constructor becomes:

```cpp
Tensor::Tensor(void* data_in, DType dtype_in, std::initializer_list<std::int32_t> shape)
    : Tensor(data_in, nullptr, dtype_in, shape) {}

Tensor::Tensor(void* data_in, const void* host_in, DType dtype_in,
               std::initializer_list<std::int32_t> shape)
    : data(data_in), host(host_in), dtype(dtype_in) {
    const auto normalized = normalize_shape(shape);
    for (int i = 0; i < 4; ++i) { ne[i] = normalized[i]; }
    set_contiguous_strides(*this);
}
```

`view` propagates the host address:

```cpp
Tensor Tensor::view(std::initializer_list<std::int32_t> shape) const {
    if (!is_contiguous()) { throw std::invalid_argument("view requires a contiguous tensor"); }

    const auto normalized = normalize_shape(shape);
    if (shape_numel(normalized) != numel()) {
        throw std::invalid_argument("view element count mismatch");
    }

    return Tensor(data, host, dtype, shape);
}
```

`reshape` delegates to `view` (already correct: `return view(shape);`).

`slice` advances `host` by the same byte offset as `data`:

```cpp
Tensor Tensor::slice(int dim, std::int32_t start, std::int32_t len) const {
    if (dim < 0 || dim >= 4) { throw std::invalid_argument("slice dim out of range"); }
    if (start < 0 || len <= 0 || start > ne[dim] || len > ne[dim] - start) {
        throw std::invalid_argument("slice range out of bounds");
    }

    Tensor out                = *this;
    const std::int64_t offset = checked_mul_i64(static_cast<std::int64_t>(start), out.nb[dim]);
    auto* bytes_ptr           = static_cast<unsigned char*>(out.data);
    if (bytes_ptr != nullptr) { bytes_ptr += offset; }
    out.data = bytes_ptr;
    auto* host_bytes_ptr = static_cast<const unsigned char*>(out.host);
    if (host_bytes_ptr != nullptr) { host_bytes_ptr += offset; }
    out.host = host_bytes_ptr;
    out.ne[dim] = len;
    return out;
}
```

`permute` copies the struct (`Tensor out = *this;`) and only reorders `ne`/`nb`, so `host` is preserved automatically; no change needed.

- [ ] **Step 3: Run the test to verify it passes**

Run: `cmake --build build --target ninfer_tensor_test`
Run: `/usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_tensor_test'`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/core/tensor.h src/core/tensor.cpp
git commit -m "feat(core): carry an optional host address on Tensor and Weight"
```

---

### Task 2b: Propagate host through the remaining whole-object derivation sites

**Files:**
- Modify: `src/core/weight.h`
- Modify: `src/ops/wrapper/gdn_input_proj.cpp`
- Test: `tests/test_tensor.cpp` (uses `as_dense`)

**Context:** `Tensor::host`/`Weight::host` mirror the base of the object. The four `Tensor` view ops propagate host (Task 2). Two other derivation sites rebuild a value-type view from base pointers and would silently drop host once a weight is host-backed: the canonical `Weight -> Tensor` projection `as_dense` (`src/core/weight.h:10-23`) and the GDN conv helper `flatten_columns` (`src/ops/wrapper/gdn_input_proj.cpp:114-116`). `row_view` in both target bindings does NOT need a change: it keeps `payload` at the object base, and `Weight::host` mirrors `payload`, so a row view correctly keeps the object-base host pointer. This task closes the two real gaps so the dual-address contract is coherent repo-wide before Task 4 populates host addresses.

- [ ] **Step 1: Add a failing test for `as_dense` host propagation**

In `tests/test_tensor.cpp`, after the existing dual-address block (after the `dual_reshaped.host` check), add:

```cpp
    ninfer::Weight dual_weight{};
    dual_weight.qtype        = ninfer::QType::BF16_CTRL;
    dual_weight.layout       = ninfer::QuantLayout::Contiguous;
    dual_weight.ndim         = 2;
    dual_weight.shape[0]     = 2;
    dual_weight.shape[1]     = 3;
    dual_weight.qdata        = base;
    dual_weight.host         = host_base;
    const ninfer::Tensor dense = ninfer::as_dense(dual_weight);
    if (dense.host != host_base) {
        ++failures;
        std::cerr << "as_dense dropped the host address\n";
    }
```

Run: `cmake --build build --target ninfer_tensor_test`
Expected: FAIL — `dense.host` is `nullptr` (the `as_dense` projection uses the 3-arg constructor).

- [ ] **Step 2: Propagate host in `as_dense`**

In `src/core/weight.h`, pass the host through the 4-arg constructor at all four rank branches:

```cpp
inline Tensor as_dense(const Weight& w) {
    const DType dt = (w.qtype == QType::FP32_CTRL) ? DType::FP32 : DType::BF16;
    void* p        = const_cast<void*>(w.qdata);
    switch (w.ndim) {
    case 1:
        return Tensor(p, w.host, dt, {w.shape[0]});
    case 2:
        return Tensor(p, w.host, dt, {w.shape[0], w.shape[1]});
    case 3:
        return Tensor(p, w.host, dt, {w.shape[0], w.shape[1], w.shape[2]});
    default:
        return Tensor(p, w.host, dt, {w.shape[0], w.shape[1], w.shape[2], w.shape[3]});
    }
}
```

- [ ] **Step 3: Propagate host in `flatten_columns`**

In `src/ops/wrapper/gdn_input_proj.cpp:114-116`:

```cpp
Tensor flatten_columns(const Tensor& tensor, std::int32_t rows, ConvGeometry geometry) {
    return Tensor(tensor.data, tensor.host, tensor.dtype, {rows, geometry.aggregate_columns});
}
```

- [ ] **Step 4: Run the tests to verify**

Run: `cmake --build build --target ninfer_tensor_test`
Run: `/usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_tensor_test'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/weight.h src/ops/wrapper/gdn_input_proj.cpp tests/test_tensor.cpp
git commit -m "feat(core): propagate host address through weight projections"
```

---

### Task 3: Failing test for dual-address views from the artifact layer

**Files:**
- Modify: `tests/test_artifact_materialization.cpp`
- Test: `tests/test_artifact_materialization.cpp`

- [ ] **Step 1: Add the failing assertions**

In `tests/test_artifact_materialization.cpp`, inside the host-fixture block (which starts at `auto host_fixture = write_host_fixture();` and ends at the closing brace after the stats check on line 235), add the following block immediately after the `host materialization statistics are incomplete` require (line 235) and before the closing `}` of the block:

```cpp
            const auto host_view = ninfer::artifact::materialized_tensor(
                host_materialized, host_tensor, ninfer::artifact::NumericFormat::BF16, {8});
            require(host_view.data == nullptr &&
                        host_view.host == host_materialized.host_data(host_tensor),
                    "host tensor view carries the wrong addresses");

            const auto host_weight = ninfer::artifact::materialized_weight(
                host_materialized, host_tensor, ninfer::artifact::NumericFormat::BF16, 8, 1);
            require(host_weight.payload == nullptr && host_weight.qdata == nullptr &&
                        host_weight.host == host_materialized.host_data(host_tensor),
                    "host weight carries the wrong addresses");

            const auto host_device_view = ninfer::artifact::materialized_tensor(
                host_materialized, host_device_tensor, ninfer::artifact::NumericFormat::BF16, {2});
            require(host_device_view.data ==
                        host_materialized.device_data(host_device_tensor) &&
                        host_device_view.host == nullptr,
                    "device tensor view carries the wrong addresses");
```

Rationale for the shapes: the host fixture's `weights/host` object is `{8}` BF16 = 16 bytes; as a contiguous weight `8` rows x `1` column = 16 bytes, matching `kHostTensor.size()`. The device object `weights/device` is `{2}` BF16 = 4 bytes, matching `kHostFixtureDeviceTensor.size()`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build --target ninfer_artifact_materialization_test`
Run: `/usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_artifact_materialization_test'`
Expected: FAIL — currently `materialized_tensor` and `materialized_weight` call `device_data(handle)` which throws for the host-only object (`object handle does not name a materialized tensor`), so the process exits 1 with that message.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_artifact_materialization.cpp
git commit -m "test(artifact): assert dual-address tensor and weight views"
```

---

### Task 4: Populate host addresses in `materialized_tensor` and `materialized_weight`

**Files:**
- Modify: `src/artifact/materializer.h`
- Modify: `src/artifact/materializer.cpp`
- Modify: `src/artifact/typed_binding.cpp`
- Test: `tests/test_artifact_materialization.cpp`

- [ ] **Step 1: Add non-throwing accessors to `MaterializedArtifact`**

In `src/artifact/materializer.h`, add two accessors after `host_data` (line 44):

```cpp
    void* device_data(ObjectHandle handle) const;
    void* host_data(ObjectHandle handle) const;
    void* device_data_or_null(ObjectHandle handle) const noexcept;
    void* host_data_or_null(ObjectHandle handle) const noexcept;
    std::span<const std::byte> resource_bytes(ObjectHandle handle) const;
    std::vector<std::byte> take_resource_bytes(ObjectHandle handle);
```

In `src/artifact/materializer.cpp`, add the implementations after `host_data` (line 83):

```cpp
void* MaterializedArtifact::device_data_or_null(ObjectHandle handle) const noexcept {
    if (handle.index >= objects_.size()) { return nullptr; }
    return objects_[handle.index].device;
}

void* MaterializedArtifact::host_data_or_null(ObjectHandle handle) const noexcept {
    if (handle.index >= objects_.size()) { return nullptr; }
    return objects_[handle.index].host;
}
```

- [ ] **Step 2: Set the host address in the binding helpers**

In `src/artifact/typed_binding.cpp`, `contiguous_weight` becomes:

```cpp
Weight contiguous_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                         NumericFormat format, std::int32_t rows, std::int32_t columns) {
    Weight out{};
    out.payload       = materialized.device_data_or_null(handle);
    out.qdata         = out.payload;
    out.host          = materialized.host_data_or_null(handle);
    out.payload_bytes = static_cast<std::uint64_t>(rows) * columns * dtype_size(dtype_for(format));
    out.qtype         = qtype_for(format);
    out.layout        = QuantLayout::Contiguous;
    out.n             = rows;
    out.k             = columns;
    out.ndim          = 2;
    out.shape[0]      = rows;
    out.shape[1]      = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}
```

`row_split_weight` becomes (guarding the plane-pointer arithmetic against a null device base for host-only objects):

```cpp
Weight row_split_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                        NumericFormat format, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const RowSplitGeometry geometry          = row_split_geometry(format, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data_or_null(handle));

    Weight out{};
    out.payload          = bytes;
    out.host             = materialized.host_data_or_null(handle);
    out.payload_bytes    = geometry.encoded_bytes;
    out.high_plane_bytes = geometry.high_plane_bytes;
    out.qtype            = qtype_for(format);
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    out.qdata            = bytes;
    out.qhigh = bytes == nullptr || geometry.high_plane_bytes == 0
                    ? nullptr
                    : bytes + geometry.high_plane_offset;
    out.scales      = bytes == nullptr ? nullptr : bytes + geometry.scale_plane_offset;
    out.n           = rows;
    out.k           = columns;
    out.group       = static_cast<std::int32_t>(geometry.group_size);
    out.scale_dtype = DType::FP16;
    out.ndim        = 2;
    out.shape[0]    = rows;
    out.shape[1]    = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = static_cast<std::int32_t>(geometry.padded_columns);
    return out;
}
```

`materialized_tensor` becomes:

```cpp
Tensor materialized_tensor(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format,
                           std::initializer_list<std::int32_t> internal_shape) {
    return Tensor(materialized.device_data_or_null(handle),
                  materialized.host_data_or_null(handle), dtype_for(format), internal_shape);
}
```

`materialized_weight` itself is unchanged (it already dispatches by `storage_layout_for(format)`).

- [ ] **Step 3: Run the tests to verify they pass**

Run: `cmake --build build --target ninfer_artifact_materialization_test ninfer_artifact_reader_test ninfer_tensor_test`
Run: `/usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_artifact_|ninfer_tensor_test'`
Expected: all PASS (reader, materialization, tensor).

- [ ] **Step 4: Commit**

```bash
git add src/artifact/materializer.h src/artifact/materializer.cpp src/artifact/typed_binding.cpp
git commit -m "feat(artifact): populate host addresses on tensor and weight views"
```

---

### Task 5: Full verification and design-doc status marker

**Files:**
- Modify: `docs/maintainer/weight-offload.md`

- [ ] **Step 1: Full-tree build**

Run: `cmake --build build`
Expected: full build succeeds with no errors.

- [ ] **Step 2: Filtered test suite**

Run: `/usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test'`
Expected: all PASS (reader, materialization, arena, tensor).

- [ ] **Step 3: Whitespace check**

Run: `git diff --check`
Expected: silent, exit 0.

- [ ] **Step 4: Update the design doc status**

In `docs/maintainer/weight-offload.md` section 5, extend the Status paragraph (currently lines 123-127) to record surface 2. The new text:

```markdown
Status: surface 1 is implemented as the first slice — binder host placement
(`TensorPlacement::Host`, host spans in `MaterializationPlan`), materializer host store
(`MaterializedArtifact::host_data(handle)` + host-bytes stats), and `bind_tensor` host dispatch.
The host store is plain host memory; pinning (`cudaHostRegister`) is deferred to the staging-arena
phase (surface 4). Surface 2 is implemented — `Tensor`/`Weight` carry an optional host address
(`Tensor::host`, `Weight::host`) propagated through the tensor view operations, and
`materialized_tensor`/`materialized_weight` populate the host side from the host store while
leaving device-side pointers null for host-only objects. The 27B residency profile (surface 3) is
not yet implemented.
```

- [ ] **Step 5: Commit**

```bash
git add docs/maintainer/weight-offload.md
git commit -m "docs(offload): mark tensor dual-address as implemented"
```

- [ ] **Step 6: Final integration review**

Run `git log --oneline -6` to confirm the five commits, then request the final integration code review of the branch before merging (finishing-a-development-branch).
