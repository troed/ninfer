# Partial FFN Residency Knob + MTP-Under-Offload Verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable number of device-resident FFN layers (`--n-ffn-layers N`, default 0 = all stream) to the 27B offload path so users can tune the VRAM/throughput tradeoff, and verify MTP decode works under FfnOffload against a llama.cpp baseline.

**Architecture:** A count `resident_ffn_layers` (public `EngineOptions` field, default 0; 0 = current FfnOffload all-stream, 64 = all resident) is threaded `EngineOptions → Package::plan_load → bind_artifact → per-layer ffn_placement`. `ffn_placement` becomes a 3-arg helper choosing `Host`/`Device` per text layer. `LoadedModelData` offload probe changes from "layer 0 host-placed" to "any layer host-placed", and `stage_mlp` skips resident layers (no `point_at_slot`, no slot-map entry) — the staging interleave already keys off `Weight::host != nullptr`, so resident layers simply never receive a staging copy. No change to `variant.cpp`, graph capture, memory summaries, or KV planning.

**Tech Stack:** C++20, CUDA 13.3, CMake/Ninja, `sm_120a`.

---

## Key existing facts (verified on master `8d10ea8`)

- `include/ninfer/types.h:85`: `WeightResidency weight_residency   = WeightResidency::AllResident;` in `EngineOptions` (the field to add `resident_ffn_layers` after). `SpeculativeBackend{None, Mtp, DFlash}` at :59; `SpeculativeOptions{backend, draft_tokens, proposal_head}` at :65-68; `ProposalHead::Optimized` used by the prefix test.
- `src/targets/qwen3_6_27b/impl/load/bindings.h:23`: `inline constexpr std::size_t kTextLayers = 64;`. `bind_artifact` declaration at :122-124: `ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile, qwen3_6::StartupFeatures features, ResidencyProfile residency = ResidencyProfile::AllResident);`
- `src/targets/qwen3_6_27b/impl/load/bindings.cpp`:
  - `ffn_placement(ResidencyProfile residency) noexcept` at :46-49 (2-arg, superseded by 3-arg).
  - `bind_groupwise_text_layers(binder, out, residency)` at :249-306, MLP binds at :299-304 (`ffn_placement(residency)` at :301, :304), loop var `layer` in scope.
  - `bind_nvfp4_text_layers(binder, out, residency)` at :308-384, MLP binds at :376-382 (`ffn_placement(residency)` at :379, :382), loop var `layer` in scope.
  - `bind_artifact` definition at :407-428 (forward-only residency, dispatch at :418-428).
  - `LoadedModelData` ctor at :485+: offload probe at :492-493 (`backing.host_data_or_null(plan.text_layers[0].mlp.gate_up.object) != nullptr`); staging arena sizing :494-503; `stage_mlp` lambda at :515-531 (current body: `if (!offload) { return; }` then buffer + two `point_at_slot` + two `push_back`).
  - `<algorithm>` already included (grep confirmed), so `std::any_of` is available.
- `src/targets/qwen3_6_27b/impl/package.cpp:99-107`: `Package::plan_load` — builds `residency` from `options.weight_residency`, then `detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(options), residency)` (4-arg).
- CLI: `apps/cli/options.h:27` `ninfer::WeightResidency weight_residency = ...;` in `struct Options`; `apps/cli/options.cpp` `--weight-residency` branch at :140-142, usage line at :86 `"       [--kv-dtype bf16|int8] [--weight-residency all|ffn] [--spec mtp|dflash --draft-tokens N]\n"`; `apps/cli/main.cpp:270` `engine_options.weight_residency = cli.weight_residency;`.
- Serve: `src/serve/serve_options.h:41` `WeightResidency weight_residency       = WeightResidency::AllResident;`; `src/serve/serve_options.cpp` `--weight-residency` branch after `--kv-dtype` (~:186), usage line at :79; `src/serve/generation_service.cpp:262` `engine_options.weight_residency    = options_.weight_residency;`. Serve parse uses `parse_nonnegative_int(require_value("--n-ffn-layers"), "n-ffn-layers")`.
- Tests: `tests/targets/qwen3_6_27b/test_residency.cpp` — `verify_residency(path, profile)` (full-offload arms) + `staged_mlp_round_matches_host(device, profile, post_mixer)` helper at :43-63 + `main()` at :216+ running per present artifact (`NINFER_QWEN3_8_27B_WEIGHTS`→GroupwiseIntW8Endpoints, `NINFER_QWEN3_6_27B_WEIGHTS`→GroupwiseInt, `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS`→Nvfp4), 77 if none. `tests/targets/qwen3_6_27b/test_engine_offload_real.cpp` — `offload_options(artifact)` sets `weight_residency = FfnOffload`; `verify_offloaded_product(engine, expected_staging_bytes)` checks `load.host_bytes > 0`, `load.host_capacity_bytes > 0`, `memory.staging.capacity_bytes == expected_staging_bytes`, `host_store_bytes > 0`, `weights.capacity_bytes/used_bytes > 0`, `kv_capacity > 0`; `exercise_offloaded_generation(engine)` runs 4-token greedy via `prepare_tokens({198,198})`, asserts `generated_token_ids.size() == 4` and `finish_reason == OutputLimit`; `exercise_artifact(artifact, expected_staging_bytes)`; `main()` per artifact with `kGroupwiseStagingBytes = 306380800`, `kNvfp4StagingBytes = 300811264`, 77 if none.
- Staging arena: `2 * staging_unit_bytes` always (independent of how many layers stream); groupwise unit = 153,190,400 B, arena = 306,380,800 B; NVFP4 arena = 300,811,264 B.
- Per-layer FFN raw payload bytes (for `LoadSummary.host_bytes` expectations): groupwise gate_up 94,699,520 + down 58,490,880 = **153,190,400**; NVFP4 gate_up 100,270,084 + down 50,135,044 = **150,405,128**. `host_bytes` = sum of `placement.bytes` for host-placed tensors. With 41 resident layers (23 streamed): groupwise host_bytes = 23 × 153,190,400 = **3,523,379,200**; NVFP4 = 23 × 150,405,128 = **3,459,317,944**.
- Full-attention text layers are at indices `layer >= 3 && (layer - 3) % 4 == 0` (3, 7, 11, ..., 63 = 16 total). `LoadedModelData` fills `full_layers` and `gdn_layers` using running counters in text-layer order.
- Environment: RTX 5060 Ti 16GB, CUDA 13.3, `/usr/bin/ctest` required, `NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer`. Master HEAD `8d10ea8`, working tree clean.
- Pre-existing GPU test failures (not regressions): `ninfer_qwen3_6_frontend_test` (missing tokenizer), `ninfer_gdn_gating_proj_test` (cooperative-launch), `ninfer_swa_test` (tolerance).

---

### Task 1: failing partial-residency and MTP tests (RED)

**Files:**
- Modify: `tests/targets/qwen3_6_27b/test_residency.cpp`
- Modify: `tests/targets/qwen3_6_27b/test_engine_offload_real.cpp`

- [x] **Step 1: Add the partial-residency arm to the residency test**

In `tests/targets/qwen3_6_27b/test_residency.cpp`, after the `verify_residency` function (before `} // namespace`), add:

```cpp
int verify_partial_residency(const std::filesystem::path& path, WeightsProfile profile) {
    constexpr std::size_t kResident = 41;
    constexpr std::size_t kStreamed = kTextLayers - kResident;

    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    ArtifactLoadPlan plan =
        bind_artifact(binder, profile, {}, ResidencyProfile::FfnOffload, kResident);
    if (plan.materialization.host_tensor_objects.size() != 2 * kStreamed) {
        std::cerr << "partial residency host tensor count mismatch: got "
                  << plan.materialization.host_tensor_objects.size() << " expected "
                  << 2 * kStreamed << '\n';
        return 1;
    }
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& text = plan.bindings.text_layers[layer];
        const bool resident       = layer < kResident;
        if (is_host_placed(plan.materialization, text.mlp.gate_up.object) == resident ||
            is_host_placed(plan.materialization, text.mlp.down.object) == resident) {
            std::cerr << "a partial-residency MLP weight has the wrong placement\n";
            return 1;
        }
    }

    ninfer::DeviceContext device(0);
    auto materialized =
        ninfer::artifact::materialize(reader, plan.materialization, device, nullptr);
    std::array<std::pair<ninfer::artifact::ObjectHandle, ninfer::artifact::ObjectHandle>,
               kStreamed>
        streamed_handles;
    for (std::size_t layer = kResident; layer < kTextLayers; ++layer) {
        streamed_handles[layer - kResident] = {
            plan.bindings.text_layers[layer].mlp.gate_up.object,
            plan.bindings.text_layers[layer].mlp.down.object};
    }
    LoadedModelData data(std::move(plan.bindings), std::move(materialized));
    if (data.runtime.staging_arena == nullptr) {
        std::cerr << "partial residency did not allocate a staging arena\n";
        return 1;
    }
    if (data.runtime.staged_weights.size() != 2 * kStreamed) {
        std::cerr << "partial slot map size mismatch: got " << data.runtime.staged_weights.size()
                  << " expected " << 2 * kStreamed << '\n';
        return 1;
    }
    const auto* arena_begin = static_cast<const std::uint8_t*>(data.runtime.staging_arena->base());
    const std::uintptr_t arena_lo = reinterpret_cast<std::uintptr_t>(arena_begin);
    const std::uintptr_t arena_hi = arena_lo + data.runtime.staging_arena->capacity();
    const std::size_t unit_bytes  = data.runtime.staging_arena->capacity() / 2;
    for (std::size_t layer = kResident; layer < kTextLayers; ++layer) {
        const auto& gate_up = data.runtime.staged_weights[2 * (layer - kResident)];
        const auto& down    = data.runtime.staged_weights[2 * (layer - kResident) + 1];
        if (gate_up.host_source !=
                data.backing.host_data(streamed_handles[layer - kResident].first) ||
            down.host_source != data.backing.host_data(streamed_handles[layer - kResident].second)) {
            std::cerr << "a partial slot map entry does not source from the host store object\n";
            return 1;
        }
        const std::uintptr_t expected_gate =
            arena_lo + static_cast<std::uintptr_t>((layer % 2) * unit_bytes);
        const std::uintptr_t expected_down =
            expected_gate + ((gate_up.bytes + 255U) & ~std::uint64_t{255});
        if (reinterpret_cast<std::uintptr_t>(gate_up.slot) != expected_gate ||
            reinterpret_cast<std::uintptr_t>(down.slot) != expected_down) {
            std::cerr << "a partial slot address does not follow the double-buffer layer layout\n";
            return 1;
        }
    }

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const bool resident = layer < kResident;
        const DensePostMixerPayload* post_mixer = nullptr;
        if (plan.bindings.text_layers[layer].is_full_attention) {
            post_mixer = &data.runtime.full_layers.at(full_index++).post_mixer;
        } else {
            post_mixer = &data.runtime.gdn_layers.at(gdn_index++).post_mixer;
        }
        if (resident) {
            if (post_mixer->gate_up.host != nullptr || post_mixer->gate_up.payload == nullptr ||
                post_mixer->down.host != nullptr || post_mixer->down.payload == nullptr) {
                std::cerr << "a resident MLP weight does not carry device-only addresses\n";
                return 1;
            }
        } else {
            if (post_mixer->gate_up.host == nullptr || post_mixer->gate_up.payload == nullptr ||
                post_mixer->down.host == nullptr || post_mixer->down.payload == nullptr) {
                std::cerr << "a streamed MLP weight does not carry host and slot addresses\n";
                return 1;
            }
            const std::uintptr_t gu =
                reinterpret_cast<std::uintptr_t>(post_mixer->gate_up.payload);
            const std::uintptr_t dn =
                reinterpret_cast<std::uintptr_t>(post_mixer->down.payload);
            if (gu < arena_lo || gu >= arena_hi || dn < arena_lo || dn >= arena_hi) {
                std::cerr << "a streamed MLP weight does not point into the staging arena\n";
                return 1;
            }
        }
    }
    if (full_index != data.runtime.full_layers.size() ||
        gdn_index != data.runtime.gdn_layers.size()) {
        std::cerr << "partial residency layer mapping is incomplete\n";
        return 1;
    }
    if (staged_mlp_round_matches_host(device, profile,
                                      data.runtime.full_layers.back().post_mixer) != 0) {
        return 1;
    }
    return 0;
}
```

(`full_layers.back()` is text layer 63, the last full-attention layer, streamed because 63 >= 41 — so `staged_mlp_round_matches_host` verifies a real streamed layer reproduces its host payload.)

In `main()`, after each `verify_residency(...) != 0` check for a present artifact, add the corresponding partial check. The existing `main()` structure calls `verify_residency(groupwise, WeightsProfile::GroupwiseIntW8Endpoints)` etc. inside `if (!std::filesystem::is_regular_file(...))` guards. Update each present-artifact branch to also call `verify_partial_residency(path, profile)` and `return 1` on nonzero, e.g.:

```cpp
    if (std::filesystem::is_regular_file(groupwise)) {
        if (verify_residency(groupwise, WeightsProfile::GroupwiseIntW8Endpoints) != 0 ||
            verify_partial_residency(groupwise, WeightsProfile::GroupwiseIntW8Endpoints) != 0) {
            return 1;
        }
    }
```

(and the same for `legacy` → `GroupwiseInt` and `nvfp4` → `Nvfp4`).

- [x] **Step 2: Add the partial and MTP arms to the engine offload test**

In `tests/targets/qwen3_6_27b/test_engine_offload_real.cpp`:

1. After `offload_options`, add:

```cpp
ninfer::EngineOptions partial_offload_options(const char* artifact) {
    ninfer::EngineOptions options = offload_options(artifact);
    options.resident_ffn_layers   = 41;
    return options;
}

ninfer::EngineOptions mtp_offload_options(const char* artifact) {
    ninfer::EngineOptions options = offload_options(artifact);
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}
```

2. After `exercise_artifact`, add:

```cpp
int exercise_partial_artifact(const char* artifact, std::size_t expected_staging_bytes,
                              std::uint64_t expected_host_bytes) {
    ninfer::Engine engine(partial_offload_options(artifact));
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.host_bytes != expected_host_bytes || load.host_capacity_bytes < expected_host_bytes) {
        std::cerr << "partial offload host store does not reflect 23 streamed FFN layers\n";
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.staging.capacity_bytes != expected_staging_bytes) {
        std::cerr << "partial staging arena capacity is " << memory.staging.capacity_bytes
                  << ", expected " << expected_staging_bytes << '\n';
        return 1;
    }
    if (memory.host_store_bytes < expected_host_bytes || memory.weights.capacity_bytes == 0 ||
        memory.kv_capacity == 0) {
        std::cerr << "partial offload Engine construction has incomplete memory backing\n";
        return 1;
    }
    return exercise_offloaded_generation(engine);
}

int exercise_mtp_artifact(const char* artifact) {
    ninfer::Engine engine(mtp_offload_options(artifact));
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 4;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    const std::vector<ninfer::TokenId> prompt{198, 198};
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), options);
    if (result.generated_token_ids.size() != 4 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit ||
        !result.speculative.enabled || result.speculative.rounds == 0 ||
        result.speculative.drafted_tokens == 0) {
        std::cerr << "MTP offload decode did not complete a draft-verify round\n";
        return 1;
    }
    return 0;
}
```

3. In `main()`, extend each present-artifact branch to also run the partial and MTP arms. Constants: `kGroupwisePartialHostBytes = 3523379200`, `kNvfp4PartialHostBytes = 3459317944`. Example (full `main` body rewrite):

```cpp
int main() {
    const char* groupwise = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    const char* legacy    = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* nvfp4     = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    constexpr std::size_t kGroupwiseStagingBytes  = 306380800;
    constexpr std::size_t kNvfp4StagingBytes      = 300811264;
    constexpr std::uint64_t kGroupwisePartialHost = 3523379200;
    constexpr std::uint64_t kNvfp4PartialHost     = 3459317944;
    if ((groupwise == nullptr || *groupwise == '\0') && (legacy == nullptr || *legacy == '\0') &&
        (nvfp4 == nullptr || *nvfp4 == '\0')) {
        std::cout << "skip: no 27B weight artifact is configured\n";
        return 77;
    }
    if (groupwise != nullptr && *groupwise != '\0') {
        if (exercise_artifact(groupwise, kGroupwiseStagingBytes) != 0 ||
            exercise_partial_artifact(groupwise, kGroupwiseStagingBytes,
                                      kGroupwisePartialHost) != 0 ||
            exercise_mtp_artifact(groupwise) != 0) {
            return 1;
        }
    }
    if (legacy != nullptr && *legacy != '\0') {
        if (exercise_artifact(legacy, kGroupwiseStagingBytes) != 0 ||
            exercise_partial_artifact(legacy, kGroupwiseStagingBytes,
                                      kGroupwisePartialHost) != 0 ||
            exercise_mtp_artifact(legacy) != 0) {
            return 1;
        }
    }
    if (nvfp4 != nullptr && *nvfp4 != '\0') {
        if (exercise_artifact(nvfp4, kNvfp4StagingBytes) != 0 ||
            exercise_partial_artifact(nvfp4, kNvfp4StagingBytes, kNvfp4PartialHost) != 0 ||
            exercise_mtp_artifact(nvfp4) != 0) {
            return 1;
        }
    }
    std::cout << "ok\n";
    return 0;
}
```

- [x] **Step 3: Build to verify it fails**

Run: `cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=120a && cmake --build build --target ninfer_qwen3_6_27b_residency_test ninfer_qwen3_6_27b_engine_offload_real_test`

Expected: FAIL to compile with exactly these error kinds:
- `test_residency.cpp: no matching function for call to 'bind_artifact(...)'` — the 5-arg call with `kResident`; current `bind_artifact` has 4 params.
- `test_engine_offload_real.cpp: 'struct ninfer::EngineOptions' has no member named 'resident_ffn_layers'` (×3 call sites).
No other error kinds. (The MTP arm itself only uses existing members, so it compiles; the whole test TU is red because of the other two arms.)

- [x] **Step 4: Commit**

```bash
git add tests/targets/qwen3_6_27b/test_residency.cpp tests/targets/qwen3_6_27b/test_engine_offload_real.cpp
git commit -m "test(targets): add partial FFN residency and MTP offload scenarios"
```

---

### Task 2: public field + target plumbing (GREEN core)

**Files:**
- Modify: `include/ninfer/types.h`
- Modify: `src/targets/qwen3_6_27b/impl/load/bindings.h`
- Modify: `src/targets/qwen3_6_27b/impl/load/bindings.cpp`
- Modify: `src/targets/qwen3_6_27b/impl/package.cpp`

- [x] **Step 1: Add the public field**

In `include/ninfer/types.h`, in `EngineOptions`, immediately after `WeightResidency weight_residency   = WeightResidency::AllResident;` (line 85), insert:

```cpp
    std::uint32_t resident_ffn_layers  = 0;
```

(aligning `=` to column 40, matching `kv_cache`/`weight_residency`.)

- [x] **Step 2: bind_artifact declaration gains the knob**

In `src/targets/qwen3_6_27b/impl/load/bindings.h`, change the `bind_artifact` declaration (lines 122-124) to:

```cpp
ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features,
                               ResidencyProfile residency = ResidencyProfile::AllResident,
                               std::uint32_t resident_ffn_layers = 0);
```

- [x] **Step 3: bindings.cpp — per-layer placement helper**

In `src/targets/qwen3_6_27b/impl/load/bindings.cpp`, replace the 2-arg `ffn_placement` (lines 46-49) with:

```cpp
artifact::TensorPlacement ffn_placement(ResidencyProfile residency, std::uint32_t resident_layers,
                                        std::size_t layer) noexcept {
    if (residency != ResidencyProfile::FfnOffload) { return artifact::TensorPlacement::Device; }
    return layer < resident_layers ? artifact::TensorPlacement::Device
                                   : artifact::TensorPlacement::Host;
}
```

- [x] **Step 4: bindings.cpp — thread through both layer binders**

Change `bind_groupwise_text_layers` signature (line 249-250) to:

```cpp
void bind_groupwise_text_layers(artifact::Binder& binder, BindingPlan& out,
                                ResidencyProfile residency, std::uint32_t resident_ffn_layers) {
```

and its MLP binds (lines 299-304) from `ffn_placement(residency)` to `ffn_placement(residency, resident_ffn_layers, layer)` at both :301 and :304.

Change `bind_nvfp4_text_layers` signature (line 308-309) to:

```cpp
void bind_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out,
                            ResidencyProfile residency, std::uint32_t resident_ffn_layers) {
```

and its MLP binds (lines 376-382) from `ffn_placement(residency)` to `ffn_placement(residency, resident_ffn_layers, layer)` at both :379 and :382.

- [x] **Step 5: bindings.cpp — bind_artifact definition gains param + validation**

Change the `bind_artifact` definition (line 407-408) to:

```cpp
ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features, ResidencyProfile residency,
                               std::uint32_t resident_ffn_layers) {
    if (resident_ffn_layers > kTextLayers) {
        throw std::invalid_argument("resident_ffn_layers must be in [0,64]");
    }
```

and update the two dispatch calls (lines 421 and 424) from `bind_groupwise_text_layers(binder, out, residency)` to `bind_groupwise_text_layers(binder, out, residency, resident_ffn_layers)` and `bind_nvfp4_text_layers(binder, out, residency, resident_ffn_layers)`.

- [x] **Step 6: bindings.cpp — LoadedModelData offload probe + stage_mlp gate**

Replace the offload probe (lines 492-493):

```cpp
    const bool offload = std::any_of(
        plan.text_layers.begin(), plan.text_layers.end(), [&](const TextLayerPlan& layer) {
            return backing.host_data_or_null(layer.mlp.gate_up.object) != nullptr;
        });
```

In the `stage_mlp` lambda (line 515-531), immediately after `if (!offload) { return; }`, add:

```cpp
        if (post_mixer.gate_up.host == nullptr) { return; }
```

This is the single enforcement point that resident layers (host == nullptr) are never re-pointed and never enter the slot map. (`std::any_of` needs `<algorithm>`, already included.)

- [x] **Step 7: package.cpp plan_load passes the knob**

In `src/targets/qwen3_6_27b/impl/package.cpp` `plan_load` (lines 99-107), change the `bind_artifact` call to:

```cpp
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, detail::bind_artifact(binder, weights_profile,
                                               qwen3_6::startup_features(options), residency,
                                               options.resident_ffn_layers)));
```

- [x] **Step 8: Build and run the gate**

Run: `cmake --build build`

Run: `NINFER_QWEN3_8_27B_WEIGHTS=$HOME/llm-models/qwen3_8_27b.ninfer /usr/bin/ctest --test-dir build --output-on-failure -R 'ninfer_qwen3_6_27b_residency_test|ninfer_qwen3_6_27b_engine_offload_real_test|ninfer_artifact_|ninfer_arena_test|ninfer_tensor_test'`

Expected: **7/7 PASS** — residency (~9 s), engine offload (~15 s), artifact_reader, artifact_materialization, tensor, arena. The two new arms pass: partial-residency asserts 46 staged weights with correct double-buffer slot layout and device-only first 41 layers; engine partial arm asserts `host_bytes == 3523379200` (23/64 of groupwise FFN bytes); the MTP arm completes a draft-verify round under FfnOffload (`speculative.enabled && rounds > 0 && drafted_tokens > 0`).

> **Note for the implementer:** the MTP arm exercises graph capture + replay of the MTP schedule with the staged FFN copies for the first time. If it fails, it is a genuine offload-speculative integration bug — diagnose and fix in this branch, then re-run. (Known risk in the design spec §8.)

- [x] **Step 9: Commit**

```bash
git add include/ninfer/types.h src/targets/qwen3_6_27b/impl/load/bindings.h src/targets/qwen3_6_27b/impl/load/bindings.cpp src/targets/qwen3_6_27b/impl/package.cpp
git commit -m "feat(engine): add the partial FFN residency knob"
```

---

### Task 3: CLI and serve flags

**Files:**
- Modify: `apps/cli/options.h`
- Modify: `apps/cli/options.cpp`
- Modify: `apps/cli/main.cpp`
- Modify: `src/serve/serve_options.h`
- Modify: `src/serve/serve_options.cpp`
- Modify: `src/serve/generation_service.cpp`

- [x] **Step 1: CLI Options field**

In `apps/cli/options.h`, in `struct Options`, immediately after `ninfer::WeightResidency weight_residency = ninfer::WeightResidency::AllResident;` (line 27), insert:

```cpp
    std::uint32_t resident_ffn_layers = 0;
```

- [x] **Step 2: CLI parse + usage**

In `apps/cli/options.cpp`, immediately after the `--weight-residency` parse branch (line 140-142), insert:

```cpp
        } else if (arg == "--n-ffn-layers") {
            options.resident_ffn_layers = parse_u32(value(arg), "n-ffn-layers");
        }
```

Extend the usage line (line 86) to:

```cpp
           "       [--kv-dtype bf16|int8] [--weight-residency all|ffn] [--n-ffn-layers N] [--spec mtp|dflash --draft-tokens N]\n"
```

- [x] **Step 3: CLI main mapping**

In `apps/cli/main.cpp`, immediately after `engine_options.weight_residency = cli.weight_residency;` (line 270), insert:

```cpp
        engine_options.resident_ffn_layers = cli.resident_ffn_layers;
```

- [x] **Step 4: serve Options field**

In `src/serve/serve_options.h`, immediately after `WeightResidency weight_residency       = WeightResidency::AllResident;` (line 41), insert:

```cpp
    std::uint32_t resident_ffn_layers  = 0;
```

- [x] **Step 5: serve parse + usage**

In `src/serve/serve_options.cpp`, immediately after the `--weight-residency` parse branch (`options.weight_residency = parse_weight_residency(require_value("--weight-residency"));`), insert:

```cpp
        } else if (arg == "--n-ffn-layers") {
            options.resident_ffn_layers = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--n-ffn-layers"), "n-ffn-layers"));
        }
```

Extend the usage line (line 79) to add `[--n-ffn-layers N]` after `[--weight-residency all|ffn]`.

- [x] **Step 6: serve EngineOptions mapping**

In `src/serve/generation_service.cpp`, immediately after `engine_options.weight_residency    = options_.weight_residency;` (line 262), insert:

```cpp
    engine_options.resident_ffn_layers = options_.resident_ffn_layers;
```

- [x] **Step 7: Build and smoke-test**

Run: `cmake --build build --target ninfer ninfer-serve`

Run: `./build/apps/ninfer --help 2>&1 | grep -E "n-ffn-layers|weight-residency"` — both flags render.

Run (real artifact, 16 GB card): `./build/apps/ninfer $HOME/llm-models/qwen3_8_27b.ninfer --weight-residency ffn --n-ffn-layers 41 --prompt "Hello" --max-new 4 --greedy --max-context 4096 2>&1 | grep -E "host store|gpu staging arena|generated tokens"`

Expected: load succeeds; `host store` ≈ 3.28 GiB (23 × 153,190,400 = 3,523,379,200 B ≈ 3.28 GiB); `gpu staging arena` 292.19 MiB unchanged; `generated tokens 4`.

Also run: `./build/apps/ninfer $HOME/llm-models/qwen3_8_27b.ninfer --weight-residency ffn --n-ffn-layers 65 --prompt "Hello" --max-new 1 2>&1 | tail -1` — expected error: `resident_ffn_layers must be in [0,64]`.

Run serve smoke: `./build/apps/ninfer-serve $HOME/llm-models/qwen3_8_27b.ninfer --weight-residency ffn --n-ffn-layers 41 --host 127.0.0.1 --port 18081 --max-context 4096 --kv-capacity 4096 2>&1 | grep -E "weights|listening"` then kill; expected `weights` line shows reduced device-resident size and server listens.

- [x] **Step 8: Commit**

```bash
git add apps/cli/options.h apps/cli/options.cpp apps/cli/main.cpp src/serve/serve_options.h src/serve/serve_options.cpp src/serve/generation_service.cpp
git commit -m "feat(cli): add the partial FFN residency knob to CLI and serve"
```

---

### Task 4: docs

**Files:**
- Modify: `docs/maintainer/weight-offload.md`
- Modify: `docs/cli.md`
- Modify: `docs/serving.md`
- Modify: `README.md`

- [x] **Step 1: weight-offload.md**

In `docs/maintainer/weight-offload.md`:

1. Top-of-file Status (lines 3-8): append a sentence after the surfaces 1-6 sentence: `A partial-residency knob (`EngineOptions::resident_ffn_layers`, CLI `--n-ffn-layers N`) keeps the first N FFN text layers device-resident while the rest stream.` (Place it so the paragraph still reads: surfaces 1-6 implemented ..., the knob ..., remaining surfaces 7-8 not yet implemented.)

2. Section 2 (residency profile) — after the `FfnOffload` profile description, add:

```text
Partial residency (`resident_ffn_layers = N`): the first N text-layer FFN
matrices (indices 0..N-1) are bound device-resident; layers N..63 stream from the
pinned host store through the staging arena. All 64 text-layer FFN matrices are
byte-identical in the registered identities (gate_up 34816x5120 + down 5120x17408;
153,190,400 B groupwise, 150,405,128 B NVFP4 per layer), so "N largest layers"
degenerates to "any N layers", and the deterministic first-N prefix is the exact
behavior. `staged_weights` shrinks to 2*(64-N) entries; the staging arena stays
2 x per-layer-unit regardless of N. Resident layers are skipped by the
`post_mixer` staging gate (`weight.host == nullptr`), so they never receive a
staging copy and their payloads stay in the backing weights arena.
```

3. Section-5 Status paragraph: append after the existing surface-6 sentence:

```text
 A partial-residency knob (`EngineOptions::resident_ffn_layers`, default 0 = all
 stream; CLI `--n-ffn-layers N`) keeps the first N FFN text layers device-resident
 and validates `N <= 64` in the 27B binder; MTP3 under FfnOffload is verified
 end-to-end.
```

- [x] **Step 2: cli.md and serving.md option rows**

In `docs/cli.md`, immediately after the `--weight-residency all\|ffn` row (line 141), insert:

```md
| `--n-ffn-layers N` | when combined with `--weight-residency ffn`, keep the first N FFN text layers device-resident and stream layers N..63 from pinned host memory; tune the VRAM/throughput tradeoff (default `0` = all FFN layers stream) | `0` |
```

In `docs/serving.md`, immediately after the `--weight-residency all\|ffn` row (~line 434), insert the same row.

- [x] **Step 3: README note**

In `README.md`, in the "Current limits" offload bullet (added by `f732c97`, ~line 301-306), extend the sentence describing `--weight-residency ffn` with: `A partial-residency knob (`--n-ffn-layers N`) keeps the first N FFN layers resident to trade VRAM against throughput.`

- [x] **Step 4: plan checkboxes**

Mark every `- [ ]` task checkbox in this file as `- [x]`.

- [x] **Step 5: verify**

Run: `git diff --check` (silent expected). Confirm only the four docs files changed.

- [x] **Step 6: Commit**

```bash
git add docs/maintainer/weight-offload.md docs/cli.md docs/serving.md README.md docs/superpowers/plans/2026-08-15-ffn-residency-knob.md
git commit -m "docs(offload): document the partial FFN residency knob"
```

---

## Out of scope (recorded for follow-on)

- Byte-budget auto-selection, non-uniform layer-size selection, AllResident/35B/KV/mechanism changes — per design spec §7.
- The llama.cpp-baseline comparison measurement (MTP3 + `--n-ffn-layers 41`) is a post-merge manual verification with hardware/workload context recorded per AGENTS.md performance-work rules, not a benchmark campaign.
