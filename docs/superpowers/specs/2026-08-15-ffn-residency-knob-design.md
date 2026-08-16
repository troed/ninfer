# Partial FFN Residency Knob + MTP-Under-Offload Verification — Design

> Status: approved design, not yet implemented.

**Goal:** Allow a user to keep a configurable number of FFN layers device-resident while streaming the rest from pinned host RAM, so the 27B dense identities can tune the VRAM/throughput tradeoff on a 16 GB-VRAM `sm_120a` GPU and beat a llama.cpp baseline (~8–10 tok/s with MTP3, ~23 of 64 FFN layers offloaded).

**Scope:** Extend the existing offload feature (surfaces 1–6) with a per-layer residency count. No new streaming mechanism — the graph-capture interleave already keys off `Weight::host != nullptr`, so resident layers simply never receive a staging copy.

---

## 1. Public surface

- `EngineOptions::resident_ffn_layers` (`std::uint32_t`, default `0`) in `include/ninfer/types.h`, placed after `weight_residency`.
  - Meaning: number of FFN text layers kept device-resident when `weight_residency == FfnOffload`.
  - `0` = all FFN layers stream (exactly today's FfnOffload behavior).
  - `64` = all FFN layers resident (degenerate to AllResident semantics for the FFN subset).
  - Ignored when `weight_residency == AllResident`; no cross-validation error (harmless no-op).
- Validation: the `[0,64]` bound belongs in the 27B `bind_artifact` implementation (the only place `kTextLayers` is in scope), throwing `std::invalid_argument("resident_ffn_layers must be in [0,64]")` when out of range. `registry.cpp validate_options` is target-generic and has no public layer-count constant, so it does not check this field (consistent with how other target-specific bounds live in the target layer).
- CLI (`apps/cli/options.*`): `--n-ffn-layers N` (parse `uint32`), mapped into `engine_options.resident_ffn_layers`. Default 0.
- Serve (`src/serve/serve_options.*` + `generation_service.cpp`): mirror `--n-ffn-layers N`, mapped into `engine_options.resident_ffn_layers`. Default 0.
- 35B-A3B: no residency support exists; the existing reject of `weight_residency != AllResident` in the 35B `plan_load` also covers any `resident_ffn_layers` setting (the knob is meaningless without FfnOffload). No additional 35B change.

## 2. Selection policy

All 64 text-layer FFN matrices are byte-identical in the current identities (gate_up 34816×5120 + down 5120×17408; 153,190,400 B groupwise, 150,405,632 B NVFP4 per layer), so "keep the N largest layers resident" degenerates to "keep any N layers resident".

**Chosen policy:** keep the **first N layers** (indices `0..N-1`) device-resident; layers `N..63` stream. This is a deterministic prefix, matches the existing per-layer bind loop index, and — because sizes are uniform — yields identical streamed-bytes-per-token to any other selection. A size-sorted selection is deliberately not implemented: it would be dead code for every registered identity (AGENTS.md: no placeholders for hypothetical models). The knob's public definition is "N FFN layers resident, largest-first", and the uniform-size prefix realization is documented here as the exact current behavior.

## 3. Plumbing (the only real code)

Threading path: `EngineOptions` → 27B `Package::plan_load` → `bind_artifact(..., ResidencyProfile, std::uint32_t resident_ffn_layers = 0)` → `bind_groupwise_text_layers` / `bind_nvfp4_text_layers` → per-layer MLP bind placement.

- `src/targets/qwen3_6_27b/impl/load/bindings.h`: `bind_artifact` declaration gains a trailing `std::uint32_t resident_ffn_layers = 0` parameter.
- `src/targets/qwen3_6_27b/impl/load/bindings.cpp`:
  - `bind_artifact` definition: new param, forwarded to both layer binders.
  - Both layer binders: gain the param; the MLP `bind_weight` calls change from `ffn_placement(residency)` to a per-layer choice. Since the bind loops already have `layer`, replace `ffn_placement(residency)` at the two groupwise MLP sites (lines ~301, ~304) and two NVFP4 sites (~379, ~382) with a small helper, e.g.:
    ```cpp
    artifact::TensorPlacement ffn_placement(ResidencyProfile residency,
                                            std::uint32_t resident_layers,
                                            std::size_t layer) noexcept {
        if (residency != ResidencyProfile::FfnOffload) { return artifact::TensorPlacement::Device; }
        return layer < resident_layers ? artifact::TensorPlacement::Device
                                       : artifact::TensorPlacement::Host;
    }
    ```
  - The existing two-arg `ffn_placement` is superseded by the three-arg form; update all call sites. Attention/GDN/other projections remain `Device` unconditionally (unchanged).
- `src/targets/qwen3_6_27b/impl/load/bindings.cpp` `LoadedModelData` ctor:
  - Offload probe: replace the current "layer 0 gate_up host-placed" check with "any text layer's gate_up is host-placed" (`std::any_of` over `plan.text_layers`). Under partial residency, layer 0 is device-placed, so the layer-0 probe would wrongly report no offload.
  - `stage_mlp` lambda: early-return per layer when `post_mixer.gate_up.host == nullptr` (resident layer → no point_at_slot, no slot-map entries). `point_at_slot` must not run for resident layers (their payloads stay in the backing weights arena).
  - Staging arena unchanged: still `2 * staging_unit_bytes` where `staging_unit_bytes` is the per-layer FFN unit (~153 MiB groupwise). The arena is sized for the largest single streaming unit and is independent of how many layers stream.
  - `runtime.staged_weights` shrinks to `2 * (64 - N)` entries under partial residency.
- No change to `variant.cpp stage_weight`, `post_mixer`, `prepare_graphs`, `program_impl.h memory_summary`, or `kv_capacity.cpp` — the staging interleave and reporting already key off `weight.host != nullptr` / `model.staging_arena != nullptr` and remain correct.

## 4. MTP-under-offload verification (work item, not code)

MTP weights are device-resident and never staged (`mtp_post_mixer` has no `stage_weight` call; MTP MLP binds use `Device` placement), so MTP3 under FfnOffload should already function. The deliverable is verification and measurement:

- Engine-level correctness: a `--spec mtp --draft-tokens 3 --weight-residency ffn` run (eager + graph) must produce a correct output sequence (draft+verify round completes; sampled/stop tokens sane).
- Performance: measure decode tok/s at `--n-ffn-layers 41` + MTP3 on the RTX 5060 Ti 16GB with the real qwen3.8 artifact, and compare against the stated llama.cpp baseline (~8–10 tok/s with 23 of 64 FFN offloaded + MTP3). Record the result with hardware/workload context (per AGENTS.md performance-work rules). This is a measurement/verification claim, not a benchmark campaign.

## 5. Tests

- `tests/targets/qwen3_6_27b/test_residency.cpp` (or a sibling): a partial-N arm asserting, under `FfnOffload` with `resident_ffn_layers == 41`:
  - `staged_weights.size() == 2 * (64 - 41) == 46`;
  - the first 41 layers' `post_mixer.gate_up/.down` are device-only (`host == nullptr`, payload in the backing weights arena), the last 23 are host-backed (`host != nullptr`, payload in the staging arena);
  - staging arena capacity unchanged (`2 * unit`).
  - `staged_mlp_round_matches_host` still passes for a streamed layer.
- `tests/targets/qwen3_6_27b/test_engine_offload_real.cpp`: add a partial-residency arm (`resident_ffn_layers = 41`) that loads + decodes 4 tokens, asserting the same summary invariants (host store > 0, staging == 2*unit). Optionally set the options with `resident_ffn_layers` and confirm `LoadSummary.host_bytes` reflects 23/64 of the FFN bytes rather than all.
- MTP: engine-level FfnOffload + `speculative = mtp(3)` correctness arm (a draft+verify round yielding a completed generation) in the engine offload test.

## 6. Docs

- `docs/maintainer/weight-offload.md`: document `resident_ffn_layers` / `--n-ffn-layers`, the prefix-selection note, the `stage_mlp` per-layer gate, and update the section-5 status (surfaces 1-6 + this knob).
- `docs/cli.md` + `docs/serving.md`: `--n-ffn-layers N` option rows.
- `README.md`: the offload note gains the knob.

## 7. Non-goals

- Byte-budget auto-selection (`--ffn-resident-bytes`) — rejected in design; explicit count only.
- Non-uniform layer-size selection — not implemented (uniform in all registered identities).
- Any change to `AllResident`, the 35B target, KV capacity planning, or the staging mechanism itself.
- Full benchmark campaign; only the single MTP3+partial comparison measurement requested.

## 8. Risks

- The `offload` probe change (layer-0 → any-layer) is the correctness-critical edit: under AllResident it must remain false (no host-placed gate_up exists → `any_of` false). Under full FfnOffload it stays true. Under partial it must be true iff any layer streams.
- `stage_mlp` per-layer gate: resident layers must never be re-pointed; the `host == nullptr` guard is the single enforcement point.
- NVFP4 and legacy-groupwise residency arms remain unexercised locally (no artifacts); the partial-N logic is profile-agnostic and covered by the groupwise real-artifact test plus code inspection.
