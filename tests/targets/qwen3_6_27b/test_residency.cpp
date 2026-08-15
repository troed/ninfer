#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <variant>

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
        if (entry.object.index == handle.index) { return true; }
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
