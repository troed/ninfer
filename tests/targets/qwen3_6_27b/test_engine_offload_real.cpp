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

} // namespace

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
