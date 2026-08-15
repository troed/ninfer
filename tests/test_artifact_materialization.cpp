#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};
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

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw-bytes-v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"name", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"name", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                        })},
        },
        "materialization");
}

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

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        validation_binder.retain_on_host(validated_resource);
        constexpr std::array<std::uint64_t, 1> validated_shape = {2};
        const auto validated_only                              = validation_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.validate_only(validated_only);
        constexpr std::array<std::uint64_t, 1> retained_shape = {4};
        const auto retained_tensor                            = validation_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        validation_binder.materialize_on_device(retained_tensor);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 3 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_capacity_bytes == kSecondTensor.size(),
                "validate-only tensor was included in the materialization plan");

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        ninfer::artifact::Binder binder(reader);

        const auto resource = binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        binder.retain_on_host(resource);
        constexpr std::array<std::uint64_t, 1> second_shape = {4};
        const auto second =
            binder.require_tensor("weights/second", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        binder.materialize_on_device(second);

        // Bind in the opposite order from the artifact. Device placement order and file read order
        // are intentionally independent, exercising the direct-I/O scatter path.
        constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
        const auto tensor =
            binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.materialize_on_device(tensor);

        const ninfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 3 && plan.host_objects.size() == 1 &&
                    plan.device_objects.size() == 2 && plan.device_capacity_bytes == 260,
                "binder produced the wrong materialization plan");

        ninfer::DeviceContext device(0);
        auto materialized = ninfer::artifact::materialize(reader, plan, device);

        std::array<std::byte, kTensor.size()> copied{};
        CUDA_CHECK(cudaMemcpy(copied.data(), materialized.device_data(tensor), copied.size(),
                              cudaMemcpyDeviceToHost));
        require(copied == kTensor, "device tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> second_copied{};
        CUDA_CHECK(cudaMemcpy(second_copied.data(), materialized.device_data(second),
                              second_copied.size(), cudaMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");

        const auto retained = materialized.resource_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 2 && stats.resource_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() &&
                    stats.retained_resource_bytes == kResource.size() &&
                    stats.file_bytes == kResource.size() +
                                            ninfer::artifact::Reader::direct_io_alignment +
                                            kSecondTensor.size(),
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");

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

            cudaPointerAttributes attributes{};
            CUDA_CHECK(cudaPointerGetAttributes(&attributes,
                                                host_materialized.host_data(host_tensor)));
            require(attributes.type == cudaMemoryTypeHost,
                    "host tensor payload is not pinned host memory");

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
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
