// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

#include "optimized_matmul_config.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

namespace {

ttnn::Shape compute_output_shape(const ttnn::Shape& input_shape_a, const ttnn::Shape& input_shape_b) {
    auto output_shape = input_shape_a;
    output_shape[-1] = input_shape_b[-1];
    return output_shape;
}

void validate_expected_buffer_layout(
    const tt::tt_metal::Buffer* buffer,
    const uint32_t expected_page_size,
    const uint32_t expected_total_size,
    const std::string_view tensor_name) {
    TT_FATAL(buffer != nullptr, "optimized_matmul requires {} buffer to exist", tensor_name);
    TT_FATAL(
        buffer->page_size() == expected_page_size,
        "optimized_matmul requires {} buffer page_size={} for optim layout, got {}",
        tensor_name,
        expected_page_size,
        buffer->page_size());
    TT_FATAL(
        buffer->size() == expected_total_size,
        "optimized_matmul requires {} buffer size={} for optim layout, got {}",
        tensor_name,
        expected_total_size,
        buffer->size());
}

Tensor create_raw_optim_output_tensor(
    const TensorSpec& tensor_spec,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const uint32_t page_size,
    const uint32_t total_size) {
    const tt::tt_metal::distributed::DeviceLocalBufferConfig device_local_buffer_config{
        .page_size = page_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated_buffer_config{
        .size = total_size,
    };
    auto mesh_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        replicated_buffer_config, device_local_buffer_config, mesh_device);

    std::vector<tt::tt_metal::distributed::MeshCoordinate> coords;
    coords.reserve(mesh_device->shape().mesh_size());
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape())) {
        coords.push_back(coord);
    }

    tt::tt_metal::DeviceStorage device_storage(std::move(mesh_buffer), coords);
    auto tensor_topology = tt::tt_metal::TensorTopology::create_fully_replicated_tensor_topology(mesh_device->shape());
    return Tensor(std::move(device_storage), tensor_spec, std::move(tensor_topology));
}

void validate_inputs(const OptimizedMatmulDeviceOperation::tensor_args_t& tensor_args) {
    using namespace tt::constants;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE && input_tensor_b.storage_type() == StorageType::DEVICE,
        "optimized_matmul requires both inputs to be on device");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr && input_tensor_b.buffer() != nullptr,
        "optimized_matmul requires both inputs to be allocated on device");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "optimized_matmul requires both inputs on same device");
    TT_FATAL(
        input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE,
        "optimized_matmul currently supports only TILE layout inputs");
    TT_FATAL(
        input_tensor_a.memory_config() == ttnn::DRAM_MEMORY_CONFIG,
        "optimized_matmul currently supports only DRAM interleaved input A; got {}",
        input_tensor_a.memory_config());
    TT_FATAL(
        input_tensor_b.memory_config() == ttnn::DRAM_MEMORY_CONFIG,
        "optimized_matmul currently supports only DRAM interleaved input B; got {}",
        input_tensor_b.memory_config());
    TT_FATAL(!input_tensor_a.is_sharded() && !input_tensor_b.is_sharded(), "optimized_matmul does not support sharded inputs");
    TT_FATAL(input_tensor_a.dtype() == input_tensor_b.dtype(), "optimized_matmul requires matching input dtypes");
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16,
        "optimized_matmul currently supports only BFLOAT16 inputs, got {}",
        input_tensor_a.dtype());
    TT_FATAL(
        input_tensor_a.device()->arch() == tt::ARCH::WORMHOLE_B0,
        "optimized_matmul currently supports only WORMHOLE_B0, got {}",
        input_tensor_a.device()->arch());
    const auto compute_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
    TT_FATAL(
        compute_grid_size.x == 8 && compute_grid_size.y == 8,
        "optimized_matmul currently supports only compute_with_storage_grid_size=8x8, got {}",
        compute_grid_size);

    const auto& a_logical_shape = input_tensor_a.logical_shape();
    const auto& b_logical_shape = input_tensor_b.logical_shape();
    const auto& a_padded_shape = input_tensor_a.padded_shape();
    const auto& b_padded_shape = input_tensor_b.padded_shape();

    TT_FATAL(
        a_logical_shape.rank() >= 2 && b_logical_shape.rank() >= 2,
        "optimized_matmul requires both inputs to be at least rank 2");
    TT_FATAL(
        a_logical_shape.rank() == b_logical_shape.rank(),
        "optimized_matmul currently supports only inputs with matching ranks, got {} and {}",
        a_logical_shape.rank(),
        b_logical_shape.rank());
    TT_FATAL(
        a_logical_shape[-1] == b_logical_shape[-2],
        "optimized_matmul requires A.K == B.K, got {} and {}",
        a_logical_shape[-1],
        b_logical_shape[-2]);
    TT_FATAL(
        a_padded_shape[-1] == b_padded_shape[-2],
        "optimized_matmul requires matching padded K dimensions, got {} and {}",
        a_padded_shape[-1],
        b_padded_shape[-2]);
    TT_FATAL(
        a_padded_shape[-2] % TILE_HEIGHT == 0 && a_padded_shape[-1] % TILE_WIDTH == 0 &&
            b_padded_shape[-2] % TILE_HEIGHT == 0 && b_padded_shape[-1] % TILE_WIDTH == 0,
        "optimized_matmul currently supports only tile-aligned padded shapes");

    const auto batch_size_a = get_batch_size(a_logical_shape);
    const auto batch_size_b = get_batch_size(b_logical_shape);
    TT_FATAL(
        batch_size_a == 1 && batch_size_b == 1,
        "optimized_matmul currently supports only batch_product == 1, got {} and {}",
        batch_size_a,
        batch_size_b);

    const auto policy = resolve_optimized_matmul_policy(input_tensor_a, input_tensor_b);
    const auto variant_spec = get_optimized_matmul_variant_spec(policy.variant_id);
    const auto resolved_config = resolve_optimized_matmul_config(input_tensor_a, input_tensor_b, policy.active_grid);
    const auto buffer_layout = resolve_optimized_matmul_buffer_layout(
        resolved_config,
        variant_spec,
        tt::tile_size(datatype_to_dataformat_converter(input_tensor_a.dtype())),
        input_tensor_a.device()->num_dram_channels());

    if (variant_spec.optimized_a_read) {
        validate_expected_buffer_layout(input_tensor_a.buffer(), buffer_layout.a_page_size, buffer_layout.a_total_size, "A");
    }
    if (variant_spec.optimized_b_read) {
        validate_expected_buffer_layout(input_tensor_b.buffer(), buffer_layout.b_page_size, buffer_layout.b_total_size, "B");
    }
}

}  // namespace

OptimizedMatmulDeviceOperation::program_factory_t OptimizedMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MultiCoreProgramFactory{};
}

void OptimizedMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        operation_attributes.output_memory_config == ttnn::DRAM_MEMORY_CONFIG,
        "optimized_matmul currently supports only DRAM interleaved output; got {}",
        operation_attributes.output_memory_config);
    TT_FATAL(
        operation_attributes.active_grid_x == 8 && operation_attributes.active_grid_y == 8,
        "optimized_matmul currently supports only an 8x8 active grid, got {}x{}",
        operation_attributes.active_grid_x,
        operation_attributes.active_grid_y);
    validate_inputs(tensor_args);
}

void OptimizedMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

OptimizedMatmulDeviceOperation::spec_return_value_t OptimizedMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_shape = compute_output_shape(
        tensor_args.input_tensor_a.logical_shape(), tensor_args.input_tensor_b.logical_shape());
    const auto padded_output_shape = compute_output_shape(
        tensor_args.input_tensor_a.padded_shape(), tensor_args.input_tensor_b.padded_shape());

    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            tensor_args.input_tensor_a.dtype(),
            PageConfig(Layout::TILE),
            operation_attributes.output_memory_config,
            output_shape,
            padded_output_shape));
}

OptimizedMatmulDeviceOperation::tensor_return_value_t OptimizedMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    const auto variant_spec =
        get_optimized_matmul_variant_spec(static_cast<OptimizedMatmulVariantId>(operation_attributes.variant_id));
    if (!variant_spec.optimized_write) {
        return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
    }

    const auto resolved_config = resolve_optimized_matmul_config(
        tensor_args.input_tensor_a,
        tensor_args.input_tensor_b,
        tt::tt_metal::CoreCoord{operation_attributes.active_grid_x, operation_attributes.active_grid_y});
    const auto buffer_layout = resolve_optimized_matmul_buffer_layout(
        resolved_config,
        variant_spec,
        tt::tile_size(datatype_to_dataformat_converter(tensor_args.input_tensor_a.dtype())),
        tensor_args.input_tensor_a.device()->num_dram_channels());
    return create_raw_optim_output_tensor(
        output_spec, tensor_args.input_tensor_a.device(), buffer_layout.c_page_size, buffer_layout.c_total_size);
}

std::tuple<OptimizedMatmulDeviceOperation::operation_attributes_t, OptimizedMatmulDeviceOperation::tensor_args_t>
OptimizedMatmulDeviceOperation::invoke(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    const auto policy = resolve_optimized_matmul_policy(input_tensor_a, input_tensor_b);
    return {
        operation_attributes_t{
            ttnn::DRAM_MEMORY_CONFIG,
            static_cast<uint32_t>(policy.variant_id),
            policy.active_grid.x,
            policy.active_grid.y},
        tensor_args_t{input_tensor_a, input_tensor_b}};
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
