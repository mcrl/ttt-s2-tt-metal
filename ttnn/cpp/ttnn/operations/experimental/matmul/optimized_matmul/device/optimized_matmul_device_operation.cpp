// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

#include "optimized_matmul_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

namespace {

OptimizedMatmulVariantSpec get_device_operation_variant_spec_from_attributes(
    const OptimizedMatmulDeviceOperation::operation_attributes_t& operation_attributes) {
    return {
        .input_a_is_dram = operation_attributes.input_a_is_dram,
        .input_b_is_dram = operation_attributes.input_b_is_dram,
        .optimized_a_read = operation_attributes.optimized_a_read,
        .optimized_b_read = operation_attributes.optimized_b_read,
        .optimized_write = operation_attributes.optimized_write,
        .packer_l1_acc = operation_attributes.packer_l1_acc,
        .optimized_write_use_generated_schedule = operation_attributes.optimized_write_use_generated_schedule,
        .active_grid = tt::tt_metal::CoreCoord{operation_attributes.active_grid_x, operation_attributes.active_grid_y},
    };
}

ttnn::Shape compute_output_shape(const ttnn::Shape& input_shape_a, const ttnn::Shape& input_shape_b) {
    auto output_shape = input_shape_a;
    output_shape[-1] = input_shape_b[-1];
    return output_shape;
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

MathFidelity resolve_optimized_matmul_math_fidelity(
    const Tensor& input_tensor_a, std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    std::optional<DeviceComputeKernelConfig> resolved_compute_kernel_config = std::nullopt;
    if (compute_kernel_config.has_value()) {
        if (input_tensor_a.storage_type() == StorageType::DEVICE && input_tensor_a.device() != nullptr) {
            resolved_compute_kernel_config =
                init_device_compute_kernel_config(input_tensor_a.device()->arch(), compute_kernel_config, MathFidelity::HiFi2);
        } else {
            resolved_compute_kernel_config = compute_kernel_config.value();
        }
    }

    auto math_fidelity = get_math_fidelity(resolved_compute_kernel_config);
    if (math_fidelity == MathFidelity::Invalid) {
        math_fidelity = MathFidelity::HiFi2;
    }

    TT_FATAL(
        math_fidelity == MathFidelity::LoFi || math_fidelity == MathFidelity::HiFi2 ||
            math_fidelity == MathFidelity::HiFi4,
        "optimized_matmul currently supports only LoFi, HiFi2, or HiFi4 math_fidelity, got {}",
        math_fidelity);
    return math_fidelity;
}

DataType resolve_optimized_matmul_output_dtype(
    const Tensor& input_tensor_a, const std::optional<const DataType>& dtype) {
    return dtype.value_or(input_tensor_a.dtype());
}

MemoryConfig resolve_optimized_matmul_output_memory_config(const std::optional<const MemoryConfig>& memory_config) {
    return memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
}

bool is_supported_optimized_matmul_dtype(const DataType dtype) {
    return dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B;
}

bool is_supported_optimized_matmul_output_memory_config(const MemoryConfig& memory_config) {
    return !memory_config.is_sharded() && memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
           (memory_config.is_dram() || memory_config.is_l1());
}

void validate_inputs(
    const OptimizedMatmulDeviceOperation::operation_attributes_t& operation_attributes,
    const OptimizedMatmulDeviceOperation::tensor_args_t& tensor_args) {
    using namespace tt::constants;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const bool input_a_is_interleaved_dram = is_interleaved_buffer_type(input_tensor_a, BufferType::DRAM);
    const bool input_a_is_interleaved_l1 = is_interleaved_buffer_type(input_tensor_a, BufferType::L1);
    const bool input_b_is_interleaved_dram = is_interleaved_buffer_type(input_tensor_b, BufferType::DRAM);
    const bool input_b_is_interleaved_l1 = is_interleaved_buffer_type(input_tensor_b, BufferType::L1);

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
        input_a_is_interleaved_dram || (input_a_is_interleaved_l1 && !operation_attributes.optimized_a_read),
        "optimized_matmul currently supports input A only as DRAM interleaved, or as interleaved L1 when optimized "
        "A read is disabled; got {}",
        input_tensor_a.memory_config());
    TT_FATAL(
        input_b_is_interleaved_dram || (input_b_is_interleaved_l1 && !operation_attributes.optimized_b_read),
        "optimized_matmul currently supports input B only as DRAM interleaved, or as interleaved L1 when optimized "
        "B read is disabled; got {}",
        input_tensor_b.memory_config());
    TT_FATAL(!input_tensor_a.is_sharded() && !input_tensor_b.is_sharded(), "optimized_matmul does not support sharded inputs");
    TT_FATAL(
        is_supported_optimized_matmul_dtype(input_tensor_a.dtype()),
        "optimized_matmul currently supports input A only as BFLOAT16 or BFLOAT8_B, got {}",
        input_tensor_a.dtype());
    TT_FATAL(
        is_supported_optimized_matmul_dtype(input_tensor_b.dtype()),
        "optimized_matmul currently supports input B only as BFLOAT16 or BFLOAT8_B, got {}",
        input_tensor_b.dtype());
    TT_FATAL(
        is_supported_optimized_matmul_dtype(operation_attributes.output_dtype),
        "optimized_matmul currently supports output dtype only as BFLOAT16 or BFLOAT8_B, got {}",
        operation_attributes.output_dtype);
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
}

}  // namespace

OptimizedMatmulDeviceOperation::program_factory_t OptimizedMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MultiCoreProgramFactory{};
}

void OptimizedMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        is_supported_optimized_matmul_output_memory_config(operation_attributes.output_memory_config),
        "optimized_matmul currently supports output only as DRAM interleaved or L1 interleaved; got {}",
        operation_attributes.output_memory_config);
    TT_FATAL(
        operation_attributes.output_memory_config.is_dram() || !operation_attributes.optimized_write,
        "optimized_matmul optimized write currently supports only DRAM interleaved output; got {}",
        operation_attributes.output_memory_config);
    const auto resolved_output_dtype = resolve_optimized_matmul_output_dtype(
        tensor_args.input_tensor_a, std::make_optional(operation_attributes.output_dtype));
    TT_FATAL(
        operation_attributes.math_fidelity == MathFidelity::LoFi ||
            operation_attributes.math_fidelity == MathFidelity::HiFi2 ||
            operation_attributes.math_fidelity == MathFidelity::HiFi4,
        "optimized_matmul currently supports only LoFi, HiFi2, or HiFi4 math_fidelity, got {}",
        operation_attributes.math_fidelity);
    TT_FATAL(
        operation_attributes.active_grid_x > 0 && operation_attributes.active_grid_x <= 8 &&
            operation_attributes.active_grid_y > 0 && operation_attributes.active_grid_y <= 8,
        "optimized_matmul currently supports only active grids within 8x8, got {}x{}",
        operation_attributes.active_grid_x,
        operation_attributes.active_grid_y);
    validate_inputs(operation_attributes, tensor_args);
    const auto resolved_variant_spec =
        resolve_optimized_matmul_variant_spec(tensor_args.input_tensor_a, tensor_args.input_tensor_b);
    const bool expected_optimized_write =
        resolved_variant_spec.optimized_write && operation_attributes.output_memory_config.is_dram();
    TT_FATAL(
        operation_attributes.input_a_dtype == tensor_args.input_tensor_a.dtype() &&
            operation_attributes.input_b_dtype == tensor_args.input_tensor_b.dtype() &&
            operation_attributes.output_dtype == resolved_output_dtype &&
            operation_attributes.input_a_is_dram == resolved_variant_spec.input_a_is_dram &&
            operation_attributes.input_b_is_dram == resolved_variant_spec.input_b_is_dram &&
            operation_attributes.optimized_a_read == resolved_variant_spec.optimized_a_read &&
            operation_attributes.optimized_b_read == resolved_variant_spec.optimized_b_read &&
            operation_attributes.optimized_write == expected_optimized_write &&
            operation_attributes.packer_l1_acc == resolved_variant_spec.packer_l1_acc &&
            operation_attributes.optimized_write_use_generated_schedule ==
                resolved_variant_spec.optimized_write_use_generated_schedule &&
            operation_attributes.active_grid_x == resolved_variant_spec.active_grid.x &&
            operation_attributes.active_grid_y == resolved_variant_spec.active_grid.y,
        "optimized_matmul operation attributes do not match the resolved variant spec");
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
            operation_attributes.output_dtype,
            PageConfig(Layout::TILE),
            operation_attributes.output_memory_config,
            output_shape,
            padded_output_shape));
}

OptimizedMatmulDeviceOperation::tensor_return_value_t OptimizedMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    const auto variant_spec = get_device_operation_variant_spec_from_attributes(operation_attributes);
    if (!variant_spec.optimized_write) {
        return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
    }

    const auto resolved_config = resolve_optimized_matmul_config(
        tensor_args.input_tensor_a,
        tensor_args.input_tensor_b,
        operation_attributes.output_dtype,
        tt::tt_metal::CoreCoord{operation_attributes.active_grid_x, operation_attributes.active_grid_y});
    const auto buffer_layout = resolve_optimized_matmul_buffer_layout(
        resolved_config,
        variant_spec,
        tt::tile_size(datatype_to_dataformat_converter(operation_attributes.input_a_dtype)),
        tt::tile_size(datatype_to_dataformat_converter(operation_attributes.input_b_dtype)),
        tt::tile_size(datatype_to_dataformat_converter(operation_attributes.output_dtype)),
        tensor_args.input_tensor_a.device()->num_dram_channels());
    return create_raw_optim_output_tensor(
        output_spec, tensor_args.input_tensor_a.device(), buffer_layout.c_page_size, buffer_layout.c_total_size);
}

std::tuple<OptimizedMatmulDeviceOperation::operation_attributes_t, OptimizedMatmulDeviceOperation::tensor_args_t>
OptimizedMatmulDeviceOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType>& dtype) {
    const auto variant_spec = resolve_optimized_matmul_variant_spec(input_tensor_a, input_tensor_b);
    const auto math_fidelity = resolve_optimized_matmul_math_fidelity(input_tensor_a, compute_kernel_config);
    const auto output_memory_config = resolve_optimized_matmul_output_memory_config(memory_config);
    const auto output_dtype = resolve_optimized_matmul_output_dtype(input_tensor_a, dtype);
    const bool use_optimized_write = variant_spec.optimized_write && output_memory_config.is_dram();
    return {
        operation_attributes_t{
            output_memory_config,
            output_dtype,
            math_fidelity,
            input_tensor_a.dtype(),
            input_tensor_b.dtype(),
            variant_spec.input_a_is_dram,
            variant_spec.input_b_is_dram,
            variant_spec.optimized_a_read,
            variant_spec.optimized_b_read,
            use_optimized_write,
            variant_spec.packer_l1_acc,
            variant_spec.optimized_write_use_generated_schedule,
            variant_spec.active_grid.x,
            variant_spec.active_grid.y},
        tensor_args_t{input_tensor_a, input_tensor_b}};
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
