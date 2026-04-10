// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul_device_operation.hpp"

#include <cstdio>
#include <fmt/format.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include "optimized_matmul_config.hpp"
#include "optimized_matmul_schedule.hpp"
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

std::optional<uint32_t> get_tensix_harvesting_mask_for_schedule(
    const tt::tt_metal::CoreCoord& active_grid, const uint32_t physical_chip_id) {
    if (uses_optimized_matmul_harvest_schedule(active_grid)) {
        return tt::tt_metal::GetTensixHarvestingMask(physical_chip_id);
    }
    return std::nullopt;
}

void log_schedule_selection_candidates(
    const OptimizedMatmulDeviceOperation::operation_attributes_t& operation_attributes,
    const OptimizedMatmulDeviceOperation::tensor_args_t& tensor_args) {
    auto* mesh_device = tensor_args.input_tensor_a.device();
    if (mesh_device == nullptr) {
        return;
    }

    const auto active_grid = tt::tt_metal::CoreCoord{
        operation_attributes.active_grid_x, operation_attributes.active_grid_y};
    [[maybe_unused]] size_t coord_idx = 0;
    for (const auto& coord :
         ttnn::device_operation::mesh_device_operation_utils::extract_tensor_coordinates(
             tensor_args, mesh_device)) {
        auto* target_device = mesh_device->get_device(coord);
        if (target_device == nullptr) {
            // fmt::print(
            //     stderr,
            //     "optimized_matmul schedule candidate[{}]: unresolved target device\n",
            //     coord_idx);
            continue;
        }

        [[maybe_unused]] const auto physical_chip_id = static_cast<uint32_t>(target_device->id());
        [[maybe_unused]] const auto pcie_device_id =
            static_cast<uint32_t>(tt::tt_metal::GetPCIeDeviceID(target_device->id()));
        [[maybe_unused]] const auto tensix_harvesting_mask =
            get_tensix_harvesting_mask_for_schedule(active_grid, physical_chip_id);
        // fmt::print(
        //     stderr,
        //     "optimized_matmul schedule candidate[{}]: mesh_coordinate={}, physical_chip_id={}, pcie_device_id={}, "
        //     "tensix_harvesting_mask={}, active_grid={}x{}, header_by_harvest={}, header_by_physical={}, "
        //     "header_by_pcie={}\n",
        //     coord_idx,
        //     coord,
        //     physical_chip_id,
        //     pcie_device_id,
        //     tensix_harvesting_mask.has_value() ? fmt::format("{:#x}", *tensix_harvesting_mask) : std::string("<none>"),
        //     active_grid.x,
        //     active_grid.y,
        //     get_optimized_matmul_schedule_header_basename(active_grid, physical_chip_id, tensix_harvesting_mask),
        //     get_optimized_matmul_schedule_header_basename(active_grid, physical_chip_id),
        //     get_optimized_matmul_schedule_header_basename(active_grid, pcie_device_id));
        ++coord_idx;
    }
    std::fflush(stderr);
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

    const auto matmul_shape_override = OptimizedMatmulDeviceOperation::get_matmul_shape_override(operation_attributes);
    const auto effective_shapes =
        resolve_optimized_matmul_effective_shapes(input_tensor_a, input_tensor_b, matmul_shape_override);
    const auto& a_logical_shape = effective_shapes.a_logical_shape;
    const auto& b_logical_shape = effective_shapes.b_logical_shape;
    const auto& a_padded_shape = effective_shapes.a_padded_shape;
    const auto& b_padded_shape = effective_shapes.b_padded_shape;

    TT_FATAL(
        a_logical_shape.rank() >= 2 && b_logical_shape.rank() >= 2,
        "optimized_matmul requires both inputs to be at least rank 2");
    if (matmul_shape_override.has_value()) {
        TT_FATAL(
            matmul_shape_override->M > 0 && matmul_shape_override->N > 0 && matmul_shape_override->K > 0,
            "optimized_matmul matmul_shape_override requires positive M, N, K, got [{}, {}, {}]",
            matmul_shape_override->M,
            matmul_shape_override->N,
            matmul_shape_override->K);
        TT_FATAL(
            input_tensor_a.padded_shape().volume() == a_padded_shape.volume(),
            "optimized_matmul matmul_shape_override must preserve A padded volume; actual {} vs override {}",
            input_tensor_a.padded_shape(),
            a_padded_shape);
        TT_FATAL(
            input_tensor_b.padded_shape().volume() == b_padded_shape.volume(),
            "optimized_matmul matmul_shape_override must preserve B padded volume; actual {} vs override {}",
            input_tensor_b.padded_shape(),
            b_padded_shape);
    }
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
    return MultiCoreMeshWorkloadFactory{};
}

std::optional<OptimizedMatmulShapeOverride> OptimizedMatmulDeviceOperation::get_matmul_shape_override(
    const operation_attributes_t& operation_attributes) {
    if (!operation_attributes.has_matmul_shape_override) {
        return std::nullopt;
    }

    return OptimizedMatmulShapeOverride{
        .M = operation_attributes.override_m,
        .N = operation_attributes.override_n,
        .K = operation_attributes.override_k,
    };
}

std::optional<OptimizedMatmulOutputShapeOverride> OptimizedMatmulDeviceOperation::get_output_shape_override(
    const operation_attributes_t& operation_attributes) {
    if (!operation_attributes.has_output_shape_override) {
        return std::nullopt;
    }

    return OptimizedMatmulOutputShapeOverride{
        .logical_shape = operation_attributes.output_override_logical_shape,
        .padded_shape = operation_attributes.output_override_padded_shape,
    };
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
    log_schedule_selection_candidates(operation_attributes, tensor_args);
    validate_inputs(operation_attributes, tensor_args);
    const auto effective_shapes = resolve_optimized_matmul_effective_shapes(
        tensor_args.input_tensor_a,
        tensor_args.input_tensor_b,
        get_matmul_shape_override(operation_attributes));
    const auto default_output_shapes = resolve_optimized_matmul_output_shapes(effective_shapes);
    const auto output_shape_override = get_output_shape_override(operation_attributes);
    if (output_shape_override.has_value()) {
        TT_FATAL(
            output_shape_override->logical_shape.volume() == default_output_shapes.logical_shape.volume(),
            "optimized_matmul output_shape_override must preserve logical volume; default {} vs override {}",
            default_output_shapes.logical_shape,
            output_shape_override->logical_shape);
        TT_FATAL(
            output_shape_override->padded_shape.volume() == default_output_shapes.padded_shape.volume(),
            "optimized_matmul output_shape_override must preserve padded volume; default {} vs override {}",
            default_output_shapes.padded_shape,
            output_shape_override->padded_shape);
    }
    const auto resolved_variant_spec = resolve_optimized_matmul_variant_spec(
        tensor_args.input_tensor_a,
        tensor_args.input_tensor_b,
        get_matmul_shape_override(operation_attributes));
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
    const auto effective_shapes = resolve_optimized_matmul_effective_shapes(
        tensor_args.input_tensor_a, tensor_args.input_tensor_b, get_matmul_shape_override(operation_attributes));
    const auto output_shapes =
        resolve_optimized_matmul_output_shapes(effective_shapes, get_output_shape_override(operation_attributes));

    return TensorSpec(
        output_shapes.logical_shape,
        TensorLayout::fromPaddedShape(
            operation_attributes.output_dtype,
            PageConfig(Layout::TILE),
            operation_attributes.output_memory_config,
            output_shapes.logical_shape,
            output_shapes.padded_shape));
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
        tt::tt_metal::CoreCoord{operation_attributes.active_grid_x, operation_attributes.active_grid_y},
        get_matmul_shape_override(operation_attributes));
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
    std::optional<std::array<uint32_t, 3>> matmul_shape_override,
    std::optional<std::vector<uint32_t>> output_shape_override,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType>& dtype) {
    const auto resolved_shape_override = make_optimized_matmul_shape_override(matmul_shape_override);
    const auto resolved_output_shape_override = make_optimized_matmul_output_shape_override(output_shape_override);
    const auto effective_shapes =
        resolve_optimized_matmul_effective_shapes(input_tensor_a, input_tensor_b, resolved_shape_override);
    const auto variant_spec = resolve_optimized_matmul_variant_spec(input_tensor_a, input_tensor_b, resolved_shape_override);
    const auto math_fidelity = resolve_optimized_matmul_math_fidelity(input_tensor_a, compute_kernel_config);
    const auto output_memory_config = resolve_optimized_matmul_output_memory_config(memory_config);
    const auto output_dtype = resolve_optimized_matmul_output_dtype(input_tensor_a, dtype);
    const bool use_optimized_write = variant_spec.optimized_write && output_memory_config.is_dram();
    const tensor_args_t tensor_args{input_tensor_a, input_tensor_b};
    auto* mesh_device = input_tensor_a.device();
    TT_FATAL(mesh_device != nullptr, "optimized_matmul requires inputs to be allocated on a mesh device");
    std::size_t schedule_selection_hash = 0;
    const auto active_grid = variant_spec.active_grid;
    for (const auto& coord :
         ttnn::device_operation::mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
        auto* target_device = mesh_device->get_device(coord);
        TT_FATAL(target_device != nullptr, "optimized_matmul could not resolve target device for {}", coord);
        const auto tensix_harvesting_mask =
            get_tensix_harvesting_mask_for_schedule(active_grid, static_cast<uint32_t>(target_device->id()));
        ttsl::hash::hash_combine(schedule_selection_hash, coord);
        if (tensix_harvesting_mask.has_value()) {
            ttsl::hash::hash_combine(schedule_selection_hash, *tensix_harvesting_mask);
        } else {
            ttsl::hash::hash_combine(schedule_selection_hash, target_device->id());
        }
    }
    return {
        operation_attributes_t{
            output_memory_config,
            output_dtype,
            math_fidelity,
            schedule_selection_hash,
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
            variant_spec.active_grid.y,
            resolved_shape_override.has_value(),
            resolved_shape_override.has_value() ? resolved_shape_override->M : 0U,
            resolved_shape_override.has_value() ? resolved_shape_override->N : 0U,
            resolved_shape_override.has_value() ? resolved_shape_override->K : 0U,
            resolved_output_shape_override.has_value(),
            resolved_output_shape_override.has_value() ? resolved_output_shape_override->logical_shape : ttnn::Shape{},
            resolved_output_shape_override.has_value() ? resolved_output_shape_override->padded_shape : ttnn::Shape{}},
        tensor_args};
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
