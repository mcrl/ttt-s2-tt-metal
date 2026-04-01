// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

namespace {

ttnn::Shape compute_output_shape(const ttnn::Shape& input_shape_a, const ttnn::Shape& input_shape_b) {
    auto output_shape = input_shape_a;
    output_shape[-1] = input_shape_b[-1];
    return output_shape;
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
        batch_size_b == 1 || batch_size_a == batch_size_b,
        "optimized_matmul currently supports only identical batch sizes or B broadcast, got {} and {}",
        batch_size_a,
        batch_size_b);

    if (batch_size_b != 1) {
        for (auto index = 0; index < a_logical_shape.rank() - 2; ++index) {
            TT_FATAL(
                a_logical_shape[index] == b_logical_shape[index],
                "optimized_matmul requires matching batch dimensions when B is not broadcast: got A{}={} and B{}={}",
                index,
                a_logical_shape[index],
                index,
                b_logical_shape[index]);
        }
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
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor_a.device());
}

std::tuple<OptimizedMatmulDeviceOperation::operation_attributes_t, OptimizedMatmulDeviceOperation::tensor_args_t>
OptimizedMatmulDeviceOperation::invoke(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    return {operation_attributes_t{ttnn::DRAM_MEMORY_CONFIG}, tensor_args_t{input_tensor_a, input_tensor_b}};
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
