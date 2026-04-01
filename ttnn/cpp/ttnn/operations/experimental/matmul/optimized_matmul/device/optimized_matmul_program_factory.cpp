// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul_device_operation.hpp"

#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

using namespace tt;
using namespace tt::constants;

OptimizedMatmulDeviceOperation::MultiCoreProgramFactory::cached_program_t
OptimizedMatmulDeviceOperation::MultiCoreProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    Program program{};

    const auto& a_shape = input_tensor_a.padded_shape();
    const auto& b_shape = input_tensor_b.padded_shape();
    const auto& c_shape = output_tensor.padded_shape();

    auto* src0_buffer = input_tensor_a.buffer();
    auto* src1_buffer = input_tensor_b.buffer();
    auto* dst_buffer = output_tensor.buffer();

    TT_FATAL(src0_buffer != nullptr && src1_buffer != nullptr, "optimized_matmul input buffers must exist");
    TT_FATAL(dst_buffer != nullptr, "optimized_matmul output buffer must exist");
    TT_FATAL(
        output_tensor.memory_config() == operation_attributes.output_memory_config,
        "optimized_matmul output memory config mismatch: expected {} but got {}",
        operation_attributes.output_memory_config,
        output_tensor.memory_config());

    const auto input_data_format = datatype_to_dataformat_converter(input_tensor_a.dtype());
    const auto weight_data_format = datatype_to_dataformat_converter(input_tensor_b.dtype());
    const auto output_data_format = datatype_to_dataformat_converter(output_tensor.dtype());

    const uint32_t input_single_tile_size = tt::tile_size(input_data_format);
    const uint32_t weight_single_tile_size = tt::tile_size(weight_data_format);
    const uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    auto* device = input_tensor_a.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    const uint32_t num_output_tiles_total = get_batch_size(c_shape) * c_shape[-2] * c_shape[-1] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    const uint32_t batch_size_a = get_batch_size(a_shape);
    const uint32_t batch_size_b = get_batch_size(b_shape);
    const bool broadcast_batch = batch_size_a > 1 && batch_size_b == 1;

    const uint32_t Mt = a_shape[-2] / TILE_HEIGHT;
    const uint32_t Kt = a_shape[-1] / TILE_WIDTH;
    const uint32_t Nt = b_shape[-1] / TILE_WIDTH;
    const uint32_t MtKt = Mt * Kt;
    const uint32_t KtNt = Kt * Nt;
    const uint32_t MtNt = Mt * Nt;

    const uint32_t src0_addr = src0_buffer->address();
    const uint32_t src1_addr = src1_buffer->address();
    const uint32_t dst_addr = dst_buffer->address();

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;

    CircularBufferConfig src0_cb_config =
        CircularBufferConfig(num_input_tiles * input_single_tile_size, {{src0_cb_index, input_data_format}})
            .set_page_size(src0_cb_index, input_single_tile_size);
    CreateCircularBuffer(program, all_cores, src0_cb_config);

    CircularBufferConfig src1_cb_config =
        CircularBufferConfig(num_input_tiles * weight_single_tile_size, {{src1_cb_index, weight_data_format}})
            .set_page_size(src1_cb_index, weight_single_tile_size);
    CreateCircularBuffer(program, all_cores, src1_cb_config);

    CircularBufferConfig output_cb_config =
        CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    CreateCircularBuffer(program, all_cores, output_cb_config);

    const uint32_t last_ktile_w = input_tensor_a.logical_shape()[-1] % TILE_WIDTH;
    std::vector<uint32_t> reader_compile_time_args = {last_ktile_w};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/device/kernels/dataflow/reader_optimized_matmul.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/device/kernels/dataflow/writer_optimized_matmul.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args_group_1 = {1, 1, Kt, num_output_tiles_per_core_group_1};
    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/device/kernels/compute/optimized_bmm.cpp",
        core_group_1,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .dst_full_sync_en = true, .compile_args = compute_args_group_1});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_group_2 = {1, 1, Kt, num_output_tiles_per_core_group_2};
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/device/kernels/compute/optimized_bmm.cpp",
            core_group_2,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4, .dst_full_sync_en = true, .compile_args = compute_args_group_2});
    }

    for (uint32_t core_index = 0, num_tiles_written = 0; core_index < num_cores; ++core_index) {
        CoreCoord core = {core_index / num_cores_y, core_index % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_THROW("optimized_matmul core not in specified core ranges");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {src0_addr,
             src1_addr,
             Mt,
             Kt,
             Nt,
             MtKt,
             KtNt,
             batch_size_a,
             static_cast<uint32_t>(broadcast_batch),
             num_tiles_written,
             num_output_tiles_per_core,
             MtNt});

        SetRuntimeArgs(program, writer_kernel_id, core, {dst_addr, num_output_tiles_per_core, num_tiles_written});
        num_tiles_written += num_output_tiles_per_core;
    }

    return {
        std::move(program),
        {
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .num_cores = num_cores,
            .num_cores_y = num_cores_y,
        }};
}

void OptimizedMatmulDeviceOperation::MultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto num_cores = cached_program.shared_variables.num_cores;
    const auto num_cores_y = cached_program.shared_variables.num_cores_y;

    auto* src0_buffer = tensor_args.input_tensor_a.buffer();
    auto* src1_buffer = tensor_args.input_tensor_b.buffer();
    auto* dst_buffer = output_tensor.buffer();

    TT_FATAL(
        output_tensor.memory_config() == operation_attributes.output_memory_config,
        "optimized_matmul output memory config mismatch: expected {} but got {}",
        operation_attributes.output_memory_config,
        output_tensor.memory_config());

    for (uint32_t core_index = 0; core_index < num_cores; ++core_index) {
        CoreCoord core = {core_index / num_cores_y, core_index % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src0_buffer->address();
            runtime_args[1] = src1_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
