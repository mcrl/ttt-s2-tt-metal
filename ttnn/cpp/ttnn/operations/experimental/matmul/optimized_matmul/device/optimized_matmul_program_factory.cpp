// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul_device_operation.hpp"

#include <cstdint>
#include <map>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include "optimized_matmul_config.hpp"
#include "optimized_matmul_schedule.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

using namespace tt;
using namespace tt::constants;

namespace {

OptimizedMatmulVariantSpec get_program_factory_variant_spec_from_attributes(
    const OptimizedMatmulDeviceOperation::operation_attributes_t& operation_attributes) {
    return {
        .input_a_is_dram = operation_attributes.input_a_is_dram,
        .optimized_a_read = operation_attributes.optimized_a_read,
        .optimized_b_read = operation_attributes.optimized_b_read,
        .optimized_write = operation_attributes.optimized_write,
        .packer_l1_acc = operation_attributes.packer_l1_acc,
        .optimized_write_use_generated_schedule = operation_attributes.optimized_write_use_generated_schedule,
        .active_grid = tt::tt_metal::CoreCoord{operation_attributes.active_grid_x, operation_attributes.active_grid_y},
    };
}

constexpr uint32_t kInputPipelineDepth = 2;
constexpr uint32_t kOutputPipelineDepth = 1;
constexpr uint32_t kSyncPageSize = 4;
tt::tt_metal::CoreRange get_active_cores(const OptimizedMatmulDeviceOperation::operation_attributes_t& attributes) {
    return tt::tt_metal::CoreRange({0, 0}, {attributes.active_grid_x - 1, attributes.active_grid_y - 1});
}

void create_data_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRange& cores,
    const uint32_t cb_index,
    const uint32_t num_tiles,
    const tt::DataFormat data_format) {
    using namespace tt::tt_metal;

    CircularBufferConfig cb_config =
        CircularBufferConfig(tt::tile_size(data_format) * num_tiles, {{cb_index, data_format}})
            .set_page_size(cb_index, tt::tile_size(data_format));
    CreateCircularBuffer(program, cores, cb_config);
}

void create_sync_circular_buffer(
    tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& cores, const uint32_t cb_index) {
    using namespace tt::tt_metal;

    CircularBufferConfig cb_config =
        CircularBufferConfig(kSyncPageSize, {{cb_index, tt::DataFormat::UInt32}}).set_page_size(cb_index, kSyncPageSize);
    CreateCircularBuffer(program, cores, cb_config);
}

std::map<std::string, std::string> create_kernel_defines(const OptimizedMatmulVariantSpec& variant_spec) {
    auto defines = get_optimized_matmul_kernel_defines(variant_spec);
    defines["TTT_DMVK_SCHEDULE_HEADER"] = get_optimized_matmul_schedule_header_include_path(variant_spec.active_grid);
    return defines;
}

}  // namespace

OptimizedMatmulDeviceOperation::MultiCoreProgramFactory::cached_program_t
OptimizedMatmulDeviceOperation::MultiCoreProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

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

    const auto variant_spec = get_program_factory_variant_spec_from_attributes(operation_attributes);
    const auto resolved_config = resolve_optimized_matmul_config(
        input_tensor_a,
        input_tensor_b,
        tt::tt_metal::CoreCoord{operation_attributes.active_grid_x, operation_attributes.active_grid_y});

    Program program{};

    const auto all_cores = get_active_cores(operation_attributes);
    const auto cb_data_format = datatype_to_dataformat_converter(input_tensor_a.dtype());

    create_data_circular_buffer(
        program, all_cores, tt::CBIndex::c_0, resolved_config.BMt * resolved_config.BKt * kInputPipelineDepth, cb_data_format);
    create_data_circular_buffer(
        program, all_cores, tt::CBIndex::c_1, resolved_config.BKt * resolved_config.BNt * kInputPipelineDepth, cb_data_format);
    create_data_circular_buffer(
        program, all_cores, tt::CBIndex::c_16, resolved_config.BMt * resolved_config.BNt * kOutputPipelineDepth, cb_data_format);
    create_data_circular_buffer(
        program, all_cores, tt::CBIndex::c_24, resolved_config.BMt * resolved_config.BNt * kOutputPipelineDepth, cb_data_format);
    create_sync_circular_buffer(program, all_cores, tt::CBIndex::c_25);
    create_sync_circular_buffer(program, all_cores, tt::CBIndex::c_26);

    const auto a_master_sem = CreateSemaphore(program, all_cores, 0);
    const auto a_slave_sem = CreateSemaphore(program, all_cores, INVALID);
    const auto b_master_sem = CreateSemaphore(program, all_cores, 0);
    const auto b_slave_sem = CreateSemaphore(program, all_cores, INVALID);
    const auto global_master_sem = CreateSemaphore(program, all_cores, 0);
    const auto global_slave_sem = CreateSemaphore(program, all_cores, INVALID);

    const std::vector<uint32_t> dataflow_compile_args = {
        a_master_sem,
        a_slave_sem,
        b_master_sem,
        b_slave_sem,
        global_master_sem,
        global_slave_sem,
        operation_attributes.active_grid_x,
        operation_attributes.active_grid_y,
        operation_attributes.input_a_is_dram ? 1U : 0U,
    };

    const auto kernel_defines = create_kernel_defines(variant_spec);

    const auto dmvk_noc0_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/device/kernels/dataflow/dmvk_optimized_matmul.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = dataflow_compile_args,
            .defines = kernel_defines});

    const auto dmvk_noc1_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/device/kernels/dataflow/dmvk_optimized_matmul.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = dataflow_compile_args,
            .defines = kernel_defines});

    const std::vector<uint32_t> compute_compile_args = {
        resolved_config.Kt / resolved_config.BKt,
        resolved_config.BMt,
        resolved_config.BNt,
        resolved_config.BKt,
        resolved_config.SBMt,
        resolved_config.SBNt,
        resolved_config.SBMt * resolved_config.SBNt,
        resolved_config.BMt * resolved_config.BNt,
        resolved_config.BMt * resolved_config.BKt,
        resolved_config.BKt * resolved_config.BNt,
    };

    const auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/device/kernels/compute/optimized_bmm.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = operation_attributes.math_fidelity,
            .compile_args = compute_compile_args,
            .defines = kernel_defines});

    const auto src0_addr = src0_buffer->address();
    const auto src1_addr = src1_buffer->address();
    const auto dst_addr = dst_buffer->address();

    for (uint32_t y = 0; y < operation_attributes.active_grid_y; ++y) {
        for (uint32_t x = 0; x < operation_attributes.active_grid_x; ++x) {
            const CoreCoord core = {x, y};
            const uint32_t row_block_start = y * resolved_config.row_nblocks_per_core;
            const uint32_t col_block_start = x * resolved_config.col_nblocks_per_core;

            const std::vector<uint32_t> dataflow_runtime_args = {
                src0_addr,
                src1_addr,
                dst_addr,
                resolved_config.Mt,
                resolved_config.Nt,
                resolved_config.Kt,
                resolved_config.BMt,
                resolved_config.BNt,
                resolved_config.BKt,
                resolved_config.SBMt,
                resolved_config.SBNt,
                row_block_start,
                col_block_start,
                resolved_config.row_nblocks_per_core,
                resolved_config.col_nblocks_per_core,
            };

            SetRuntimeArgs(program, dmvk_noc0_kernel_id, core, dataflow_runtime_args);
            SetRuntimeArgs(program, dmvk_noc1_kernel_id, core, dataflow_runtime_args);
            SetRuntimeArgs(
                program,
                compute_kernel_id,
                core,
                {row_block_start, col_block_start, resolved_config.row_nblocks_per_core, resolved_config.col_nblocks_per_core});
        }
    }

    return {
        std::move(program),
        {
            .dmvk_noc0_kernel_id = dmvk_noc0_kernel_id,
            .dmvk_noc1_kernel_id = dmvk_noc1_kernel_id,
            .active_grid_x = operation_attributes.active_grid_x,
            .active_grid_y = operation_attributes.active_grid_y,
        }};
}

void OptimizedMatmulDeviceOperation::MultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto* src0_buffer = tensor_args.input_tensor_a.buffer();
    auto* src1_buffer = tensor_args.input_tensor_b.buffer();
    auto* dst_buffer = output_tensor.buffer();

    TT_FATAL(
        output_tensor.memory_config() == operation_attributes.output_memory_config,
        "optimized_matmul output memory config mismatch: expected {} but got {}",
        operation_attributes.output_memory_config,
        output_tensor.memory_config());

    for (uint32_t y = 0; y < cached_program.shared_variables.active_grid_y; ++y) {
        for (uint32_t x = 0; x < cached_program.shared_variables.active_grid_x; ++x) {
            const CoreCoord core = {x, y};

            for (const auto kernel_id : {
                     cached_program.shared_variables.dmvk_noc0_kernel_id,
                     cached_program.shared_variables.dmvk_noc1_kernel_id}) {
                auto& runtime_args = GetRuntimeArgs(program, kernel_id, core);
                runtime_args[0] = src0_buffer->address();
                runtime_args[1] = src1_buffer->address();
                runtime_args[2] = dst_buffer->address();
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
