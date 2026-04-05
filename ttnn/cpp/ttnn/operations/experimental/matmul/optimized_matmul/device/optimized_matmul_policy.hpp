// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

struct OptimizedMatmulVariantSpec {
    bool input_a_is_dram;
    bool input_b_is_dram;
    bool optimized_a_read;
    bool optimized_b_read;
    bool optimized_write;
    bool packer_l1_acc;
    bool optimized_write_use_generated_schedule;
    tt::tt_metal::CoreCoord active_grid;
};

inline bool is_interleaved_buffer_type(const Tensor& tensor, const BufferType buffer_type) {
    const auto& memory_config = tensor.memory_config();
    return !tensor.is_sharded() && memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
           memory_config.buffer_type() == buffer_type;
}

inline std::map<std::string, std::string> get_optimized_matmul_kernel_defines(
    const OptimizedMatmulVariantSpec& variant_spec) {
    return {
        {"OPTIMIZED_A_READ", variant_spec.optimized_a_read ? "1" : "0"},
        {"OPTIMIZED_B_READ", variant_spec.optimized_b_read ? "1" : "0"},
        {"OPTIMIZED_WRITE", variant_spec.optimized_write ? "1" : "0"},
        {"OPTIMIZED_WRITE_USE_GENERATED_SCHEDULE", variant_spec.optimized_write_use_generated_schedule ? "1" : "0"},
        {"PACKER_L1_ACC", variant_spec.packer_l1_acc ? "1" : "0"},
        {"SKIP_READ", "0"},
        {"SKIP_BCAST", "0"},
        {"SKIP_WRITE", "0"},
        {"SKIP_COMPUTE", "0"},
    };
}

inline OptimizedMatmulVariantSpec resolve_optimized_matmul_variant_spec(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    auto* mesh_device = input_tensor_a.device();
    TT_FATAL(mesh_device != nullptr, "optimized_matmul requires input A to be allocated on a mesh device");

    const auto& a_shape = input_tensor_a.padded_shape();
    const auto& b_shape = input_tensor_b.padded_shape();
    const auto mt = a_shape[-2] / tt::constants::TILE_HEIGHT;
    const auto nt = b_shape[-1] / tt::constants::TILE_WIDTH;
    const auto kt = a_shape[-1] / tt::constants::TILE_WIDTH;

    (void)mt;
    (void)nt;
    (void)kt;

    const bool input_a_is_dram = is_interleaved_buffer_type(input_tensor_a, BufferType::DRAM);
    const bool input_a_is_interleaved_l1 = is_interleaved_buffer_type(input_tensor_a, BufferType::L1);
    const bool input_b_is_dram = is_interleaved_buffer_type(input_tensor_b, BufferType::DRAM);
    const bool input_b_is_interleaved_l1 = is_interleaved_buffer_type(input_tensor_b, BufferType::L1);

    auto active_grid = tt::tt_metal::CoreCoord{8, 8};
    auto row_only_active_grid = tt::tt_metal::CoreCoord{8, 1};
    auto col_only_active_grid = tt::tt_metal::CoreCoord{1, 8};
    bool optimized_write_use_generated_schedule = false;

    if (mesh_device->arch() == tt::ARCH::BLACKHOLE) {
        // Blackhole experiments in this fork use a fixed 12x10 active grid.
        active_grid = tt::tt_metal::CoreCoord{12, 10};
        row_only_active_grid = tt::tt_metal::CoreCoord{12, 1};
        col_only_active_grid = tt::tt_metal::CoreCoord{1, 10};
    }

    // When both Mt and Nt are 1, prefer the Mt==1 rule.
    if (mt == 1) {
        optimized_write_use_generated_schedule = true;
        active_grid = row_only_active_grid;
    } else if (nt == 1) {
        optimized_write_use_generated_schedule = true;
        active_grid = col_only_active_grid;
    }

    return OptimizedMatmulVariantSpec{
        .input_a_is_dram = input_a_is_dram,
        .input_b_is_dram = input_b_is_dram,
        .optimized_a_read = !input_a_is_interleaved_l1,
        .optimized_b_read = !input_b_is_interleaved_l1,
        .optimized_write = true,
        .packer_l1_acc = true,
        .optimized_write_use_generated_schedule = optimized_write_use_generated_schedule,
        .active_grid = active_grid,
    };
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
