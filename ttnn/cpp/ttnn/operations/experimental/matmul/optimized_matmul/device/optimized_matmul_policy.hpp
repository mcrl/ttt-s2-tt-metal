// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

struct OptimizedMatmulShapeOverride {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

struct OptimizedMatmulEffectiveShapes {
    ttnn::Shape a_logical_shape;
    ttnn::Shape b_logical_shape;
    ttnn::Shape a_padded_shape;
    ttnn::Shape b_padded_shape;
};

struct OptimizedMatmulOutputShapeOverride {
    ttnn::Shape logical_shape;
    ttnn::Shape padded_shape;
};

struct OptimizedMatmulOutputShapes {
    ttnn::Shape logical_shape;
    ttnn::Shape padded_shape;
};

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

inline uint32_t round_up_u32(const uint32_t value, const uint32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

inline std::optional<OptimizedMatmulShapeOverride> make_optimized_matmul_shape_override(
    const std::optional<std::array<uint32_t, 3>>& matmul_shape_override) {
    if (!matmul_shape_override.has_value()) {
        return std::nullopt;
    }

    return OptimizedMatmulShapeOverride{
        .M = matmul_shape_override->at(0),
        .N = matmul_shape_override->at(1),
        .K = matmul_shape_override->at(2),
    };
}

inline std::optional<OptimizedMatmulOutputShapeOverride> make_optimized_matmul_output_shape_override(
    const std::optional<std::vector<uint32_t>>& output_shape_override) {
    if (!output_shape_override.has_value()) {
        return std::nullopt;
    }

    TT_FATAL(
        output_shape_override->size() >= 2,
        "optimized_matmul output_shape_override must have rank >= 2, got {}",
        output_shape_override->size());

    auto logical_shape_dims = tt::stl::SmallVector<uint32_t>(output_shape_override->begin(), output_shape_override->end());
    const auto logical_shape = ttnn::Shape{std::move(logical_shape_dims)};
    auto padded_shape = logical_shape;
    padded_shape[-2] = round_up_u32(logical_shape[-2], tt::constants::TILE_HEIGHT);
    padded_shape[-1] = round_up_u32(logical_shape[-1], tt::constants::TILE_WIDTH);

    return OptimizedMatmulOutputShapeOverride{
        .logical_shape = logical_shape,
        .padded_shape = padded_shape,
    };
}

inline OptimizedMatmulEffectiveShapes resolve_optimized_matmul_effective_shapes(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<OptimizedMatmulShapeOverride>& matmul_shape_override = std::nullopt) {
    if (!matmul_shape_override.has_value()) {
        return {
            .a_logical_shape = input_tensor_a.logical_shape(),
            .b_logical_shape = input_tensor_b.logical_shape(),
            .a_padded_shape = input_tensor_a.padded_shape(),
            .b_padded_shape = input_tensor_b.padded_shape(),
        };
    }

    auto a_logical_shape = input_tensor_a.logical_shape();
    auto b_logical_shape = input_tensor_b.logical_shape();
    auto a_padded_shape = input_tensor_a.padded_shape();
    auto b_padded_shape = input_tensor_b.padded_shape();

    a_logical_shape[-2] = matmul_shape_override->M;
    a_logical_shape[-1] = matmul_shape_override->K;
    b_logical_shape[-2] = matmul_shape_override->K;
    b_logical_shape[-1] = matmul_shape_override->N;

    a_padded_shape[-2] = round_up_u32(matmul_shape_override->M, tt::constants::TILE_HEIGHT);
    a_padded_shape[-1] = round_up_u32(matmul_shape_override->K, tt::constants::TILE_WIDTH);
    b_padded_shape[-2] = round_up_u32(matmul_shape_override->K, tt::constants::TILE_HEIGHT);
    b_padded_shape[-1] = round_up_u32(matmul_shape_override->N, tt::constants::TILE_WIDTH);

    return {
        .a_logical_shape = std::move(a_logical_shape),
        .b_logical_shape = std::move(b_logical_shape),
        .a_padded_shape = std::move(a_padded_shape),
        .b_padded_shape = std::move(b_padded_shape),
    };
}

inline OptimizedMatmulOutputShapes resolve_optimized_matmul_output_shapes(
    const OptimizedMatmulEffectiveShapes& effective_shapes,
    const std::optional<OptimizedMatmulOutputShapeOverride>& output_shape_override = std::nullopt) {
    if (output_shape_override.has_value()) {
        return {
            .logical_shape = output_shape_override->logical_shape,
            .padded_shape = output_shape_override->padded_shape,
        };
    }

    auto logical_shape = effective_shapes.a_logical_shape;
    auto padded_shape = effective_shapes.a_padded_shape;
    logical_shape[-1] = effective_shapes.b_logical_shape[-1];
    padded_shape[-1] = effective_shapes.b_padded_shape[-1];

    return {
        .logical_shape = std::move(logical_shape),
        .padded_shape = std::move(padded_shape),
    };
}

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
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<OptimizedMatmulShapeOverride>& matmul_shape_override = std::nullopt) {
    const auto effective_shapes =
        resolve_optimized_matmul_effective_shapes(input_tensor_a, input_tensor_b, matmul_shape_override);
    const auto& a_shape = effective_shapes.a_padded_shape;
    const auto& b_shape = effective_shapes.b_padded_shape;
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
    bool optimized_write_use_generated_schedule = false;

    // When both Mt and Nt are 1, prefer the Mt==1 rule.
    if (mt == 1) {
        optimized_write_use_generated_schedule = true;
        active_grid = tt::tt_metal::CoreCoord{8, 1};
    } else if (nt == 1) {
        optimized_write_use_generated_schedule = true;
        active_grid = tt::tt_metal::CoreCoord{1, 8};
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
