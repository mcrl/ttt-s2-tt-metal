// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

enum class OptimizedMatmulVariantId : uint32_t {
    StandardInterleaved = 0,
    OptimizedABWriteHardcoded = 1,
    OptimizedABWriteGenerated = 2,
};

struct OptimizedMatmulVariantSpec {
    bool optimized_a_read;
    bool optimized_b_read;
    bool optimized_write;
    bool packer_l1_acc;
    bool optimized_write_use_generated_schedule;
};

struct OptimizedMatmulPolicy {
    OptimizedMatmulVariantId variant_id;
    tt::tt_metal::CoreCoord active_grid;
};

inline OptimizedMatmulVariantSpec get_optimized_matmul_variant_spec(const OptimizedMatmulVariantId variant_id) {
    switch (variant_id) {
        case OptimizedMatmulVariantId::StandardInterleaved:
            return {
                .optimized_a_read = false,
                .optimized_b_read = false,
                .optimized_write = false,
                .packer_l1_acc = true,
                .optimized_write_use_generated_schedule = false,
            };
        case OptimizedMatmulVariantId::OptimizedABWriteHardcoded:
            return {
                .optimized_a_read = true,
                .optimized_b_read = true,
                .optimized_write = true,
                .packer_l1_acc = true,
                .optimized_write_use_generated_schedule = false,
            };
        case OptimizedMatmulVariantId::OptimizedABWriteGenerated:
            return {
                .optimized_a_read = true,
                .optimized_b_read = true,
                .optimized_write = true,
                .packer_l1_acc = true,
                .optimized_write_use_generated_schedule = true,
            };
    }

    TT_THROW("Unsupported optimized_matmul variant id {}", static_cast<uint32_t>(variant_id));
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

inline OptimizedMatmulPolicy resolve_optimized_matmul_policy(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    const auto& a_shape = input_tensor_a.padded_shape();
    const auto& b_shape = input_tensor_b.padded_shape();
    const auto mt = a_shape[-2] / tt::constants::TILE_HEIGHT;
    const auto nt = b_shape[-1] / tt::constants::TILE_WIDTH;
    const auto kt = a_shape[-1] / tt::constants::TILE_WIDTH;

    (void)mt;
    (void)nt;
    (void)kt;

    // Phase 1 defaults to the fully optimized WH8x8 path.
    // Callers are expected to provide tensors whose underlying device buffers
    // already obey the optim raw A/B layout contract.
    return {
        .variant_id = OptimizedMatmulVariantId::OptimizedABWriteHardcoded,
        .active_grid = tt::tt_metal::CoreCoord{8, 8},
    };
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
