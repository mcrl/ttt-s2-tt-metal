// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::matmul {

struct OptimizedMatmulOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        std::optional<std::array<uint32_t, 3>> matmul_shape_override = std::nullopt,
        std::optional<std::vector<uint32_t>> output_shape_override = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const DataType>& dtype = std::nullopt);
};

}  // namespace ttnn::operations::experimental::matmul

namespace ttnn::experimental {

constexpr auto optimized_matmul = ttnn::register_operation<
    "ttnn::experimental::optimized_matmul",
    ttnn::operations::experimental::matmul::OptimizedMatmulOperation>();

}  // namespace ttnn::experimental
