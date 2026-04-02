// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::matmul {

struct OptimizedMatmulOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::matmul

namespace ttnn::experimental {

constexpr auto optimized_matmul = ttnn::register_operation<
    "ttnn::experimental::optimized_matmul",
    ttnn::operations::experimental::matmul::OptimizedMatmulOperation>();

}  // namespace ttnn::experimental
