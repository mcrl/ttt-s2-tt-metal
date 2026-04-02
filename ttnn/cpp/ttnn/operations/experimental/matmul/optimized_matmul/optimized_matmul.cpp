// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul.hpp"

#include "device/optimized_matmul_device_operation.hpp"

namespace ttnn::operations::experimental::matmul {

ttnn::Tensor OptimizedMatmulOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::optimized_matmul(input_tensor_a, input_tensor_b, compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::matmul
