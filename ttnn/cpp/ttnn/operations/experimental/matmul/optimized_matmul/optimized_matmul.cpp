// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul.hpp"

#include "device/optimized_matmul_device_operation.hpp"

namespace ttnn::operations::experimental::matmul {

ttnn::Tensor OptimizedMatmulOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<std::array<uint32_t, 3>> matmul_shape_override,
    std::optional<std::vector<uint32_t>> output_shape_override,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType>& dtype) {
    return ttnn::prim::optimized_matmul(
        input_tensor_a,
        input_tensor_b,
        matmul_shape_override,
        output_shape_override,
        compute_kernel_config,
        memory_config,
        dtype);
}

}  // namespace ttnn::operations::experimental::matmul
