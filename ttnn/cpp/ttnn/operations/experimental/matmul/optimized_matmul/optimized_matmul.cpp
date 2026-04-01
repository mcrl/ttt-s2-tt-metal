// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul.hpp"

#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::operations::experimental::matmul {

ttnn::Tensor OptimizedMatmulOperation::invoke(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    return ttnn::matmul(input_tensor_a, input_tensor_b);
}

}  // namespace ttnn::operations::experimental::matmul
