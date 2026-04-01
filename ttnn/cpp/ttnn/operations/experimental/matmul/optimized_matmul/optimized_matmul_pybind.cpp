// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul_pybind.hpp"

#include <pybind11/pybind11.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/matmul/optimized_matmul/optimized_matmul.hpp"

namespace ttnn::operations::experimental::matmul::detail {

namespace py = pybind11;

void bind_optimized_matmul(py::module& module) {
    using OperationType = decltype(ttnn::experimental::optimized_matmul);

    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::optimized_matmul,
        R"doc(
            Performs matrix multiplication for two tensors through the experimental optimized_matmul API.

            This native op is intentionally registered as a distinct experimental entrypoint with its own
            minimal device program path under optimized_matmul/.

            Current constraints:
            - input_tensor_a must use ttnn.DRAM_MEMORY_CONFIG
            - input_tensor_b must use ttnn.DRAM_MEMORY_CONFIG
            - output is always produced with ttnn.DRAM_MEMORY_CONFIG
            - both inputs must be device TILE tensors
        )doc",
        ttnn::pybind_overload_t{
            [](const OperationType& self, const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
                return self(input_tensor_a, input_tensor_b);
            },
            py::arg("input_tensor_a").noconvert(),
            py::arg("input_tensor_b").noconvert()});
}

}  // namespace ttnn::operations::experimental::matmul::detail
