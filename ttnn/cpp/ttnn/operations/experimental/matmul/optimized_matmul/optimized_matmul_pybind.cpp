// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimized_matmul_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
            - input_tensor_a must use DRAM interleaved memory, or interleaved L1 to trigger a standard-read fallback
            - input_tensor_b must use DRAM interleaved memory, or interleaved L1 to trigger a standard-read fallback
            - input_tensor_a, input_tensor_b, and dtype independently support BFLOAT16 and BFLOAT8_B
            - output supports ttnn.DRAM_MEMORY_CONFIG and ttnn.L1_MEMORY_CONFIG
            - L1 output triggers a standard-write fallback
            - both inputs must be device TILE tensors
            - dtype is optional and controls output dtype; when omitted it defaults to input_tensor_a.dtype()
            - compute_kernel_config is optional and currently only math_fidelity is consumed
            - matmul_shape_override is optional and, when provided as [M, N, K], makes optimized_matmul behave
              as if A had shape [..., M, K] and B had shape [..., K, N] without requiring the tensor metadata to
              match those dimensions
            - output_shape_override is optional and, when provided as an integer tuple, sets the returned tensor
              metadata shape directly while keeping kernel execution based on matmul_shape_override
            - supported math_fidelity values are LoFi, HiFi2, and HiFi4
        )doc",
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input_tensor_a,
               const Tensor& input_tensor_b,
               const std::optional<std::array<uint32_t, 3>>& matmul_shape_override,
               const std::optional<std::vector<uint32_t>>& output_shape_override,
               const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const DataType>& dtype) {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    matmul_shape_override,
                    output_shape_override,
                    compute_kernel_config,
                    memory_config,
                    dtype);
            },
            py::arg("input_tensor_a").noconvert(),
            py::arg("input_tensor_b").noconvert(),
            py::arg("matmul_shape_override") = std::nullopt,
            py::arg("output_shape_override") = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("dtype") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::matmul::detail
