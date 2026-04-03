// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "optimized_matmul_policy.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

struct OptimizedMatmulDeviceOperation {
    struct operation_attributes_t {
        MemoryConfig output_memory_config;
        MathFidelity math_fidelity;
        bool input_a_is_dram;
        bool optimized_a_read;
        bool optimized_b_read;
        bool optimized_write;
        bool packer_l1_acc;
        bool optimized_write_use_generated_schedule;
        uint32_t active_grid_x;
        uint32_t active_grid_y;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "output_memory_config",
            "math_fidelity",
            "input_a_is_dram",
            "optimized_a_read",
            "optimized_b_read",
            "optimized_write",
            "packer_l1_acc",
            "optimized_write_use_generated_schedule",
            "active_grid_x",
            "active_grid_y");
        auto attribute_values() const {
            return std::forward_as_tuple(
                output_memory_config,
                math_fidelity,
                input_a_is_dram,
                optimized_a_read,
                optimized_b_read,
                optimized_write,
                packer_l1_acc,
                optimized_write_use_generated_schedule,
                active_grid_x,
                active_grid_y);
        }
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCoreProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle dmvk_noc0_kernel_id;
            tt::tt_metal::KernelHandle dmvk_noc1_kernel_id;
            uint32_t active_grid_x;
            uint32_t active_grid_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MultiCoreProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul

namespace ttnn::prim {
constexpr auto optimized_matmul = ttnn::register_operation<
    "ttnn::prim::optimized_matmul",
    ttnn::operations::experimental::matmul::optimized_matmul::OptimizedMatmulDeviceOperation>();
}  // namespace ttnn::prim
