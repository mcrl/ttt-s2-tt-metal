// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>

#include "optimized_matmul_policy.hpp"
#include "optimized_matmul_schedule.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::matmul::optimized_matmul {

struct OptimizedMatmulConfig {
    uint32_t Mt;
    uint32_t Nt;
    uint32_t Kt;
    uint32_t BMt;
    uint32_t BNt;
    uint32_t BKt;
    uint32_t SBMt;
    uint32_t SBNt;
    uint32_t total_row_blocks;
    uint32_t total_col_blocks;
    uint32_t row_nblocks_per_core;
    uint32_t col_nblocks_per_core;
};

struct OptimizedMatmulBufferLayout {
    uint32_t a_page_size;
    uint32_t a_total_size;
    uint32_t b_page_size;
    uint32_t b_total_size;
    uint32_t c_page_size;
    uint32_t c_total_size;
};

inline uint32_t ceil_div_u32(const uint32_t value, const uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline OptimizedMatmulConfig resolve_optimized_matmul_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::CoreCoord& active_grid,
    const std::optional<OptimizedMatmulShapeOverride>& matmul_shape_override = std::nullopt) {
    using namespace ttnn::operations::matmul;

    const auto effective_shapes =
        resolve_optimized_matmul_effective_shapes(input_tensor_a, input_tensor_b, matmul_shape_override);
    const auto virtual_input_tensor_a =
        input_tensor_a.reshape(effective_shapes.a_logical_shape, effective_shapes.a_padded_shape);
    const auto virtual_input_tensor_b =
        input_tensor_b.reshape(effective_shapes.b_logical_shape, effective_shapes.b_padded_shape);
    const auto program_config = resolve_matmul_2d_reuse_program_config(
        virtual_input_tensor_a,
        virtual_input_tensor_b,
        std::nullopt,
        Matmul{.output_dtype = output_dtype},
        std::nullopt);

    const auto& a_shape = effective_shapes.a_padded_shape;
    const auto& b_shape = effective_shapes.b_padded_shape;

    const auto mt = static_cast<uint32_t>(a_shape[-2] / tt::constants::TILE_HEIGHT);
    const auto nt = static_cast<uint32_t>(b_shape[-1] / tt::constants::TILE_WIDTH);
    const auto kt = static_cast<uint32_t>(a_shape[-1] / tt::constants::TILE_WIDTH);

    const auto bmt = static_cast<uint32_t>(program_config.out_block_h);
    const auto bnt = static_cast<uint32_t>(program_config.out_block_w);
    const auto bkt = static_cast<uint32_t>(program_config.in0_block_w);
    const auto sbmt = static_cast<uint32_t>(program_config.out_subblock_h);
    const auto sbnt = static_cast<uint32_t>(program_config.out_subblock_w);

    TT_FATAL(bmt > 0 && bnt > 0 && bkt > 0, "optimized_matmul resolved an invalid block config");
    TT_FATAL(sbmt > 0 && sbnt > 0, "optimized_matmul resolved an invalid subblock config");
    TT_FATAL(bmt % sbmt == 0 && bnt % sbnt == 0, "optimized_matmul requires block sizes divisible by subblocks");
    TT_FATAL(kt % bkt == 0, "optimized_matmul requires K tiles {} to be divisible by BKt {}", kt, bkt);

    const auto total_row_blocks = ceil_div_u32(mt, bmt);
    const auto total_col_blocks = ceil_div_u32(nt, bnt);

    return {
        .Mt = mt,
        .Nt = nt,
        .Kt = kt,
        .BMt = bmt,
        .BNt = bnt,
        .BKt = bkt,
        .SBMt = sbmt,
        .SBNt = sbnt,
        .total_row_blocks = total_row_blocks,
        .total_col_blocks = total_col_blocks,
        .row_nblocks_per_core = ceil_div_u32(total_row_blocks, active_grid.y),
        .col_nblocks_per_core = ceil_div_u32(total_col_blocks, active_grid.x),
    };
}

inline uint32_t split_balanced_items(const uint32_t total_items, const uint32_t chunks_per_core, const uint32_t chunk_idx) {
    if (chunk_idx >= chunks_per_core) {
        return 0;
    }

    const uint32_t base_items = total_items / chunks_per_core;
    const uint32_t remainder = total_items % chunks_per_core;
    return chunk_idx < remainder ? base_items + 1 : base_items;
}

inline uint32_t split_front_loaded_items(
    const uint32_t total_items, const uint32_t chunks_per_core, const uint32_t chunk_idx) {
    if (chunk_idx >= chunks_per_core) {
        return 0;
    }

    const uint32_t base_items = total_items / chunks_per_core;
    if (chunk_idx == 0) {
        return total_items - base_items * (chunks_per_core - 1);
    }
    return base_items;
}

inline OptimizedMatmulBufferLayout resolve_optimized_matmul_buffer_layout(
    const OptimizedMatmulConfig& config,
    const OptimizedMatmulVariantSpec& variant_spec,
    const uint32_t a_tile_size,
    const uint32_t b_tile_size,
    const uint32_t c_tile_size,
    const uint32_t dram_bank_count) {
    const auto schedule_metadata = get_optimized_matmul_schedule_metadata(variant_spec.active_grid);
    const uint32_t logical_a_size = a_tile_size * config.Mt * config.Kt;
    const uint32_t logical_b_size = b_tile_size * config.Kt * config.Nt;
    const uint32_t logical_c_size = c_tile_size * config.Mt * config.Nt;
    const uint32_t num_kblocks = config.Kt / config.BKt;
    const uint32_t repetitions_a = config.row_nblocks_per_core * num_kblocks;
    const uint32_t repetitions_b = config.col_nblocks_per_core * num_kblocks;
    const uint32_t repetitions_c = config.row_nblocks_per_core * config.col_nblocks_per_core;
    const uint32_t a_tiles_per_block = config.BMt * config.BKt;
    const uint32_t b_tiles_per_block = config.BKt * config.BNt;
    const uint32_t n_subblocks = (config.BMt / config.SBMt) * (config.BNt / config.SBNt);
    const uint32_t subblock_bytes = c_tile_size * config.SBMt * config.SBNt;

    auto set_slot_layout = [dram_bank_count](
                               const bool enabled,
                               const uint32_t page_size_if_disabled,
                               const uint32_t logical_size,
                               const uint32_t repetitions,
                               const uint32_t valid_slots_per_bank_count,
                               const uint32_t slot_bytes) {
        std::pair<uint32_t, uint32_t> result{};
        if (!enabled) {
            result.first = page_size_if_disabled;
            result.second = logical_size;
            return result;
        }

        result.first = repetitions * valid_slots_per_bank_count * slot_bytes;
        result.second = result.first * dram_bank_count;
        return result;
    };

    const auto [a_page_size, a_total_size] = set_slot_layout(
        variant_spec.optimized_a_read,
        a_tile_size,
        logical_a_size,
        repetitions_a,
        schedule_metadata.a_read_valid_slots_per_bank_count,
        split_balanced_items(a_tiles_per_block, schedule_metadata.a_read_chunks_per_core, 0) * a_tile_size);

    const auto [b_page_size, b_total_size] = set_slot_layout(
        variant_spec.optimized_b_read,
        b_tile_size,
        logical_b_size,
        repetitions_b,
        schedule_metadata.b_read_valid_slots_per_bank_count,
        split_balanced_items(b_tiles_per_block, schedule_metadata.b_read_chunks_per_core, 0) * b_tile_size);

    const uint32_t c_chunks = variant_spec.optimized_write_use_generated_schedule
                                  ? schedule_metadata.c_write_chunks_per_core
                                  : schedule_metadata.c_write_hardcoded_num_phase;
    const uint32_t c_valid_slots_per_bank_count = variant_spec.optimized_write_use_generated_schedule
                                                      ? schedule_metadata.c_write_valid_slots_per_bank_count
                                                      : schedule_metadata.c_write_hardcoded_valid_slots_per_bank_count;
    const auto [c_page_size, c_total_size] = set_slot_layout(
        variant_spec.optimized_write,
        c_tile_size,
        logical_c_size,
        repetitions_c,
        c_valid_slots_per_bank_count,
        split_front_loaded_items(n_subblocks, c_chunks, 0) * subblock_bytes);

    return {
        .a_page_size = a_page_size,
        .a_total_size = a_total_size,
        .b_page_size = b_page_size,
        .b_total_size = b_total_size,
        .c_page_size = c_page_size,
        .c_total_size = c_total_size,
    };
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
