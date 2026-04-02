// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::matmul::optimized_matmul {

namespace wormhole_b0_8x8_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_schedule.hpp"
}  // namespace wormhole_b0_8x8_schedule

namespace wormhole_b0_8x1_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_schedule.hpp"
}  // namespace wormhole_b0_8x1_schedule

namespace wormhole_b0_1x8_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_schedule.hpp"
}  // namespace wormhole_b0_1x8_schedule

struct OptimizedMatmulScheduleMetadata {
    uint32_t a_read_chunks_per_core;
    uint32_t a_read_valid_slots_per_bank_count;
    uint32_t b_read_chunks_per_core;
    uint32_t b_read_valid_slots_per_bank_count;
    uint32_t c_write_chunks_per_core;
    uint32_t c_write_valid_slots_per_bank_count;
    uint32_t c_write_hardcoded_num_phase;
    uint32_t c_write_hardcoded_valid_slots_per_bank_count;
};

inline OptimizedMatmulScheduleMetadata get_optimized_matmul_schedule_metadata(
    const tt::tt_metal::CoreCoord& active_grid) {
    if (active_grid.x == 8 && active_grid.y == 8) {
        return {
            .a_read_chunks_per_core = wormhole_b0_8x8_schedule::A_READ_CHUNKS_PER_CORE,
            .a_read_valid_slots_per_bank_count = wormhole_b0_8x8_schedule::A_READ_valid_slots_per_bank_count,
            .b_read_chunks_per_core = wormhole_b0_8x8_schedule::B_READ_CHUNKS_PER_CORE,
            .b_read_valid_slots_per_bank_count = wormhole_b0_8x8_schedule::B_READ_valid_slots_per_bank_count,
            .c_write_chunks_per_core = wormhole_b0_8x8_schedule::C_WRITE_CHUNKS_PER_CORE,
            .c_write_valid_slots_per_bank_count = wormhole_b0_8x8_schedule::C_WRITE_valid_slots_per_bank_count,
            .c_write_hardcoded_num_phase = wormhole_b0_8x8_schedule::NUM_PHASE,
            .c_write_hardcoded_valid_slots_per_bank_count =
                wormhole_b0_8x8_schedule::C_WRITE_HARDCODED_valid_slots_per_bank_count,
        };
    }

    if (active_grid.x == 8 && active_grid.y == 1) {
        return {
            .a_read_chunks_per_core = wormhole_b0_8x1_schedule::A_READ_CHUNKS_PER_CORE,
            .a_read_valid_slots_per_bank_count = wormhole_b0_8x1_schedule::A_READ_valid_slots_per_bank_count,
            .b_read_chunks_per_core = wormhole_b0_8x1_schedule::B_READ_CHUNKS_PER_CORE,
            .b_read_valid_slots_per_bank_count = wormhole_b0_8x1_schedule::B_READ_valid_slots_per_bank_count,
            .c_write_chunks_per_core = wormhole_b0_8x1_schedule::C_WRITE_CHUNKS_PER_CORE,
            .c_write_valid_slots_per_bank_count = wormhole_b0_8x1_schedule::C_WRITE_valid_slots_per_bank_count,
            .c_write_hardcoded_num_phase = wormhole_b0_8x1_schedule::NUM_PHASE,
            .c_write_hardcoded_valid_slots_per_bank_count =
                wormhole_b0_8x1_schedule::C_WRITE_HARDCODED_valid_slots_per_bank_count,
        };
    }

    if (active_grid.x == 1 && active_grid.y == 8) {
        return {
            .a_read_chunks_per_core = wormhole_b0_1x8_schedule::A_READ_CHUNKS_PER_CORE,
            .a_read_valid_slots_per_bank_count = wormhole_b0_1x8_schedule::A_READ_valid_slots_per_bank_count,
            .b_read_chunks_per_core = wormhole_b0_1x8_schedule::B_READ_CHUNKS_PER_CORE,
            .b_read_valid_slots_per_bank_count = wormhole_b0_1x8_schedule::B_READ_valid_slots_per_bank_count,
            .c_write_chunks_per_core = wormhole_b0_1x8_schedule::C_WRITE_CHUNKS_PER_CORE,
            .c_write_valid_slots_per_bank_count = wormhole_b0_1x8_schedule::C_WRITE_valid_slots_per_bank_count,
            .c_write_hardcoded_num_phase = wormhole_b0_1x8_schedule::NUM_PHASE,
            .c_write_hardcoded_valid_slots_per_bank_count =
                wormhole_b0_1x8_schedule::C_WRITE_HARDCODED_valid_slots_per_bank_count,
        };
    }

    TT_THROW(
        "optimized_matmul does not have a vendored schedule for active grid {}x{}",
        active_grid.x,
        active_grid.y);
}

inline std::string get_optimized_matmul_schedule_header_include_path(const tt::tt_metal::CoreCoord& active_grid) {
    if (active_grid.x == 8 && active_grid.y == 8) {
        return "\"schedules/wormhole_b0_8x8_schedule.hpp\"";
    }
    if (active_grid.x == 8 && active_grid.y == 1) {
        return "\"schedules/wormhole_b0_8x1_schedule.hpp\"";
    }
    if (active_grid.x == 1 && active_grid.y == 8) {
        return "\"schedules/wormhole_b0_1x8_schedule.hpp\"";
    }

    TT_THROW(
        "optimized_matmul does not have a vendored schedule header for active grid {}x{}",
        active_grid.x,
        active_grid.y);
}

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
