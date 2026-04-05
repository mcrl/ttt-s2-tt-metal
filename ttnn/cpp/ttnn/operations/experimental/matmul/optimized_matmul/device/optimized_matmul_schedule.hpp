// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::matmul::optimized_matmul {

#ifndef FORCE_INLINE
#define TTNN_OPTIMIZED_MATMUL_SCHEDULE_FORCE_INLINE_DEFINED
#define FORCE_INLINE inline
#endif

namespace wormhole_b0_8x8_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_schedule.hpp"
}  // namespace wormhole_b0_8x8_schedule

namespace wormhole_b0_8x8_0_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_0_schedule.hpp"
}  // namespace wormhole_b0_8x8_0_schedule

namespace wormhole_b0_8x8_1_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_1_schedule.hpp"
}  // namespace wormhole_b0_8x8_1_schedule

namespace wormhole_b0_8x8_2_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_2_schedule.hpp"
}  // namespace wormhole_b0_8x8_2_schedule

namespace wormhole_b0_8x8_3_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_3_schedule.hpp"
}  // namespace wormhole_b0_8x8_3_schedule

namespace wormhole_b0_8x8_4_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_4_schedule.hpp"
}  // namespace wormhole_b0_8x8_4_schedule

namespace wormhole_b0_8x8_5_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_5_schedule.hpp"
}  // namespace wormhole_b0_8x8_5_schedule

namespace wormhole_b0_8x8_6_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_6_schedule.hpp"
}  // namespace wormhole_b0_8x8_6_schedule

namespace wormhole_b0_8x8_7_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x8_7_schedule.hpp"
}  // namespace wormhole_b0_8x8_7_schedule

namespace wormhole_b0_8x1_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_schedule.hpp"
}  // namespace wormhole_b0_8x1_schedule

namespace wormhole_b0_8x1_0_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_0_schedule.hpp"
}  // namespace wormhole_b0_8x1_0_schedule

namespace wormhole_b0_8x1_1_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_1_schedule.hpp"
}  // namespace wormhole_b0_8x1_1_schedule

namespace wormhole_b0_8x1_2_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_2_schedule.hpp"
}  // namespace wormhole_b0_8x1_2_schedule

namespace wormhole_b0_8x1_3_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_3_schedule.hpp"
}  // namespace wormhole_b0_8x1_3_schedule

namespace wormhole_b0_8x1_4_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_4_schedule.hpp"
}  // namespace wormhole_b0_8x1_4_schedule

namespace wormhole_b0_8x1_5_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_5_schedule.hpp"
}  // namespace wormhole_b0_8x1_5_schedule

namespace wormhole_b0_8x1_6_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_6_schedule.hpp"
}  // namespace wormhole_b0_8x1_6_schedule

namespace wormhole_b0_8x1_7_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_8x1_7_schedule.hpp"
}  // namespace wormhole_b0_8x1_7_schedule

namespace wormhole_b0_1x8_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_schedule.hpp"
}  // namespace wormhole_b0_1x8_schedule

namespace wormhole_b0_1x8_0_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_0_schedule.hpp"
}  // namespace wormhole_b0_1x8_0_schedule

namespace wormhole_b0_1x8_1_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_1_schedule.hpp"
}  // namespace wormhole_b0_1x8_1_schedule

namespace wormhole_b0_1x8_2_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_2_schedule.hpp"
}  // namespace wormhole_b0_1x8_2_schedule

namespace wormhole_b0_1x8_3_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_3_schedule.hpp"
}  // namespace wormhole_b0_1x8_3_schedule

namespace wormhole_b0_1x8_4_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_4_schedule.hpp"
}  // namespace wormhole_b0_1x8_4_schedule

namespace wormhole_b0_1x8_5_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_5_schedule.hpp"
}  // namespace wormhole_b0_1x8_5_schedule

namespace wormhole_b0_1x8_6_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_6_schedule.hpp"
}  // namespace wormhole_b0_1x8_6_schedule

namespace wormhole_b0_1x8_7_schedule {
#include "kernels/dataflow/schedules/wormhole_b0_1x8_7_schedule.hpp"
}  // namespace wormhole_b0_1x8_7_schedule

#ifdef TTNN_OPTIMIZED_MATMUL_SCHEDULE_FORCE_INLINE_DEFINED
#undef FORCE_INLINE
#undef TTNN_OPTIMIZED_MATMUL_SCHEDULE_FORCE_INLINE_DEFINED
#endif

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

constexpr bool operator==(
    const OptimizedMatmulScheduleMetadata& lhs, const OptimizedMatmulScheduleMetadata& rhs) {
    return lhs.a_read_chunks_per_core == rhs.a_read_chunks_per_core &&
           lhs.a_read_valid_slots_per_bank_count == rhs.a_read_valid_slots_per_bank_count &&
           lhs.b_read_chunks_per_core == rhs.b_read_chunks_per_core &&
           lhs.b_read_valid_slots_per_bank_count == rhs.b_read_valid_slots_per_bank_count &&
           lhs.c_write_chunks_per_core == rhs.c_write_chunks_per_core &&
           lhs.c_write_valid_slots_per_bank_count == rhs.c_write_valid_slots_per_bank_count &&
           lhs.c_write_hardcoded_num_phase == rhs.c_write_hardcoded_num_phase &&
           lhs.c_write_hardcoded_valid_slots_per_bank_count == rhs.c_write_hardcoded_valid_slots_per_bank_count;
}

#define TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(NAME)                         \
    OptimizedMatmulScheduleMetadata {                                         \
        .a_read_chunks_per_core = NAME::A_READ_CHUNKS_PER_CORE,              \
        .a_read_valid_slots_per_bank_count = NAME::A_READ_valid_slots_per_bank_count, \
        .b_read_chunks_per_core = NAME::B_READ_CHUNKS_PER_CORE,              \
        .b_read_valid_slots_per_bank_count = NAME::B_READ_valid_slots_per_bank_count, \
        .c_write_chunks_per_core = NAME::C_WRITE_CHUNKS_PER_CORE,            \
        .c_write_valid_slots_per_bank_count = NAME::C_WRITE_valid_slots_per_bank_count, \
        .c_write_hardcoded_num_phase = NAME::NUM_PHASE,                      \
        .c_write_hardcoded_valid_slots_per_bank_count =                      \
            NAME::C_WRITE_HARDCODED_valid_slots_per_bank_count,              \
    }

constexpr auto wormhole_b0_8x8_metadata = TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x8_schedule);
constexpr auto wormhole_b0_8x1_metadata = TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x1_schedule);
constexpr auto wormhole_b0_1x8_metadata = TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_1x8_schedule);

static_assert(
    wormhole_b0_8x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x8_0_schedule),
    "8x8 chip-specific schedule metadata must match the generic 8x8 schedule");
static_assert(
    wormhole_b0_8x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x8_1_schedule),
    "8x8 chip-specific schedule metadata must match the generic 8x8 schedule");
static_assert(
    wormhole_b0_8x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x8_2_schedule),
    "8x8 chip-specific schedule metadata must match the generic 8x8 schedule");
static_assert(
    wormhole_b0_8x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x8_3_schedule),
    "8x8 chip-specific schedule metadata must match the generic 8x8 schedule");
static_assert(
    wormhole_b0_8x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x8_4_schedule),
    "8x8 chip-specific schedule metadata must match the generic 8x8 schedule");
static_assert(
    wormhole_b0_8x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x8_5_schedule),
    "8x8 chip-specific schedule metadata must match the generic 8x8 schedule");
static_assert(
    wormhole_b0_8x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x8_6_schedule),
    "8x8 chip-specific schedule metadata must match the generic 8x8 schedule");
static_assert(
    wormhole_b0_8x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x8_7_schedule),
    "8x8 chip-specific schedule metadata must match the generic 8x8 schedule");
static_assert(
    wormhole_b0_8x1_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x1_0_schedule),
    "8x1 chip-specific schedule metadata must match the generic 8x1 schedule");
static_assert(
    wormhole_b0_8x1_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x1_1_schedule),
    "8x1 chip-specific schedule metadata must match the generic 8x1 schedule");
static_assert(
    wormhole_b0_8x1_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x1_2_schedule),
    "8x1 chip-specific schedule metadata must match the generic 8x1 schedule");
static_assert(
    wormhole_b0_8x1_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x1_3_schedule),
    "8x1 chip-specific schedule metadata must match the generic 8x1 schedule");
static_assert(
    wormhole_b0_8x1_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x1_4_schedule),
    "8x1 chip-specific schedule metadata must match the generic 8x1 schedule");
static_assert(
    wormhole_b0_8x1_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x1_5_schedule),
    "8x1 chip-specific schedule metadata must match the generic 8x1 schedule");
static_assert(
    wormhole_b0_8x1_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x1_6_schedule),
    "8x1 chip-specific schedule metadata must match the generic 8x1 schedule");
static_assert(
    wormhole_b0_8x1_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_8x1_7_schedule),
    "8x1 chip-specific schedule metadata must match the generic 8x1 schedule");
static_assert(
    wormhole_b0_1x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_1x8_0_schedule),
    "1x8 chip-specific schedule metadata must match the generic 1x8 schedule");
static_assert(
    wormhole_b0_1x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_1x8_1_schedule),
    "1x8 chip-specific schedule metadata must match the generic 1x8 schedule");
static_assert(
    wormhole_b0_1x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_1x8_2_schedule),
    "1x8 chip-specific schedule metadata must match the generic 1x8 schedule");
static_assert(
    wormhole_b0_1x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_1x8_3_schedule),
    "1x8 chip-specific schedule metadata must match the generic 1x8 schedule");
static_assert(
    wormhole_b0_1x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_1x8_4_schedule),
    "1x8 chip-specific schedule metadata must match the generic 1x8 schedule");
static_assert(
    wormhole_b0_1x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_1x8_5_schedule),
    "1x8 chip-specific schedule metadata must match the generic 1x8 schedule");
static_assert(
    wormhole_b0_1x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_1x8_6_schedule),
    "1x8 chip-specific schedule metadata must match the generic 1x8 schedule");
static_assert(
    wormhole_b0_1x8_metadata == TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA(wormhole_b0_1x8_7_schedule),
    "1x8 chip-specific schedule metadata must match the generic 1x8 schedule");

inline OptimizedMatmulScheduleMetadata get_optimized_matmul_schedule_metadata(
    const tt::tt_metal::CoreCoord& active_grid) {
    if (active_grid.x == 8 && active_grid.y == 8) {
        return wormhole_b0_8x8_metadata;
    }

    if (active_grid.x == 8 && active_grid.y == 1) {
        return wormhole_b0_8x1_metadata;
    }

    if (active_grid.x == 1 && active_grid.y == 8) {
        return wormhole_b0_1x8_metadata;
    }

    TT_THROW(
        "optimized_matmul does not have a vendored schedule for active grid {}x{}",
        active_grid.x,
        active_grid.y);
}

inline std::string get_optimized_matmul_schedule_header_basename(
    const tt::tt_metal::CoreCoord& active_grid, const std::optional<uint32_t> physical_chip_id = std::nullopt) {
    if (active_grid.x == 8 && active_grid.y == 8) {
        if (physical_chip_id.has_value() && physical_chip_id.value() < 8) {
            return "wormhole_b0_8x8_" + std::to_string(physical_chip_id.value()) + "_schedule.hpp";
        }
        return "wormhole_b0_8x8_schedule.hpp";
    }
    if (active_grid.x == 8 && active_grid.y == 1) {
        if (physical_chip_id.has_value() && physical_chip_id.value() < 8) {
            return "wormhole_b0_8x1_" + std::to_string(physical_chip_id.value()) + "_schedule.hpp";
        }
        return "wormhole_b0_8x1_schedule.hpp";
    }
    if (active_grid.x == 1 && active_grid.y == 8) {
        if (physical_chip_id.has_value() && physical_chip_id.value() < 8) {
            return "wormhole_b0_1x8_" + std::to_string(physical_chip_id.value()) + "_schedule.hpp";
        }
        return "wormhole_b0_1x8_schedule.hpp";
    }

    TT_THROW(
        "optimized_matmul does not have a vendored schedule header for active grid {}x{}",
        active_grid.x,
        active_grid.y);
}

inline std::string get_optimized_matmul_schedule_header_include_path(
    const tt::tt_metal::CoreCoord& active_grid, const std::optional<uint32_t> physical_chip_id = std::nullopt) {
    return "\"schedules/" + get_optimized_matmul_schedule_header_basename(active_grid, physical_chip_id) + "\"";
}

#undef TTNN_OPTIMIZED_MATMUL_SCHEDULE_METADATA

}  // namespace ttnn::operations::experimental::matmul::optimized_matmul
