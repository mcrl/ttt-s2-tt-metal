// Data Movement Kernel for Matrix Multiplication
// Single source file launched on both RISCV_0 (NOC_0) and RISCV_1 (NOC_1)

#include <cstdint>

#include "dataflow_api.h"
#include "optimized_matmul_dataflow_ops.hpp"

#ifndef TTT_DMVK_SCHEDULE_HEADER
#error "TTT_DMVK_SCHEDULE_HEADER must be defined"
#endif

#include TTT_DMVK_SCHEDULE_HEADER
// Runtime arguments
uint32_t A_DRAM_base_addr;
uint32_t B_DRAM_base_addr;
uint32_t C_DRAM_base_addr;
uint32_t Mt, Nt, Kt;
uint32_t BMt, BNt, BKt;
uint32_t SBMt, SBNt;
uint32_t row_bidx0, col_bidx0;
uint32_t row_nblocks, col_nblocks;

// Compile-time arguments
uint32_t Amaster_sem, Aslave_sem;
uint32_t Bmaster_sem, Bslave_sem;
uint32_t global_master_sem, global_slave_sem;
uint32_t active_PW, active_PH;

// Circular buffer IDs
constexpr uint32_t A_cb = tt::CBIndex::c_0;
constexpr uint32_t B_cb = tt::CBIndex::c_1;
constexpr uint32_t C_cb = tt::CBIndex::c_16;

// Derived constants (computed once in parse_args)
uint32_t num_kblocks;                      // Number of BK-sized K blocks.
uint32_t nblocks;                          // Number of output blocks owned by this core.
uint32_t A_tiles_per_block;                // Tiles in one A block.
uint32_t B_tiles_per_block;                // Tiles in one B block.
uint32_t sb_cols;                          // Subblock columns inside one C block.
uint32_t n_subblocks;                      // Total subblocks in one C block.
uint32_t sb_tiles;                         // Tiles in one C subblock.
uint32_t A_tile_bytes;                     // Bytes in one A tile.
uint32_t B_tile_bytes;                     // Bytes in one B tile.
uint32_t C_tile_bytes;                     // Bytes in one C tile.

uint32_t A_l1_base_addr;                   // L1 base address for packed A block data.
uint32_t B_l1_base_addr;                   // L1 base address for packed B block data.
uint32_t C_l1_base_addr;                   // L1 base address for packed C block data.
uint32_t A_block_bytes;                    // Bytes in one full A block.
uint32_t B_block_bytes;                    // Bytes in one full B block.
uint32_t total_row_blocks;                 // Real output block rows after ceil-div.
uint32_t total_col_blocks;                 // Real output block cols after ceil-div.
uint32_t A_valid_slots_stride;             // Valid A slots per bank per repetition.
uint32_t B_valid_slots_stride;             // Valid B slots per bank per repetition.
uint32_t C_valid_slots_stride;             // Valid generated-C slots per bank per repetition.
uint32_t C_hardcoded_valid_slots_stride;   // Valid hardcoded-C slots per bank per repetition.
uint32_t A_slot_bytes;                     // Fixed slot size for one A chunk.
uint32_t B_slot_bytes;                     // Fixed slot size for one B chunk.
uint32_t C_slot_bytes;                     // Fixed slot size for one generated-C chunk.
uint32_t C_hardcoded_slot_bytes;           // Fixed slot size for one hardcoded-C phase.
uint32_t A_repetition_stride_bytes;        // Bytes consumed by one A repetition in one bank page.
uint32_t B_repetition_stride_bytes;        // Bytes consumed by one B repetition in one bank page.
uint32_t C_repetition_stride_bytes;        // Bytes consumed by one generated-C repetition in one bank page.
uint32_t C_hardcoded_repetition_stride_bytes;  // Bytes consumed by one hardcoded-C repetition in one bank page.
uint32_t A_bank_page_bytes;                // Total bytes in one bank page for A.
uint32_t B_bank_page_bytes;                // Total bytes in one bank page for B.
uint32_t C_bank_page_bytes;                // Total bytes in one bank page for generated C.
uint32_t C_hardcoded_bank_page_bytes;      // Total bytes in one bank page for hardcoded C.

// Core coordinate and NOC setup (initialized in kernel_main)
uint32_t x, y;
uint8_t my_noc;
uint8_t A_read_noc;
uint8_t A_bcast_noc;
uint8_t B_read_noc;
uint8_t B_bcast_noc;
uint8_t C_write_noc;
uint8_t sync_noc = 0;

// Options (passed via kernel defines from optim_options.hpp):
// OPTIMIZED_A_READ, OPTIMIZED_B_READ, OPTIMIZED_WRITE, PACKER_L1_ACC
// SKIP_READ, SKIP_BCAST, SKIP_WRITE, SKIP_COMPUTE

FORCE_INLINE bool is_A_reader() {
    return my_noc == A_read_noc && x == (OPTIMIZED_A_READ ? A_MASTER_COL[y] : 0);
}

FORCE_INLINE bool is_B_reader() {
    return my_noc == B_read_noc && y == (OPTIMIZED_B_READ ? B_MASTER_ROW[x] : 0);
}

FORCE_INLINE bool is_A_sender() {
    return my_noc == A_bcast_noc && x == (OPTIMIZED_A_READ ? A_MASTER_COL[y] : 0);
}

FORCE_INLINE bool is_A_receiver() {
    return my_noc == A_bcast_noc && x != (OPTIMIZED_A_READ ? A_MASTER_COL[y] : 0);
}

FORCE_INLINE bool is_B_sender() {
    return my_noc == B_bcast_noc && y == (OPTIMIZED_B_READ ? B_MASTER_ROW[x] : 0);
}

FORCE_INLINE bool is_B_receiver() {
    return my_noc == B_bcast_noc && y != (OPTIMIZED_B_READ ? B_MASTER_ROW[x] : 0);
}

FORCE_INLINE bool is_C_writer() {
    return my_noc == C_write_noc;
}

FORCE_INLINE bool is_sync_worker() {
    return my_noc == sync_noc;
}

FORCE_INLINE void sync_all() {
    dmvk_barrier();
    if (is_sync_worker()) {
        sync_tensix_cores(GRID_START, GRID_START, GRID_START, GRID_START,
                          active_PW - 1, active_PH - 1, global_master_sem,
                          global_slave_sem, sync_noc);
    }
    dmvk_barrier();
}

FORCE_INLINE uint32_t ceil_div(const uint32_t value, const uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

FORCE_INLINE uint32_t get_valid_block_h(const uint32_t row_bidx) {
    // Returns the number of real tile rows in this logical output block; 0 means padded-only.
    if (row_bidx >= total_row_blocks) {
        return 0;
    }

    const uint32_t remaining_rows = Mt - row_bidx * BMt;
    return remaining_rows < BMt ? remaining_rows : BMt;
}

FORCE_INLINE uint32_t get_valid_block_w(const uint32_t col_bidx) {
    // Returns the number of real tile cols in this logical output block; 0 means padded-only.
    if (col_bidx >= total_col_blocks) {
        return 0;
    }

    const uint32_t remaining_cols = Nt - col_bidx * BNt;
    return remaining_cols < BNt ? remaining_cols : BNt;
}

FORCE_INLINE uint32_t get_valid_subblock_extent(const uint32_t block_offset,
                                                const uint32_t block_extent,
                                                const uint32_t subblock_extent) {
    if (block_offset >= block_extent) {
        return 0;
    }

    const uint32_t remaining_extent = block_extent - block_offset;
    return remaining_extent < subblock_extent ? remaining_extent
                                              : subblock_extent;
}

// A_read: Only called when OPTIMIZED_A_READ=false
// (standard interleaved, per-tile DRAM transaction)
FORCE_INLINE void A_read(uint32_t row_bidx, uint32_t k_bidx,
                         uint32_t A_l1_ptr, uint8_t noc) {
    if (!is_A_reader()) return;
    const uint32_t valid_block_h = get_valid_block_h(row_bidx);
    InterleavedAddrGen<true> addrgen = {.bank_base_address = A_DRAM_base_addr,
                                        .page_size = A_tile_bytes};
    for (uint32_t t = 0; t < A_tiles_per_block; t++) {
        uint32_t h = t / BKt;
        uint32_t ki = t % BKt;
        uint32_t mt_global = row_bidx * BMt + h;
        uint32_t kt_global = k_bidx * BKt + ki;
        uint32_t tile_index = mt_global * Kt + kt_global;
        uint32_t cb_ptr = A_l1_ptr + t * A_tile_bytes;
        uint64_t noc_addr = addrgen.get_noc_addr(tile_index, 0, noc);
#if !SKIP_READ
        if (h < valid_block_h) {
            noc_async_read(noc_addr, cb_ptr, A_tile_bytes, noc);
        }
#endif
    }
}

// B_read: Only called when OPTIMIZED_B_READ=false
// (standard interleaved, per-tile DRAM transaction)
FORCE_INLINE void B_read(uint32_t col_bidx, uint32_t k_bidx,
                         uint32_t B_l1_ptr, uint8_t noc) {
    if (!is_B_reader()) return;
    const uint32_t valid_block_w = get_valid_block_w(col_bidx);
    InterleavedAddrGen<true> addrgen = {.bank_base_address = B_DRAM_base_addr,
                                        .page_size = B_tile_bytes};
    for (uint32_t t = 0; t < B_tiles_per_block; t++) {
        uint32_t ki = t / BNt;
        uint32_t w = t % BNt;
        uint32_t kt_global = k_bidx * BKt + ki;
        uint32_t nt_global = col_bidx * BNt + w;
        uint32_t tile_index = kt_global * Nt + nt_global;
        uint32_t cb_ptr = B_l1_ptr + t * B_tile_bytes;
        uint64_t noc_addr = addrgen.get_noc_addr(tile_index, 0, noc);
#if !SKIP_READ
        if (w < valid_block_w) {
            noc_async_read(noc_addr, cb_ptr, B_tile_bytes, noc);
        }
#endif
    }
}

FORCE_INLINE uint32_t split_chunk_items(const uint32_t total_items,
                                        const uint32_t chunks_per_core,
                                        const uint32_t chunk_idx) {
    if (chunk_idx >= chunks_per_core) {
        return 0;
    }

    const uint32_t base_items = total_items / chunks_per_core;
    const uint32_t remainder = total_items % chunks_per_core;
    return chunk_idx < remainder ? base_items + 1 : base_items;
}

FORCE_INLINE uint32_t A_repetition_idx(const uint32_t row_bidx,
                                       const uint32_t k_bidx) {
    return (row_bidx - row_bidx0) * num_kblocks + k_bidx;
}

FORCE_INLINE uint32_t B_repetition_idx(const uint32_t col_bidx,
                                       const uint32_t k_bidx) {
    return (col_bidx - col_bidx0) * num_kblocks + k_bidx;
}

FORCE_INLINE uint32_t C_repetition_idx(const uint32_t row_bidx,
                                       const uint32_t col_bidx) {
    return (row_bidx - row_bidx0) * col_nblocks + (col_bidx - col_bidx0);
}

// A_read_optimized: compact scheduled multi-bank reads over fixed-size slots.
FORCE_INLINE void A_read_optimized(uint32_t row_bidx, uint32_t k_bidx,
                                   uint32_t A_l1_ptr) {
    if (x != A_MASTER_COL[y]) {
        return;
    }
    const uint32_t repetition_base_bytes =
        A_repetition_idx(row_bidx, k_bidx) * A_repetition_stride_bytes;
    InterleavedAddrGen<true> A_addrgen = {.bank_base_address = A_DRAM_base_addr,
                                          .page_size = A_bank_page_bytes};
    uint32_t A_idx = 0;

    for (uint32_t transfer_idx = 0; transfer_idx < A_READ_CHUNKS_PER_CORE;
         ++transfer_idx) {
        const uint32_t chunk_idx = transfer_idx;
        const uint32_t read_tiles = split_chunk_items(
            A_tiles_per_block, A_READ_CHUNKS_PER_CORE, chunk_idx);
        if (read_tiles == 0) {
            continue;
        }

        const uint32_t transfer_noc = A_READ_core_noc[y][x][transfer_idx];
        if (transfer_noc == my_noc) {
            uint64_t noc_addr =
                A_addrgen.get_noc_addr(
                    A_READ_core_bank[y][x][transfer_idx],
                    repetition_base_bytes +
                        A_READ_core_slot_ordinal[y][x][transfer_idx] * A_slot_bytes,
                    transfer_noc);
#if !SKIP_READ
            noc_async_read(noc_addr, A_l1_ptr + A_idx * A_tile_bytes,
                           read_tiles * A_tile_bytes, transfer_noc);
#endif
        }
        A_idx += read_tiles;
    }
}

// B_read_optimized: compact scheduled multi-bank reads over fixed-size slots.
FORCE_INLINE void B_read_optimized(uint32_t col_bidx, uint32_t k_bidx,
                                   uint32_t B_l1_ptr) {
    if (y != B_MASTER_ROW[x]) {
        return;
    }
    const uint32_t repetition_base_bytes =
        B_repetition_idx(col_bidx, k_bidx) * B_repetition_stride_bytes;
    InterleavedAddrGen<true> B_addrgen = {.bank_base_address = B_DRAM_base_addr,
                                          .page_size = B_bank_page_bytes};
    uint32_t B_idx = 0;

    for (uint32_t transfer_idx = 0; transfer_idx < B_READ_CHUNKS_PER_CORE;
         ++transfer_idx) {
        const uint32_t chunk_idx = transfer_idx;
        const uint32_t read_tiles = split_chunk_items(
            B_tiles_per_block, B_READ_CHUNKS_PER_CORE, chunk_idx);
        if (read_tiles == 0) {
            continue;
        }

        const uint32_t transfer_noc = B_READ_core_noc[y][x][transfer_idx];
        if (transfer_noc == my_noc) {
            uint64_t noc_addr =
                B_addrgen.get_noc_addr(
                    B_READ_core_bank[y][x][transfer_idx],
                    repetition_base_bytes +
                        B_READ_core_slot_ordinal[y][x][transfer_idx] * B_slot_bytes,
                    transfer_noc);
#if !SKIP_READ
            noc_async_read(noc_addr, B_l1_ptr + B_idx * B_tile_bytes,
                           read_tiles * B_tile_bytes, transfer_noc);
#endif
        }
        B_idx += read_tiles;
    }
}

// C_write: Only called when OPTIMIZED_WRITE=false (standard interleaved, single NOC writer)
FORCE_INLINE void C_write(uint32_t row_bidx, uint32_t col_bidx, uint8_t noc) {
    InterleavedAddrGen<true> addrgen = {.bank_base_address = C_DRAM_base_addr,
                                        .page_size = C_tile_bytes};
    // Real tile extent of this padded output block after clipping the bottom/right tail.
    const uint32_t block_h = get_valid_block_h(row_bidx);
    const uint32_t block_w = get_valid_block_w(col_bidx);
    for (uint32_t sb = 0; sb < n_subblocks; sb++) {
        uint32_t bh = (sb / sb_cols) * SBMt;
        uint32_t bw = (sb % sb_cols) * SBNt;
        cb_wait_front(C_cb, sb_tiles);
        uint32_t l1_read_addr = get_read_ptr(C_cb);
        const uint32_t valid_subblock_h =
            get_valid_subblock_extent(bh, block_h, SBMt);
        const uint32_t valid_subblock_w =
            get_valid_subblock_extent(bw, block_w, SBNt);
        for (uint32_t t = 0; t < sb_tiles; t++) {
            uint32_t h = t / SBNt;
            uint32_t w = t % SBNt;
            uint32_t tile_row = row_bidx * BMt + bh + h;
            uint32_t tile_col = col_bidx * BNt + bw + w;
            uint32_t tile_index = tile_row * Nt + tile_col;
            uint64_t noc_addr = addrgen.get_noc_addr(tile_index, 0, noc);
#if !SKIP_WRITE
            if (h < valid_subblock_h && w < valid_subblock_w) {
                noc_async_write(l1_read_addr, noc_addr, C_tile_bytes, noc);
            }
#endif
            l1_read_addr += C_tile_bytes;
        }
        noc_async_write_barrier(noc);
        cb_pop_front(C_cb, sb_tiles);
    }
}

FORCE_INLINE uint32_t C_write_chunk_subblocks(const uint32_t chunk_idx) {
    if (chunk_idx >= C_WRITE_CHUNKS_PER_CORE) {
        return 0;
    }

    const uint32_t base_subblocks = n_subblocks / C_WRITE_CHUNKS_PER_CORE;
    if (chunk_idx == 0) {
        return n_subblocks - base_subblocks * (C_WRITE_CHUNKS_PER_CORE - 1);
    }
    return base_subblocks;
}

FORCE_INLINE uint32_t C_write_hardcoded_phase_subblocks(const uint32_t phase) {
    uint32_t phase_subblocks = n_subblocks / NUM_PHASE;
    if (phase == 0) {
        phase_subblocks =
            n_subblocks - phase_subblocks * (NUM_PHASE - 1);
    }
    return phase_subblocks;
}

FORCE_INLINE uint32_t C_write_chunk_subblock_start(const uint32_t chunk_idx) {
    if (chunk_idx == 0 || chunk_idx >= C_WRITE_CHUNKS_PER_CORE) {
        return 0;
    }

    const uint32_t first_chunk = C_write_chunk_subblocks(0);
    const uint32_t base_subblocks = n_subblocks / C_WRITE_CHUNKS_PER_CORE;
    return first_chunk + (chunk_idx - 1) * base_subblocks;
}

FORCE_INLINE uint32_t C_write_hardcoded_phase_start(const uint32_t phase) {
    if (phase == 0 || phase >= NUM_PHASE) {
        return 0;
    }

    const uint32_t first_phase = C_write_hardcoded_phase_subblocks(0);
    const uint32_t base_subblocks = n_subblocks / NUM_PHASE;
    return first_phase + (phase - 1) * base_subblocks;
}

// C_write_optimized_hardcode: Legacy phased bank/NOC schedule.
FORCE_INLINE void C_write_optimized_hardcode(uint32_t row_bidx,
                                             uint32_t col_bidx) {
    const uint32_t repetition_base_bytes =
        C_repetition_idx(row_bidx, col_bidx) * C_hardcoded_repetition_stride_bytes;
    InterleavedAddrGen<true> addrgen = {.bank_base_address = C_DRAM_base_addr,
                                        .page_size = C_hardcoded_bank_page_bytes};
    // Real tile extent of this padded output block after clipping the bottom/right tail.
    const uint32_t block_h = get_valid_block_h(row_bidx);
    const uint32_t block_w = get_valid_block_w(col_bidx);
    for (uint32_t phase = 0; phase < NUM_PHASE; phase++) {
        const uint32_t phase_subblocks = C_write_hardcoded_phase_subblocks(phase);
        if (phase_subblocks == 0) {
            continue;
        }
        const uint32_t subblock_start = C_write_hardcoded_phase_start(phase);
        const uint32_t C_phase_bank = C_WRITE_BANK[phase][y][x];
        const uint32_t C_phase_noc = C_WRITE_NOC[phase][y][x];
        const uint32_t bank_offset =
            repetition_base_bytes +
            C_WRITE_HARDCODED_slot_ordinal[phase][y][x] * C_hardcoded_slot_bytes;

        dmvk_barrier();

        if (my_noc == C_phase_noc) {
            // Only the designated RISC handles this phase's CB consumption and writes
            for (uint32_t sb = 0; sb < phase_subblocks; sb++) {
                uint32_t current_sb = subblock_start + sb;
                uint32_t bh = (current_sb / sb_cols) * SBMt;
                uint32_t bw = (current_sb % sb_cols) * SBNt;
                cb_wait_front(C_cb, sb_tiles);
                uint32_t l1_read_addr =
                    C_l1_base_addr + current_sb * sb_tiles * C_tile_bytes;
                const uint32_t slot_sb_offset = sb * sb_tiles * C_tile_bytes;
                const uint64_t slot_base_noc_addr =
                    addrgen.get_noc_addr(C_phase_bank,
                                         bank_offset + slot_sb_offset,
                                         C_phase_noc);
                const uint32_t valid_subblock_h =
                    get_valid_subblock_extent(bh, block_h, SBMt);
                const uint32_t valid_subblock_w =
                    get_valid_subblock_extent(bw, block_w, SBNt);
                for (uint32_t t = 0; t < sb_tiles; t++) {
                    uint32_t h = t / SBNt;
                    uint32_t w = t % SBNt;
                    const uint64_t noc_addr =
                        slot_base_noc_addr + static_cast<uint64_t>(t) * C_tile_bytes;
#if !SKIP_WRITE
                    if (h < valid_subblock_h && w < valid_subblock_w) {
                        noc_async_write(l1_read_addr, noc_addr, C_tile_bytes,
                                        C_phase_noc);
                    }
#endif
                    l1_read_addr += C_tile_bytes;
                }
                noc_async_write_barrier(C_phase_noc);
                cb_pop_front(C_cb, sb_tiles);
            }
        }
    }
}

// C_write_optimized_schedule: Replay generated sparse per-core C-write events.
FORCE_INLINE void C_write_optimized_schedule(uint32_t row_bidx,
                                             uint32_t col_bidx) {
    const uint32_t repetition_base_bytes =
        C_repetition_idx(row_bidx, col_bidx) * C_repetition_stride_bytes;
    InterleavedAddrGen<true> addrgen = {.bank_base_address = C_DRAM_base_addr,
                                        .page_size = C_bank_page_bytes};
    // Real tile extent of this padded output block after clipping the bottom/right tail.
    const uint32_t block_h = get_valid_block_h(row_bidx);
    const uint32_t block_w = get_valid_block_w(col_bidx);
    uint32_t next_chunk_idx = 0;

    for (uint32_t t = 0; t < C_WRITE_SCHED_T; ++t) {
        dmvk_barrier();

        if (next_chunk_idx < C_WRITE_CHUNKS_PER_CORE &&
            C_WRITE_core_time[y][x][next_chunk_idx] == t) {
            const uint32_t scheduled_bank = C_WRITE_core_bank[y][x][next_chunk_idx];
            const uint32_t scheduled_noc = C_WRITE_core_noc[y][x][next_chunk_idx];
            const uint32_t chunk_subblocks =
                C_write_chunk_subblocks(next_chunk_idx);
            const uint32_t subblock_start =
                C_write_chunk_subblock_start(next_chunk_idx);
            const uint32_t bank_offset =
                repetition_base_bytes +
                C_WRITE_core_slot_ordinal[y][x][next_chunk_idx] * C_slot_bytes;

            if (scheduled_noc == my_noc) {
                for (uint32_t sb = 0; sb < chunk_subblocks; ++sb) {
                    const uint32_t current_sb = subblock_start + sb;
                    const uint32_t bh = (current_sb / sb_cols) * SBMt;
                    const uint32_t bw = (current_sb % sb_cols) * SBNt;
                    cb_wait_front(C_cb, sb_tiles);
                    uint32_t l1_read_addr =
                        C_l1_base_addr + current_sb * sb_tiles * C_tile_bytes;
                    const uint32_t slot_sb_offset = sb * sb_tiles * C_tile_bytes;
                    const uint32_t valid_subblock_h =
                        get_valid_subblock_extent(bh, block_h, SBMt);
                    const uint32_t valid_subblock_w =
                        get_valid_subblock_extent(bw, block_w, SBNt);

                    for (uint32_t tile = 0; tile < sb_tiles; ++tile) {
                        const uint32_t h = tile / SBNt;
                        const uint32_t w = tile % SBNt;
                        uint64_t noc_addr =
                            addrgen.get_noc_addr(scheduled_bank,
                                                 bank_offset + slot_sb_offset +
                                                     tile * C_tile_bytes,
                                                 scheduled_noc);
#if !SKIP_WRITE
                        if (h < valid_subblock_h && w < valid_subblock_w) {
                            noc_async_write(l1_read_addr, noc_addr, C_tile_bytes,
                                            scheduled_noc);
                        }
#endif
                        l1_read_addr += C_tile_bytes;
                    }

                    noc_async_write_barrier(scheduled_noc);
                    cb_pop_front(C_cb, sb_tiles);
                }
                noc_async_write_barrier(scheduled_noc);
            }
            ++next_chunk_idx;
        }

        dmvk_barrier();
    }
}

void parse_args() {
    // Runtime arguments
    // [A_DRAM_base_addr, B_DRAM_base_addr, C_DRAM_base_addr, Mt, Nt, Kt, BMt, BNt, BKt, SBMt,
    //  SBNt, row_bidx0, col_bidx0, row_nblocks, col_nblocks]
    A_DRAM_base_addr = get_arg_val<uint32_t>(0);
    B_DRAM_base_addr = get_arg_val<uint32_t>(1);
    C_DRAM_base_addr = get_arg_val<uint32_t>(2);
    Mt = get_arg_val<uint32_t>(3);
    Nt = get_arg_val<uint32_t>(4);
    Kt = get_arg_val<uint32_t>(5);
    BMt = get_arg_val<uint32_t>(6);
    BNt = get_arg_val<uint32_t>(7);
    BKt = get_arg_val<uint32_t>(8);
    SBMt = get_arg_val<uint32_t>(9);
    SBNt = get_arg_val<uint32_t>(10);
    row_bidx0 = get_arg_val<uint32_t>(11);
    col_bidx0 = get_arg_val<uint32_t>(12);
    row_nblocks = get_arg_val<uint32_t>(13);
    col_nblocks = get_arg_val<uint32_t>(14);

    // Compile-time arguments
    // [Amaster_sem, Aslave_sem, Bmaster_sem, Bslave_sem,
    //  global_master_sem, global_slave_sem, active_PW, active_PH]
    Amaster_sem = get_compile_time_arg_val(0);
    Aslave_sem = get_compile_time_arg_val(1);
    Bmaster_sem = get_compile_time_arg_val(2);
    Bslave_sem = get_compile_time_arg_val(3);
    global_master_sem = get_compile_time_arg_val(4);
    global_slave_sem = get_compile_time_arg_val(5);
    active_PW = get_compile_time_arg_val(6);
    active_PH = get_compile_time_arg_val(7);

    // Derived constants
    num_kblocks = Kt / BKt;
    nblocks = row_nblocks * col_nblocks;
    A_tiles_per_block = BMt * BKt;
    B_tiles_per_block = BKt * BNt;
    sb_cols = BNt / SBNt;
    n_subblocks = (BMt / SBMt) * sb_cols;
    sb_tiles = SBMt * SBNt;

    A_l1_base_addr = get_write_ptr(A_cb);
    B_l1_base_addr = get_write_ptr(B_cb);
    C_l1_base_addr = get_write_ptr(C_cb);

    dmvk_barrier();
    A_tile_bytes = get_tile_size(A_cb);
    B_tile_bytes = get_tile_size(B_cb);
    C_tile_bytes = get_tile_size(C_cb);
    A_block_bytes = A_tiles_per_block * A_tile_bytes;
    B_block_bytes = B_tiles_per_block * B_tile_bytes;
    total_row_blocks = ceil_div(Mt, BMt);
    total_col_blocks = ceil_div(Nt, BNt);
    A_valid_slots_stride = A_READ_valid_slots_per_bank_count;
    B_valid_slots_stride = B_READ_valid_slots_per_bank_count;
    C_valid_slots_stride = C_WRITE_valid_slots_per_bank_count;
    C_hardcoded_valid_slots_stride =
        C_WRITE_HARDCODED_valid_slots_per_bank_count;
    A_slot_bytes =
        split_chunk_items(A_tiles_per_block, A_READ_CHUNKS_PER_CORE, 0) *
        A_tile_bytes;
    B_slot_bytes =
        split_chunk_items(B_tiles_per_block, B_READ_CHUNKS_PER_CORE, 0) *
        B_tile_bytes;
    C_slot_bytes = C_write_chunk_subblocks(0) * sb_tiles * C_tile_bytes;
    C_hardcoded_slot_bytes =
        C_write_hardcoded_phase_subblocks(0) * sb_tiles * C_tile_bytes;
    A_repetition_stride_bytes = A_valid_slots_stride * A_slot_bytes;
    B_repetition_stride_bytes = B_valid_slots_stride * B_slot_bytes;
    C_repetition_stride_bytes = C_valid_slots_stride * C_slot_bytes;
    C_hardcoded_repetition_stride_bytes =
        C_hardcoded_valid_slots_stride * C_hardcoded_slot_bytes;
    A_bank_page_bytes = row_nblocks * num_kblocks * A_repetition_stride_bytes;
    B_bank_page_bytes = col_nblocks * num_kblocks * B_repetition_stride_bytes;
    C_bank_page_bytes = row_nblocks * col_nblocks * C_repetition_stride_bytes;
    C_hardcoded_bank_page_bytes =
        row_nblocks * col_nblocks * C_hardcoded_repetition_stride_bytes;

    dmvk_barrier();
    // for (uint32_t i = 0; i < 10; i++) {
    //     sync_all();
    // }
}

void kernel_main() {
    // =========================================================================
    // SETUP
    // =========================================================================
    x = worker_logical_x();
    y = worker_logical_y();
    my_noc = noc_index;

    A_read_noc = 1;
    A_bcast_noc = 1;
    B_read_noc = 0;
    B_bcast_noc = 0;
    sync_noc = 0;
    parse_args();
#if defined(ARCH_BLACKHOLE)
    const uint32_t c_split = (active_PW + 1) / 2;
    C_write_noc = x < c_split ? 1 : 0;
#else
    C_write_noc = x > 4 ? 0 : 1;  // C write NoC half-split
#endif

    // =========================================================================
    // MAIN LOOP
    // =========================================================================
    for (uint32_t bidx = 0; bidx < nblocks; bidx++) {
        uint32_t row_bidx = row_bidx0 + bidx / col_nblocks;
        uint32_t col_bidx = col_bidx0 + bidx % col_nblocks;

        // ---------------------------------------------------------------------
        // K LOOP
        // ---------------------------------------------------------------------
        for (uint32_t k_bidx = 0; k_bidx < num_kblocks; k_bidx++) {
            // Keep the manual double-buffer slot aligned with the CB FIFO
            // across output blocks. When num_kblocks is odd, the next block
            // starts on the opposite half of the buffer.
            const uint32_t input_buffer_slot =
                (bidx * num_kblocks + k_bidx) & 0x1;
            uint32_t A_l1_ptr = input_buffer_slot == 0
                                    ? A_l1_base_addr
                                    : A_l1_base_addr + A_block_bytes;
            uint32_t B_l1_ptr = input_buffer_slot == 0
                                    ? B_l1_base_addr
                                    : B_l1_base_addr + B_block_bytes;

            // -----------------------------------------------------------------
            // READ A and B
            // -----------------------------------------------------------------
            // Reserve CB space before issuing reads that fill the block.
#if OPTIMIZED_A_READ
            if (my_noc == A_read_noc) {
                cb_reserve_back(A_cb, A_tiles_per_block);
            }
#else
            if (is_A_reader()) {
                cb_reserve_back(A_cb, A_tiles_per_block);
            }
#endif  // #if OPTIMIZED_A_READ

#if OPTIMIZED_B_READ
            if (my_noc == B_read_noc) {
                cb_reserve_back(B_cb, B_tiles_per_block);
            }
#else
            if (is_B_reader()) {
                cb_reserve_back(B_cb, B_tiles_per_block);
            }
#endif  // #if OPTIMIZED_B_READ

            // dmvk_barrier();

            // Read A block into the reserved CB space.
#if OPTIMIZED_A_READ
            A_read_optimized(row_bidx, k_bidx, A_l1_ptr);
#else   // !OPTIMIZED_A_READ
            A_read(row_bidx, k_bidx, A_l1_ptr, A_read_noc);
#endif  // #if OPTIMIZED_A_READ

            // Read B block into the reserved CB space.
#if OPTIMIZED_B_READ
            B_read_optimized(col_bidx, k_bidx, B_l1_ptr);
#else   // !OPTIMIZED_B_READ
            B_read(col_bidx, k_bidx, B_l1_ptr, B_read_noc);
#endif  // #if OPTIMIZED_B_READ

            noc_async_read_barrier(my_noc);
#if OPTIMIZED_A_READ || OPTIMIZED_B_READ || OPTIMIZED_WRITE
            dmvk_barrier();
#endif

            // -----------------------------------------------------------------
            // BCAST A
            // -----------------------------------------------------------------
            if (is_A_sender()) {
                cb_push_back(A_cb, A_tiles_per_block);
#if !SKIP_BCAST
                broadcast_v2(A_l1_ptr, A_tiles_per_block * A_tile_bytes,
                             OPTIMIZED_A_READ ? A_MASTER_COL[y] : active_PW - 1,
                             y, GRID_START, y, active_PW - 1, y,
                             Amaster_sem, Aslave_sem, A_bcast_noc);
#endif  // !SKIP_BCAST
            }
            if (is_A_receiver()) {
#if !SKIP_BCAST
                broadcast_v2(A_l1_ptr, A_tiles_per_block * A_tile_bytes,
                             OPTIMIZED_A_READ ? A_MASTER_COL[y] : active_PW - 1,
                             y, GRID_START, y, active_PW - 1, y,
                             Amaster_sem, Aslave_sem, A_bcast_noc);
#endif  // !SKIP_BCAST
                cb_push_back(A_cb, A_tiles_per_block);
            }
            // -----------------------------------------------------------------
            // BCAST B
            // -----------------------------------------------------------------
            if (is_B_sender()) {
                cb_push_back(B_cb, B_tiles_per_block);
#if !SKIP_BCAST
                broadcast_v2(B_l1_ptr, B_tiles_per_block * B_tile_bytes, x,
                             OPTIMIZED_B_READ ? B_MASTER_ROW[x] : 0,
                             x, GRID_START, x, active_PH - 1,
                             Bmaster_sem, Bslave_sem, B_bcast_noc);
#endif  // !SKIP_BCAST
            }
            if (is_B_receiver()) {
#if !SKIP_BCAST
                broadcast_v2(B_l1_ptr, B_tiles_per_block * B_tile_bytes, x,
                             OPTIMIZED_B_READ ? B_MASTER_ROW[x] : 0,
                             x, GRID_START, x, active_PH - 1,
                             Bmaster_sem, Bslave_sem, B_bcast_noc);
#endif  // !SKIP_BCAST
                cb_push_back(B_cb, B_tiles_per_block);
            }
#if !(OPTIMIZED_A_READ || OPTIMIZED_B_READ || OPTIMIZED_WRITE)
            dmvk_barrier();
#endif
        }

        // ---------------------------------------------------------------------
        // C WRITE
        // ---------------------------------------------------------------------
#if OPTIMIZED_WRITE
#if OPTIMIZED_WRITE_USE_GENERATED_SCHEDULE
        C_write_optimized_schedule(row_bidx, col_bidx);
#else
        C_write_optimized_hardcode(row_bidx, col_bidx);
#endif
#else   // !OPTIMIZED_WRITE
        if (is_C_writer()) {
            C_write(row_bidx, col_bidx, C_write_noc);
        }
#endif  // #if OPTIMIZED_WRITE
    }

}
