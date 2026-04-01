// Compute Kernel for Block Matrix Multiplication
// Tensix compute kernel that performs block-based tile matrix multiplication.
// Block geometry is passed as compile-time args so the inner loops can be
// specialized per resolved matmul configuration.
//
// Options (passed via kernel defines from optim_options.hpp):
// PACKER_L1_ACC, SKIP_COMPUTE

#include "compute_kernel_api/matmul.h"

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

void MAIN {
    const uint32_t row_bidx0 = get_arg_val<uint32_t>(0);
    const uint32_t col_bidx0 = get_arg_val<uint32_t>(1);
    const uint32_t row_nblocks = get_arg_val<uint32_t>(2);
    const uint32_t col_nblocks = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_ki_iterations = get_compile_time_arg_val(0);
    constexpr uint32_t BMt = get_compile_time_arg_val(1);
    constexpr uint32_t BNt = get_compile_time_arg_val(2);
    constexpr uint32_t BKt = get_compile_time_arg_val(3);
    constexpr uint32_t SBMt = get_compile_time_arg_val(4);
    constexpr uint32_t SBNt = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_size = get_compile_time_arg_val(6);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t a_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t b_block_tiles = get_compile_time_arg_val(9);

    constexpr uint32_t A_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t B_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t C_cb_id = tt::CBIndex::c_16;
    constexpr uint32_t Cbuffer_cb_id = tt::CBIndex::c_24;

    mm_block_init(A_cb_id, B_cb_id, Cbuffer_cb_id, false, BNt, BMt, BKt);
    unary_op_init_common(Cbuffer_cb_id, C_cb_id);

    for (uint32_t block_idx_row = row_bidx0;
         block_idx_row < row_bidx0 + row_nblocks; block_idx_row++) {
        for (uint32_t block_idx_col = col_bidx0;
             block_idx_col < col_bidx0 + col_nblocks; block_idx_col++) {
#if !PACKER_L1_ACC
            bool enable_reload = false;
#endif

            // Inner loop: accumulate across K dimension
            for (uint32_t ki_iter = 0; ki_iter < num_ki_iterations; ki_iter++) {
                bool is_first_ki = (ki_iter == 0);
                bool is_last_ki = (ki_iter == num_ki_iterations - 1);

                // Wait for input tiles from A and B
                cb_wait_front(A_cb_id, a_block_tiles);
                cb_wait_front(B_cb_id, b_block_tiles);

                // Process output block in subblock units (SBMt × SBNt tiles)
                for (uint32_t h = 0; h < BMt; h += SBMt) {
                    for (uint32_t w = 0; w < BNt; w += SBNt) {
                        tile_regs_acquire();

#if PACKER_L1_ACC
                        // With packer L1 acc: only reload on last iteration
                        if (is_last_ki && num_ki_iterations > 1) {
                            copy_tile_to_dst_init_short(Cbuffer_cb_id);
                            cb_wait_front(Cbuffer_cb_id, subblock_size);
                            copy_block_matmul_partials(Cbuffer_cb_id, 0, 0,
                                                       subblock_size);
                            cb_pop_front(Cbuffer_cb_id, subblock_size);
                        }
#else
                        // Without packer L1 acc: reload every iteration except first
                        if (enable_reload) {
                            copy_tile_to_dst_init_short(Cbuffer_cb_id);
                            cb_wait_front(Cbuffer_cb_id, subblock_size);
                            copy_block_matmul_partials(Cbuffer_cb_id, 0, 0,
                                                       subblock_size);
                            cb_pop_front(Cbuffer_cb_id, subblock_size);
                        }
#endif

#if !SKIP_COMPUTE
                        mm_block_init_short(A_cb_id, B_cb_id, false, SBNt, SBMt,
                                            BKt);
                        // Compute (SBMt × BKt) @ (BKt × SBNt) → (SBMt × SBNt)
                        for (uint32_t k = 0; k < BKt; k++) {
                            uint32_t A_cb_tile_idx = h * BKt + k;
                            uint32_t B_cb_tile_idx = k * BNt + w;
                            matmul_block(A_cb_id, B_cb_id, A_cb_tile_idx,
                                         B_cb_tile_idx, 0, false, SBNt, SBMt,
                                         BKt);
                        }
#endif

                        tile_regs_commit();
                        tile_regs_wait();

                        // Pack results
                        if (is_last_ki) {
                            // Last iteration: pack to final output buffer
#if PACKER_L1_ACC
                            PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                            cb_reserve_back(C_cb_id, subblock_size);
                            pack_tile_block(0, C_cb_id, subblock_size);
                            cb_push_back(C_cb_id, subblock_size);
                        } else {
                            // Not last iteration: pack to intermediate buffer
#if PACKER_L1_ACC
                            if (is_first_ki) {
                                PACK((llk_pack_reconfig_l1_acc(0)));
                            } else if (ki_iter == 1) {
                                PACK((llk_pack_reconfig_l1_acc(1)));
                            }
#endif
                            cb_reserve_back(Cbuffer_cb_id, subblock_size);
                            pack_tile_block(0, Cbuffer_cb_id, subblock_size);
                            cb_push_back(Cbuffer_cb_id, subblock_size);
                        }

                        tile_regs_release();
                    }  // subblock w loop
                }  // subblock h loop

#if PACKER_L1_ACC
                // Pop CB to update FIFO pointer (except before last iteration)
                if (!is_last_ki && ki_iter < num_ki_iterations - 2) {
                    cb_wait_front(Cbuffer_cb_id, block_num_tiles);
                    cb_pop_front(Cbuffer_cb_id, block_num_tiles);
                }
#else
                // Standard spill/reload: always reload after first iteration
                if (num_ki_iterations > 1) {
                    enable_reload = true;
                }
#endif

                // Release input tiles
                cb_pop_front(A_cb_id, a_block_tiles);
                cb_pop_front(B_cb_id, b_block_tiles);

            }  // ki_iter loop
        }  // block_idx_col loop
    }  // block_idx_row loop
}
}  // namespace NAMESPACE
