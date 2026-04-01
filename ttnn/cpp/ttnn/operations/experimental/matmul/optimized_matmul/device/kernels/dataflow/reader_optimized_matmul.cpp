// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "pad_tile.hpp"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);
    uint32_t KtNt = get_arg_val<uint32_t>(6);
    uint32_t batch = get_arg_val<uint32_t>(7);
    uint32_t bcast_B = get_arg_val<uint32_t>(8);
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(9);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(10);
    uint32_t MtNt = get_arg_val<uint32_t>(11);

    constexpr uint32_t in0_last_ktile_w = get_compile_time_arg_val(0);
    constexpr auto src0_args = TensorAccessorArgs<1>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    uint32_t itileA = output_tile_start_id / Nt * Kt;

    uint32_t outbatch = output_tile_start_id % MtNt;
    uint32_t itileB_batch = output_tile_start_id % Nt;
    uint32_t itileB = itileB_batch;
    if (bcast_B == 0) {
        itileB += output_tile_start_id / MtNt * KtNt;
    }

    const auto s0 = TensorAccessor(src0_args, src0_addr, in0_tile_bytes);
    const auto s1 = TensorAccessor(src1_args, src1_addr, in1_tile_bytes);

    for (uint32_t n = 0; n < num_output_tiles; ++n) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            {
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(itileA, s0, l1_write_addr_in0);
                noc_async_read_barrier();
                if constexpr (in0_last_ktile_w > 0) {
                    if (kt == Kt - 1) {
                        const DataFormat in0_data_format = get_dataformat(cb_id_in0);
                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(l1_write_addr_in0);
                    }
                }
                cb_push_back(cb_id_in0, onetile);
            }

            {
                cb_reserve_back(cb_id_in1, onetile);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(itileB, s1, l1_write_addr_in1);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, onetile);
            }

            itileA += 1;
            itileB += Nt;
        }

        outbatch += 1;
        itileB_batch += 1;
        itileB -= KtNt;
        itileB += 1;

        if (itileB_batch == Nt) {
            itileB_batch = 0;
            itileB -= Nt;
            if (outbatch == MtNt) {
                if (bcast_B == 0) {
                    itileB += KtNt;
                }
                outbatch = 0;
            }
        } else {
            itileA -= Kt;
        }
    }
}
