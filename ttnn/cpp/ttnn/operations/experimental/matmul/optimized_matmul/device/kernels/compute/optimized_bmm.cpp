// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"

using std::uint32_t;

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    const uint32_t batch = get_compile_time_arg_val(0);
    const uint32_t Mt = get_compile_time_arg_val(1);
    const uint32_t Kt = get_compile_time_arg_val(2);
    const uint32_t Nt = get_compile_time_arg_val(3);

    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    for (uint32_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (uint32_t mt_index = 0; mt_index < Mt; ++mt_index) {
            for (uint32_t nt_index = 0; nt_index < Nt; ++nt_index) {
                acquire_dst();
                for (uint32_t kt_index = 0; kt_index < Kt; ++kt_index) {
                    cb_wait_front(tt::CBIndex::c_0, onetile);
                    cb_wait_front(tt::CBIndex::c_1, onetile);

                    matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, false);

                    cb_pop_front(tt::CBIndex::c_0, onetile);
                    cb_pop_front(tt::CBIndex::c_1, onetile);
                }

                cb_reserve_back(tt::CBIndex::c_16, onetile);
                pack_tile(0, tt::CBIndex::c_16);
                cb_push_back(tt::CBIndex::c_16, onetile);
                release_dst();
            }
        }
    }
}
}  // namespace NAMESPACE
