// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>

template <
    typename T,
    uint32_t num_elements_unpadded_w,
    uint32_t num_elements_unpadded_h,
    uint32_t num_faces_w,
    uint32_t num_faces_h,
    uint32_t face_h,
    uint32_t face_w>
void fill_pad_face(T* tile_ptr, T fill_value) {
    using namespace tt::constants;

    constexpr uint32_t face_offset = (face_h * num_faces_w + face_w) * FACE_HW;
    auto face_ptr = tile_ptr + face_offset;

    constexpr uint32_t face_w_offset = face_w * FACE_WIDTH;
    constexpr uint32_t face_pad_w = (num_elements_unpadded_w <= face_w_offset) ? FACE_WIDTH
                                    : (num_elements_unpadded_w >= face_w_offset + FACE_WIDTH)
                                        ? 0
                                        : face_w_offset + FACE_WIDTH - num_elements_unpadded_w;

    if constexpr (face_pad_w > 0) {
#pragma unroll
        for (uint32_t row = 0; row < FACE_HEIGHT; ++row) {
            auto row_ptr = face_ptr + row * FACE_WIDTH;
            for (uint32_t col = FACE_WIDTH - face_pad_w; col < FACE_WIDTH; ++col) {
                row_ptr[col] = fill_value;
            }
        }
    }

    constexpr uint32_t face_h_offset = face_h * FACE_HEIGHT;
    constexpr uint32_t face_pad_h = (num_elements_unpadded_h <= face_h_offset) ? FACE_HEIGHT
                                    : (num_elements_unpadded_h >= face_h_offset + FACE_HEIGHT)
                                        ? 0
                                        : face_h_offset + FACE_HEIGHT - num_elements_unpadded_h;

    if constexpr (face_pad_h > 0) {
#pragma unroll
        for (uint32_t row = FACE_HEIGHT - face_pad_h; row < FACE_HEIGHT; ++row) {
            auto row_ptr = face_ptr + row * FACE_WIDTH;
            for (uint32_t col = 0; col < FACE_WIDTH; ++col) {
                row_ptr[col] = fill_value;
            }
        }
    }
}

template <
    typename T,
    uint32_t num_elements_unpadded_w,
    uint32_t num_elements_unpadded_h,
    uint32_t num_faces_w = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH,
    uint32_t num_faces_h = tt::constants::TILE_HEIGHT / tt::constants::FACE_HEIGHT>
void fill_pad_tile(uint32_t l1_tile_ptr, T fill_value) {
    auto tile_ptr = reinterpret_cast<T*>(l1_tile_ptr);

    fill_pad_face<T, num_elements_unpadded_w, num_elements_unpadded_h, num_faces_w, num_faces_h, 0, 0>(
        tile_ptr, fill_value);

    if constexpr (num_faces_w > 1) {
        fill_pad_face<T, num_elements_unpadded_w, num_elements_unpadded_h, num_faces_w, num_faces_h, 0, 1>(
            tile_ptr, fill_value);
    }

    if constexpr (num_faces_h > 1) {
        fill_pad_face<T, num_elements_unpadded_w, num_elements_unpadded_h, num_faces_w, num_faces_h, 1, 0>(
            tile_ptr, fill_value);
    }

    if constexpr (num_faces_w > 1 && num_faces_h > 1) {
        fill_pad_face<T, num_elements_unpadded_w, num_elements_unpadded_h, num_faces_w, num_faces_h, 1, 1>(
            tile_ptr, fill_value);
    }
}

template <DataFormat in0_data_format, uint32_t in0_last_ktile_w>
void pad_last_ktile(uint32_t l1_write_addr_in0) {
    using namespace tt::constants;
    if constexpr (in0_data_format == DataFormat::Float32) {
        fill_pad_tile<uint32_t, in0_last_ktile_w, TILE_HEIGHT>(l1_write_addr_in0, 0);
    } else if constexpr (in0_data_format == DataFormat::Float16_b) {
        fill_pad_tile<uint16_t, in0_last_ktile_w, TILE_HEIGHT>(l1_write_addr_in0, 0);
    }
}
