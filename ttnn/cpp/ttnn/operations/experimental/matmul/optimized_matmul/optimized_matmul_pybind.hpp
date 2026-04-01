// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::matmul::detail {

void bind_optimized_matmul(pybind11::module& module);

}  // namespace ttnn::operations::experimental::matmul::detail
