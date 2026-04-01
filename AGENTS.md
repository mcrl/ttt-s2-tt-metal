# AGENTS.md

## Repository Purpose

This repository is a `tt-metal` fork used to demonstrate and validate matrix multiplication optimization performance for the `ttt-s2` project.

For the main `ttt-s2` project code and reference context, see:

`$HOME/ttt-s2`

Before starting work in this repository, always read:

`$HOME/ttt-s2/AGENTS.md`

## Origin

The canonical origin for this fork is:

`https://github.com/mcrl/ttt-s2-tt-metal`

## Build Rules

- Always build with the `ttt` conda environment activated.
- Always run the build from the repository root.
- Always use the root build script:

```bash
source /home/jinpyo/miniconda3/etc/profile.d/conda.sh
conda activate ttt
./build_metal.sh
```

- Do not run `cmake` commands directly.
- If you need to verify a change, use `./build_metal.sh` from the root directory instead of invoking the build system manually.

## Experimental Operation Added In This Fork

This fork adds a new native experimental TTNN operation:

`ttnn.experimental.optimized_matmul`

Current public API:

```python
ttnn.experimental.optimized_matmul(A, B)
```

## optimized_matmul Implementation Notes

`optimized_matmul` is implemented as a new C++ native experimental operation, not as a Python wrapper.

At the moment, the operation is exposed as its own experimental entry point and currently delegates internally to `ttnn::matmul`. This keeps the API distinct while making it straightforward to replace the implementation later with a dedicated optimized kernel or specialized execution path.

### Primary implementation files

- `ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/optimized_matmul.hpp`
- `ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/optimized_matmul.cpp`
- `ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/optimized_matmul_pybind.hpp`
- `ttnn/cpp/ttnn/operations/experimental/matmul/optimized_matmul/optimized_matmul_pybind.cpp`

### Integration files modified for registration and build wiring

- `ttnn/cpp/ttnn/operations/experimental/experimental_pybind.cpp`
- `ttnn/cpp/ttnn/operations/experimental/matmul/CMakeLists.txt`
- `ttnn/CMakeLists.txt`

### Python-side support files modified

- `ttnn/ttnn/operations/matmul.py`

This file was updated to attach a golden function for `ttnn.experimental.optimized_matmul`.

### Test file modified

- `tests/ttnn/unit_tests/operations/matmul/test_experimental.py`

This file was updated with a basic regression test for `ttnn.experimental.optimized_matmul`.

## Guidance For Future Work

- If you change the implementation of `optimized_matmul`, update both the C++ registration path and the Python golden-function coverage.
- Keep `optimized_matmul` build-integrated through `./build_metal.sh`.
- Do not bypass the repository build script even for small TTNN changes.
