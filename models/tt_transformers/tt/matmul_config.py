import os

import ttnn


def parse_active_core_grid_override(device):
    active_pw_env = os.environ.get("TTT_ACTIVE_PW")
    active_ph_env = os.environ.get("TTT_ACTIVE_PH")
    if (active_pw_env is None) != (active_ph_env is None):
        raise ValueError("TTT_ACTIVE_PW and TTT_ACTIVE_PH must be set together")

    if active_pw_env is None:
        return None

    try:
        active_pw = int(active_pw_env)
        active_ph = int(active_ph_env)
    except ValueError as exc:
        raise ValueError(
            f"Invalid active grid override TTT_ACTIVE_PW={active_pw_env}, TTT_ACTIVE_PH={active_ph_env}"
        ) from exc

    device_grid = device.compute_with_storage_grid_size()
    if active_pw <= 0 or active_ph <= 0 or active_pw > device_grid.x or active_ph > device_grid.y:
        raise ValueError(
            f"Active grid override {active_pw}x{active_ph} exceeds device grid {device_grid.x}x{device_grid.y}"
        )

    return ttnn.CoreGrid(y=active_ph, x=active_pw)


def get_matmul_override_kwargs(ttnn_a, ttnn_b, *, mesh_device, compute_kernel_config, **kwargs):
    active_core_grid = parse_active_core_grid_override(mesh_device)
    program_config = ttnn.resolve_matmul_2d_reuse_program_config(
        ttnn_a,
        ttnn_b,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=active_core_grid,
    )
    if (
        active_core_grid is not None
        and (
            program_config.compute_with_storage_grid_size.x != active_core_grid.x
            or program_config.compute_with_storage_grid_size.y != active_core_grid.y
        )
    ):
        raise RuntimeError(
            "Active grid override "
            f"{active_core_grid.x}x{active_core_grid.y} was not honored; "
            "resolved program_config uses "
            f"{program_config.compute_with_storage_grid_size.x}x"
            f"{program_config.compute_with_storage_grid_size.y}"
        )

    matmul_kwargs = dict(kwargs)
    matmul_kwargs["compute_kernel_config"] = compute_kernel_config
    matmul_kwargs["program_config"] = program_config
    matmul_kwargs.pop("core_grid", None)
    return matmul_kwargs


def ensure_dram_interleaved(tensor):
    if tensor.is_sharded():
        return ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)

    memory_config = tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        return ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)

    return tensor


def get_prefill_dram_matmul_inputs_and_kwargs(ttnn_a, ttnn_b, *, mesh_device, compute_kernel_config, **kwargs):
    ttnn_a = ensure_dram_interleaved(ttnn_a)
    ttnn_b = ensure_dram_interleaved(ttnn_b)
    matmul_kwargs = get_matmul_override_kwargs(
        ttnn_a,
        ttnn_b,
        mesh_device=mesh_device,
        compute_kernel_config=compute_kernel_config,
        **kwargs,
    )
    matmul_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    return ttnn_a, ttnn_b, matmul_kwargs
