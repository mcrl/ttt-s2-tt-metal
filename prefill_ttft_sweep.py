#!/usr/bin/env python3

import argparse
import csv
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SYSTEM_LIBSTDCPP = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"


def ensure_system_libstdcpp():
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if SYSTEM_LIBSTDCPP in ld_preload:
        return
    if not Path(SYSTEM_LIBSTDCPP).exists():
        return
    if os.environ.get("TTT_PREFILL_TTFT_SWEEP_REEXECED") == "1":
        return

    new_env = os.environ.copy()
    new_env["LD_PRELOAD"] = f"{SYSTEM_LIBSTDCPP}:{ld_preload}" if ld_preload else SYSTEM_LIBSTDCPP
    new_env["TTT_PREFILL_TTFT_SWEEP_REEXECED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)


logger = None
torch = None
ttnn = None
load_inputs = None
normalize_page_params_for_context = None
prepare_generator_args = None
get_base_model_name = None
preprocess_inputs_prefill = None
Generator = None
DecodersPrecision = None
get_updated_device_params = None


def lazy_imports():
    global logger
    global torch
    global ttnn
    global load_inputs
    global normalize_page_params_for_context
    global prepare_generator_args
    global get_base_model_name
    global preprocess_inputs_prefill
    global Generator
    global DecodersPrecision
    global get_updated_device_params

    if ttnn is not None:
        return

    import torch as torch_module
    from loguru import logger as logger_module

    import ttnn as ttnn_module
    from models.tt_transformers.demo.simple_text_demo import (
        load_inputs as load_inputs_fn,
        normalize_page_params_for_context as normalize_page_params_for_context_fn,
        prepare_generator_args as prepare_generator_args_fn,
    )
    from models.tt_transformers.tt.common import (
        get_base_model_name as get_base_model_name_fn,
        preprocess_inputs_prefill as preprocess_inputs_prefill_fn,
    )
    from models.tt_transformers.tt.generator import Generator as GeneratorClass
    from models.tt_transformers.tt.model_config import DecodersPrecision as DecodersPrecisionClass
    from tests.scripts.common import get_updated_device_params as get_updated_device_params_fn

    torch = torch_module
    logger = logger_module
    ttnn = ttnn_module
    load_inputs = load_inputs_fn
    normalize_page_params_for_context = normalize_page_params_for_context_fn
    prepare_generator_args = prepare_generator_args_fn
    get_base_model_name = get_base_model_name_fn
    preprocess_inputs_prefill = preprocess_inputs_prefill_fn
    Generator = GeneratorClass
    DecodersPrecision = DecodersPrecisionClass
    get_updated_device_params = get_updated_device_params_fn


MESH_SHAPES = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

TRACE_REGION_SIZE_OVERRIDES = {
    ("Llama-3.1-8B", "N150"): 25_000_000,
    ("Llama-3.1-8B", "N300"): 38_000_000,
    ("Llama-3.1-8B", "T3K"): 50_000_000,
    ("Llama-3.1-8B", "TG"): 50_000_000,
    ("Llama-3.1-70B", "T3K"): 90_000_000,
    ("Llama-3.1-70B", "TG"): 90_000_000,
    ("Llama-3.3-70B", "T3K"): 80_000_000,
    ("Llama-3.3-70B", "TG"): 80_000_000,
    ("Qwen2.5-72B", "T3K"): 70_000_000,
    ("Qwen2.5-72B", "TG"): 70_000_000,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Measure TTFT across one or more prefill lengths.")
    parser.add_argument("--model", default=os.getenv("MODEL", "Llama-3.1-70B-Instruct"))
    parser.add_argument("--mesh-device", choices=sorted(MESH_SHAPES), default=os.getenv("MESH_DEVICE", "T3K"))
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--tt-cache-path", default=None)
    parser.add_argument(
        "--input-prompts",
        default="models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("B", "32")))
    parser.add_argument("--data-parallel", type=int, default=1)
    parser.add_argument("--max-generated-tokens", type=int, default=int(os.getenv("GEN_TOKENS", "128")))
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--seqs", default=None)
    parser.add_argument("--page-block-size", type=int, default=32)
    parser.add_argument("--page-max-num-blocks-per-dp", type=int, default=1024)
    parser.add_argument("--optimizations", choices=("performance", "accuracy"), default="performance")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--append-output", action="store_true")
    return parser.parse_args()


def parse_seq_lens(args):
    if args.seqs:
        seq_lens = [int(token.strip()) for token in args.seqs.split(",") if token.strip()]
        if not seq_lens:
            raise ValueError("No sequence lengths were provided")
        return seq_lens

    if args.seq_len is not None:
        return [args.seq_len]

    return [int(os.getenv("S", "1024"))]


def default_output_csv():
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M")
    return Path("generated/perf_experiments") / f"prefill_ttft_sweep_{timestamp}.csv"


def configure_env(args):
    hf_model = args.hf_model or f"/shared/models/{args.model}"
    tt_cache_path = args.tt_cache_path or str(Path.home() / ".cache" / "ttt-weight-cache" / args.model)

    os.environ["MODEL"] = args.model
    os.environ["MESH_DEVICE"] = args.mesh_device
    os.environ["HF_MODEL"] = hf_model
    os.environ["TT_CACHE_PATH"] = tt_cache_path


def get_optimizations_factory(name):
    if name == "performance":
        return lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)
    if name == "accuracy":
        return lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name)
    raise ValueError(f"Unsupported optimizations mode: {name}")


def get_trace_region_size(model_name, mesh_device_name):
    base_model_name = get_base_model_name(model_name)
    return TRACE_REGION_SIZE_OVERRIDES.get((base_model_name, mesh_device_name), 50_000_000)


def normalize_fabric_config(fabric_config, mesh_shape):
    if fabric_config is not True:
        return fabric_config

    num_devices = mesh_shape.mesh_size()
    if num_devices == 1:
        return None

    cluster_type = ttnn.cluster.get_cluster_type()
    if cluster_type == ttnn.cluster.ClusterType.GALAXY:
        return ttnn.FabricConfig.FABRIC_1D_RING
    return ttnn.FabricConfig.FABRIC_1D


def open_mesh_device(mesh_device_name, model_name):
    mesh_shape = ttnn.MeshShape(*MESH_SHAPES[mesh_device_name])
    device_params = {
        "fabric_config": True,
        "trace_region_size": get_trace_region_size(model_name, mesh_device_name),
        "num_command_queues": 1,
    }
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    fabric_config = normalize_fabric_config(fabric_config, mesh_shape)

    if fabric_config:
        if reliability_mode is None:
            reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT
        if fabric_tensix_config is None:
            fabric_tensix_config = ttnn.FabricTensixConfig.DISABLED
        ttnn.set_fabric_config(fabric_config, reliability_mode, None, fabric_tensix_config)

    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)
    return mesh_device, fabric_config


def close_mesh_device(mesh_device, fabric_config):
    if mesh_device is not None:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)

    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    args = parse_args()
    ensure_system_libstdcpp()
    lazy_imports()

    seq_lens = parse_seq_lens(args)
    max_requested_seq_len = max(seq_lens)
    max_seq_len = args.max_seq_len or max(max_requested_seq_len + args.max_generated_tokens, 128)
    output_csv = Path(args.output_csv) if args.output_csv else default_output_csv()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    configure_env(args)

    global_batch_size = args.batch_size * args.data_parallel
    page_params = normalize_page_params_for_context(
        {
            "page_block_size": args.page_block_size,
            "page_max_num_blocks_per_dp": args.page_max_num_blocks_per_dp,
        },
        max_seq_len=max_seq_len,
        batch_size=args.batch_size,
        paged_attention=True,
    )

    input_prompts, _ = load_inputs(args.input_prompts, global_batch_size, instruct=True)
    optimizations = get_optimizations_factory(args.optimizations)

    mesh_device = None
    fabric_config = None

    try:
        mesh_device, fabric_config = open_mesh_device(args.mesh_device, args.model)
        model_args, model, page_table, tt_kv_cache, tokenizer, processor = prepare_generator_args(
            num_devices=mesh_device.get_num_devices(),
            data_parallel=args.data_parallel,
            mesh_device=mesh_device,
            instruct=True,
            global_batch_size=global_batch_size,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            page_params=page_params,
            paged_attention=True,
            num_layers=args.num_layers,
        )

        csv_mode = "a" if args.append_output and output_csv.exists() else "w"
        should_write_header = csv_mode == "w" or output_csv.stat().st_size == 0

        with output_csv.open(csv_mode, newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "model",
                    "mesh_device",
                    "batch_size",
                    "data_parallel",
                    "optimizations",
                    "requested_seq_len",
                    "prefill_seq_len",
                    "compile_s",
                    "inference_s",
                    "ttft_ms",
                    "prefill_tok_s",
                ],
            )
            if should_write_header:
                writer.writeheader()
                csv_file.flush()

            for seq_len in seq_lens:
                logger.info(f"Running TTFT case: seq_len={seq_len}")

                generator = Generator(
                    model,
                    model_args,
                    mesh_device,
                    processor=processor,
                    tokenizer=tokenizer,
                )

                try:
                    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
                        input_prompts,
                        tokenizer,
                        model_args,
                        instruct=True,
                        max_generated_tokens=args.max_generated_tokens,
                        max_prefill_len=max_seq_len,
                        fixed_prefill_len=seq_len,
                    )
                    del encoded_prompts

                    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)
                    actual_prefill_seq_len = generator._get_prefill_seq_len(max(decoding_pos))

                    compile_start = time.perf_counter()
                    logits = generator.prefill_forward_text(
                        input_tokens_prefill_pt,
                        page_table=page_table,
                        kv_cache=tt_kv_cache,
                        prompt_lens=decoding_pos,
                        enable_trace=True,
                    )
                    _ = torch.argmax(logits, dim=-1)
                    ttnn.synchronize_device(mesh_device)
                    compile_s = time.perf_counter() - compile_start

                    inference_start = time.perf_counter()
                    logits = generator.prefill_forward_text(
                        input_tokens_prefill_pt,
                        page_table=page_table,
                        kv_cache=tt_kv_cache,
                        prompt_lens=decoding_pos,
                        enable_trace=True,
                    )
                    _ = torch.argmax(logits, dim=-1)
                    ttnn.synchronize_device(mesh_device)
                    inference_s = time.perf_counter() - inference_start

                    ttft_ms = inference_s / global_batch_size * 1000.0
                    prefill_tok_s = prefill_lens[0] / inference_s * global_batch_size

                    row = {
                        "model": args.model,
                        "mesh_device": args.mesh_device,
                        "batch_size": args.batch_size,
                        "data_parallel": args.data_parallel,
                        "optimizations": args.optimizations,
                        "requested_seq_len": seq_len,
                        "prefill_seq_len": actual_prefill_seq_len,
                        "compile_s": f"{compile_s:.6f}",
                        "inference_s": f"{inference_s:.6f}",
                        "ttft_ms": f"{ttft_ms:.2f}",
                        "prefill_tok_s": f"{prefill_tok_s:.2f}",
                    }
                    writer.writerow(row)
                    csv_file.flush()
                    logger.info(
                        "Recorded result: "
                        f"requested_seq_len={seq_len}, prefill_seq_len={actual_prefill_seq_len}, ttft_ms={ttft_ms:.2f}"
                    )
                finally:
                    generator.release_prefill_traces()
                    del generator
                    gc.collect()

        logger.info(f"Saved TTFT sweep results to {output_csv}")
    finally:
        close_mesh_device(mesh_device, fabric_config)


if __name__ == "__main__":
    main()
