#!/bin/bash

set -uo pipefail

export SEQS=${SEQS:-256,512,1024,2048,4096}

# Hardcode the system/model matrix here.
SYSTEM_MODEL_MATRIX=(
    "N150|Llama-3.2-3B-Instruct"
    "N150|Qwen2.5-7B"
    "N150|Llama-3.1-8B-Instruct"
    "T3K|Qwen3-32B"
    "T3K|Llama-3.1-70B-Instruct"
    "T3K|Qwen2.5-72B"
    "P150|Llama-3.2-3B-Instruct"
    "P150|Qwen2.5-7B"
    "P150|Llama-3.1-8B-Instruct"
    "P150x4|Qwen3-32B"
    "P150x4|Llama-3.1-70B"
    "P150x4|Qwen2.5-72B"
)

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LOGURU_LEVEL=${LOGURU_LEVEL:-INFO}
export TTT_OPTIMIZED_MATMUL_TRANSPOSED=${TTT_OPTIMIZED_MATMUL_TRANSPOSED:-1}
export TTT_MATH_FIDELITY=${TTT_MATH_FIDELITY:-LoFi}

export B=${B:-32}
export GEN_TOKENS=${GEN_TOKENS:-128}
export NUM_LAYERS=${NUM_LAYERS:-}

get_visible_devices() {
    local system="$1"
    if [ "$system" = "P150" ] || [ "$system" = "N150" ]; then
        echo "0"
    else
        echo "0,1,2,3"
    fi
}

ttft_for_seq_from_csv() {
    local csv_path="$1"
    local seq="$2"
    python - "$csv_path" "$seq" <<'PY'
import csv
import os
import sys

csv_path = sys.argv[1]
requested_seq = int(sys.argv[2])
ttft_ms = ""

if os.path.exists(csv_path):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            requested_seq_len = row.get("requested_seq_len", "").strip()
            row_ttft_ms = row.get("ttft_ms", "").strip()
            if not requested_seq_len or not row_ttft_ms:
                continue
            if int(requested_seq_len) == requested_seq:
                ttft_ms = row_ttft_ms

print(ttft_ms)
PY
}

run_prefill_case() {
    local system="$1"
    local model="$2"
    local visible_devices="$3"
    local optim_flag="$4"
    local seq="$5"
    local output_csv="$6"
    local case_max_s="$((seq + GEN_TOKENS))"

    rm -rf ~/.cache/tt-metal-cache
    tt-smi -r
    local reset_exit_code=$?
    if [ "$reset_exit_code" -ne 0 ]; then
        return "$reset_exit_code"
    fi

    env \
        TTT_OPTIMIZED_MATMUL="$optim_flag" \
        MODEL="$model" \
        MESH_DEVICE="$system" \
        TT_VISIBLE_DEVICES="$visible_devices" \
        HF_MODEL="/shared/models/$model" \
        TT_CACHE_PATH="$HOME/.cache/ttt-weight-cache/$model" \
        python -m tt_lock python prefill_ttft_sweep.py \
            --model "$model" \
            --mesh-device "$system" \
            --batch-size "$B" \
            --seq-len "$seq" \
            --max-generated-tokens "$GEN_TOKENS" \
            --max-seq-len "$case_max_s" \
            --output-csv "$output_csv" \
            --append-output \
            "${NUM_LAYER_ARGS[@]}"
}

TIMESTAMP="$(TZ=Asia/Seoul date +%Y%m%d_%H%M)"
OUT_DIR="generated/perf_experiments/perf_matrix_${TIMESTAMP}"
RAW_DIR="${OUT_DIR}/raw"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

mkdir -p "$RAW_DIR"

echo "system,model,B,S,implementation,ttft" > "$SUMMARY_CSV"

NUM_LAYER_ARGS=()
if [ -n "$NUM_LAYERS" ]; then
    NUM_LAYER_ARGS+=(--num-layers "$NUM_LAYERS")
fi

for entry in "${SYSTEM_MODEL_MATRIX[@]}"; do
    IFS='|' read -r system model <<< "$entry"
    model_safe="${model//\//_}"
    visible_devices="$(get_visible_devices "$system")"

    echo "Running matrix entry: system=${system}, model=${model}, TT_VISIBLE_DEVICES=${visible_devices}"

    noopt_csv="${RAW_DIR}/${system}_${model_safe}_noopt.csv"
    opt_csv="${RAW_DIR}/${system}_${model_safe}_opt.csv"
    rm -f "$noopt_csv" "$opt_csv"

    for optim_flag in 0 1; do
        if [ "$optim_flag" -eq 1 ]; then
            mode_name="opt"
            implementation_name="optim"
            output_csv="$opt_csv"
        else
            mode_name="noopt"
            implementation_name="ttnn"
            output_csv="$noopt_csv"
        fi

        echo "  -> mode=${mode_name}, seqs=${SEQS}"

        IFS=',' read -r -a seq_list <<< "$SEQS"
        for seq in "${seq_list[@]}"; do
            seq="${seq//[[:space:]]/}"
            if [ -z "$seq" ]; then
                continue
            fi

            echo "     -> seq=${seq}"
            if run_prefill_case "$system" "$model" "$visible_devices" "$optim_flag" "$seq" "$output_csv"; then
                echo "     -> seq=${seq} completed"
                ttft_value="$(ttft_for_seq_from_csv "$output_csv" "$seq")"
                echo "${system},${model},${B},${seq},${implementation_name},${ttft_value}" >> "$SUMMARY_CSV"
                echo "     -> summary recorded: implementation=${implementation_name}, seq=${seq}, ttft=${ttft_value}"
            else
                case_exit_code=$?
                echo "     -> seq=${seq} failed with exit code ${case_exit_code}; keeping partial CSV and continuing"
                echo "${system},${model},${B},${seq},${implementation_name},FAIL(${case_exit_code})" >> "$SUMMARY_CSV"
                echo "     -> summary recorded: implementation=${implementation_name}, seq=${seq}, ttft=FAIL(${case_exit_code})"
            fi
        done
    done

done

echo "Saved summary to ${SUMMARY_CSV}"
