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
    # "P150|Llama-3.2-3B-Instruct"
    # "P150|Qwen2.5-7B"
    # "P150|Llama-3.1-8B-Instruct"
    # "P150x4|Qwen3-32B"
    # "P150x4|Llama-3.1-70B"
    # "P150x4|Qwen2.5-72B"
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

ttft_from_case_csv() {
    local csv_path="$1"
    python - "$csv_path" <<'PY'
import csv
import os
import sys

csv_path = sys.argv[1]
ttft_ms = ""

if os.path.exists(csv_path):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row_ttft_ms = row.get("ttft_ms", "").strip()
            if not row_ttft_ms:
                continue
            ttft_ms = row_ttft_ms
            break

print(ttft_ms)
PY
}

append_case_csv_to_raw() {
    local case_csv="$1"
    local raw_csv="$2"

    if [ ! -f "$case_csv" ]; then
        return 1
    fi

    if [ ! -f "$raw_csv" ] || [ ! -s "$raw_csv" ]; then
        cat "$case_csv" > "$raw_csv"
    else
        tail -n +2 "$case_csv" >> "$raw_csv"
    fi
}

run_prefill_case() {
    local system="$1"
    local model="$2"
    local visible_devices="$3"
    local optim_flag="$4"
    local seq="$5"
    local case_csv="$6"
    local log_path="$7"
    local case_max_s="$((seq + GEN_TOKENS))"

    (
        echo "system=${system} model=${model} optim_flag=${optim_flag} seq=${seq} batch=${B}"
        echo "case_csv=${case_csv}"
        echo "log_path=${log_path}"

        rm -rf ~/.cache/tt-metal-cache
        tt-smi -r || exit $?

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
                --output-csv "$case_csv" \
                "${NUM_LAYER_ARGS[@]}"
    ) 2>&1 | tee "$log_path"

    return "${PIPESTATUS[0]}"
}

TIMESTAMP="$(TZ=Asia/Seoul date +%Y%m%d_%H%M)"
OUT_DIR="generated/perf_experiments/perf_matrix_${TIMESTAMP}"
RAW_DIR="${OUT_DIR}/raw"
CASE_DIR="${OUT_DIR}/case_csv"
LOG_DIR="${OUT_DIR}/logs"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

mkdir -p "$RAW_DIR" "$CASE_DIR" "$LOG_DIR"

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

            case_csv="${CASE_DIR}/${system}_${model_safe}_B${B}_S${seq}_${implementation_name}.csv"
            log_path="${LOG_DIR}/${system}_${model_safe}_B${B}_S${seq}_${implementation_name}.log"
            rm -f "$case_csv" "$log_path"

            echo "     -> seq=${seq}"
            if run_prefill_case "$system" "$model" "$visible_devices" "$optim_flag" "$seq" "$case_csv" "$log_path"; then
                echo "     -> seq=${seq} completed"
                append_case_csv_to_raw "$case_csv" "$output_csv"
                ttft_value="$(ttft_from_case_csv "$case_csv")"
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
