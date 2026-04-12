#!/bin/bash

set -uo pipefail

export SEQS=${SEQS:-512}

# Hardcode the system/model matrix here.
SYSTEM_MODEL_MATRIX=(
    "N150|Llama-3.2-3B-Instruct"
    "N150|Qwen2.5-7B"
    # "N150|Llama-3.1-8B-Instruct"
    "T3K|Qwen3-32B"
    "T3K|Llama-3.1-70B-Instruct"
    # "T3K|Qwen2.5-72B"
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
export STOP_AT_EOS=${STOP_AT_EOS:-0}
export NUM_LAYERS=${NUM_LAYERS:-}

if [ "$GEN_TOKENS" -lt 2 ]; then
    echo "GEN_TOKENS must be >= 2 for decode performance runs because the first decode iteration is compile."
    exit 1
fi

get_visible_devices() {
    local system="$1"
    if [ "$system" = "P150" ] || [ "$system" = "N150" ]; then
        echo "0"
    else
        echo "0,1,2,3"
    fi
}

metric_from_case_csv() {
    local csv_path="$1"
    local field_name="$2"
    python - "$csv_path" "$field_name" <<'PY'
import csv
import os
import sys

csv_path = sys.argv[1]
field_name = sys.argv[2]
value = ""

if os.path.exists(csv_path):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            value = row.get(field_name, "").strip()
            if value:
                break

print(value)
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

write_case_csv_from_log() {
    local log_path="$1"
    local case_csv="$2"
    local system="$3"
    local model="$4"
    local batch_size="$5"
    local seq="$6"
    local gen_tokens="$7"
    local max_seq_len="$8"
    local implementation="$9"

    python - "$log_path" "$case_csv" "$system" "$model" "$batch_size" "$seq" "$gen_tokens" "$max_seq_len" "$implementation" <<'PY'
import csv
import re
import sys
from pathlib import Path

(
    log_path,
    case_csv,
    system,
    model,
    batch_size,
    seq,
    gen_tokens,
    max_seq_len,
    implementation,
) = sys.argv[1:]

log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")

float_pattern = r"([0-9]+(?:\.[0-9]+)?)"
speed_matches = re.findall(
    rf"Average speed: {float_pattern}ms @ {float_pattern} tok/s/user \({float_pattern} tok/s throughput\)",
    log_text,
)
ttft_matches = re.findall(rf"Average Time to First Token \(TTFT\): {float_pattern}ms", log_text)

if not speed_matches:
    raise SystemExit("Could not find decode speed metrics in log")

avg_decode_ms, decode_tok_s_user, decode_tok_s = speed_matches[-1]
ttft_ms = ttft_matches[-1] if ttft_matches else ""

with open(case_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "system",
            "model",
            "batch_size",
            "requested_seq_len",
            "max_generated_tokens",
            "max_seq_len",
            "implementation",
            "ttft_ms",
            "avg_decode_ms",
            "decode_tok_s_user",
            "decode_tok_s",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "system": system,
            "model": model,
            "batch_size": batch_size,
            "requested_seq_len": seq,
            "max_generated_tokens": gen_tokens,
            "max_seq_len": max_seq_len,
            "implementation": implementation,
            "ttft_ms": ttft_ms,
            "avg_decode_ms": avg_decode_ms,
            "decode_tok_s_user": decode_tok_s_user,
            "decode_tok_s": decode_tok_s,
        }
    )
PY
}

run_decode_case() {
    local system="$1"
    local model="$2"
    local visible_devices="$3"
    local optim_flag="$4"
    local seq="$5"
    local case_csv="$6"
    local log_path="$7"
    local case_max_s="$((seq + GEN_TOKENS))"

    (
        echo "system=${system} model=${model} optim_flag=${optim_flag} seq=${seq} batch=${B} gen_tokens=${GEN_TOKENS}"
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
            FIX_PREFILL_LEN="$seq" \
            python -m tt_lock pytest --tb=short -s \
                models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
                -k "batch-32 and performance" \
                --mode decode \
                --max_generated_tokens "$GEN_TOKENS" \
                --max_seq_len "$case_max_s" \
                --batch_size "$B" \
                --stop_at_eos "$STOP_AT_EOS" \
                "${NUM_LAYER_ARGS[@]}"
    ) 2>&1 | tee "$log_path"

    return "${PIPESTATUS[0]}"
}

TIMESTAMP="$(TZ=Asia/Seoul date +%Y%m%d_%H%M)"
OUT_DIR="generated/perf_experiments/decode_perf_matrix_${TIMESTAMP}"
RAW_DIR="${OUT_DIR}/raw"
CASE_DIR="${OUT_DIR}/case_csv"
LOG_DIR="${OUT_DIR}/logs"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

mkdir -p "$RAW_DIR" "$CASE_DIR" "$LOG_DIR"

echo "system,model,B,S,gen_tokens,max_seq_len,implementation,ttft_ms,avg_decode_ms,decode_tok_s_user,decode_tok_s" > "$SUMMARY_CSV"

NUM_LAYER_ARGS=()
if [ -n "$NUM_LAYERS" ]; then
    NUM_LAYER_ARGS+=(--num_layers "$NUM_LAYERS")
fi

for entry in "${SYSTEM_MODEL_MATRIX[@]}"; do
    IFS='|' read -r system model <<< "$entry"
    model_safe="${model//\//_}"
    visible_devices="$(get_visible_devices "$system")"

    echo "Running matrix entry: system=${system}, model=${model}, TT_VISIBLE_DEVICES=${visible_devices}"

    noopt_csv="${RAW_DIR}/${system}_${model_safe}_decode_noopt.csv"
    opt_csv="${RAW_DIR}/${system}_${model_safe}_decode_opt.csv"
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

            case_max_s="$((seq + GEN_TOKENS))"
            case_csv="${CASE_DIR}/${system}_${model_safe}_B${B}_S${seq}_${implementation_name}.csv"
            log_path="${LOG_DIR}/${system}_${model_safe}_B${B}_S${seq}_${implementation_name}.log"
            rm -f "$case_csv" "$log_path"

            echo "     -> seq=${seq}"
            if run_decode_case "$system" "$model" "$visible_devices" "$optim_flag" "$seq" "$case_csv" "$log_path"; then
                if write_case_csv_from_log \
                    "$log_path" \
                    "$case_csv" \
                    "$system" \
                    "$model" \
                    "$B" \
                    "$seq" \
                    "$GEN_TOKENS" \
                    "$case_max_s" \
                    "$implementation_name"; then
                    echo "     -> seq=${seq} completed"
                    append_case_csv_to_raw "$case_csv" "$output_csv"
                    ttft_value="$(metric_from_case_csv "$case_csv" "ttft_ms")"
                    avg_decode_ms_value="$(metric_from_case_csv "$case_csv" "avg_decode_ms")"
                    decode_tok_s_user_value="$(metric_from_case_csv "$case_csv" "decode_tok_s_user")"
                    decode_tok_s_value="$(metric_from_case_csv "$case_csv" "decode_tok_s")"
                    echo "${system},${model},${B},${seq},${GEN_TOKENS},${case_max_s},${implementation_name},${ttft_value},${avg_decode_ms_value},${decode_tok_s_user_value},${decode_tok_s_value}" >> "$SUMMARY_CSV"
                    echo "     -> summary recorded: implementation=${implementation_name}, seq=${seq}, decode_tok_s_user=${decode_tok_s_user_value}"
                else
                    parse_exit_code=$?
                    echo "     -> seq=${seq} completed but log parsing failed with exit code ${parse_exit_code}"
                    echo "${system},${model},${B},${seq},${GEN_TOKENS},${case_max_s},${implementation_name},FAIL(PARSE_${parse_exit_code}),FAIL(PARSE_${parse_exit_code}),FAIL(PARSE_${parse_exit_code}),FAIL(PARSE_${parse_exit_code})" >> "$SUMMARY_CSV"
                    echo "     -> summary recorded: implementation=${implementation_name}, seq=${seq}, decode_tok_s_user=FAIL(PARSE_${parse_exit_code})"
                fi
            else
                case_exit_code=$?
                echo "     -> seq=${seq} failed with exit code ${case_exit_code}; keeping partial log and continuing"
                echo "${system},${model},${B},${seq},${GEN_TOKENS},${case_max_s},${implementation_name},FAIL(${case_exit_code}),FAIL(${case_exit_code}),FAIL(${case_exit_code}),FAIL(${case_exit_code})" >> "$SUMMARY_CSV"
                echo "     -> summary recorded: implementation=${implementation_name}, seq=${seq}, decode_tok_s_user=FAIL(${case_exit_code})"
            fi
        done
    done
done

echo "Saved summary to ${SUMMARY_CSV}"
