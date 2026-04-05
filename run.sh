export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

export MESH_DEVICE=N150
export HF_MODEL=/shared/models/Llama-3.2-3B-Instruct
export TT_CACHE_PATH=$HOME/.cache/ttt-weight-cache

export B=${B:-32}
export S=${S:-512}
export MAX_S=${MAX_S:-1040}
export GEN_TOKENS=${GEN_TOKENS:-128}

export TTT_OPTIMIZED_MATMUL_TRANSPOSED=${TTT_OPTIMIZED_MATMUL_TRANSPOSED:-1}

export FIX_PREFILL_LEN=$S
export MODE=decode

python -m tt_lock pytest --tb=short -s \
  models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
  -k "batch-32 and performance" \
  --mode $MODE \
  --max_generated_tokens $GEN_TOKENS \
  --max_seq_len $MAX_S \
  --batch_size $B


# python -m tt_lock python -m tracy -r -p -v -m pytest --tb=short -s \
#   models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
#   -k "batch-32 and performance" \
#   --mode $MODE \
#   --max_generated_tokens 1 \
#   --max_seq_len $MAX_S \
#   --batch_size $B \
#   --num_layers 1
