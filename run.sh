export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

export TTT_OPTIMIZED_MATMUL=0

if [ "$TTT_OPTIMIZED_MATMUL" -eq 1 ]; then
    MATMUL_MODE="optimized"
else
    MATMUL_MODE="naive"
fi

export MODEL=Llama-3.1-70B-Instruct
export MESH_DEVICE=T3K #N150, T3K, P150, P150x4, TG
export HF_MODEL=/shared/models/$MODEL
export TT_CACHE_PATH=$HOME/.cache/ttt-weight-cache/$MODEL/$MATMUL_MODE

export B=${B:-32}
export S=${S:-1024}
export MAX_S=${MAX_S:-1500}
export GEN_TOKENS=${GEN_TOKENS:-1}

export FIX_PREFILL_LEN=$S
# export MODE=decode

python -m tt_lock pytest --tb=short -s \
  models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
  -k "batch-32 and performance" \
  --max_generated_tokens $GEN_TOKENS \
  --max_seq_len $MAX_S \
  --batch_size $B \
# --mode $MODE

: << "END"
python -m tt_lock python -m tracy -r -p -v -m pytest --tb=short -s \
  models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
  -k "batch-32 and performance" \
  --mode $MODE \
  --max_generated_tokens 1 \
  --max_seq_len $MAX_S \
  --batch_size $B \
  --num_layers 1
END