export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6


export MESH_DEVICE=N150
export HF_MODEL=/shared/models/Llama-3.2-3B-Instruct
export TT_CACHE_PATH=$HOME/.cache/ttt-weight-cache
export FIX_PREFILL_LEN=1024
export MAX_SEQ_LEN=1040

# python -m tracy -r -p -v -m pytest -s \
#   models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
#   -k "batch-32 and performance" \
#   --mode prefill \
#   --max_generated_tokens 1 \
#   --max_seq_len $MAX_SEQ_LEN \
#   --batch_size 32 \
#   --num_layers 1

pytest --tb=short -s \
  models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
  -k "batch-32 and performance" \
  --mode prefill \
  --max_generated_tokens 1 \
  --max_seq_len $MAX_SEQ_LEN \
  --batch_size 32