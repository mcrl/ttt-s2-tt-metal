export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6


MESH_DEVICE=N150 \
HF_MODEL=/shared/models/Llama-3.2-3B-Instruct \
python -m tracy -r -p -v -m pytest -s \
  models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
  -k "batch-32 and performance" \
  --mode prefill \
  --max_generated_tokens 1 \
  --max_seq_len 256 \
  --batch_size 32 \
  --num_layers 1 \
  |& tee -a log