# T3K Model Test Notes

This branch keeps a minimal README focused on local model bring-up and test commands.

## Environment

- Conda env: `ttt`
- Model cache / checkpoints: `/shared/models`
- Test entrypoint: [simple_text_demo.py](tt-metal/models/tt_transformers/demo/simple_text_demo.py)

Base command used for stock T3K end-to-end testing:

```bash
cd $TT_METAL_HOME

MESH_DEVICE=T3K \
HF_MODEL=/shared/models/<MODEL_DIR> \
timeout 1800s \
conda run -n ttt \
pytest -s models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --max_generated_tokens 1 \
  --max_seq_len 256 \
  --batch_size 1
```

## Tested Models

The following models were run on full T3K with the stock `tt_transformers` path and passed end-to-end.

| Model | HF_MODEL | Result |
| --- | --- | --- |
| Llama-3.1-8B-Instruct | `/shared/models/Llama-3.1-8B-Instruct` | PASS |
| Llama-3.2-3B-Instruct | `/shared/models/Llama-3.2-3B-Instruct` | PASS |
| Qwen3-32B | `/shared/models/Qwen3-32B` | PASS |
| Mistral-7B-Instruct-v0.3 | `/shared/models/Mistral-7B-Instruct-v0.3` | PASS |

## Commands Used

### Llama-3.1-8B-Instruct

```bash
cd $TT_METAL_HOME

MESH_DEVICE=T3K \
HF_MODEL=/shared/models/Llama-3.1-8B-Instruct \
timeout 900s \
pytest -s models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --max_generated_tokens 1 \
  --max_seq_len 256 \
  --batch_size 1
```

### Llama-3.2-3B-Instruct

```bash
cd $TT_METAL_HOME

MESH_DEVICE=T3K \
HF_MODEL=/shared/models/Llama-3.2-3B-Instruct \
timeout 900s \
pytest -s models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --max_generated_tokens 1 \
  --max_seq_len 256 \
  --batch_size 1
```

### Qwen3-32B

```bash
cd $TT_METAL_HOME

MESH_DEVICE=T3K \
HF_MODEL=/shared/models/Qwen3-32B \
timeout 1800s \
pytest -s models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --max_generated_tokens 1 \
  --max_seq_len 256 \
  --batch_size 1
```

### Mistral-7B-Instruct-v0.3

```bash
cd $TT_METAL_HOME

MESH_DEVICE=T3K \
HF_MODEL=/shared/models/Mistral-7B-Instruct-v0.3 \
timeout 900s \
pytest -s models/tt_transformers/demo/simple_text_demo.py \
  -k "performance and batch-1" \
  --max_generated_tokens 1 \
  --max_seq_len 256 \
  --batch_size 1
```
