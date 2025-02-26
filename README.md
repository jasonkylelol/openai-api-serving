# openai-api-serving

- running local open-source model like glm-4 and qwen2.5 with OpenAI compatible API
- also with embedding and rerank support

## examples:
```bash
export MODEL_ROOT=/data/huggingface/models
export LLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/
export EMBEDDING_MODEL=maidalun1020/bce-embedding-base_v1/
export RERANK_MODEL=maidalun1020/bce-reranker-base_v1/
CUDA_VISIBLE_DEVICES=2,3 python3 openai_api_all_in_one.py
```