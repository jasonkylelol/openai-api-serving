# openai-api-serving

- running local open-source model like glm-4 and qwen2.5 with OpenAI compatible API
- also with embedding support

## examples:
```bash
LLM_MODEL=Qwen/Qwen2.5-14B-Instruct EMBEDDING_MODEL=TencentBAC/Conan-embedding-v1 CUDA_VISIBLE_DEVICES=1,2 python3 openai_api_all_in_one.py

LLM_MODEL=Qwen/Qwen2.5-7B-Instruct EMBEDDING_MODEL=TencentBAC/Conan-embedding-v1 CUDA_VISIBLE_DEVICES=1 python3 openai_api_all_in_one.py
```