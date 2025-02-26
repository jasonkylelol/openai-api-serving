from pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

workdir /workspace

# add models models
add requirements.txt requirements.txt
run pip install -r requirements.txt

add openai_api_all_in_one.py openai_api_all_in_one.py
add openai_api_app.py openai_api_app.py
add openai_api_protocol.py openai_api_protocol.py
add openai_api_glm4_app.py openai_api_glm4_app.py
add openai_api_qwen2_app.py openai_api_qwen2_app.py
add openai_api_embedding_app.py openai_api_embedding_app.py
add openai_api_rerank_app.py openai_api_rerank_app.py

env MODEL_ROOT=/workspace/models
# env LLM_MODEL=llm
# env EMBEDDING_MODEL=embedding

CMD ["python3", "openai_api_all_in_one.py"]

# docker build -t graphrag/openai-compatible-api:v4.0 .
