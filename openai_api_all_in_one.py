import sys, os, signal
import uvicorn
from vllm import AsyncEngineArgs
from fastapi import FastAPI, Response, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import torch
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse
from openai_api_app import App
from openai_api_protocol import (
    ModelList, ChatCompletionRequest, ChatCompletionResponse,
    CreateEmbeddingRequest, CreateEmbeddingResponse
)

MODEL_ROOT = os.getenv("MODEL_ROOT", "/root/huggingface/models")
LLM_MODEL_PATH = f"{MODEL_ROOT}/{os.getenv('LLM_MODEL')}"
EMBEDDING_MODEL_PATH = f"{MODEL_ROOT}/{os.getenv('EMBEDDING_MODEL')}"
SERVER_PORT = int(os.getenv("SERVER_PORT", 8060))
MAX_MODEL_LENGTH = 1024 * 16

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

svr, llm_app, embedding_app = None, App, None
llm_router = APIRouter()
embedding_router = APIRouter()

def llm_init():
    tensor_parallel_size = torch.cuda.device_count()
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    engine_args = AsyncEngineArgs(
        model=LLM_MODEL_PATH,
        tokenizer=LLM_MODEL_PATH,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=False,
        disable_log_requests=True,
        max_model_len=MAX_MODEL_LENGTH,
        enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_MODEL_LENGTH,
        disable_custom_all_reduce=True,
    )
    print(f"[AsyncEngineArgs] tensor_parallel_size: {engine_args.tensor_parallel_size} "
        f"gpu_memory_utilization: {engine_args.gpu_memory_utilization} ")
    
    global llm_app

    if LLM_MODEL_PATH.lower().count("glm-4") > 0:
        from openai_api_glm4_app import GLM4App
        llm_app = GLM4App(engine_args)
    elif LLM_MODEL_PATH.lower().count("qwen2.5") > 0:
        from openai_api_qwen2_app import Qwen2App
        llm_app = Qwen2App(engine_args)
    else:
        raise RuntimeError(f"invalid LLM model: {LLM_MODEL_PATH}")


def embedding_init():
    global embedding_app

    from openai_api_embedding_app import EmbeddingApp
    embedding_app = EmbeddingApp(EMBEDDING_MODEL_PATH)


@llm_router.get("/health")
async def health() -> Response:
    return await llm_app.health()


@llm_router.get("/v1/models", response_model=ModelList)
async def list_models():
    return await llm_app.list_models()


@llm_router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    return await llm_app.create_chat_completion(request)


@embedding_router.post("/v1/embeddings", response_model=CreateEmbeddingResponse)
async def create_embedding(request: CreateEmbeddingRequest):
    return await embedding_app.create_embedding(request)


def handler():
    app = FastAPI(
        title="OpenAI Compatible API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(llm_router)
    app.include_router(embedding_router)

    host = "0.0.0.0"
    config = uvicorn.Config(app=app, host=host, port=SERVER_PORT)
    global svr

    svr = uvicorn.Server(config)
    print(f"[AllInOne] start api server on {host}:{SERVER_PORT}", flush=True)
    svr.run()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("[Main] shutting down...", flush=True)
        if svr: svr.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    llm_init()
    embedding_init()
    handler()

    print("[Main] exited", flush=True)
