from typing import List, Union
from starlette.concurrency import run_in_threadpool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings
)
import torch
import pydantic
from transformers import AutoTokenizer
from openai_api_protocol import (
    CreateEmbeddingRequest, CreateEmbeddingResponse, Embedding, UsageInfo
)

NORMALIZE_EMBEDDINGS = "1"
E5_EMBED_INSTRUCTION = "passage: "
E5_QUERY_INSTRUCTION = "query: "
BGE_EN_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

class EmbeddingApp:
    def __init__(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}", flush=True)

        self.model_name = model_path
        print(f"Loading embedding model: {self.model_name}", flush=True)
        encode_kwargs = {
            "normalize_embeddings": NORMALIZE_EMBEDDINGS
        }
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if "e5" in self.model_name:
            self.embeddings = HuggingFaceInstructEmbeddings(model_name=self.model_name,
                embed_instruction=E5_EMBED_INSTRUCTION,
                query_instruction=E5_QUERY_INSTRUCTION,
                encode_kwargs=encode_kwargs,
                model_kwargs={"device": device})
        elif "bge-" in self.model_name and "-en" in self.model_name:
            self.embeddings = HuggingFaceBgeEmbeddings(model_name=self.model_name,
                query_instruction=BGE_EN_QUERY_INSTRUCTION,
                encode_kwargs=encode_kwargs,
                model_kwargs={"device": device})
        elif "bge-" in self.model_name and "-zh" in self.model_name:
            self.embeddings = HuggingFaceBgeEmbeddings(model_name=self.model_name,
                query_instruction=BGE_ZH_QUERY_INSTRUCTION,
                encode_kwargs=encode_kwargs,
                model_kwargs={"device": device})
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name,
                encode_kwargs=encode_kwargs,
                model_kwargs={"device": device})


    async def create_embedding(self, request: CreateEmbeddingRequest):
        print(f"[Embeddings] request: {request}", flush=True)
        if pydantic.__version__ > '2.0.0':
            return await run_in_threadpool(
                self._create_embedding, **request.model_dump(exclude={"user", "model", "model_config", "dimensions"})
            )
        else:
            return await run_in_threadpool(
                self._create_embedding, **request.dict(exclude={"user", "model", "model_config", "dimensions"})
            )


    def _create_embedding(self, input: Union[str, List[str]]):
        model_name_short = self.model_name.split("/")[-1]
        if isinstance(input, str):
            tokens = self.tokenizer.tokenize(input)
            return CreateEmbeddingResponse(
                data=[
                    Embedding(embedding=self.embeddings.embed_query(input), object="embedding", index=0)
                ],
                model=model_name_short,
                object='list',
                usage=UsageInfo(prompt_tokens=len(tokens), total_tokens=len(tokens))
            )
        else:
            data = [Embedding(embedding=embedding, object="embedding", index=i)
                for i, embedding in enumerate(self.embeddings.embed_documents(input))]
            total_tokens = 0
            for text in input:
                total_tokens += len(self.tokenizer.tokenize(text))
            return CreateEmbeddingResponse(
                data=data,
                model=model_name_short,
                object='list',
                usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens)
            )

