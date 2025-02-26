from starlette.concurrency import run_in_threadpool
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from openai_api_protocol import RerankRequest, RerankResponse, RerankResult, RerankDoc


NORMALIZE_EMBEDDINGS = "1"
E5_EMBED_INSTRUCTION = "passage: "
E5_QUERY_INSTRUCTION = "query: "
BGE_EN_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

class RerankApp:
    def __init__(self, model_path):
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self.device = torch.device(f'cuda:{device_count-1}' if device_count > 1 else 'cuda')
        else:
            self.device = torch.device('cpu')

        self.model_name = model_path
        print(f"Loading rerank device: {self.device} model: {self.model_name}", flush=True)
        
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            model_path, device_map=self.device)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, device_map=self.device, low_cpu_mem_usage=True)
        self.rerank_model = self.rerank_model.eval()


    async def rerank(self, request: RerankRequest):
        print(f"[Rerank] request: {request}", flush=True)
        return await run_in_threadpool(
            self._rerank, **request.model_dump()
        )


    def _rerank(self, model, query, documents, top_n, return_documents = True):
        if not top_n:
            top_n = len(documents)
        if len(documents) < 2:
            return documents
        if not self.rerank_model:
            return documents[:top_n]

        pairs = []
        for idx, document in enumerate(documents):
            pairs.append([query, document])
        rerank_results = []
        
        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True,
                return_tensors='pt', max_length=512).to(self.device)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = torch.sigmoid(scores)
            scores = scores.tolist()
        
        # print(f"scores: {scores}")
        combined_list = list(zip(documents, scores))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
        for idx, item in enumerate(sorted_combined_list):
            if idx >= top_n:
                break
            document = item[0]
            score = item[1]

            rerank_results.append(RerankResult(
                index=idx,
                relevance_score=score,
                document=RerankDoc(text=document),
            ))

        print(f"[rerank] fetched {len(rerank_results)} docs")
        return RerankResponse(
            model=model,
            results=rerank_results)
