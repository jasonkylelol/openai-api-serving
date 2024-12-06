from abc import ABC, abstractmethod
from fastapi import Response
from openai_api_protocol import ModelList, ChatCompletionRequest

class App(ABC):
    @abstractmethod
    async def health(self) -> Response:
        pass


    @abstractmethod
    async def list_models(self) -> ModelList:
        pass


    @abstractmethod
    async def create_chat_completion(self, request: ChatCompletionRequest):
        pass