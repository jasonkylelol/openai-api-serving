import time
import torch
import re
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from fastapi import HTTPException, Response
from transformers import AutoTokenizer
from sse_starlette.sse import EventSourceResponse
from openai_api_app import App
from openai_api_protocol import (
    generate_id, UsageInfo, DeltaMessage, ChatMessage,
    ModelCard, ModelList, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
)

class Qwen2App(App):
    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(engine_args.tokenizer, trust_remote_code=True)


    async def health(self) -> Response:
        try:
            await self.engine.check_health()
        except Exception as e:
            raise HTTPException(status_code=500, detail="model not ready") from e
        return Response(status_code=200)


    async def list_models(self) -> ModelList:
        model_card = ModelCard(id="Qwen2.5")
        return ModelList(data=[model_card])


    async def create_chat_completion(self, request: ChatCompletionRequest):
        if len(request.messages) < 1 or request.messages[-1].role == "assistant":
            raise HTTPException(status_code=400, detail="Invalid request")

        gen_params = dict(
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            echo=False,
            stream=request.stream,
            repetition_penalty=request.repetition_penalty,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )
        print(f"----- request -----\n{gen_params}\n", flush=True)

        if request.stream:
            stream_generator = self.generate_stream(request.model, gen_params)
            return EventSourceResponse(stream_generator, media_type="text/event-stream")
        
        usage = UsageInfo()
        response_text, finish_reason = "", ""
        async for response in self.generate_stream(request.model, gen_params):
            chunk = ChatCompletionResponse.model_validate_json(response)
            usage = chunk.usage
            finish_reason = chunk.choices[0].finish_reason
            response_text += chunk.choices[0].delta.content

        message = ChatMessage(
            role="assistant",
            content=response_text,
        )
        if message.content and isinstance(message.content, str):
            message.content = message.content.strip()
            prefix = "```json"
            if message.content.startswith(prefix):
                message.content = message.content[len(prefix):]
                message.content = message.content.replace("\n", "")
                message.content = message.content.replace("```", "")
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )
        return ChatCompletionResponse(
            model=request.model,
            choices=[choice_data],
            object="chat.completion",
            usage=usage
        )


    @torch.inference_mode()
    async def generate_stream(self, model_id, params):
        messages = params["messages"]
        tools = params["tools"]
        tool_choice = params["tool_choice"]
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_tokens", 8192))
        created_time = int(time.time())
        response_id = generate_id('chatcmpl-', 29)
        system_fingerprint = generate_id('fp_', 9)
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        params_dict = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": -1,
            # "stop_token_ids": [151645],
            "ignore_eos": False,
            "max_tokens": max_new_tokens,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }
        sampling_params = SamplingParams(**params_dict)
        generate_text = ""
        think_skip = False
        async for output in self.engine.generate(
            inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):

            finish_reason = output.outputs[0].finish_reason
            output_len = len(output.outputs[0].token_ids)
            input_len = len(output.prompt_token_ids)
            output_text = output.outputs[0].text

            delta_text = output_text[len(generate_text):]
            generate_text = output_text
            usage = UsageInfo()
            task_usage = UsageInfo.model_validate({
                    "prompt_tokens": input_len,
                    "completion_tokens": output_len,
                    "total_tokens": output_len + input_len
                })
            for usage_key, usage_value in task_usage.model_dump().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

            # if delta_text.strip() == "<think>":
            #     think_skip = True
            # elif delta_text.strip() == "</think>":
            #     think_skip = False
            #     continue
            # if think_skip:
            #     continue

            message = DeltaMessage(
                    content=delta_text,
                    role="assistant",
                    function_call=None,
                )
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=finish_reason
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                id=response_id,
                choices=[choice_data],
                created=created_time,
                system_fingerprint=system_fingerprint,
                object="chat.completion.chunk",
                usage=usage,
            )
            yield chunk.model_dump_json(exclude_unset=True)

        # generate_text = re.sub(r'<think>.*?</think>', "", generate_text, flags=re.DOTALL)
        print(f"----- response -----\n{generate_text}\n", flush=True)
        # gc.collect()
        # torch.cuda.empty_cache()
