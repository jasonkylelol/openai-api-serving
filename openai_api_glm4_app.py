import time
from asyncio.log import logger
import re
import gc
import json
import torch
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from fastapi import HTTPException, Response
from typing import List, Union
from transformers import AutoTokenizer
from sse_starlette.sse import EventSourceResponse
from openai_api_app import App
from openai_api_protocol import (
    generate_id, UsageInfo, DeltaMessage, ChatMessage,
    ModelCard, ModelList, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChoiceDeltaToolCallFunction, ChatCompletionMessageToolCall, FunctionCall,
)

class GLM4App(App):
    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(engine_args.tokenizer, trust_remote_code=True)


    async def health(self) -> Response:
        try:
            await self.engine.check_health()
        except Exception as e:
            raise HTTPException(status_code=500, detail="model not ready") from e
        return Response(status_code=200)


    async def list_models(self):
        model_card = ModelCard(id="glm-4")
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
        print(f"----- request -----\n{gen_params}", flush=True)

        if request.stream:
            predict_stream_generator = self.predict_stream(request.model, gen_params)
            output = await anext(predict_stream_generator)
            if output:
                return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
            logger.debug(f"First result output: \n{output}")

            function_call = None
            if output and request.tools:
                try:
                    function_call = self.process_response(output, request.tools, use_tool=True)
                except:
                    logger.warning("Failed to parse tool call")

            if isinstance(function_call, dict):
                function_call = ChoiceDeltaToolCallFunction(**function_call)
                generate = self.parse_output_text(request.model, output, function_call=function_call)
                return EventSourceResponse(generate, media_type="text/event-stream")
            else:
                return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
        response = ""
        async for response in self.generate_stream_glm4(gen_params):
            pass

        if response["text"].startswith("\n"):
            response["text"] = response["text"][1:]
        response["text"] = response["text"].strip()

        usage = UsageInfo()

        function_call, finish_reason = None, "stop"
        tool_calls = None
        if request.tools:
            try:
                function_call = self.process_response(response["text"], request.tools, use_tool=True)
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {e}")
        if isinstance(function_call, dict):
            finish_reason = "tool_calls"
            function_call_response = ChoiceDeltaToolCallFunction(**function_call)
            function_call_instance = FunctionCall(
                name=function_call_response.name,
                arguments=function_call_response.arguments
            )
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=generate_id('call_', 24),
                    function=function_call_instance,
                    type="function")]

        message = ChatMessage(
            role="assistant",
            content=None if tool_calls else response["text"],
            function_call=None,
            tool_calls=tool_calls,
        )

        if message.content and isinstance(message.content, str):
            prefix = "```json\n"
            if message.content.startswith(prefix):
                message.content = message.content[len(prefix):]
                message.content = message.content.replace("\n", "")
                message.content = message.content.replace("```", "")

        print(f"----- response -----\n{message}\n", flush=True)

        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )
        task_usage = UsageInfo.model_validate(response["usage"])
        for usage_key, usage_value in task_usage.model_dump().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        return ChatCompletionResponse(
            model=request.model,
            choices=[choice_data],
            object="chat.completion",
            usage=usage
        )


    async def predict_stream(self, model_id, gen_params):
        output = ""
        is_function_call = False
        has_send_first_chunk = False
        created_time = int(time.time())
        function_name = None
        response_id = generate_id('chatcmpl-', 29)
        system_fingerprint = generate_id('fp_', 9)
        tools = {tool['function']['name'] for tool in gen_params['tools']} if gen_params['tools'] else {}
        delta_text = ""
        full_text = ""
        async for new_response in self.generate_stream_glm4(gen_params):
            decoded_unicode = new_response["text"]
            delta_text += decoded_unicode[len(output):]
            full_text += delta_text
            output = decoded_unicode
            lines = output.strip().split("\n")

            # 检查是否为工具
            # 这是一个简单的工具比较函数，不能保证拦截所有非工具输出的结果，比如参数未对齐等特殊情况。
            ##TODO 如果你希望做更多处理，可以在这里进行逻辑完善。

            if not is_function_call and len(lines) >= 2:
                first_line = lines[0].strip()
                if first_line in tools:
                    is_function_call = True
                    function_name = first_line
                    delta_text = lines[1]

            # 工具调用返回
            if is_function_call:
                if not has_send_first_chunk:
                    function_call = {"name": function_name, "arguments": ""}
                    tool_call = ChatCompletionMessageToolCall(
                        index=0,
                        id=generate_id('call_', 24),
                        function=FunctionCall(**function_call),
                        type="function"
                    )
                    message = DeltaMessage(
                        content=None,
                        role="assistant",
                        function_call=None,
                        tool_calls=[tool_call]
                    )
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=message,
                        finish_reason=None
                    )
                    chunk = ChatCompletionResponse(
                        model=model_id,
                        id=response_id,
                        choices=[choice_data],
                        created=created_time,
                        system_fingerprint=system_fingerprint,
                        object="chat.completion.chunk"
                    )
                    yield ""
                    yield chunk.model_dump_json(exclude_unset=True)
                    has_send_first_chunk = True

                function_call = {"name": None, "arguments": delta_text}
                delta_text = ""
                tool_call = ChatCompletionMessageToolCall(
                    index=0,
                    id=None,
                    function=FunctionCall(**function_call),
                    type="function"
                )
                message = DeltaMessage(
                    content=None,
                    role=None,
                    function_call=None,
                    tool_calls=[tool_call]
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=None
                )
                chunk = ChatCompletionResponse(
                    model=model_id,
                    id=response_id,
                    choices=[choice_data],
                    created=created_time,
                    system_fingerprint=system_fingerprint,
                    object="chat.completion.chunk"
                )
                yield chunk.model_dump_json(exclude_unset=True)

            # 用户请求了 Function Call 但是框架还没确定是否为Function Call
            elif (gen_params["tools"] and gen_params["tool_choice"] != "none") or is_function_call:
                continue

            # 常规返回
            else:
                finish_reason = new_response.get("finish_reason", None)
                if not has_send_first_chunk:
                    message = DeltaMessage(
                        content="",
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
                        object="chat.completion.chunk"
                    )
                    yield chunk.model_dump_json(exclude_unset=True)
                    has_send_first_chunk = True

                message = DeltaMessage(
                    content=delta_text,
                    role="assistant",
                    function_call=None,
                )
                delta_text = ""
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
                    object="chat.completion.chunk"
                )
                yield chunk.model_dump_json(exclude_unset=True)

        # 工具调用需要额外返回一个字段以对齐 OpenAI 接口
        if is_function_call:
            yield ChatCompletionResponse(
                model=model_id,
                id=response_id,
                system_fingerprint=system_fingerprint,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            content=None,
                            role=None,
                            function_call=None,
                        ),
                        finish_reason="tool_calls"
                    )],
                created=created_time,
                object="chat.completion.chunk",
                usage=None
            ).model_dump_json(exclude_unset=True)
        elif delta_text != "":
            message = DeltaMessage(
                content="",
                role="assistant",
                function_call=None,
            )
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=None
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                id=response_id,
                choices=[choice_data],
                created=created_time,
                system_fingerprint=system_fingerprint,
                object="chat.completion.chunk"
            )
            yield chunk.model_dump_json(exclude_unset=True)
        
            finish_reason = 'stop'
            message = DeltaMessage(
                content=delta_text,
                role="assistant",
                function_call=None,
            )
            delta_text = ""
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
                object="chat.completion.chunk"
            )
            yield chunk.model_dump_json(exclude_unset=True)
            yield '[DONE]'
        else:
            yield '[DONE]'
        print(f"----- streaming response -----\n{full_text}\n", flush=True)


    async def parse_output_text(self, model_id: str, value: str,
        function_call: ChoiceDeltaToolCallFunction = None):
        delta = DeltaMessage(role="assistant", content=value)
        if function_call is not None:
            delta.function_call = function_call

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            choices=[choice_data],
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        yield '[DONE]'


    def process_response(self, output: str, tools: dict | List[dict] = None, use_tool: bool = False) -> Union[str, dict]:
        lines = output.strip().split("\n")
        arguments_json = None
        special_tools = ["cogview", "simple_browser"]
        tools = {tool['function']['name'] for tool in tools} if tools else {}

        # 这是一个简单的工具比较函数，不能保证拦截所有非工具输出的结果，比如参数未对齐等特殊情况。
        ##TODO 如果你希望做更多判断，可以在这里进行逻辑完善。

        if len(lines) >= 2 and lines[1].startswith("{"):
            function_name = lines[0].strip()
            arguments = "\n".join(lines[1:]).strip()
            if function_name in tools or function_name in special_tools:
                try:
                    arguments_json = json.loads(arguments)
                    is_tool_call = True
                except json.JSONDecodeError:
                    is_tool_call = function_name in special_tools

                if is_tool_call and use_tool:
                    content = {
                        "name": function_name,
                        "arguments": json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments,
                                                ensure_ascii=False)
                    }
                    if function_name == "simple_browser":
                        search_pattern = re.compile(r'search\("(.+?)"\s*,\s*recency_days\s*=\s*(\d+)\)')
                        match = search_pattern.match(arguments)
                        if match:
                            content["arguments"] = json.dumps({
                                "query": match.group(1),
                                "recency_days": int(match.group(2))
                            }, ensure_ascii=False)
                    elif function_name == "cogview":
                        content["arguments"] = json.dumps({
                            "prompt": arguments
                        }, ensure_ascii=False)

                    return content
        return output.strip()


    @torch.inference_mode()
    async def generate_stream_glm4(self, params):
        messages = params["messages"]
        tools = params["tools"]
        tool_choice = params["tool_choice"]
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_tokens", 8192))

        messages = self.process_messages(messages, tools=tools, tool_choice=tool_choice)
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        params_dict = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": -1,
            "stop_token_ids": [151329, 151336, 151338],
            "ignore_eos": False,
            "max_tokens": max_new_tokens,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }
        sampling_params = SamplingParams(**params_dict)
        async for output in self.engine.generate(inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
            output_len = len(output.outputs[0].token_ids)
            input_len = len(output.prompt_token_ids)
            ret = {
                "text": output.outputs[0].text,
                "usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": output_len,
                    "total_tokens": output_len + input_len
                },
                "finish_reason": output.outputs[0].finish_reason,
            }
            yield ret
        gc.collect()
        torch.cuda.empty_cache()


    def process_messages(self, messages, tools=None, tool_choice="none"):
        _messages = messages
        processed_messages = []
        msg_has_sys = False

        def filter_tools(tool_choice, tools):
            function_name = tool_choice.get('function', {}).get('name', None)
            if not function_name:
                return []
            filtered_tools = [
                tool for tool in tools
                if tool.get('function', {}).get('name') == function_name
            ]
            return filtered_tools

        if tool_choice != "none":
            if isinstance(tool_choice, dict):
                tools = filter_tools(tool_choice, tools)
            if tools:
                processed_messages.append(
                    {
                        "role": "system",
                        "content": None,
                        "tools": tools
                    }
                )
                msg_has_sys = True

        if isinstance(tool_choice, dict) and tools:
            processed_messages.append(
                {
                    "role": "assistant",
                    "metadata": tool_choice["function"]["name"],
                    "content": ""
                }
            )

        for m in _messages:
            role, content, func_call = m.role, m.content, m.function_call
            tool_calls = getattr(m, 'tool_calls', None)

            if role == "function":
                processed_messages.append(
                    {
                        "role": "observation",
                        "content": content
                    }
                )
            elif role == "tool":
                processed_messages.append(
                    {
                        "role": "observation",
                        "content": content,
                        "function_call": True
                    }
                )
            elif role == "assistant":
                if tool_calls:
                    for tool_call in tool_calls:
                        processed_messages.append(
                            {
                                "role": "assistant",
                                "metadata": tool_call.function.name,
                                "content": tool_call.function.arguments
                            }
                        )
                else:
                    for response in content.split("\n"):
                        if "\n" in response:
                            metadata, sub_content = response.split("\n", maxsplit=1)
                        else:
                            metadata, sub_content = "", response
                        processed_messages.append(
                            {
                                "role": role,
                                "metadata": metadata,
                                "content": sub_content.strip()
                            }
                        )
            else:
                if role == "system" and msg_has_sys:
                    msg_has_sys = False
                    continue
                processed_messages.append({"role": role, "content": content})

        if not tools or tool_choice == "none":
            for m in _messages:
                if m.role == 'system':
                    processed_messages.insert(0, {"role": m.role, "content": m.content})
                    break
        return processed_messages
