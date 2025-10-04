import os
import json
from typing import List
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq  # 👈🏽 Aqui usamos o cliente Groq

# Imports locais
from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.tools import get_current_weather

# 🔐 Carregar variáveis
load_dotenv(".env.local")

app = FastAPI()

# 🧠 Inicializar cliente GROQ
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY") or "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",
)

# 📦 Modelo da requisição
class Request(BaseModel):
    messages: List[ClientMessage]

# 🧰 Tools disponíveis
available_tools = {
    "get_current_weather": get_current_weather,
}

# ⚙️ Função que faz stream da resposta
def stream_text(messages: List, protocol: str = 'data'):
    draft_tool_calls = []
    draft_tool_calls_index = -1

    stream = client.chat.completions.create(
        messages=messages,
        model="llama-3.1-70b-versatile",  # 👈🏽 Usa modelo Groq
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather at a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number", "description": "The latitude of the location"},
                            "longitude": {"type": "number", "description": "The longitude of the location"},
                        },
                        "required": ["latitude", "longitude"],
                    },
                },
            },
        ],
    )

    # 🌀 Stream de resposta em tempo real
    for chunk in stream:
        for choice in chunk.choices:
            if choice.finish_reason == "stop":
                continue

            elif choice.finish_reason == "tool_calls":
                for tool_call in draft_tool_calls:
                    yield f'9:{{"toolCallId":"{tool_call["id"]}","toolName":"{tool_call["name"]}","args":{tool_call["arguments"]}}}\n'

                for tool_call in draft_tool_calls:
                    tool_result = available_tools[tool_call["name"]](**json.loads(tool_call["arguments"]))
                    yield f'a:{{"toolCallId":"{tool_call["id"]}","toolName":"{tool_call["name"]}","args":{tool_call["arguments"]},"result":{json.dumps(tool_result)}}}\n'

            elif choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    id = tool_call.id
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments

                    if id is not None:
                        draft_tool_calls_index += 1
                        draft_tool_calls.append({"id": id, "name": name, "arguments": ""})
                    else:
                        draft_tool_calls[draft_tool_calls_index]["arguments"] += arguments

            else:
                yield f'0:{json.dumps(choice.delta.content)}\n'

        if not chunk.choices:
            usage = chunk.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            yield (
                f'e:{{"finishReason":"{"tool-calls" if len(draft_tool_calls) > 0 else "stop"}",'
                f'"usage":{{"promptTokens":{prompt_tokens},"completionTokens":{completion_tokens}}},'
                f'"isContinued":false}}\n'
            )


# 🔥 Endpoint principal
@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    messages = request.messages
    groq_messages = convert_to_openai_messages(messages)

    response = StreamingResponse(stream_text(groq_messages, protocol))
    response.headers['x-vercel-ai-data-stream'] = 'v1'
    return response
    
