import json
import yaml
import httpx
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse # Adicionado StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

CONFIG_PATH = "models.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MODEL_MAP: Dict[str, Dict[str, Any]] = CONFIG["models"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models")
async def list_models():
    models_list = []
    for model_id in MODEL_MAP.keys():
        models_list.append({
            "id": model_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": "local-gateway"
        })
    return {"object": "list", "data": models_list}

def get_backend(model_name: str) -> Dict[str, Any]:
    if model_name not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")
    return MODEL_MAP[model_name]

# FUNÇÃO AUXILIAR PARA STREAMING
# No seu proxy_stream, tente garantir que o iter_bytes seja lido sem buffering agressivo
async def proxy_stream(url: str, body: dict):
    client = httpx.AsyncClient(timeout=None)
    async def event_generator():
        try:
            async with client.stream("POST", url, json=body) as response:
                # O segredo está em ler os chunks conforme eles chegam
                async for chunk in response.aiter_raw(): 
                    yield chunk
        finally:
            await client.aclose()
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")

    if not model_name:
        raise HTTPException(status_code=400, detail="'model' field is required")

    backend = get_backend(model_name)
    endpoint = backend["endpoint"]

    # --- HIGIENIZAÇÃO DO BODY PARA AGENTES (Evita o "Empty Response") ---
    # Mantém o OpenWebUI e n8n funcionando, mas remove chaves que o llama-server rejeita
    allowed_keys = {
        "model", "messages", "temperature", "top_p", "stream", 
        "max_tokens", "stop", "presence_penalty", "frequency_penalty"
    }
    clean_body = {k: v for k, v in body.items() if k in allowed_keys}
    # --------------------------------------------------------------------

    if backend["type"] in ("text", "code", "vision"):
        # Se o cliente pedir streaming, repassamos o stream usando o body limpo
        if body.get("stream", False):
            return await proxy_stream(f"{endpoint}/v1/chat/completions", clean_body)

        # Caso contrário, resposta normal usando o body limpo
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{endpoint}/v1/chat/completions", json=clean_body)
            return JSONResponse(status_code=resp.status_code, content=resp.json())

    raise HTTPException(status_code=500, detail="Unsupported backend type")

@app.post("/v1/images/captions/{model_name}")
async def vision_captions(model_name: str, request: Request):
    backend = get_backend(model_name)
    endpoint = backend["endpoint"]
    async with httpx.AsyncClient(timeout=None) as client:
        content_type = request.headers.get("Content-Type")
        body = await request.body()
        resp = await client.post(
            f"{endpoint}/v1/images/captions",
            content=body,
            headers={"Content-Type": content_type}
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())