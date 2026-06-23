# 🚀 LLM Local Stack - Infraestrutura de IA Local

Esta documentação descreve a implementação de um stack completo de IA rodando localmente via **WSL2 (Ubuntu)**. O ecossistema utiliza `llama.cpp` para inferência de alta performance em CPU, um **Gateway FastAPI** para roteamento de modelos e scripts de gerenciamento para facilitar o uso no dia a dia.

---

## 📁 0. Estrutura de Pastas Real

### Instalação

**Clone o repositório:**
  ```bash
  git clone https://github.com/denilsonbomjesus/local-llm-cpu-stack.git
  cd local-llm-cpu-stack
  ```

> Nota: Recomanda-se alocar o projeto na raiz do perfil de usuário Linux. Ao baixar/ clonar o repositório, troque o nome do diretório principal de "local-llm-cpu-stack" para "llm-stack", para se adequar melhor aos comandos.

O projeto está organizado para manter modelos, servidores e scripts isolados:

```bash
~/llm-stack
  ├── llama.cpp/               # Código-fonte e binários compilados
  ├── models/                  # Arquivos .gguf organizados por categoria
  │   ├── text/                # Qwen2.5-1.5B, Gemma 2 2B Abliterated
  │   ├── code/                # Qwen2.5-Coder, Ministral-3-3B-Instruct-2512
  │   └── vision/              # Qwen2.5-VL (Modelo + mmproj)
  ├── vision/                  # Servidor Python para o Qwen-VL
  ├── gateway/                 # Roteador FastAPI (Porta 9000)
  ├── scripts/                 # Scripts de controle (manage.sh, chat.sh)
  ├── logs/                    # Logs de execução (opcional)
  ├── external_config.yaml     # Cardápio de modelos para ferramentas externas
  └── README.md                # Esta documentação
```

---

## 📦 1. Instalação base no WSL2 (Ubuntu)
### 1.1. Dependências de sistema
```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
  build-essential cmake git pkg-config \
  libssl-dev zlib1g-dev \
  python3 python3-venv python3-pip \
  curl wget \
  tmux \
  jq
```

Isso cobre compilação do `llama.cpp`, Python para gateway/vision, `tmux` para rodar em background. [github](http://github.com/ggml-org/llama.cpp)

Se quiser Node.js (opcional, não vou usar aqui):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

***
## 🧠 2. Clonar e compilar `llama.cpp` (CPU-only)
Vou usar o repositório oficial `ggml-org/llama.cpp`. [github](http://github.com/ggml-org/llama.cpp)
### 2.1. Clone
```bash
cd ~/llm-stack
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```
### 2.2. Build otimizado (Release, com suporte a K‑quants e HTTP)
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"
```

Isso gera binários em `~/llm-stack/llama.cpp/build/` (`llama-cli`, `llama-server`, etc.). [github](http://github.com/ggml-org/llama.cpp)

> Nota: não vamos ativar CUDA/CL aqui para manter a simplicidade e foco em CPU; a performance nos modelos escolhidos é excelente mesmo sem aceleração de GPU dedicada.

***
## 📥 3. Download dos modelos (WSL2)
Vou padronizar todos em `~/llm-stack/models`.

```bash
mkdir -p ~/llm-stack/models/{text,code,vision}
cd ~/llm-stack/models
```
### 3.1. Texto / agentes
#### Qwen2.5‑1.5B‑Instruct‑Q4_K_M.gguf

Repo oficial GGUF: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`. [huggingface](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)

Arquivo:

- `qwen2.5-1.5b-instruct-q4_k_m.gguf` (4‑bit K‑quant, recomendado para laptops). [skywork](https://skywork.ai/blog/models/qwen2-5-1-5b-instruct-gguf-free-chat-online-skywork-ai/)

```bash
cd ~/llm-stack/models/text
wget -O qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

#### Gemma-2-2b-it-abliterated-q4_k_m.gguf

Repo: `bartowski/gemma-2-2b-it-abliterated-GGUF`. [huggingface](https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF/resolve/main/gemma-2-2b-it-abliterated-Q4_K_M.gguf)

Arquivo:

- `Gemma-2-2b-it-abliterated-gguf` (~1.6 GB). [huggingface](https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF/resolve/main/gemma-2-2b-it-abliterated-Q4_K_M.gguf)

```bash
cd ~/llm-stack/models/text
wget -O gemma-2-2b-it-abliterated-q4_k_m.gguf \
  https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF/resolve/main/gemma-2-2b-it-abliterated-Q4_K_M.gguf
```
### 3.2. Código
#### Qwen2.5‑Coder‑1.5B‑Instruct‑Q4_K_M.gguf

Repo GGUF: `bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF`. [huggingface](https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF)

Arquivo recomendado: `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` (4‑bit, “default size for must use cases, *recommended*”). [huggingface](https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF)

```bash
cd ~/llm-stack/models/code
wget -O qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf
```

#### Ministral-3-3b-instruct-2512-q4_k_m.gguf

Repo: `unsloth/Ministral-3-3B-Instruct-2512-GGUF`. [huggingface](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf)

Arquivo:

- `Ministral-3-3B-Instruct-2512-GGUF` (Q4_K_M = 4.5 bpw, “balanced quality – recommended”). [huggingface](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf)

```bash
cd ~/llm-stack/models/code
wget -O ministral-3-3b-instruct-2512-q4_k_m.gguf \
  https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
```

#### Download do mmproj 
O arquivo é os "olhos" do Ministral-3-3B-Instruct-2512.

Execute este comando:
```bash
wget -O ~/llm-stack/models/code/ministral-3-3b-instruct-2512-mmproj-f16.gguf \
  https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/mmproj-F16.gguf
```

### 3.3. Visão – 

#### Qwen2.5‑VL‑3B‑Abliterated‑Caption‑it (GGUF)

Repo: `prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF`. [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF)

Arquivo 3B disponível na tree GGUF:

- `Qwen2.5-VL-3B-Abliterated-Caption-it.IQ4_XS.gguf` (4‑bit IQ4_XS, otimizada, ainda 4‑bit; e explicitamente Abliterated / Uncensored). [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF/tree/main/Qwen2.5-VL-3B-Abliterated-Caption-it-GGUF)

```bash
cd ~/llm-stack/models/vision
wget -O qwen2.5-vl-3b-abliterated-caption-it-iq4_xs.gguf \
  https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF/resolve/main/Qwen2.5-VL-3B-Abliterated-Caption-it-GGUF/Qwen2.5-VL-3B-Abliterated-Caption-it.IQ4_XS.gguf
```

#### Download do mmproj 
O arquivo é os "olhos" do Qwen2.5-VL-3B.

Execute este comando:
```bash
wget -O ~/llm-stack/models/vision/qwen2.5-vl-3b-abliterated-caption-it.mmproj-Q8_0.gguf \
  https://huggingface.co/lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf
```

> Obs.: não vi Q4_K_M explicitamente na tree, mas IQ4_XS é uma variante 4‑bit compacta, adequada para CPU. [huggingface](https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF)

***
## 🧠 2b. O que é Q4_K_M / IQ4_XS?
- **Q4_K_M**: quantização 4‑bit em “super‑blocos” com estatísticas por bloco, 4.5 bits por peso, trade‑off ótimo de qualidade vs RAM – a opção normalmente recomendada para uso geral. [huggingface](https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF)
- **IQ4_XS**: variante “imatrix” super‑compacta 4‑bit, com compressão adicional e pequena perda de qualidade, muito usada em modelos pequenos para reduzir footprint. [huggingface](https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF)

Alternativas:

- **Q5_K_M** (quando existir) → +qualidade, +RAM, +latência. [huggingface](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- **Q8_0** → quase sem perda, mas praticamente dobra memória e custo. [inference.readthedocs](https://inference.readthedocs.io/en/v0.15.4/models/builtin/llm/qwen2.5-instruct.html)

Para o seu notebook, Q4_K_M / IQ4_XS é o sweet spot mesmo. [skywork](https://skywork.ai/blog/models/qwen2-5-1-5b-instruct-gguf-free-chat-online-skywork-ai/)

***
## 🚀 4. Execução dos modelos com `manage.sh`
Para simplificar o gerenciamento de múltiplos modelos, utilizamos o script `./scripts/manage.sh`. Ele automatiza a criação de sessões `tmux` e garante que cada modelo rode na porta correta.

### 4.1. Como iniciar um modelo
No terminal (WSL2), execute:
```bash
./scripts/manage.sh start [modelo]
```

**Opções disponíveis:**
- `qwen-text`          → Porta 8001 (Qwen 2.5 1.5B)
- `gemma2`             → Porta 8002 (Gemma 2 2B)
- `qwen-coder`         → Porta 8003 (Qwen Coder 1.5B)
- `ministral-agent`    → Porta 8004 (Ministral 3 3B)
- `ministral-vision`   → Porta 8005 (Ministral 3 3B)
- `vision`             → Porta 8010 (Qwen-VL Python Server)
- `gateway`            → Porta 9000 (O Roteador Central)

### 4.2. Comandos úteis do Manager
- `./scripts/manage.sh status`      → Vê o que está rodando.
- `./scripts/manage.sh stop-all`    → Mata todos os serviços e libera a RAM.
- `./scripts/manage.sh stop gemma2` → Para apenas um modelo específico.

***
## 🌐 5. API HTTP OpenAI‑like do `llama-server`
O `llama-server` expõe endpoints OpenAI‑compatíveis, como `/v1/chat/completions`. [github](http://github.com/ggml-org/llama.cpp)

Teste rápido (no WSL2):

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct-q4_k_m",
    "messages": [
      {"role": "user", "content": "Resuma em 3 pontos a função de um sistema operacional."}
    ],
    "max_tokens": 128
  }'
```

Isso já deve retornar JSON no formato OpenAI. [learn.arm](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/llama-server/)

> O campo `"model"` aqui é ignorado pelo `llama-server` (ele já está “fixo” no binário), mas vamos usar esse campo no *gateway* para roteamento.  

***
## 🧠 10. Visão – servidores
### 10.1. Qwen2.5‑VL‑3B‑Abliterated (caption sem censura, em WSL2)
Para visão, hoje não há suporte multimodal de Qwen‑VL direto no `llama.cpp` vanilla, então vamos fazer um micro‑servidor Python usando Hugging Face Transformers, rodando CPU‑only (rápido o suficiente para imagens pontuais). [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF)

#### 10.1.1. Ambiente Python de visão (WSL2)

```bash
cd ~/llm-stack/vision
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install "transformers>=4.40.0" "accelerate" "torch" "safetensors" pillow fastapi uvicorn[standard]
```

> Torch CPU simples é suficiente para o seu caso; se quiser otimizar mais tarde, dá para trocar backend.  

#### 10.1.2. Servidor FastAPI para Qwen‑VL Abliterated

`~/llm-stack/vision/qwen_vl_server.py`:

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io

from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "prithivMLmods/Qwen2.5-VL-3B-Abliterated-Caption-GGUF"

app = FastAPI()

print("Loading Qwen2.5-VL-3B-Abliterated model...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, trust_remote_code=True)

class CaptionRequest(BaseModel):
    prompt: Optional[str] = "Describe this image in detail."

@app.post("/v1/images/captions")
async def caption_image(request: CaptionRequest = None, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prompt = (request.prompt if request and request.prompt else
              "Describe this image in detail.")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=128)
    caption = processor.batch_decode(out, skip_special_tokens=True)[0]
    return {"caption": caption}
```

> Observação: o modelo base do repo é Qwen2.5‑VL Abliterated; o arquivo GGUF é usado para `llama.cpp`, mas aqui usamos o modelo HF original (não‑GGUF) do mesmo repo; isso te dá um endpoint dedicado de caption. [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF/tree/main/Qwen2.5-VL-3B-Abliterated-Caption-it-GGUF)

Rodar:

```bash
cd ~/llm-stack/vision
source venv/bin/activate
tmux new-session -d -s qwen-vl \
  "uvicorn qwen_vl_server:app --host 0.0.0.0 --port 8010"
```

Teste:

```bash
curl -X POST "http://localhost:8010/v1/images/captions" \
  -F "file=@/caminho/para/sua_imagem.jpg"
```

***

## 🌐 6. Gateway / Model Router (WSL2, FastAPI)
Vamos fazer um gateway OpenAI‑style em **WSL2** que expõe `/v1/chat/completions` e encaminha para o servidor certo com base no campo `"model"`. [llamastack.github](https://llamastack.github.io/docs/providers/openai)
### 6.1. Ambiente Python para gateway
```bash
cd ~/llm-stack/gateway
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install fastapi uvicorn[standard] httpx pyyaml
```
### 6.2. Config YAML de modelos (para n8n e roteador)
`~/llm-stack/gateway/models.yaml`:

```yaml
models:
  qwen2.5-1.5b-instruct:
    type: text
    endpoint: http://localhost:8001

  gemma-2-2b-abliterated:
    type: text
    endpoint: http://localhost:8002

  qwen2.5-coder-1.5b:
    type: code
    endpoint: http://localhost:8003

  ministral-agent:
    type: code
    endpoint: http://localhost:8004

  ministral-vision:
    type: vision
    endpoint: http://localhost:8005

  qwen2.5-vl-3b-abliterated:
    type: vision
    endpoint: http://localhost:8010
```

> Esse mesmo YAML você pode reutilizar no n8n para ter uma tabela de endpoints centralizada.
### 6.3. Gateway FastAPI – roteador OpenAI‑style
`~/llm-stack/gateway/gateway_server.py`:

```python
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
```

> Mantive `/v1/chat/completions` centralizado para texto/código (onde a compatibilidade OpenAI importa mais para n8n/AnythingLLM/etc.). Para visão, recomendo chamar o endpoint da própria visão diretamente (até porque cada um tem formato diferente). [github](http://github.com/ggml-org/llama.cpp)

Rodar gateway:

```bash
cd ~/llm-stack/gateway
source venv/bin/activate
tmux new-session -d -s router \
  "uvicorn gateway_server:app --host 0.0.0.0 --port 9000"
```

***
## 🔗 8. Exemplo de consumo via HTTP (curl)
### 8.1. Chat com Qwen‑Instruct (via gateway)
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [
      {"role": "system", "content": "Você é um assistente técnico conciso."},
      {"role": "user", "content": "Explique brevemente o que é uma syscall."}
    ],
    "max_tokens": 128,
    "temperature": 0.4
  }' | jq
```
### 8.2. Chat com Gemma-2-2b-abliterated
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2-2b-abliterated",
    "messages": [
      {"role": "user", "content": "Vamos conversar sobre design de sistemas distribuídos."}
    ],
    "max_tokens": 256
  }' | jq
```
### 8.3. Auxiliar de código (Qwen‑Coder) no mesmo endpoint
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b",
    "messages": [
      {"role": "system", "content": "Você é um assistente de código. Responda com código e breve explicação."},
      {"role": "user", "content": "Escreva uma função em C que inverta uma string in-place."}
    ],
    "max_tokens": 256
  }' | jq
```

***
## ⚙️ 9. Otimização prática (`-t`, `-b`, `-ngl`, memória)
**`-t` (threads)**  
- Comece com `8`. Se ver que a CPU não chega a 100% em todos os núcleos, pode subir para `10`.  
- Muito alto → overhead de agendamento sem ganho real.

**`-b` (batch size)**  
- 256 é seguro; tentar `-b 512` pode aumentar tok/s, mas consome mais RAM.  
- Observe com `htop` / `time`.  

**`-ngl` (GPU layers)**  
- 0 (CPU only) é o padrão aqui; o foco total é na performance via instruções vetoriais da CPU no WSL2.

**Memória por modelo** (aprox):

- Qwen2.5‑1.5B‑Instruct‑Q4_K_M → ~1.1 GB + KV‑cache (até ~2–3 GB com contexto grande). [huggingface](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- Qwen2.5‑Coder‑1.5B‑Q4_K_M → ~1.0 GB + cache. [huggingface](https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF)

Com 16 GB dá para rodar **3–4 modelos 1.5–3B** em Q4 simultâneos + sistema + n8n, desde que não exagere em contextos gigantes em todos ao mesmo tempo. [skywork](https://skywork.ai/blog/models/qwen2-5-1-5b-instruct-gguf-free-chat-online-skywork-ai/)

Para medir performance real:

```bash
# uma geração de teste com estatísticas
./llama.cpp/build/bin/llama-cli \
  -m ./models/text/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -p "Teste de throughput." -n 256 -t 8 -c 4096 -b 256 -ngl 0 \
  --log-disable
```

Observe tempo total / tokens gerados.


***
## 📁 10. Organização Final do Projeto
No WSL2, a estrutura consolidada é:

```text
~/llm-stack
  ├── llama.cpp/               # Binários compilados (llama-server, llama-cli)
  ├── models/
  │   ├── text/                # .gguf de texto (Qwen, Gemma2)
  │   ├── code/                # .gguf de código (Qwen Coder, Ministral, Ministral + mmproj)
  │   └── vision/              # .gguf de visão (Qwen-VL + mmproj)
  ├── vision/                  # venv + qwen_vl_server.py
  ├── gateway/                 # venv + gateway_server.py + models.yaml
  ├── scripts/                 
  │   ├── manage.sh            # Script principal de controle (Start/Stop/Status)
  │   └── chat.sh              # Script para chat interativo via CLI
  ├── external_config.yaml     # Configurações para n8n / ferramentas externas
  └── logs/                    # Arquivos de log (opcional)
```

***

## 🔄 11. Automação e Uso Diário
Não é necessário rodar comandos manuais do `llama.cpp` ou `tmux`. Use os scripts na pasta `scripts/`.

### 11.1. Gerenciador de Serviços (`manage.sh`)
O `manage.sh` é o "controle remoto" do seu stack.

**Comandos:**
- `./scripts/manage.sh start [alvo]`   → Inicia um modelo/serviço no background.
- `./scripts/manage.sh stop [alvo]`    → Para um serviço específico.
- `./scripts/manage.sh stop-all`       → Desliga tudo e limpa a memória RAM.
- `./scripts/manage.sh status`         → Lista quais sessões estão ativas.

### 11.2. Chat Interativo via Terminal (`chat.sh`)
Se você quiser conversar com um modelo diretamente pelo terminal (sem passar pela rede/gateway), use o `chat.sh`.

**Uso:**
```bash
./scripts/chat.sh [modelo]
```
*Nota: O modelo escolhido deve estar PARADO no `manage.sh` para evitar conflito de memória.*

---

## ⚠️ 12. Problemas comuns e como resolver
### 12.1. Erros de compilação do `llama.cpp`
- Falta de `cmake` ou `build-essential` → instale com `apt` como acima. [github](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- Versão antiga de `cmake` → se Ubuntu for muito velho, instale `cmake` mais novo via `snap` ou script oficial (no seu caso 22.04 está ok).
### 12.2. “out of memory” ao carregar modelo
- Acontece se tentar subir modelos demais ou com contexto muito grande.  
- Ajuste:
  - Reduza `-c 4096` para `2048` no modelo menos crítico.  
  - Feche alguns servers (por ex. mantenha só 2 ativos).  
  - Garanta que não haja outros processos pesados no Windows competindo por RAM com o WSL2.
### 12.3. Porta ocupada
- Mensagem: `bind: Address already in use`.  
- Use `ss -tulpen | grep 8001` para ver quem está na porta; mate o processo ou mude a porta no comando.
### 12.4. Modelo não carregando (arquivo não encontrado)
- Cheque path completo:
  - `ls ~/llm-stack/models/text`  
  - Compare com caminho em `-m`.  
- Se baixar de novo, confira se `wget` não salvou com nome diferente (`?download=1` etc.).
### 12.5. Gateway retornando 400 “Unknown model”
- O campo `"model"` no JSON deve bater com a chave em `models.yaml` (`qwen2.5-1.5b-instruct`, `qwen2.5-coder-1.5b`, etc.).
### 12.6. Qwen‑VL lento
- VLMs são mais pesados que LLMs puros; use para tarefas pontuais (captioning, não chat longo). [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF)
