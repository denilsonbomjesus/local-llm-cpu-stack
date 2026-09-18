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
  ├── llama.cpp/               # Código-fonte e binários compilados (build/CPU/)
  ├── models/                  # Arquivos .gguf organizados por categoria
  │   ├── text/                # Gemma 4 (E2B/E4B), Dolphin3.0
  │   ├── code/                # Qwen2.5-Coder, LFM 2.5, Nanbeige4.2, MiniCPM5-1B
  │   ├── vision/              # Qwen2.5-VL (Modelo + mmproj)
  │   └── bonsai/              # Bonsai 27B 1-bit + Ternary Bonsai 27B
  ├── gateway/                 # Roteador FastAPI (Porta 9000)
  ├── scripts/                 # Scripts de controle (manage.sh, chat.sh, webui.sh)
  ├── external/                # Presets (catálogo de modelos da WebUI)
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

Isso cobre compilação do `llama.cpp` (GGML nativo CPU), Python para o gateway e `tmux` para rodar em background. [github](http://github.com/ggml-org/llama.cpp)

Se quiser Node.js (opcional, não vou usar aqui):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

***
## 🧠 2. Clonar e compilar `llama.cpp` (GGML nativo CPU)
Vou usar o repositório oficial `ggml-org/llama.cpp`. [github](http://github.com/ggml-org/llama.cpp)
### 2.1. Clone
```bash
cd ~/llm-stack
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```
### 2.2. Build (Release, GGML nativo CPU)
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"
```

Isso gera binários em `~/llm-stack/llama.cpp/build/` (`llama-cli`, `llama-server`, etc.). [github](http://github.com/ggml-org/llama.cpp)

> Nota: não vamos ativar CUDA/OpenVINO aqui para manter a simplicidade e foco em CPU. O backend OpenVINO foi removido deste stack: ele produzia saída corrompida / erros de shape em modelos Qwen, e nenhum modelo atual o utiliza — todos rodam no build GGML nativo CPU. Se um dia quiser testar OpenVINO de novo, recompile com `-DGGML_OPENVINO=ON` em um diretório separado.

***
## 📥 3. Download dos modelos (WSL2)
Vou padronizar todos em `~/llm-stack/models`.

```bash
mkdir -p ~/llm-stack/models/{text,code,vision}
cd ~/llm-stack/models
```
### 3.1. Texto / agentes

#### Dolphin3.0-Llama3.2-3B (Q4_K_M)

O **Dolphin3.0** é um modelo da Cognitive Computations baseado no Llama 3.2-3B, conhecido por ser uncensored e versátil. Ótimo para tarefas gerais, chatbot e código simples.

Repo: `dphn/Dolphin3.0-Llama3.2-3B` (safetensors original). GGUF convertido por `bartowski/Dolphin3.0-Llama3.2-3B-GGUF`. [huggingface](https://huggingface.co/bartowski/Dolphin3.0-Llama3.2-3B-GGUF)

- **Arquivo:** `Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf` (1.9 GB)
- **Qualidade:** Excelente para chat geral, código simples, tarefas do dia a dia
- **RAM (4K ctx):** ~2.9 GB ✅ cabe em 8 GB WSL2
- **Velocidade CPU (i5):** ~20-30 tok/s ⚡ Muito rápido
- **Sem thinking mode** (Llama 3.2 base, sem tokens especiais de raciocínio)

```bash
cd ~/llm-stack/models/text

wget -O Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Dolphin3.0-Llama3.2-3B-GGUF/resolve/main/Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf
```


### 3.2. Código
#### Qwen2.5‑Coder‑3B‑Instruct‑Q4_K_M.gguf

Repo oficial GGUF: `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF`. [huggingface](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)

Arquivo: `qwen2.5-coder-3b-instruct-q4_k_m.gguf` (2.0 GB, Q4_K_M, ~3B params, recomendado). [huggingface](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)

```bash
cd ~/llm-stack/models/code
wget -O qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf
```


#### Liquid LFM 2.5-1.2B-Instruct (Q8_0)

O **LFM 2.5 (Liquid Foundation Model 2.5)** é um modelo compacto de 1.2B parâmetros da LiquidAI, otimizado para código e raciocínio com qualidade excepcional graças à quantização Q8_0 (8-bit, quase sem perda).

Repo: `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`. [huggingface](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)

- **Arquivo:** `LFM2.5-1.2B-Instruct-Q8_0.gguf` (1.2 GB)
- **Qualidade:** Excelente (Q8_0 = quase perda zero, ~99% do FP16)
- **RAM (4K ctx):** ~2.2 GB ✅ cabe com folga em 8 GB WSL2
- **Velocidade CPU (i5):** ~25-40 tok/s ⚡ Muito rápido
- **Sem thinking mode** (modelo base, sem tokens especiais de raciocínio)

```bash
cd ~/llm-stack/models/code

wget -O LFM2.5-1.2B-Instruct-Q8_0.gguf \
  https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q8_0.gguf
```

#### Nanbeige4.2-3B (Q4_K_M)

O **Nanbeige4.2-3B** é um modelo da OWAO baseado no Nanbeige, com 3B parâmetros. Otimizado para código, raciocínio e tarefas gerais com qualidade excelente em Q4_K_M.

Repo: `owao/Nanbeige4.2-3B-GGUF`. [huggingface](https://huggingface.co/owao/Nanbeige4.2-3B-GGUF)

- **Arquivo:** `Nanbeige4.2-3B-Q4_K_M.gguf` (2.4 GB)
- **Qualidade:** Muito boa (Q4_K_M = balanço qualidade/RAM)
- **RAM (4K ctx):** ~3.4 GB ✅ cabe em 8 GB WSL2
- **Velocidade CPU (i5):** ~18-30 tok/s ⚡ Rápido
- **Com thinking mode** (tags `<think>`/`</think>` ativadas por padrão)
- Use `nanbeige-3b-nothink` para desligar o thinking e obter respostas mais diretas

```bash
cd ~/llm-stack/models/code

wget -O Nanbeige4.2-3B-Q4_K_M.gguf \
  https://huggingface.co/owao/Nanbeige4.2-3B-GGUF/resolve/main/Nanbeige4.2-3B-Q4_K_M.gguf
```

#### MiniCPM5-1B-Agentic-Tooluse (Q8_0)

O **MiniCPM5-1B-Agentic-Tooluse** é um modelo compacto de 1B parâmetros da OpenBMB, fine-tuned com Nemotron-DPO para ferramentas XML e fluxos agentivos. Baseado em LlamaForCausalLM, nativo 128K de contexto.

Repo: `ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-GGUF`. [huggingface](https://huggingface.co/ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-GGUF)

- **Arquivo:** `MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf` (1.1 GB)
- **Qualidade:** Excelente (Q8_0 = quase perda zero, ~99% do FP16)
- **RAM (4K ctx):** ~2.1 GB ✅ cabe com folga em 8 GB WSL2
- **Velocidade CPU (i5):** ~30-50 tok/s ⚡ Muito rápido
- **Com thinking mode** (tags `<think>`/`</think>` via `enable_thinking`)
- Use `minicpm5-1b-nothink` para desligar o thinking e obter respostas mais diretas
- **128K contexto nativo** (recomendado manter 4K-8K para CPU)

```bash
cd ~/llm-stack/models/code

wget -O MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf \
  https://huggingface.co/ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-GGUF/resolve/main/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf
```

### 3.3. Visão

#### Qwen2.5-VL-3B-Uncensored (Q4_K_M)

O **Qwen2.5-VL-3B-Uncensored** (também chamado de Abliterated) é uma versão do Qwen2.5-VL-3B-Instruct com as recusas de conteúdo removidas (uncensored). Usa `llama.cpp` multimodal nativo com suporte a imagens via mmproj.

Repo: `mradermacher/Qwen2.5-VL-3B-Instruct-abliterated-GGUF`. [huggingface](https://huggingface.co/mradermacher/Qwen2.5-VL-3B-Instruct-abliterated-GGUF)

- **Modelo:** `qwen2.5-vl-3b-uncensored-Q4_K_M.gguf` (1.8 GB) — Q4_K_M, qualidade/recomendado
- **mmproj:** `qwen2.5-vl-3b-uncensored-mmproj-Q8_0.gguf` (1.3 GB Q8_0, visão)
- **RAM (4K ctx):** ~3.1 GB total (1.8 GB modelo + 1.3 GB mmproj) ✅ cabe em 8 GB WSL2
- **Velocidade CPU (i5):** ~15-25 tok/s
- **Short name (chat.sh/manage.sh):** `qwen-vl-3b-uncensored`
- **Porta:** 8010

> **Sobre mmproj Q8_0:** O repositório `mradermacher` não inclui arquivos mmproj. Usamos o mmproj Q8_0 (f16 convertido) do repo `lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF`, que é totalmente compatível por ser a mesma arquitetura. O mmproj (projeção do encoder de visão) só existe em formatos f16/Q8_0 — não há versão Q4_K_M.

```bash
cd ~/llm-stack/models/vision

# Modelo principal (Q4_K_M)
wget -O qwen2.5-vl-3b-uncensored-Q4_K_M.gguf \
  https://huggingface.co/mradermacher/Qwen2.5-VL-3B-Instruct-abliterated-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-abliterated.Q4_K_M.gguf

# mmproj (compatível, Q8_0)
wget -O qwen2.5-vl-3b-uncensored-mmproj-Q8_0.gguf \
  https://huggingface.co/lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf
```

### 3.5. Gemma 4 (E2B & E4B)

Os modelos **Gemma 4** da Google representam um novo patamar de eficiência com a arquitetura PLE (Pyramid-Like Expansion). São modelos multimodais (texto + visão) com suporte nativo a GGUF.

> **Pré-requisito:** llama.cpp versão b10173+ (mainline recente). Ambos usam `qat-q4_0`.

#### Gemma 4 E2B (2.3B efetivos)

Repo: `google/gemma-4-E2B-it-qat-q4_0-gguf`. [huggingface](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)

- **Arquivo:** `gemma-4-E2B_q4_0-it.gguf` (3.2 GB)
- **Qualidade:** Excelente para conversação, resumos e classificação
- **RAM (4K ctx, texto puro):** ~4.2 GB ✅ cabe em 8 GB WSL2
- **RAM (4K ctx, com mmproj):** ~5.2 GB ✅
- **Velocidade CPU (i5):** ~22-35 tok/s ⚡ Muito rápido
- **mmproj:** `gemma-4-E2B-it-mmproj.gguf` (942 MB, para visão)

```bash
cd ~/llm-stack/models/text

# Modelo principal
wget -O gemma-4-E2B_q4_0-it.gguf \
  https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf

# mmproj (opcional — só se for usar visão)
wget -O gemma-4-E2B-it-mmproj.gguf \
  https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B-it-mmproj.gguf
```

#### Gemma 4 E4B (4.5B efetivos)

Repo: `google/gemma-4-E4B-it-qat-q4_0-gguf`. [huggingface](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)

- **Arquivo:** `gemma-4-E4B_q4_0-it.gguf` (4.9 GB)
- **Qualidade:** Superior em lógica, programação e raciocínio
- **RAM (4K ctx, texto puro):** ~5.9 GB ⚠️ apertado em 8 GB WSL2
- **RAM (4K ctx, com mmproj):** ~6.9 GB ⚠️ recomendado 10 GB WSL2
- **Velocidade CPU (i5):** ~10-15 tok/s
- **mmproj:** `gemma-4-E4B-it-mmproj.gguf` (946 MB, para visão)

```bash
cd ~/llm-stack/models/text

# Modelo principal
wget -O gemma-4-E4B_q4_0-it.gguf \
  https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf

# mmproj (opcional)
wget -O gemma-4-E4B-it-mmproj.gguf \
  https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B-it-mmproj.gguf
```

> **Dica de RAM:** Use as variantes **sem mmproj** (gemma-4-e2b, gemma-4-e4b) para economizar ~1 GB quando não for usar imagens. As variantes **-vision** (gemma-4-e2b-vision, gemma-4-e4b-vision) carregam o mmproj para suporte multimodal.

> **Sobre thinking mode:** Gemma 4 tem pensamento interno via `<|think|>` tokens. Use as variantes `-nothink` para desligar e obter respostas mais diretas.

***
## 🧠 2b. O que é Q4_K_M / IQ4_XS?
- **Q4_K_M**: quantização 4‑bit em “super‑blocos” com estatísticas por bloco, 4.5 bits por peso, trade‑off ótimo de qualidade vs RAM – a opção normalmente recomendada para uso geral. [huggingface](https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF)- **IQ4_XS**: variante “imatrix” super‑compacta 4‑bit, com compressão adicional e pequena perda de qualidade, muito usada em modelos pequenos para reduzir footprint.

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
- `qwen-coder-3b`       → Porta 8003 (Qwen2.5-Coder 3B Q4_K_M ~12-20 tok/s)
- `qwen-vl-3b-uncensored` → Porta 8010 (Qwen2.5-VL-3B abliterado Q4_K_M ~15-25 tok/s)
- `bonsai-27b`         → Porta 8011 (Bonsai 27B 1-bit ~4-8 tok/s)
- `bonsai-27b-nothink` → Porta 8013 (Bonsai 27B, thinking OFF)
- `ternary-bonsai-27b` → Porta 8012 (Ternary Bonsai 27B ~2-5 tok/s)
- `ternary-bonsai-27b-nothink` → Porta 8014 (Ternary Bonsai 27B, thinking OFF)
- `gemma-4-e2b`        → Porta 8021 (Gemma 4 E2B 2.3B ~22-35 tok/s, texto puro)
- `gemma-4-e2b-nothink`→ Porta 8023 (Gemma 4 E2B, thinking OFF)
- `gemma-4-e2b-vision` → Porta 8031 (Gemma 4 E2B + visão)
- `gemma-4-e2b-vision-nothink`→ Porta 8033 (Gemma 4 E2B + visão, thinking OFF)
- `gemma-4-e4b`        → Porta 8022 (Gemma 4 E4B 4.5B ~10-15 tok/s, texto puro)
- `gemma-4-e4b-nothink`→ Porta 8024 (Gemma 4 E4B, thinking OFF)
- `gemma-4-e4b-vision` → Porta 8032 (Gemma 4 E4B + visão)
- `gemma-4-e4b-vision-nothink`→ Porta 8034 (Gemma 4 E4B + visão, thinking OFF)
- `dolphin3-3b`        → Porta 8041 (Dolphin3.0 Llama3.2-3B Q4_K_M ~20-30 tok/s)
- `lfm2.5-1.2b`        → Porta 8061 (Liquid LFM 2.5-1.2B Q8_0 ~25-40 tok/s)
- `nanbeige-3b`        → Porta 8071 (Nanbeige4.2-3B Q4_K_M ~18-30 tok/s)
- `nanbeige-3b-nothink`→ Porta 8073 (Nanbeige4.2-3B, thinking OFF)
- `minicpm5-1b`        → Porta 8081 (MiniCPM5-1B Nemotron-DPO Q8_0 ~30-50 tok/s)
- `minicpm5-1b-nothink`→ Porta 8083 (MiniCPM5-1B, thinking OFF)
- `gateway`            → Porta 9000 (O Roteador Central)

### 4.2. Comandos úteis do Manager
- `./scripts/manage.sh status`      → Vê o que está rodando.
- `./scripts/manage.sh stop-all`    → Mata todos os serviços e libera a RAM.


***
## 🌐 5. API HTTP OpenAI‑like do `llama-server`
O `llama-server` expõe endpoints OpenAI‑compatíveis, como `/v1/chat/completions`. [github](http://github.com/ggml-org/llama.cpp)

Teste rápido (no WSL2):

```bash
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-coder-3b",
    "messages": [
      {"role": "user", "content": "Resuma em 3 pontos a função de um sistema operacional."}
    ],
    "max_tokens": 128
  }'
```

Isso já deve retornar JSON no formato OpenAI. [learn.arm](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/llama-server/)

> O campo `"model"` aqui é ignorado pelo `llama-server` (ele já está “fixo” no binário), mas vamos usar esse campo no *gateway* para roteamento.  

***
## 🧠 10. Visão – modelos
### 10.1. Visão via llama.cpp (GGUF multimodal nativo)
O `llama.cpp` suporta modelos multimodais nativamente via GGUF (modelo + mmproj). Todos os modelos de visão do stack usam a **mesma rota** `/v1/chat/completions` com formato OpenAI de imagem (`image_url`). O servidor Python legado (`vision/`) foi removido — visão consolidada no `llama-server`/gateway.

**Modelos de visão disponíveis:**

| Short name | Porta | Descrição |
|---|---|---|
| `qwen-vl-3b-uncensored` | 8010 | Qwen2.5-VL 3B abliterado (Q4_K_M + mmproj) |
| `gemma-4-e2b-vision` | 8031 | Gemma 4 E2B + visão |
| `gemma-4-e2b-vision-nothink` | 8033 | Gemma 4 E2B + visão (thinking OFF) |
| `gemma-4-e4b-vision` | 8032 | Gemma 4 E4B + visão |
| `gemma-4-e4b-vision-nothink` | 8034 | Gemma 4 E4B + visão (thinking OFF) |

**Iniciar um modelo de visão (via manage.sh):**
```bash
./scripts/manage.sh start qwen-vl-3b-uncensored   # ou gemma-4-e2b-vision, gemma-4-e4b-vision...
```

**Chat interativo (via chat.sh):**
```bash
./scripts/chat.sh qwen-vl-3b-uncensored
# Depois use /image caminho/da/imagem.jpg dentro do chat
```

**Direto na porta do modelo:**
```bash
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-3b-uncensored",
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}},
        {"type": "text", "text": "Descreva esta imagem em detalhes."}
      ]}
    ],
    "max_tokens": 256
  }'
```

**Via gateway (qualquer modelo de visão, porta 9000):**
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e2b-vision",   # troque por qwen-vl-3b-uncensored, gemma-4-e4b-vision...
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}},
        {"type": "text", "text": "Descreva esta imagem em detalhes."}
      ]}
    ],
    "max_tokens": 256
  }'
```

> O gateway encaminha para a porta certa com base no `"model"` (todos os modelos de visão estão registrados em `gateway/models.yaml` e `external_config.yaml`).

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
  qwen-coder-3b:
    type: code
    endpoint: http://localhost:8003

  qwen-vl-3b-uncensored:
    type: vision
    endpoint: http://localhost:8010

  bonsai-27b:
    type: text
    endpoint: http://localhost:8011

  bonsai-27b-nothink:
    type: text
    endpoint: http://localhost:8013

  ternary-bonsai-27b:
    type: text
    endpoint: http://localhost:8012

  ternary-bonsai-27b-nothink:
    type: text
    endpoint: http://localhost:8014

  gemma-4-e2b:
    type: text
    endpoint: http://localhost:8021

  gemma-4-e2b-nothink:
    type: text
    endpoint: http://localhost:8023

  gemma-4-e2b-vision:
    type: vision
    endpoint: http://localhost:8031

  gemma-4-e2b-vision-nothink:
    type: vision
    endpoint: http://localhost:8033

  gemma-4-e4b:
    type: text
    endpoint: http://localhost:8022

  gemma-4-e4b-nothink:
    type: text
    endpoint: http://localhost:8024

  gemma-4-e4b-vision:
    type: vision
    endpoint: http://localhost:8032

  gemma-4-e4b-vision-nothink:
    type: vision
    endpoint: http://localhost:8034

  dolphin3-3b:
    type: text
    endpoint: http://localhost:8041

  lfm2.5-1.2b:
    type: code
    endpoint: http://localhost:8061

  nanbeige-3b:
    type: code
    endpoint: http://localhost:8071

  nanbeige-3b-nothink:
    type: code
    endpoint: http://localhost:8073

  minicpm5-1b:
    type: code
    endpoint: http://localhost:8081

  minicpm5-1b-nothink:
    type: code
    endpoint: http://localhost:8083
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

# NOTA SOBRE VISÃO:
# Todos os modelos de visão (qwen-vl-3b-uncensored, gemma-4-e2b-vision, etc.)
# rodam via llama-server multimodal e usam a MESMA rota /v1/chat/completions,
# enviando a imagem no formato OpenAI (image_url com file:// ou data URI).
# Não existe mais rota /v1/images/captions: o llama-server não a implementa.
# (A antiga rota /v1/images/captions foi removida junto com o servidor Python legado vision/.)
```

> Todos os modelos — texto, código E visão — passam por `/v1/chat/completions` (compatível OpenAI). Os modelos de visão recebem a imagem via `image_url` no conteúdo da mensagem. A rota legada `/v1/images/captions` e o servidor Python `vision/` foram removidos.

Rodar gateway:

```bash
cd ~/llm-stack/gateway
source venv/bin/activate
tmux new-session -d -s router \
  "uvicorn gateway_server:app --host 0.0.0.0 --port 9000"
```

***
## 🔗 8. Exemplo de consumo via HTTP (curl)
### 8.1. Chat com Qwen‑Coder (via gateway)
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-coder-3b",
    "messages": [
      {"role": "system", "content": "Você é um assistente técnico conciso."},
      {"role": "user", "content": "Explique brevemente o que é uma syscall."}
    ],
    "max_tokens": 128,
    "temperature": 0.4
  }' | jq
```
### 8.2. Chat com Nanbeige4.2-3B
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanbeige-3b",
    "messages": [
      {"role": "user", "content": "Vamos conversar sobre design de sistemas distribuídos."}
    ],
    "max_tokens": 256
  }' | jq
```
### 8.3. Chat com Bonsai 27B (via gateway)
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bonsai-27b",
    "messages": [
      {"role": "system", "content": "Você é um assistente técnico conciso."},
      {"role": "user", "content": "Explique brevemente o que é uma syscall."}
    ],
    "max_tokens": 128,
    "temperature": 0.4
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

- **Qwen2.5‑Coder‑3B‑Q4_K_M** → ~2.0 GB + ~1 GB overhead + KV cache (~3.0 GB total em 4K ctx)
- **Bonsai 27B 1-bit** → ~3.9 GB + ~1.3 GB overhead + KV cache (~5.2 GB total em 4K ctx)
- **Ternary Bonsai 27B** → ~7.2 GB + ~1.2 GB overhead + KV cache (~8.4 GB total em 4K ctx)
- **Gemma 4 E2B (texto)** → ~3.2 GB + ~1 GB overhead + KV cache (~4.2 GB total em 4K ctx)
- **Gemma 4 E2B (visão)** → ~4.2 GB + ~1 GB overhead + KV cache (~5.2 GB total em 4K ctx)
- **Gemma 4 E4B (texto)** → ~4.9 GB + ~1 GB overhead + KV cache (~5.9 GB total em 4K ctx)
- **Gemma 4 E4B (visão)** → ~4.9 GB + ~1 GB overhead + KV cache (~6.9 GB total em 4K ctx)
- **LFM 2.5 1.2B Q8_0** → ~1.2 GB + ~1 GB overhead + KV cache (~2.2 GB total em 4K ctx)
- **Nanbeige4.2-3B Q4_K_M** → ~2.4 GB + ~1 GB overhead + KV cache (~3.4 GB total em 4K ctx)
- **Qwen2.5-VL-Uncensored Q4_K_M + mmproj** → ~1.8 GB + 1.3 GB + KV cache (~3.1 GB total em 4K ctx)
- **MiniCPM5-1B Q8_0** → ~1.1 GB + ~1 GB overhead + KV cache (~2.1 GB total em 4K ctx)

Com 16 GB dá para rodar **3–4 modelos 1.5–3B** em Q4 simultâneos + sistema + n8n, desde que não exagere em contextos gigantes em todos ao mesmo tempo. [skywork](https://skywork.ai/blog/models/qwen2-5-1-5b-instruct-gguf-free-chat-online-skywork-ai/)

Para medir performance real:

```bash
# uma geração de teste com estatísticas
./llama.cpp/build/CPU/bin/llama-cli \
  -m ./models/code/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  -p "Teste de throughput." -n 256 -t 8 -c 4096 -b 256 -ngl 0 \
  --log-disable
```

Observe tempo total / tokens gerados.


***
## 📁 10. Organização Final do Projeto
No WSL2, a estrutura consolidada é:

```text
~/llm-stack
  ├── llama.cpp/               # Binários compilados (llama-server, llama-cli em build/CPU/bin/)
  ├── models/
  │   ├── text/                # .gguf de texto (Gemma 4 E2B/E4B, Dolphin3.0)
  │   ├── code/                # .gguf de código (Qwen Coder, LFM 2.5, Nanbeige4.2, MiniCPM5)
  │   ├── vision/              # .gguf de visão (Qwen-VL + mmproj)
  │   └── bonsai/              # Bonsai 27B 1-bit + Ternary Bonsai 27B
  ├── gateway/                 # venv + gateway_server.py + models.yaml
  ├── scripts/                 
  │   ├── manage.sh            # Script principal de controle (Start/Stop/Status)
  │   ├── chat.sh              # Script para chat interativo via CLI
  │   └── webui.sh             # Interface gráfica de chat (WebUI, porta 8080)
  ├── external/                # Presets (webui-models.ini — catálogo da WebUI)
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

### 11.3. Interface Gráfica de Chat (WebUI do llama.cpp — `webui.sh`)
Além do terminal (`chat.sh`), o stack tem a **interface gráfica de chat oficial do `llama.cpp`**, embutida no próprio `llama-server` (estilo ChatGPT). Ela roda em **modo router**: um único processo serve **todos os modelos do catálogo** de uma vez, carregando cada um na RAM **sob demanda** — você só gasta memória do modelo que estiver usando no momento.

**Como abrir:**
```bash
./scripts/webui.sh start
```

Depois abra no navegador (funciona no Windows ou dentro do WSL):
```
http://localhost:8080
```
> O WSL2 encaminha `localhost` automaticamente para o Windows. Tudo é 100% local — nenhum dado sai da sua máquina.

**Comandos (`webui.sh`):**
- `./scripts/webui.sh start`   → Sobe a interface na porta 8080.
- `./scripts/webui.sh stop`    → Para a interface e libera a RAM.
- `./scripts/webui.sh restart` → Reinicia (necessário após editar o catálogo).
- `./scripts/webui.sh status`  → Mostra se está ativa e lista os modelos do catálogo.
- `./scripts/webui.sh logs`    → Log do servidor em tempo real (Ctrl+C para sair).

**O que a interface tem:**
- 💬 Chat com **histórico de conversas** salvo no navegador (sidebar com múltiplas conversas)
- 🔄 **Seletor de modelo** com os 20 short names do catálogo `external/webui-models.ini` (mesmos nomes do manage.sh/chat.sh)
- 🖼️ **Upload de imagens** nos modelos de visão (Gemma 4 `-vision`, Qwen-VL) — basta anexar no chat
- 📎 Anexos de texto/CSV como contexto; renderização de Markdown e blocos de código com destaque
- 🎛️ Controles de geração: temperature, top_p, max_tokens e system prompt por conversa
- 🧠 Controle de raciocínio (thinking) nos modelos que suportam
- 📐 Respostas estruturadas: JSON Schema e grammar (GBNF) direto pela UI
- 🌗 Tema claro/escuro e atalhos de teclado

**Como funciona (modo router):**
- 1 sessão tmux única (`webui`) e 1 porta (8080) — diferente do `manage.sh`, que usa 1 processo/porta por modelo (80xx).
- O catálogo `external/webui-models.ini` define cada modelo: GGUF, variações `-nothink` (`reasoning = off`), `-vision` (`mmproj`), além de `threads = 8`, `batch = 256` e contexto por modelo (mesmos valores do manage.sh).
- `--models-max 2`: no máximo 2 modelos carregados na RAM ao mesmo tempo; para abrir outro, descarregue um (pela própria UI ou via `POST /models/unload`).
- Teste rápido via API:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dolphin3-3b","messages":[{"role":"user","content":"olá"}]}'
```

**RAM:** a memória é compartilhada com os servidores do `manage.sh`. Antes de usar os Bonsai 27B (~5-8 GB), pare os outros servidores (`./scripts/manage.sh stop-all`). Modelos pequenos (1-3B) convivem bem com a WebUI ativa.

**Problemas comuns da WebUI:**
- Página "não abre" ao testar com curl → a UI embutida é servida **comprimida (gzip)**; navegadores sempre enviam o header aceito. Teste com `curl -H "Accept-Encoding: gzip" http://localhost:8080/` — se vier `200 text/html`, a interface está OK e o problema é o cliente.
- Modelo novo não aparece → o router lê o catálogo no boot: edite `external/webui-models.ini` e faça `./scripts/webui.sh restart`.
- `out of memory` ao carregar um modelo → feche servidores do manage.sh ou escolha um modelo menor; se precisar, reduza o `ctx-size` da seção no INI.
- Modelo de visão falha ao carregar → confira se o mmproj existe no caminho declarado no INI (`ls models/vision/`).

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
- O campo `"model"` no JSON deve bater com a chave em `models.yaml` (`qwen-coder-3b`, `nanbeige-3b`, `bonsai-27b`, etc.).
### 12.6. Qwen‑VL lento
- VLMs são mais pesados que LLMs puros; use para tarefas pontuais (captioning, não chat longo).
