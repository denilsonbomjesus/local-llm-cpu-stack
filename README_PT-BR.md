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
  │   ├── text/                # Qwen2.5-1.5B, Gemma 2 2B, Gemma 4 (E2B/E4B), Dolphin3.0
  │   ├── code/                # Qwen2.5-Coder, Ministral-3-3B-Instruct-2512, LFM 2.5, Nanbeige4.2
  │   ├── vision/              # Qwen2.5-VL (Modelo + mmproj)
  │   └── bonsai/              # Bonsai 27B 1-bit + Ternary Bonsai 27B
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

# OpenCL (obrigatório para compilar com OpenVINO e usar a GPU Intel Iris Xe)
sudo apt install -y \
  ocl-icd-opencl-dev opencl-headers opencl-clhpp-headers intel-opencl-icd
```

Isso cobre compilação do `llama.cpp` (com ou sem OpenVINO), Python para gateway/vision, `tmux` para rodar em background. [github](http://github.com/ggml-org/llama.cpp)

Se quiser Node.js (opcional, não vou usar aqui):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

***
## 🧠 2. Clonar e compilar `llama.cpp` (CPU-only + OpenVINO opcional)
Vou usar o repositório oficial `ggml-org/llama.cpp`. [github](http://github.com/ggml-org/llama.cpp)
### 2.1. Clone
```bash
cd ~/llm-stack
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```
### 2.2. Build otimizado (Release, com suporte a K‑quants e HTTP, sem aceleração extra)
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"
```

Isso gera binários em `~/llm-stack/llama.cpp/build/` (`llama-cli`, `llama-server`, etc.). [github](http://github.com/ggml-org/llama.cpp)

> Nota: não vamos ativar CUDA/CL aqui para manter a simplicidade e foco em CPU; a performance nos modelos escolhidos é excelente mesmo sem aceleração de GPU dedicada.

### 2.3. Build com OpenVINO (para Intel CPU/GPU — RECOMENDADO para seu hardware)

Se você tem um processador Intel (especialmente 11ª/12ª/13ª geração ou Intel Core Ultra) com placa integrada Intel Iris Xe, esta opção oferece:
- **Prompt processing ~2-3x mais rápido** (primeira resposta chega mais rápido)
- **Token generation 10-30% mais rápido** (texto gerado por segundo)
- **Suporte a AVX-VNNI** (instruções vetoriais otimizadas da Intel)
- **Offloading para Iris Xe iGPU** via OpenCL

#### 2.3.1. Instalar OpenVINO Runtime
```bash
cd ~
# Download do OpenVINO 2026.2.1 para Ubuntu 24.04
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.2.1/linux/openvino_toolkit_ubuntu24_2026.2.1.21919.ede283a88e3_x86_64.tgz -O openvino.tgz

# Extrair para /opt/intel/
sudo mkdir -p /opt/intel/openvino_2026.2.1
sudo tar -xzf openvino.tgz -C /opt/intel/openvino_2026.2.1 --strip-components=1
sudo ln -sfn /opt/intel/openvino_2026.2.1 /opt/intel/openvino
rm openvino.tgz

# Instalar dependências de runtime do OpenVINO
sudo -E /opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh -y
```

#### 2.3.2. Compilar llama.cpp com OpenVINO
```bash
cd ~/llm-stack/llama.cpp

# Ativar ambiente OpenVINO
source /opt/intel/openvino/setupvars.sh

# Limpar build anterior (se existir)
rm -rf build

# Configurar com CMake (usando Ninja para compilação mais rápida)
cmake -B build/ReleaseOV -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENVINO=ON

# Compilar (12 threads)
cmake --build build/ReleaseOV --parallel
```

Isso gera binários em `~/llm-stack/llama.cpp/build/ReleaseOV/bin/`.

#### 2.3.3. Adicionar ao ~/.bashrc (automático)
Adicione estas linhas ao final do `~/.bashrc`:

```bash
# === OpenVINO ===
if [ -f /opt/intel/openvino/setupvars.sh ]; then
    source /opt/intel/openvino/setupvars.sh
fi
# Usar GPU (Iris Xe) por padrão. Fallback para CPU se GPU indisponível
export GGML_OPENVINO_DEVICE="${GGML_OPENVINO_DEVICE:-GPU}"
```

> ⚠️ **Importante sobre modelos Qwen**: Os modelos Qwen (Qwen2.5, Qwen-Coder, Qwen-VL) têm uma limitação conhecida no backend OpenVINO: a execução **stateful na GPU falha**. Os scripts `manage.sh` e `chat.sh` já incluem uma detecção automática que define `GGML_OPENVINO_STATEFUL_EXECUTION=0` para modelos Qwen e `=1` para os demais (Ministral, Gemma).

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

#### Lexi-Llama-3-8B-Uncensored (Q4_K_M)

O **Lexi-Llama-3-8B-Uncensored** é um modelo Llama 3 8B fine-tuned pela comunidade para ser uncensored e versátil. Ótimo para tarefas complexas, raciocínio e respostas sem restrições.

Repo: `Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF`. [huggingface](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF)

- **Arquivo:** `Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf` (4.6 GB)
- **Qualidade:** Excelente para tarefas complexas, raciocínio, coding, chat uncensored
- **RAM (4K ctx):** ~6.1 GB ⚠️ apertado em 8 GB WSL2
- **Velocidade CPU (i5):** ~8-15 tok/s
- **Sem thinking mode** (Llama 3 base, sem tokens especiais de raciocínio)

```bash
cd ~/llm-stack/models/text

wget -O Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf \
  https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF/resolve/main/Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf
```

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
#### Qwen2.5‑Coder‑3B‑Instruct‑Q4_K_M.gguf

Repo oficial GGUF: `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF`. [huggingface](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)

Arquivo: `qwen2.5-coder-3b-instruct-q4_k_m.gguf` (2.0 GB, Q4_K_M, ~3B params, recomendado). [huggingface](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)

```bash
cd ~/llm-stack/models/code
wget -O qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf
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
- Use `nanbeige-nothink` para desligar o thinking e obter respostas mais diretas

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
- Use `minicpm5-nothink` para desligar o thinking e obter respostas mais diretas
- **128K contexto nativo** (recomendado manter 4K-8K para CPU)

```bash
cd ~/llm-stack/models/code

wget -O MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf \
  https://huggingface.co/ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-GGUF/resolve/main/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf
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

> **Dica de RAM:** Use as variantes **sem mmproj** (gemma4-e2b, gemma4-e4b) para economizar ~1 GB quando não for usar imagens. As variantes **-vision** (gemma4-e2b-vision, gemma4-e4b-vision) carregam o mmproj para suporte multimodal.

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
-  `qwen-text`          → Porta 8001 (Qwen 2.5 1.5B)
- `gemma2`             → Porta 8002 (Gemma 2 2B)- `qwen-coder`         → Porta 8003 (Qwen Coder 3B Q4_K_M ~12-20 tok/s)
- `ministral-agent`    → Porta 8004 (Ministral 3 3B)
- `ministral-vision`   → Porta 8005 (Ministral 3 3B)
- `vision`             → Porta 8010 (Qwen-VL Python Server)
- `bonsai-27b`         → Porta 8011 (Bonsai 27B 1-bit ~4-8 tok/s)
- `bonsai-27b-nothink` → Porta 8013 (Bonsai 27B, thinking OFF)
- `ternary-bonsai`     → Porta 8012 (Ternary Bonsai 27B ~2-5 tok/s)
- `ternary-bonsai-nothink` → Porta 8014 (Ternary Bonsai, thinking OFF)
- `gemma4-e2b`        → Porta 8021 (Gemma 4 E2B 2.3B ~22-35 tok/s, texto puro)
- `gemma4-e2b-nothink`→ Porta 8023 (Gemma 4 E2B, thinking OFF)
- `gemma4-e2b-vision` → Porta 8031 (Gemma 4 E2B + visão)
- `gemma4-e2b-vision-nothink`→ Porta 8033 (Gemma 4 E2B + visão, thinking OFF)
- `gemma4-e4b`        → Porta 8022 (Gemma 4 E4B 4.5B ~10-15 tok/s, texto puro)
- `gemma4-e4b-nothink`→ Porta 8024 (Gemma 4 E4B, thinking OFF)
- `dolphin3`          → Porta 8041 (Dolphin3.0 Llama3.2-3B Q4_K_M ~20-30 tok/s)
- `lexi8b`            → Porta 8051 (Lexi-Llama-3-8B-Uncensored Q4_K_M ~8-15 tok/s)
- `lfm25`             → Porta 8061 (Liquid LFM 2.5 1.2B Q8_0 ~25-40 tok/s)
- `nanbeige`          → Porta 8071 (Nanbeige4.2-3B Q4_K_M ~18-30 tok/s)
- `nanbeige-nothink`   → Porta 8073 (Nanbeige4.2-3B, thinking OFF)
- `minicpm5`           → Porta 8081 (MiniCPM5-1B Nemotron-DPO Q8_0 ~30-50 tok/s)
- `minicpm5-nothink`    → Porta 8083 (MiniCPM5-1B, thinking OFF)
- `gemma4-e4b-vision` → Porta 8032 (Gemma 4 E4B + visão)
- `gemma4-e4b-vision-nothink`→ Porta 8034 (Gemma 4 E4B + visão, thinking OFF)
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

  qwen2.5-coder-3b:
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

  bonsai-27b-1bit:
    type: text
    endpoint: http://localhost:8011

  bonsai-27b-1bit-nothink:
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
    "model": "qwen2.5-coder-3b",
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

- Qwen2.5‑1.5B‑Instruct‑Q4_K_M → ~1.1 GB + KV‑cache (até ~2–3 GB com contexto grande). [huggingface](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)- **Qwen2.5‑Coder‑3B‑Q4_K_M** → ~2.0 GB + ~1 GB overhead + KV cache (~3.0 GB total em 4K ctx)
- **Bonsai 27B 1-bit** → ~3.9 GB + ~1.3 GB overhead + KV cache (~5.2 GB total em 4K ctx)
- **Ternary Bonsai 27B** → ~7.2 GB + ~1.2 GB overhead + KV cache (~8.4 GB total em 4K ctx)
- **Gemma 4 E2B (texto)** → ~3.2 GB + ~1 GB overhead + KV cache (~4.2 GB total em 4K ctx)
- **Gemma 4 E2B (visão)** → ~4.2 GB + ~1 GB overhead + KV cache (~5.2 GB total em 4K ctx)
- **Gemma 4 E4B (texto)** → ~4.9 GB + ~1 GB overhead + KV cache (~5.9 GB total em 4K ctx)
- **Gemma 4 E4B (visão)** → ~4.9 GB + ~1 GB overhead + KV cache (~6.9 GB total em 4K ctx)
- **LFM 2.5 1.2B Q8_0** → ~1.2 GB + ~1 GB overhead + KV cache (~2.2 GB total em 4K ctx)
- **Nanbeige4.2-3B Q4_K_M** → ~2.4 GB + ~1 GB overhead + KV cache (~3.4 GB total em 4K ctx)

Com 16 GB dá para rodar **3–4 modelos 1.5–3B** em Q4 simultâneos + sistema + n8n, desde que não exagere em contextos gigantes em todos ao mesmo tempo. [skywork](https://skywork.ai/blog/models/qwen2-5-1-5b-instruct-gguf-free-chat-online-skywork-ai/)

Para medir performance real:

```bash
# uma geração de teste com estatísticas
./llama.cpp/build/ReleaseOV/bin/llama-cli \
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
  ├── llama.cpp/               # Binários compilados (llama-server, llama-cli em build/ReleaseOV/bin/)os compilados (llama-server, llama-cli)
  ├── models/
  │   ├── text/                # .gguf de texto (Qwen, Gemma2, Gemma 4 E2B/E4B)
  │   ├── code/                # .gguf de código (Qwen Coder, Ministral, Ministral + mmproj, LFM 2.5, Nanbeige4.2)
  │   ├── vision/              # .gguf de visão (Qwen-VL + mmproj)
  │   └── bonsai/              # Bonsai 27B 1-bit + Ternary Bonsai 27B
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
- O campo `"model"` no JSON deve bater com a chave em `models.yaml` (`qwen2.5-1.5b-instruct`, `qwen2.5-coder-3b`, etc.).
### 12.6. Qwen‑VL lento
- VLMs são mais pesados que LLMs puros; use para tarefas pontuais (captioning, não chat longo). [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF)
