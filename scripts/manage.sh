#!/usr/bin/env bash

BASE=~/llm-stack

# ============================================================
# Build único: GGML nativo (CPU)
#   O build OpenVINO (build/ReleaseOV) foi removido do stack:
#   nenhum modelo atual o utilizava (todos caíam em fallback
#   para CPU por bugs de encoding/shape). Todos os modelos
#   rodam no build CPU principal.
# ============================================================
LLAMA="$BASE/llama.cpp/build/CPU/bin/llama-server"

# ============================================================
# Locale: Força UTF-8 para evitar que caracteres acentuados
# (português, etc.) apareçam como '?' no terminal.
# ============================================================
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Configurações de Hardware (Otimizadas para seu i5-1235U)
THREADS=8
CTX=4096
BATCH=256

# ============================================================
# helper: monta o comando do servidor para um modelo
# Padrão de nomes: <fabricante>-<modelo>-<tamanho>[-variante]
#   -nothink → desliga o modo de raciocínio (--reasoning off)
#   -vision  → carrega o mmproj (suporte a imagens)
# ============================================================
server_cmd_for() {
    local llama="$LLAMA"
    case "$1" in
        qwen-coder-3b)        echo "$llama -m $BASE/models/code/qwen2.5-coder-3b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8003" ;;
        qwen-vl-3b-uncensored) echo "$llama -m $BASE/models/vision/qwen2.5-vl-3b-uncensored-Q4_K_M.gguf --mmproj $BASE/models/vision/qwen2.5-vl-3b-uncensored-mmproj-Q8_0.gguf -t $THREADS -c $CTX --port 8010" ;;
        bonsai-27b)            echo "$llama -m $BASE/models/bonsai/Bonsai-27B-Q1_0.gguf -t $THREADS -c $CTX -b $BATCH --port 8011" ;;
        bonsai-27b-nothink)    echo "$llama -m $BASE/models/bonsai/Bonsai-27B-Q1_0.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8013" ;;
        ternary-bonsai-27b)    echo "$llama -m $BASE/models/bonsai/Ternary-Bonsai-27B-Q2_g64.gguf -t $THREADS -c $CTX -b $BATCH --port 8012" ;;
        ternary-bonsai-27b-nothink) echo "$llama -m $BASE/models/bonsai/Ternary-Bonsai-27B-Q2_g64.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8014" ;;
        gemma-4-e2b)           echo "$llama -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --port 8021" ;;
        gemma-4-e2b-nothink)   echo "$llama -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8023" ;;
        gemma-4-e4b)           echo "$llama -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --port 8022" ;;
        gemma-4-e4b-nothink)   echo "$llama -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8024" ;;
        gemma-4-e2b-vision)    echo "$llama -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E2B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --port 8031" ;;
        gemma-4-e2b-vision-nothink) echo "$llama -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E2B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8033" ;;
        gemma-4-e4b-vision)    echo "$llama -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E4B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --port 8032" ;;
        gemma-4-e4b-vision-nothink) echo "$llama -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E4B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8034" ;;
        dolphin3-3b)           echo "$llama -m $BASE/models/text/Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH --port 8041" ;;
        lfm2.5-1.2b)           echo "$llama -m $BASE/models/code/LFM2.5-1.2B-Instruct-Q8_0.gguf -t $THREADS -c $CTX -b $BATCH --port 8061" ;;
        nanbeige-3b)           echo "$llama -m $BASE/models/code/Nanbeige4.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH --port 8071" ;;
        nanbeige-3b-nothink)   echo "$llama -m $BASE/models/code/Nanbeige4.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8073" ;;
        minicpm5-1b)           echo "$llama -m $BASE/models/code/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf -t $THREADS -c $CTX -b $BATCH --port 8081" ;;
        minicpm5-1b-nothink)   echo "$llama -m $BASE/models/code/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8083" ;;
        *) echo "" ;;
    esac
}

# ============================================================
# start_model: inicia o servidor em sessão TMUX dedicada
# (não há mais fallback dinâmico: backend é sempre GGML CPU)
# ============================================================
start_model() {
    local model="$1"
    local session="$model"

    # --- Caso especial: gateway (não usa llama.cpp) ---
    if [ "$model" = "gateway" ]; then
        cd "$BASE/gateway"
        source venv/bin/activate 2>/dev/null || true
        tmux new-session -d -s gateway "uvicorn gateway_server:app --host 0.0.0.0 --port 9000"
        echo "🌐 Gateway (Router) iniciado na porta 9000"
        cd "$BASE"
        return
    fi

    # Monta o comando do servidor
    local cmd
    cmd=$(server_cmd_for "$model")
    if [ -z "$cmd" ]; then
        echo "Uso: ./manage.sh start {qwen-coder-3b|qwen-vl-3b-uncensored|bonsai-27b|bonsai-27b-nothink|ternary-bonsai-27b|ternary-bonsai-27b-nothink|gemma-4-e2b|gemma-4-e2b-nothink|gemma-4-e2b-vision|gemma-4-e2b-vision-nothink|gemma-4-e4b|gemma-4-e4b-nothink|gemma-4-e4b-vision|gemma-4-e4b-vision-nothink|dolphin3-3b|lfm2.5-1.2b|nanbeige-3b|nanbeige-3b-nothink|minicpm5-1b|minicpm5-1b-nothink|gateway}"
        return
    fi

    tmux new-session -d -s "$session" "$cmd"
    local port
    port=$(echo "$cmd" | sed 's/.*--port \([0-9]\+\).*/\1/')
    echo "✅ ${model} (GGML nativo CPU) iniciado na porta ${port}"
}

function stop_model() {
    tmux kill-session -t "$1" 2>/dev/null && echo "🛑 $1 parado." || echo "⚠️ $1 não estava rodando."
}

function status() {
    echo "--- Status dos Modelos (Sessões TMUX) ---"
    tmux ls 2>/dev/null | grep -E "qwen-coder-3b|qwen-vl-3b-uncensored|bonsai-27b|bonsai-27b-nothink|ternary-bonsai-27b|ternary-bonsai-27b-nothink|gemma-4-e2b|gemma-4-e2b-nothink|gemma-4-e2b-vision|gemma-4-e2b-vision-nothink|gemma-4-e4b|gemma-4-e4b-nothink|gemma-4-e4b-vision|gemma-4-e4b-vision-nothink|dolphin3-3b|lfm2.5-1.2b|nanbeige-3b|nanbeige-3b-nothink|minicpm5-1b|minicpm5-1b-nothink|gateway" || echo "Nenhum serviço rodando no momento."
}

# Lógica Principal do Script
case $1 in
    start)
        start_model "$2"
        ;;
    stop)
        stop_model "$2"
        ;;
    stop-all)
        echo "Finalizando todos os serviços..."
        for s in qwen-coder-3b qwen-vl-3b-uncensored bonsai-27b bonsai-27b-nothink ternary-bonsai-27b ternary-bonsai-27b-nothink gemma-4-e2b gemma-4-e2b-nothink gemma-4-e2b-vision gemma-4-e2b-vision-nothink gemma-4-e4b gemma-4-e4b-nothink gemma-4-e4b-vision gemma-4-e4b-vision-nothink dolphin3-3b lfm2.5-1.2b nanbeige-3b nanbeige-3b-nothink minicpm5-1b minicpm5-1b-nothink gateway; do
            stop_model "$s"
        done
        ;;
    status)
        status
        ;;
    *)
        echo "================================================================="
        echo "       🤖 LLM STACK MANAGER - GUIA DE COMANDOS 🤖"
        echo "================================================================="
        echo "Uso: ./manage.sh [comando] [modelo/serviço]"
        echo ""
        echo "COMANDOS:"
        echo "  start [alvo]    - Inicia o serviço em uma sessão TMUX dedicada"
        echo "  stop [alvo]     - Finaliza um serviço específico e libera a RAM"
        echo "  stop-all        - Finaliza TODOS os serviços ativos de uma vez"
        echo "  status          - Lista os serviços que estão rodando no momento"
        echo ""
        echo "ALVOS DISPONÍVEIS (Modelos & Serviços):"
        echo "  qwen-coder-3b            - Qwen2.5-Coder 3B (Q4_K_M) [Porta 8003]"
        echo "  qwen-vl-3b-uncensored    - Qwen2.5-VL 3B abliterado (sem censura) Q4_K_M [Porta 8010]"
        echo "  bonsai-27b               - Bonsai 27B 1-bit (89.5% FP16) ~4-8 tok/s [Porta 8011]"
        echo "  bonsai-27b-nothink       - Bonsai 27B 1-bit (thinking OFF) [Porta 8013]"
        echo "  ternary-bonsai-27b       - Bonsai 27B ternário (94.6% FP16) ~2-5 tok/s [Porta 8012]"
        echo "  ternary-bonsai-27b-nothink - Bonsai 27B ternário (thinking OFF) [Porta 8014]"
        echo "  gemma-4-e2b              - Gemma 4 E2B 2.3B ~22-35 tok/s [Porta 8021]"
        echo "  gemma-4-e2b-nothink      - Gemma 4 E2B (thinking OFF) [Porta 8023]"
        echo "  gemma-4-e2b-vision       - Gemma 4 E2B + visão (mmproj) [Porta 8031]"
        echo "  gemma-4-e2b-vision-nothink - Gemma 4 E2B visão (thinking OFF) [Porta 8033]"
        echo "  gemma-4-e4b              - Gemma 4 E4B 4.5B ~10-15 tok/s [Porta 8022]"
        echo "  gemma-4-e4b-nothink      - Gemma 4 E4B (thinking OFF) [Porta 8024]"
        echo "  gemma-4-e4b-vision       - Gemma 4 E4B + visão (mmproj) [Porta 8032]"
        echo "  gemma-4-e4b-vision-nothink - Gemma 4 E4B visão (thinking OFF) [Porta 8034]"
        echo "  dolphin3-3b              - Dolphin3.0 Llama3.2-3B Q4_K_M [Porta 8041]"
        echo "  lfm2.5-1.2b              - Liquid LFM 2.5-1.2B Q8_0 [Porta 8061]"
        echo "  nanbeige-3b              - Nanbeige4.2-3B Q4_K_M [Porta 8071]"
        echo "  nanbeige-3b-nothink      - Nanbeige4.2-3B (thinking OFF) [Porta 8073]"
        echo "  minicpm5-1b              - MiniCPM5-1B Nemotron-DPO Q8_0 [Porta 8081]"
        echo "  minicpm5-1b-nothink      - MiniCPM5-1B (thinking OFF) [Porta 8083]"
        echo "  gateway                  - Roteador Central (FastAPI) [Porta 9000]"
        echo ""
        echo "EXEMPLOS PRÁTICOS:"
        echo "  ./manage.sh start qwen-coder-3b       # Para começar a programar"
        echo "  ./manage.sh start qwen-vl-3b-uncensored  # Qwen2.5-VL 3B abliterado (visão sem censura)"
        echo "  ./manage.sh start bonsai-27b           # Bonsai 27B 1-bit (3.8 GB)"
        echo "  ./manage.sh start bonsai-27b-nothink   # Bonsai 27B 1-bit (thinking OFF)"
        echo "  ./manage.sh start ternary-bonsai-27b   # Bonsai 27B ternário (7.2 GB)"
        echo "  ./manage.sh start gemma-4-e2b          # Gemma 4 E2B 2.3B (3.2 GB)"
        echo "  ./manage.sh start gemma-4-e2b-vision   # Gemma 4 E2B + visão (4.2 GB)"
        echo "  ./manage.sh start gemma-4-e4b          # Gemma 4 E4B 4.5B (4.9 GB)"
        echo "  ./manage.sh start gemma-4-e4b-vision   # Gemma 4 E4B + visão (5.9 GB)"
        echo "  ./manage.sh start dolphin3-3b          # Dolphin3.0 Llama3.2-3B (1.9 GB) [Porta 8041]"
        echo "  ./manage.sh start lfm2.5-1.2b          # Liquid LFM 2.5-1.2B Q8_0 (1.2 GB) [Porta 8061]"
        echo "  ./manage.sh start nanbeige-3b          # Nanbeige4.2-3B Q4_K_M (2.4 GB) [Porta 8071]"
        echo "  ./manage.sh start nanbeige-3b-nothink  # Nanbeige4.2-3B (thinking OFF) [Porta 8073]"
        echo "  ./manage.sh start minicpm5-1b          # MiniCPM5-1B Nemotron-DPO Q8_0 (1.1 GB) [Porta 8081]"
        echo "  ./manage.sh start minicpm5-1b-nothink  # MiniCPM5-1B (thinking OFF) [Porta 8083]"
        echo "  ./manage.sh stop nanbeige-3b           # Para finalizar modelo específico"
        echo "  ./manage.sh stop-all                   # Para finalizar todos os modelos"
        echo "  ./manage.sh status                     # Para ver o que está ativo"
        echo "================================================================="
        ;;
esac
