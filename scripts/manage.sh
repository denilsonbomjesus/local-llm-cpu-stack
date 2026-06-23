#!/usr/bin/env bash

BASE=~/llm-stack
LLAMA="$BASE/llama.cpp/build/bin/llama-server"

# Configurações de Hardware (Otimizadas para seu i5-1235U)
THREADS=8
CTX=4096
BATCH=256

function start_model() {
    case $1 in
        qwen-text)
            tmux new-session -d -s qwen-text "$LLAMA -m $BASE/models/text/qwen2.5-1.5b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8001"
            echo "✅ Qwen-Text (GGUF) iniciado na porta 8001"
            ;;
        gemma2)
            tmux new-session -d -s gemma2 "$LLAMA -m $BASE/models/text/gemma-2-2b-it-abliterated-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8002"
            echo "✅ Gemma-2 (GGUF) iniciado na porta 8002"
            ;;
        qwen-coder)
            tmux new-session -d -s qwen-coder "$LLAMA -m $BASE/models/code/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8003"
            echo "✅ Qwen-Coder (GGUF) iniciado na porta 8003"
            ;;
        ministral-agent)
            # Foco em Código/Texto (Porta 8004) - Sem Visão para economizar RAM
            tmux new-session -d -s ministral-agent "$LLAMA -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8004"
            echo "✅ Ministral-Agent (Texto/Código) iniciado na porta 8004"
            ;;
        ministral-vision)
            # Modo Completo (Porta 8005) - Texto + Visão
            tmux new-session -d -s ministral-vision "$LLAMA -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf --mmproj $BASE/models/code/ministral-3-3b-instruct-2512-mmproj-f16.gguf -t $THREADS -c $CTX -b $BATCH --port 8006"
            echo "👁️ Ministral-Vision (Texto + Visão) iniciado na porta 8005"
            ;;
        # vision)
        #     # Rota Python/Transformers conforme sua implementação (Porta 8010)
        #     cd $BASE/vision
        #     source venv/bin/activate
        #     tmux new-session -d -s vision "uvicorn qwen_vl_server:app --host 0.0.0.0 --port 8010"
        #     echo "👁️ Qwen-VL (Python Server) iniciado na porta 8010"
        #     ;;
        vision)
            # USANDO O MOTOR C++ (Muito mais leve que o Python)
            tmux new-session -d -s vision "$LLAMA -m $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it-iq4_xs.gguf --mmproj $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it.mmproj-Q8_0.gguf -t $THREADS -c $CTX --port 8010"
            echo "✅ Qwen-VL (Motor C++ GGUF) iniciado na porta 8010"
            ;;
        gateway)
            cd $BASE/gateway
            source venv/bin/activate
            tmux new-session -d -s gateway "uvicorn gateway_server:app --host 0.0.0.0 --port 9000"
            echo "🌐 Gateway (Router) iniciado na porta 9000"
            ;;
        *)
            echo "Uso: ./manage.sh start {qwen-text|gemma2|qwen-coder|deepseek|vision|gateway}"
            ;;
    esac
}

function stop_model() {
    tmux kill-session -t $1 2>/dev/null && echo "🛑 $1 parado." || echo "⚠️ $1 não estava rodando."
}

function status() {
    echo "--- Status dos Modelos (Sessões TMUX) ---"
    tmux ls 2>/dev/null | grep -E "qwen-text|gemma2|qwen-coder|ministral-agent|ministral-vision|vision|gateway" || echo "Nenhum serviço rodando no momento."
}

# Lógica Principal do Script
case $1 in
    start)
        start_model $2
        ;;
    stop)
        stop_model $2
        ;;
    stop-all)
        echo "Finalizando todos os serviços..."
        for s in qwen-text gemma2 qwen-coder ministral-agent ministral-vision vision gateway; do
            stop_model $s
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
        echo "  qwen-text           - Chat e Lógica (Qwen 2.5 1.5B) [Porta 8001]"
        echo "  gemma2              - Chat Geral Sem Censura (Gemma 2 2B) [Porta 8002]"
        echo "  qwen-coder          - Especialista em Programação (Qwen) [Porta 8003]"
        echo "  ministral-agent     - Agente e Código (Ministral 3 3B) [Porta 8004]"
        echo "  ministral-vision    - Texto + Visão (Ministral 3 3B) [Porta 8005]"
        echo "  vision              - Servidor de Visão (Qwen-VL) [Porta 8010]"
        echo "  gateway             - Roteador Central (FastAPI) [Porta 9000]"
        echo ""
        echo "EXEMPLOS PRÁTICOS:"
        echo "  ./manage.sh start qwen-coder   # Para começar a programar"
        echo "  ./manage.sh start vision       # Para analisar imagens"
        echo "  ./manage.sh stop gemma2        # Para finalizar modelo"
        echo "  ./manage.sh stop-all           # Para finalizar todos os modelos"
        echo "  ./manage.sh status             # Para ver o que está ativo"
        echo "================================================================="
        ;;
esac