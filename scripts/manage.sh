#!/usr/bin/env bash

# Salva argumentos ANTES do source (setupvars.sh consome $@)
_SAVED_ARGS=("$@")

BASE=~/llm-stack

# ============================================================
# Dual-build paths:
#   build/ReleaseOV → OpenVINO acelerado (Qwen, Ministral)
#   build/CPU       → GGML nativo     (Gemma 2, modelos incompativeis)
# ============================================================
LLAMA_OV="$BASE/llama.cpp/build/ReleaseOV/bin/llama-server"
LLAMA_CPU="$BASE/llama.cpp/build/CPU/bin/llama-server"
LLAMA="$LLAMA_OV"  # default

# ============================================================
# OpenVINO Configuration (Intel Core i5 + Iris Xe)
# ============================================================
# Source OpenVINO environment if available
if [ -f /opt/intel/openvino/setupvars.sh ]; then
    source /opt/intel/openvino/setupvars.sh
fi

# Restaura argumentos (setupvars.sh consome $@ com shift)
set -- "${_SAVED_ARGS[@]}"
unset _SAVED_ARGS

# ------------------------------------------------------------
# Device mapping FINAL - validado no i5-1235U Iris Xe:
#   ⚠️  Qwen-Text                             → OpenVINO GPU (stateless)
#                                            → encoding bug em alguns modelos
#   ❌  Qwen-Coder (qwen-coder)              → OpenVINO corrompe saida     → GGML nativo CPU
#   ❌  Qwen-VL   (vision)                   → OpenVINO shape mismatch     → GGML nativo CPU
#   ❌  Ministral (agent, vision)            → OpenVINO incompativel       → GGML nativo CPU
#   ❌  Gemma 2                              → OpenVINO incompativel       → GGML nativo CPU
#
# Conclusao: OpenVINO tem bugs de encoding/shape nos builds atuais.
# So Qwen-Text funciona (parcialmente) no GPU. Demais vao para CPU.
# ------------------------------------------------------------
configure_openvino_for_model() {
    local model_type="$1"
    case "$model_type" in
        qwen-text)
            # Qwen-Text: OpenVINO GPU com stateless (pode ter encoding bug)
            export GGML_OPENVINO_DEVICE="${GGML_OPENVINO_DEVICE:-GPU}"
            export GGML_OPENVINO_STATEFUL_EXECUTION=0
            LLAMA="$LLAMA_OV"
            echo "ℹ️  Qwen-Text: OpenVINO GPU (stateless)"
            ;;
        qwen-coder)
            # Qwen-Coder: OpenVINO GPU corrompe saida de texto
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Qwen-Coder: OpenVINO GPU corrompe output. Usando GGML nativo (CPU)."
            ;;
        vision)
            # Qwen-VL: OpenVINO INCOMPATIVEL (tensor shape mismatch no GPU)
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Qwen-VL: OpenVINO incompativel. Usando GGML nativo (CPU)."
            ;;
        ministral-agent|ministral-vision)
            # Ministral: OpenVINO INCOMPATIVEL (GPU crash + CPU shape mismatch)
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Ministral: OpenVINO incompativel. Usando GGML nativo (CPU)."
            ;;
        *)
            # Gemma 2: OpenVINO INCOMPATIVEL
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Gemma 2: OpenVINO incompativel. Usando GGML nativo (CPU)."
            ;;
    esac
}
# ============================================================

# Configurações de Hardware (Otimizadas para seu i5-1235U)
THREADS=8
CTX=4096
BATCH=256

# ============================================================
# helper: monta o comando do servidor para um modelo
# ============================================================
server_cmd_for() {
    local model="$1"
    local llama="$LLAMA"
    case "$model" in
        qwen-text)  echo "$llama -m $BASE/models/text/qwen2.5-1.5b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8001" ;;
        gemma2)     echo "$llama -m $BASE/models/text/gemma-2-2b-it-abliterated-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8002" ;;
        qwen-coder) echo "$llama -m $BASE/models/code/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8003" ;;
        ministral-agent) echo "$llama -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8004" ;;
        ministral-vision) echo "$llama -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf --mmproj $BASE/models/code/ministral-3-3b-instruct-2512-mmproj-f16.gguf -t $THREADS -c $CTX -b $BATCH --port 8005" ;;
        vision)     echo "$llama -m $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it-iq4_xs.gguf --mmproj $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it.mmproj-Q8_0.gguf -t $THREADS -c $CTX --port 8010" ;;
        *)          echo "" ;;
    esac
}

# ============================================================
# start_model: inicia o servidor com fallback automatico
# Tenta OpenVINO primeiro (se aplicavel). Se detectar erro,
# reinicia com GGML nativo (CPU) sem intervencao do usuario.
# ============================================================
start_model() {
    local model="$1"
    local session="$model"

    # --- Caso especial: gateway (nao usa llama.cpp) ---
    if [ "$model" = "gateway" ]; then
        cd "$BASE/gateway"
        source venv/bin/activate 2>/dev/null || true
        tmux new-session -d -s gateway "uvicorn gateway_server:app --host 0.0.0.0 --port 9000"
        echo "🌐 Gateway (Router) iniciado na porta 9000"
        cd "$BASE"
        return
    fi

    # Passo 1: configura o backend preferido
    configure_openvino_for_model "$model"
    local using_ov=false
    [ "$LLAMA" = "$LLAMA_OV" ] && using_ov=true

    # Passo 2: monta e executa o comando
    local cmd
    cmd=$(server_cmd_for "$model")
    if [ -z "$cmd" ]; then
        echo "Uso: ./manage.sh start {qwen-text|gemma2|qwen-coder|ministral-agent|ministral-vision|vision|gateway}"
        return
    fi

    # Mostra qual backend esta sendo usado
    local backend_label
    if $using_ov; then
        backend_label="OpenVINO GPU"
    else
        backend_label="GGML nativo CPU"
    fi

    tmux new-session -d -s "$session" "$cmd"
    local port
    port=$(echo "$cmd" | sed 's/.*--port \([0-9]\+\).*/\1/')
    echo "✅ ${model^} (${backend_label}) iniciado na porta ${port}"

    # Passo 3: fallback dinâmico — so para modelos OpenVINO
    if $using_ov; then
        sleep 6
        if tmux capture-pane -t "$session" -p 2>/dev/null | grep -qi "ov::Exception\|shape incompatible\|Compute error\|failed to decode"; then
            echo "⚠️  OpenVINO falhou para '$model'! Aplicando fallback para GGML nativo (CPU)..."
            tmux kill-session -t "$session" 2>/dev/null
            sleep 1

            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"

            cmd=$(server_cmd_for "$model")
            tmux new-session -d -s "$session" "$cmd"
            echo "⚠️  Fallback: $model agora rodando em GGML nativo (CPU) na porta ${port}"
        fi
    fi
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