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
#   ✅  Bonsai 27B 1-bit                     → Q1_0 nativo (mainline)     → GGML nativo CPU
#   ✅  Ternary Bonsai 27B                   → Q2_0_g64 nativo (mainline) → GGML nativo CPU
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
        bonsai-27b|bonsai-27b-nothink)
            # Bonsai 27B 1-bit: Q1_0_g128 — mainline llama.cpp
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Bonsai 27B 1-bit: GGML nativo CPU (Q1_0)"
            ;;
        ternary-bonsai|ternary-bonsai-nothink)
            # Ternary Bonsai 27B: Q2_0_g64 — mainline llama.cpp
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Ternary Bonsai 27B: GGML nativo CPU (Q2_0_g64)"
            ;;
        gemma4-e2b|gemma4-e4b|gemma4-e2b-nothink|gemma4-e4b-nothink|gemma4-e2b-vision|gemma4-e4b-vision|gemma4-e2b-vision-nothink|gemma4-e4b-vision-nothink)
            # Gemma 4 (E2B/E4B): qat-q4_0 — suportado nativamente
            # Variantes -vision incluem mmproj para suporte multimodal
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Gemma 4: GGML nativo CPU (qat-q4_0)"
            ;;
        dolphin3)
            # Dolphin3.0-Llama3.2-3B: base Llama 3.2, Q4_K_M
            # 1.9 GB, excelente para tarefas gerais e código
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Dolphin3.0 (Llama3.2-3B): GGML nativo CPU (Q4_K_M)"
            ;;
        lexi8b)
            # Lexi-Llama-3-8B-Uncensored: Llama 3 8B uncensored, Q4_K_M
            # 4.6 GB, modelo grande e versátil
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Lexi-8B (Llama-3-8B-Uncensored): GGML nativo CPU (Q4_K_M)"
            ;;
        *)
            # Gemma 2 e outros: OpenVINO INCOMPATIVEL
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
        bonsai-27b)           echo "$llama -m $BASE/models/bonsai/Bonsai-27B-Q1_0.gguf -t $THREADS -c $CTX -b $BATCH --port 8011" ;;
        bonsai-27b-nothink)    echo "$llama -m $BASE/models/bonsai/Bonsai-27B-Q1_0.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8013" ;;
        ternary-bonsai)        echo "$llama -m $BASE/models/bonsai/Ternary-Bonsai-27B-Q2_g64.gguf -t $THREADS -c $CTX -b $BATCH --port 8012" ;;
        ternary-bonsai-nothink) echo "$llama -m $BASE/models/bonsai/Ternary-Bonsai-27B-Q2_g64.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8014" ;;
        gemma4-e2b)            echo "$llama -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --port 8021" ;;
        gemma4-e2b-nothink)     echo "$llama -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8023" ;;
        gemma4-e4b)            echo "$llama -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --port 8022" ;;
        gemma4-e4b-nothink)     echo "$llama -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8024" ;;
        gemma4-e2b-vision)      echo "$llama -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E2B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --port 8031" ;;
        gemma4-e2b-vision-nothink) echo "$llama -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E2B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8033" ;;
        gemma4-e4b-vision)      echo "$llama -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E4B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --port 8032" ;;
        gemma4-e4b-vision-nothink) echo "$llama -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E4B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8034" ;;
        dolphin3)  echo "$llama -m $BASE/models/text/Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH --port 8041" ;;
        lexi8b)    echo "$llama -m $BASE/models/text/Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH --port 8051" ;;
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
        echo "Uso: ./manage.sh start {qwen-text|gemma2|qwen-coder|ministral-agent|ministral-vision|vision|bonsai-27b|bonsai-27b-nothink|ternary-bonsai|ternary-bonsai-nothink|gemma4-e2b|gemma4-e2b-nothink|gemma4-e2b-vision|gemma4-e2b-vision-nothink|gemma4-e4b|gemma4-e4b-nothink|gemma4-e4b-vision|gemma4-e4b-vision-nothink|dolphin3|lexi8b|gateway}"
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
    tmux ls 2>/dev/null | grep -E "qwen-text|gemma2|qwen-coder|ministral-agent|ministral-vision|vision|bonsai-27b|bonsai-27b-nothink|ternary-bonsai|ternary-bonsai-nothink|gemma4-e2b|gemma4-e2b-nothink|gemma4-e2b-vision|gemma4-e2b-vision-nothink|gemma4-e4b|gemma4-e4b-nothink|gemma4-e4b-vision|gemma4-e4b-vision-nothink|dolphin3|lexi8b|gateway" || echo "Nenhum serviço rodando no momento."
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
        for s in qwen-text gemma2 qwen-coder ministral-agent ministral-vision vision bonsai-27b bonsai-27b-nothink ternary-bonsai ternary-bonsai-nothink gemma4-e2b gemma4-e2b-nothink gemma4-e2b-vision gemma4-e2b-vision-nothink gemma4-e4b gemma4-e4b-nothink gemma4-e4b-vision gemma4-e4b-vision-nothink dolphin3 lexi8b gateway; do
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
        echo "  bonsai-27b           - 27B 1-bit (89.5% FP16) ~4-8 tok/s [Porta 8011]"
        echo "  bonsai-27b-nothink    - 27B 1-bit (thinking OFF) resposta direta [Porta 8013]"
        echo "  ternary-bonsai        - 27B ternário (94.6% FP16) ~2-5 tok/s [Porta 8012]"
        echo "  ternary-bonsai-nothink - 27B ternário (thinking OFF) resposta direta [Porta 8014]"
        echo "  gemma4-e2b              - Gemma 4 E2B 2.3B ~22-35 tok/s [Porta 8021]"
        echo "  gemma4-e2b-nothink       - Gemma 4 E2B (thinking OFF) resposta direta [Porta 8023]"
        echo "  gemma4-e2b-vision        - Gemma 4 E2B + visão (mmproj) [Porta 8031]"
        echo "  gemma4-e2b-vision-nothink - Gemma 4 E2B visão (thinking OFF) [Porta 8033]"
        echo "  gemma4-e4b              - Gemma 4 E4B 4.5B ~10-15 tok/s [Porta 8022]"
        echo "  gemma4-e4b-nothink       - Gemma 4 E4B (thinking OFF) resposta direta [Porta 8024]"
        echo "  gemma4-e4b-vision        - Gemma 4 E4B + visão (mmproj) [Porta 8032]"
        echo "  gemma4-e4b-vision-nothink - Gemma 4 E4B visão (thinking OFF) [Porta 8034]"
        echo "  dolphin3             - Dolphin3.0 Llama3.2-3B Q4_K_M [Porta 8041]"
        echo "  lexi8b               - Lexi-Llama-3-8B-Uncensored Q4_K_M [Porta 8051]"
        echo "  gateway             - Roteador Central (FastAPI) [Porta 9000]"
        echo ""
        echo "EXEMPLOS PRÁTICOS:"
        echo "  ./manage.sh start qwen-coder      # Para começar a programar"
        echo "  ./manage.sh start vision          # Para analisar imagens"
        echo "  ./manage.sh start bonsai-27b           # 27B 1-bit (3.8 GB)"
        echo "  ./manage.sh start bonsai-27b-nothink    # 27B 1-bit (thinking OFF)"
        echo "  ./manage.sh start ternary-bonsai        # 27B ternário (7.2 GB)"
        echo "  ./manage.sh start ternary-bonsai-nothink # 27B ternário (thinking OFF)"
        echo "  ./manage.sh start gemma4-e2b              # Gemma 4 E2B 2.3B (3.2 GB)"
        echo "  ./manage.sh start gemma4-e2b-nothink       # Gemma 4 E2B (thinking OFF)"
        echo "  ./manage.sh start gemma4-e2b-vision        # Gemma 4 E2B + visão (4.2 GB)"
        echo "  ./manage.sh start gemma4-e2b-vision-nothink # Gemma 4 E2B (thinking OFF) + visão"
        echo "  ./manage.sh start gemma4-e4b              # Gemma 4 E4B 4.5B (4.9 GB)"
        echo "  ./manage.sh start gemma4-e4b-nothink       # Gemma 4 E4B (thinking OFF)"
        echo "  ./manage.sh start gemma4-e4b-vision        # Gemma 4 E4B + visão (5.9 GB)"
        echo "  ./manage.sh start gemma4-e4b-vision-nothink # Gemma 4 E4B (thinking OFF) + visão"
        echo "  ./manage.sh start dolphin3        # Dolphin3.0 Llama3.2-3B (1.9 GB) [Porta 8041]"
        echo "  ./manage.sh start lexi8b          # Lexi-Llama-3-8B-Uncensored (4.6 GB) [Porta 8051]"
        echo "  ./manage.sh stop gemma2           # Para finalizar modelo"
        echo "  ./manage.sh stop-all              # Para finalizar todos os modelos"
        echo "  ./manage.sh status                # Para ver o que está ativo"
        echo "================================================================="
        ;;
esac