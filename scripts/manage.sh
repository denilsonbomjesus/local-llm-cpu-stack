#!/usr/bin/env bash

# Salva argumentos ANTES do source (setupvars.sh consome $@)
_SAVED_ARGS=("$@")

BASE=~/llm-stack

# ============================================================
# Dual-build paths:
#   build/ReleaseOV → OpenVINO acelerado (Qwen, Ministral)
#   build/CPU       → GGML nativo     (modelos sem OpenVINO)
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
#   ❌  Qwen-Coder (qwen-coder)              → OpenVINO corrompe saida     → GGML nativo CPU
#   ❌  Qwen-VL   (qwen-vl-uncensored)        → OpenVINO shape mismatch     → GGML nativo CPU
#   ✅  Bonsai 27B 1-bit                     → Q1_0 nativo (mainline)     → GGML nativo CPU
#   ✅  Ternary Bonsai 27B                   → Q2_0_g64 nativo (mainline) → GGML nativo CPU
#
# Conclusao: OpenVINO tem bugs de encoding/shape nos builds atuais.
# Nenhum modelo atual usa OpenVINO. Todos vao para GGML nativo CPU.
# ------------------------------------------------------------
configure_openvino_for_model() {
    local model_type="$1"
    case "$model_type" in
        qwen-coder)
            # Qwen-Coder: OpenVINO GPU corrompe saida de texto
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Qwen-Coder: OpenVINO GPU corrompe output. Usando GGML nativo (CPU)."
            ;;
        qwen-vl-uncensored)
            # Qwen-VL uncensored (abliterated): OpenVINO INCOMPATIVEL
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Qwen-VL Uncensored: OpenVINO incompativel. Usando GGML nativo (CPU)."
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
        lfm25)
            # Liquid LFM 2.5-1.2B-Instruct: 1.2B params, Q8_0 (8-bit)
            # 1.2 GB, máxima qualidade, excelente para código e raciocínio
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  LFM 2.5 (Liquid 1.2B): GGML nativo CPU (Q8_0)"
            ;;
        nanbeige|nanbeige-nothink|minicpm5|minicpm5-nothink)
            # Nanbeige4.2-3B: modelo compacto de 3B params da OWAO
            # 2.4 GB, Q4_K_M, excelente para código e raciocínio
            # Variante -nothink desliga reasoning tags (<think>...</think>)
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Nanbeige4.2-3B: GGML nativo CPU (Q4_K_M)"
            ;;
        *)
            # Fallback: modelos sem configuracao especifica vao para CPU
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            LLAMA="$LLAMA_CPU"
            echo "ℹ️  Modelo: GGML nativo CPU (fallback)."
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
        qwen-coder) echo "$llama -m $BASE/models/code/qwen2.5-coder-3b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH --port 8003" ;;
        qwen-vl-uncensored) echo "$llama -m $BASE/models/vision/qwen2.5-vl-3b-uncensored-Q4_K_M.gguf --mmproj $BASE/models/vision/qwen2.5-vl-3b-uncensored-mmproj-Q8_0.gguf -t $THREADS -c $CTX --port 8010" ;;
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
        lfm25)    echo "$llama -m $BASE/models/code/LFM2.5-1.2B-Instruct-Q8_0.gguf -t $THREADS -c $CTX -b $BATCH --port 8061" ;;
        nanbeige) echo "$llama -m $BASE/models/code/Nanbeige4.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH --port 8071" ;;
        nanbeige-nothink) echo "$llama -m $BASE/models/code/Nanbeige4.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8073" ;;
        minicpm5) echo "$llama -m $BASE/models/code/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf -t $THREADS -c $CTX -b $BATCH --port 8081" ;;
        minicpm5-nothink) echo "$llama -m $BASE/models/code/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off --port 8083" ;;
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
        echo "Uso: ./manage.sh start {qwen-coder|qwen-vl-uncensored|bonsai-27b|bonsai-27b-nothink|ternary-bonsai|ternary-bonsai-nothink|gemma4-e2b|gemma4-e2b-nothink|gemma4-e2b-vision|gemma4-e2b-vision-nothink|gemma4-e4b|gemma4-e4b-nothink|gemma4-e4b-vision|gemma4-e4b-vision-nothink|dolphin3|lfm25|nanbeige|nanbeige-nothink|minicpm5|minicpm5-nothink|gateway}"
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
    tmux ls 2>/dev/null | grep -E "qwen-coder|qwen-vl-uncensored|bonsai-27b|bonsai-27b-nothink|ternary-bonsai|ternary-bonsai-nothink|gemma4-e2b|gemma4-e2b-nothink|gemma4-e2b-vision|gemma4-e2b-vision-nothink|gemma4-e4b|gemma4-e4b-nothink|gemma4-e4b-vision|gemma4-e4b-vision-nothink|dolphin3|lfm25|nanbeige|nanbeige-nothink|minicpm5|minicpm5-nothink|gateway" || echo "Nenhum serviço rodando no momento."
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
        for s in qwen-coder qwen-vl-uncensored bonsai-27b bonsai-27b-nothink ternary-bonsai ternary-bonsai-nothink gemma4-e2b gemma4-e2b-nothink gemma4-e2b-vision gemma4-e2b-vision-nothink gemma4-e4b gemma4-e4b-nothink gemma4-e4b-vision gemma4-e4b-vision-nothink dolphin3 lfm25 nanbeige nanbeige-nothink minicpm5 minicpm5-nothink gateway; do
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
        echo "  qwen-coder          - Especialista em Programação (Qwen) [Porta 8003]"
        echo "  qwen-vl-uncensored   - Qwen2.5-VL 3B abliterado (sem censura) Q4_K_M [Porta 8010]"
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
        echo "  lfm25                - Liquid LFM 2.5 1.2B Q8_0 [Porta 8061]"
        echo "  nanbeige             - Nanbeige4.2-3B Q4_K_M [Porta 8071]"
        echo "  nanbeige-nothink     - Nanbeige4.2-3B (thinking OFF) resposta direta [Porta 8073]"
        echo "  minicpm5             - MiniCPM5-1B Nemotron-DPO Q8_0 [Porta 8081]"
        echo "  minicpm5-nothink     - MiniCPM5-1B (thinking OFF) ferramentas [Porta 8083]"
        echo "  gateway             - Roteador Central (FastAPI) [Porta 9000]"
        echo ""
        echo "EXEMPLOS PRÁTICOS:"
        echo "  ./manage.sh start qwen-coder      # Para começar a programar"
        echo "  ./manage.sh start qwen-vl-uncensored  # Qwen2.5-VL 3B abliterado (visão sem censura)"
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
        echo "  ./manage.sh start lfm25           # Liquid LFM 2.5 1.2B Q8_0 (1.2 GB) [Porta 8061]"
        echo "  ./manage.sh start nanbeige        # Nanbeige4.2-3B Q4_K_M (2.4 GB) [Porta 8071]"
        echo "  ./manage.sh start nanbeige-nothink # Nanbeige4.2-3B (thinking OFF) [Porta 8073]"
        echo "  ./manage.sh start minicpm5         # MiniCPM5-1B Nemotron-DPO Q8_0 (1.1 GB) [Porta 8081]"
        echo "  ./manage.sh start minicpm5-nothink  # MiniCPM5-1B (thinking OFF) [Porta 8083]"
        echo "  ./manage.sh stop nanbeige         # Para finalizar modelo específico"
        echo "  ./manage.sh stop-all              # Para finalizar todos os modelos"
        echo "  ./manage.sh status                # Para ver o que está ativo"
        echo "================================================================="
        ;;
esac