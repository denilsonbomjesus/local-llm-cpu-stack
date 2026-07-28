#!/usr/bin/env bash

# Salva o argumento do modelo ANTES do source (setupvars.sh consome $@)
MODEL_ARG="$1"

BASE=~/llm-stack

# ============================================================
# Dual-build paths:
#   build/ReleaseOV → OpenVINO acelerado (Qwen, Ministral)
#   build/CPU       → GGML nativo (Gemma 2 e outros nao suportados)
# ============================================================
CLI_OV="$BASE/llama.cpp/build/ReleaseOV/bin/llama-cli"
CLI_CPU="$BASE/llama.cpp/build/CPU/bin/llama-cli"
CLI="$CLI_OV"  # default

# ============================================================
# OpenVINO Configuration (Intel Core i5 + Iris Xe)
# ============================================================
# Source OpenVINO environment if available
if [ -f /opt/intel/openvino/setupvars.sh ]; then
    source /opt/intel/openvino/setupvars.sh
fi

# Restaura o argumento (setupvars.sh consome $@ com shift)
set -- "$MODEL_ARG"

# ------------------------------------------------------------
# Device mapping FINAL - validado no i5-1235U Iris Xe:
#   ✅  Qwen (qwen-text, qwen-coder, vision) → OpenVINO GPU (stateless)
#   ❌  Ministral (agent, vision) → OpenVINO GPU/CPU crash/shape mismatch → GGML nativo
#   ❌  Gemma 2                    → OpenVINO incompativel                → GGML nativo
#
# Conclusao: OpenVINO trouxe ZERO ganho para CPU. So vale a pena na GPU.
# ------------------------------------------------------------
configure_openvino_for_model() {
    local model_type="$1"
    case "$model_type" in
        qwen-text|qwen-coder|vision)
            # Qwen: OpenVINO GPU com stateless ✅
            export GGML_OPENVINO_DEVICE="${GGML_OPENVINO_DEVICE:-GPU}"
            export GGML_OPENVINO_STATEFUL_EXECUTION=0
            CLI="$CLI_OV"
            echo "ℹ️  Qwen: OpenVINO GPU (stateless)"
            ;;
        ministral-agent|ministral-vision)
            # Ministral: OpenVINO INCOMPATIVEL (GPU crash + CPU shape mismatch)
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Ministral: OpenVINO incompativel. Usando GGML nativo (CPU)."
            ;;
        *)
            # Gemma 2: OpenVINO INCOMPATIVEL
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Gemma 2: OpenVINO incompativel. Usando GGML nativo (CPU)."
            ;;
    esac
}
# ============================================================

# Configurações de Hardware (Idênticas ao manage.sh para consistência)
THREADS=8
CTX=4096
BATCH=256

# ============================================================
# helper: monta o comando de chat para um modelo
# ============================================================
chat_cmd_for() {
    local model="$1"
    local cli="$CLI"
    case "$model" in
        qwen-text)  echo "$cli -m $BASE/models/text/qwen2.5-1.5b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        gemma2)     echo "$cli -m $BASE/models/text/gemma-2-2b-it-abliterated-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        qwen-coder) echo "$cli -m $BASE/models/code/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        ministral-agent) echo "$cli -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        ministral-vision) echo "$cli -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf --mmproj $BASE/models/code/ministral-3-3b-instruct-2512-mmproj-f16.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        vision)     echo "$cli -m $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it-iq4_xs.gguf --mmproj $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it.mmproj-Q8_0.gguf -t $THREADS -c $CTX" ;;
        *)          echo "" ;;
    esac
}

# ============================================================
# run_chat: inicia o chat interativo
# Usa o backend definido por configure_openvino_for_model (mapeamento estatico)
# ============================================================
run_chat() {
    local model="$1"

    # Verificar se o servidor desse modelo já está rodando no TMUX
    if tmux ls 2>/dev/null | grep -q "$model"; then
        echo "⚠️ ERRO: O servidor '$model' está ativo no manage.sh!"
        echo "Pare o servidor primeiro com: ./manage.sh stop $model"
        exit 1
    fi

    # Configura OpenVINO e seleciona o build correto
    configure_openvino_for_model "$model"

    local cmd
    cmd=$(chat_cmd_for "$model")
    if [ -z "$cmd" ]; then
        show_help
        return
    fi

    # --- CHAT INTERATIVO ---
    $cmd -cnv --log-disable
}

function show_help() {
    echo "================================================================="
    echo "       💬 LLM TERMINAL CHAT - MODO INTERATIVO 💬"
    echo "================================================================="
    echo "Uso: ./chat.sh [modelo]"
    echo ""
    echo "MODELOS DISPONÍVEIS:"
    echo "  qwen-text          - Chat rápido Qwen 2.5"
    echo "  gemma2             - Chat inteligente Gemma 2"
    echo "  qwen-coder         - Chat focado em programação"
    echo "  ministral-agent    - Chat inteligente/código (Ministral)"
    echo "  ministral-vision   - Chat inteligente/Visão (Ministral)"
    echo "  vision             - Chat de visão (Análise de imagens via Terminal)"
    echo "                       - Formatos: JPG, PNG, WEBP (PDF/DOCX não suportados)"
    echo "                       - Caminho Windows: /mnt/c/Users/Nome/Pictures/foto.jpg (/mnt/c/Users/denil/...)"
    echo "                       - Caminho Linux:   /home/user/llm-stack/foto.jpg (/home/denilsonbj/...)"
    echo "                       - Uso: Digite /image com o caminho da imagem, depois o prompt do chat."
    echo ""
    echo "DICA: Certifique-se que o modelo está DESLIGADO no manage.sh"
    echo "================================================================="
}

if [ -z "$1" ]; then
    show_help
else
    run_chat $1
fi