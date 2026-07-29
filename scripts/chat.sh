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

# ============================================================
# Locale: Força UTF-8 para evitar que caracteres acentuados
# (português, etc.) apareçam como '?' no terminal.
# Definido APOS setupvars.sh para evitar que ele sobrescreva.
# C.UTF-8 é embutido na glibc e funciona em praticamente
# todos os sistemas Linux modernos.
# ============================================================
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
# ============================================================
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
            CLI="$CLI_OV"
            echo "ℹ️  Qwen-Text: OpenVINO GPU (stateless)"
            ;;
        qwen-coder)
            # Qwen-Coder: OpenVINO GPU corrompe saida de texto
            # O backend OpenVINO GPU decodifica tokens incorretamente,
            # fazendo com que caracteres acentuados virem '?' no terminal.
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Qwen-Coder: OpenVINO GPU corrompe output. Usando GGML nativo (CPU)."
            ;;
        vision)
            # Qwen-VL: OpenVINO INCOMPATIVEL (tensor shape mismatch no GPU)
            # O modelo multimodal tem shapes dinamicos que o OpenVINO GPU nao gerencia.
            # Erro tipico: espera [1,1,2,256] mas recebe [1,2,2,128]
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Qwen-VL: OpenVINO incompativel. Usando GGML nativo (CPU)."
            ;;
        ministral-agent|ministral-vision)
            # Ministral: OpenVINO INCOMPATIVEL (GPU crash + CPU shape mismatch)
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Ministral: OpenVINO incompativel. Usando GGML nativo (CPU)."
            ;;
        bonsai-27b|bonsai-27b-nothink)
            # Bonsai 27B 1-bit: Q1_0_g128 — suportado nativamente no mainline llama.cpp
            # 3.8 GB, 89.5% do FP16, 262K contexto
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Bonsai 27B 1-bit: GGML nativo CPU (Q1_0)"
            ;;
        ternary-bonsai|ternary-bonsai-nothink)
            # Ternary Bonsai 27B: Q2_0_g64 — suportado nativamente no mainline llama.cpp
            # 7.2 GB, 94.6% do FP16 — requer 10 GB WSL2
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Ternary Bonsai 27B: GGML nativo CPU (Q2_0_g64)"
            ;;
        gemma4-e2b|gemma4-e4b|gemma4-e2b-nothink|gemma4-e4b-nothink|gemma4-e2b-vision|gemma4-e4b-vision|gemma4-e2b-vision-nothink|gemma4-e4b-vision-nothink)
            # Gemma 4 (E2B/E4B): qat-q4_0 — suportado nativamente
            # E2B: 2.3B efetivos, 3.2 GB, 22-35 tok/s
            # E4B: 4.5B efetivos, 4.9 GB, 10-15 tok/s
            # Variantes -vision incluem mmproj para suporte multimodal
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Gemma 4: GGML nativo CPU (qat-q4_0)"
            ;;
        dolphin3)
            # Dolphin3.0-Llama3.2-3B: base Llama 3.2, Q4_K_M
            # 1.9 GB, excelente para tarefas gerais e código
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Dolphin3.0 (Llama3.2-3B): GGML nativo CPU (Q4_K_M)"
            ;;
        lexi8b)
            # Lexi-Llama-3-8B-Uncensored: Llama 3 8B uncensored, Q4_K_M
            # 4.6 GB, modelo grande e versátil para tarefas complexas
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Lexi-8B (Llama-3-8B-Uncensored): GGML nativo CPU (Q4_K_M)"
            ;;
        lfm25)
            # Liquid LFM 2.5-1.2B-Instruct: 1.2B params, Q8_0 (8-bit)
            # 1.2 GB, máxima qualidade, excelente para código e raciocínio
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  LFM 2.5 (Liquid 1.2B): GGML nativo CPU (Q8_0)"
            ;;
        nanbeige|nanbeige-nothink|minicpm5|minicpm5-nothink)
            # Nanbeige4.2-3B: modelo compacto de 3B params da OWAO
            # 2.4 GB, Q4_K_M, excelente para código e raciocínio
            # Variante -nothink desliga reasoning tags (<think>...</think>)
            unset GGML_OPENVINO_DEVICE
            unset GGML_OPENVINO_STATEFUL_EXECUTION
            CLI="$CLI_CPU"
            echo "ℹ️  Nanbeige4.2-3B: GGML nativo CPU (Q4_K_M)"
            ;;
        *)
            # Gemma 2 e outros: OpenVINO INCOMPATIVEL
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
        qwen-coder) echo "$cli -m $BASE/models/code/qwen2.5-coder-3b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        ministral-agent) echo "$cli -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        ministral-vision) echo "$cli -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf --mmproj $BASE/models/code/ministral-3-3b-instruct-2512-mmproj-f16.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        vision)     echo "$cli -m $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it-iq4_xs.gguf --mmproj $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it.mmproj-Q8_0.gguf -t $THREADS -c $CTX" ;;
        bonsai-27b)           echo "$cli -m $BASE/models/bonsai/Bonsai-27B-Q1_0.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        bonsai-27b-nothink)    echo "$cli -m $BASE/models/bonsai/Bonsai-27B-Q1_0.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        ternary-bonsai)        echo "$cli -m $BASE/models/bonsai/Ternary-Bonsai-27B-Q2_g64.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        ternary-bonsai-nothink) echo "$cli -m $BASE/models/bonsai/Ternary-Bonsai-27B-Q2_g64.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        gemma4-e2b)            echo "$cli -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        gemma4-e2b-nothink)     echo "$cli -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        gemma4-e4b)            echo "$cli -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        gemma4-e4b-nothink)     echo "$cli -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        gemma4-e2b-vision)      echo "$cli -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E2B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        gemma4-e2b-vision-nothink) echo "$cli -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E2B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        gemma4-e4b-vision)      echo "$cli -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E4B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        gemma4-e4b-vision-nothink) echo "$cli -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E4B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        dolphin3)  echo "$cli -m $BASE/models/text/Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        lexi8b)    echo "$cli -m $BASE/models/text/Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        lfm25)    echo "$cli -m $BASE/models/code/LFM2.5-1.2B-Instruct-Q8_0.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        nanbeige) echo "$cli -m $BASE/models/code/Nanbeige4.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        minicpm5) echo "$cli -m $BASE/models/code/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        minicpm5-nothink) echo "$cli -m $BASE/models/code/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
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
    echo "  bonsai-27b           - 27B 1-bit (89.5% FP16) ~4-8 tok/s [3.8 GB]"
    echo "  bonsai-27b-nothink    - 27B 1-bit (thinking OFF) resposta direta [3.8 GB]"
    echo "  ternary-bonsai        - 27B ternário (94.6% FP16) ~2-5 tok/s [7.2 GB]"
    echo "  ternary-bonsai-nothink - 27B ternário (thinking OFF) resposta direta [7.2 GB]"
    echo "  gemma4-e2b              - Gemma 4 E2B 2.3B (qat-q4_0) ~22-35 tok/s [3.2 GB]"
    echo "  gemma4-e2b-nothink       - Gemma 4 E2B (thinking OFF) resposta direta [3.2 GB]"
    echo "  gemma4-e2b-vision        - Gemma 4 E2B + visão (mmproj) [3.2 GB + 1 GB]"
    echo "  gemma4-e2b-vision-nothink - Gemma 4 E2B visão (thinking OFF) [3.2 GB + 1 GB]"
    echo "  gemma4-e4b              - Gemma 4 E4B 4.5B (qat-q4_0) ~10-15 tok/s [4.9 GB]"
    echo "  gemma4-e4b-nothink       - Gemma 4 E4B (thinking OFF) resposta direta [4.9 GB]"
    echo "  gemma4-e4b-vision        - Gemma 4 E4B + visão (mmproj) [4.9 GB + 1 GB]"
    echo "  gemma4-e4b-vision-nothink - Gemma 4 E4B visão (thinking OFF) [4.9 GB + 1 GB]"
    echo "  dolphin3             - Dolphin3.0 Llama3.2-3B Q4_K_M ~20-30 tok/s [1.9 GB]"
    echo "  lexi8b               - Lexi-Llama-3-8B-Uncensored Q4_K_M ~8-15 tok/s [4.6 GB]"
    echo "  lfm25                - Liquid LFM 2.5 1.2B Q8_0 ~25-40 tok/s [1.2 GB]"
    echo "  nanbeige             - Nanbeige4.2-3B Q4_K_M ~18-30 tok/s [2.4 GB]"
    echo "  nanbeige-nothink     - Nanbeige4.2-3B (thinking OFF) resposta direta [2.4 GB]"
    echo "  minicpm5             - MiniCPM5-1B Nemotron-DPO Q8_0 ~30-50 tok/s [1.1 GB]"
    echo "  minicpm5-nothink     - MiniCPM5-1B (thinking OFF) ferramentas diretas [1.1 GB]"
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