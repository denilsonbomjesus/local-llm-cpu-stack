#!/usr/bin/env bash

BASE=~/llm-stack

# ============================================================
# Build único: GGML nativo (CPU)
#   O build OpenVINO (build/ReleaseOV) foi removido do stack:
#   nenhum modelo atual o utilizava (todos caíam em fallback
#   para CPU por bugs de encoding/shape). Todos os modelos
#   rodam no build CPU principal.
# ============================================================
CLI="$BASE/llama.cpp/build/CPU/bin/llama-cli"

# ============================================================
# Locale: Força UTF-8 para evitar que caracteres acentuados
# (português, etc.) apareçam como '?' no terminal.
# C.UTF-8 é embutido na glibc e funciona em praticamente
# todos os sistemas Linux modernos.
# ============================================================
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Configurações de Hardware (Idênticas ao manage.sh para consistência)
THREADS=8
CTX=4096
BATCH=256

# ============================================================
# helper: monta o comando de chat para um modelo
# Padrão de nomes: <fabricante>-<modelo>-<tamanho>[-variante]
#   -nothink → desliga o modo de raciocínio (--reasoning off)
#   -vision  → carrega o mmproj (suporte a imagens)
# ============================================================
chat_cmd_for() {
    local cli="$CLI"
    case "$1" in
        qwen-coder-3b)        echo "$cli -m $BASE/models/code/qwen2.5-coder-3b-instruct-q4_k_m.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        qwen-vl-3b-uncensored) echo "$cli -m $BASE/models/vision/qwen2.5-vl-3b-uncensored-Q4_K_M.gguf --mmproj $BASE/models/vision/qwen2.5-vl-3b-uncensored-mmproj-Q8_0.gguf -t $THREADS -c $CTX" ;;
        bonsai-27b)            echo "$cli -m $BASE/models/bonsai/Bonsai-27B-Q1_0.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        bonsai-27b-nothink)    echo "$cli -m $BASE/models/bonsai/Bonsai-27B-Q1_0.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        ternary-bonsai-27b)    echo "$cli -m $BASE/models/bonsai/Ternary-Bonsai-27B-Q2_g64.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        ternary-bonsai-27b-nothink) echo "$cli -m $BASE/models/bonsai/Ternary-Bonsai-27B-Q2_g64.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        gemma-4-e2b)           echo "$cli -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        gemma-4-e2b-nothink)   echo "$cli -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        gemma-4-e4b)           echo "$cli -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        gemma-4-e4b-nothink)   echo "$cli -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        gemma-4-e2b-vision)    echo "$cli -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E2B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        gemma-4-e2b-vision-nothink) echo "$cli -m $BASE/models/text/gemma-4-E2B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E2B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        gemma-4-e4b-vision)    echo "$cli -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E4B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        gemma-4-e4b-vision-nothink) echo "$cli -m $BASE/models/text/gemma-4-E4B_q4_0-it.gguf --mmproj $BASE/models/text/gemma-4-E4B-it-mmproj.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        dolphin3-3b)           echo "$cli -m $BASE/models/text/Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        lfm2.5-1.2b)           echo "$cli -m $BASE/models/code/LFM2.5-1.2B-Instruct-Q8_0.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        nanbeige-3b)           echo "$cli -m $BASE/models/code/Nanbeige4.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        nanbeige-3b-nothink)   echo "$cli -m $BASE/models/code/Nanbeige4.2-3B-Q4_K_M.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        minicpm5-1b)           echo "$cli -m $BASE/models/code/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf -t $THREADS -c $CTX -b $BATCH" ;;
        minicpm5-1b-nothink)   echo "$cli -m $BASE/models/code/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf -t $THREADS -c $CTX -b $BATCH --jinja --reasoning off" ;;
        *) echo "" ;;
    esac
}

# ============================================================
# run_chat: inicia o chat interativo
# Usa sempre o build CPU (não há mais backend OpenVINO).
# ============================================================
run_chat() {
    local model="$1"

    # Verificar se o servidor desse modelo já está rodando no TMUX
    if tmux ls 2>/dev/null | grep -q "$model"; then
        echo "⚠️ ERRO: O servidor '$model' está ativo no manage.sh!"
        echo "Pare o servidor primeiro com: ./manage.sh stop $model"
        exit 1
    fi

    local cmd
    cmd=$(chat_cmd_for "$model")
    if [ -z "$cmd" ]; then
        show_help
        return
    fi

    # --- CHAT INTERATIVO ---
    # Verbosidade de log em TRACE (-lv 4) para ativar o memory breakdown
    # no encerramento (o breakdown usa LOG_TRC, suprimido no nível padrão).
    # Os logs do servidor vão para stderr (arquivo de log) para não poluir
    # o chat; o texto do modelo e os timings [Prompt/Generation] vão para
    # stdout e permanecem visíveis no terminal.
    local log_file
    log_file="/tmp/llama-chat-${model}.log"
    # Garante que /tmp está gravável antes de redirecionar stderr
    if ! : > "$log_file" 2>/dev/null; then
        log_file="/dev/null"
    fi

    $cmd -cnv -lv 4 2>>"$log_file"

    # --- RESUMO DE CONTEXTO/MEMÓRIA AO ENCERRAR ---
    # Extrai do log o breakdown de memória (model/context/compute) e os
    # timings finais da última geração, exibidos após o chat terminar.
    if [ -s "$log_file" ]; then
        echo ""
        echo "═══════════════════════════════════════════"
        echo "   📊 RESUMO DE CONTEXTO / MEMÓRIA"
        echo "═══════════════════════════════════════════"
        # Timings da última geração (prompt eval / eval / total)
        grep "slot print_timing" "$log_file" | tail -5
        echo ""
        # Memory breakdown (janela de contexto em MiB): o cabeçalho e a linha
        # do Host usam o prefixo common_memory_breakdown_print (2 linhas por
        # ocorrência — startup e exit). tail -4 pega as 2 do encerramento.
        grep "common_memory_breakdown_print" "$log_file" | tail -4
        echo "═══════════════════════════════════════════"
    fi
}

function show_help() {
    echo "================================================================="
    echo "       💬 LLM TERMINAL CHAT - MODO INTERATIVO 💬"
    echo "================================================================="
    echo "Uso: ./chat.sh [modelo]"
    echo ""
    echo "MODELOS DISPONÍVEIS:"
    echo "  qwen-coder-3b           - Qwen2.5-Coder 3B (Q4_K_M) ~12-20 tok/s [2.0 GB]"
    echo "  qwen-vl-3b-uncensored   - Qwen2.5-VL 3B abliterado (sem censura) Q4_K_M ~15-25 tok/s [3.1 GB]"
    echo "  bonsai-27b              - Bonsai 27B 1-bit (89.5% FP16) ~4-8 tok/s [3.8 GB]"
    echo "  bonsai-27b-nothink      - Bonsai 27B 1-bit (thinking OFF) resposta direta [3.8 GB]"
    echo "  ternary-bonsai-27b      - Bonsai 27B ternário (94.6% FP16) ~2-5 tok/s [7.2 GB]"
    echo "  ternary-bonsai-27b-nothink - Bonsai 27B ternário (thinking OFF) resposta direta [7.2 GB]"
    echo "  gemma-4-e2b             - Gemma 4 E2B 2.3B (qat-q4_0) ~22-35 tok/s [3.2 GB]"
    echo "  gemma-4-e2b-nothink     - Gemma 4 E2B (thinking OFF) resposta direta [3.2 GB]"
    echo "  gemma-4-e2b-vision      - Gemma 4 E2B + visão (mmproj) [3.2 GB + 1 GB]"
    echo "  gemma-4-e2b-vision-nothink - Gemma 4 E2B visão (thinking OFF) [3.2 GB + 1 GB]"
    echo "  gemma-4-e4b             - Gemma 4 E4B 4.5B (qat-q4_0) ~10-15 tok/s [4.9 GB]"
    echo "  gemma-4-e4b-nothink     - Gemma 4 E4B (thinking OFF) resposta direta [4.9 GB]"
    echo "  gemma-4-e4b-vision      - Gemma 4 E4B + visão (mmproj) [4.9 GB + 1 GB]"
    echo "  gemma-4-e4b-vision-nothink - Gemma 4 E4B visão (thinking OFF) [4.9 GB + 1 GB]"
    echo "  dolphin3-3b             - Dolphin3.0 Llama3.2-3B Q4_K_M ~20-30 tok/s [1.9 GB]"
    echo "  lfm2.5-1.2b             - Liquid LFM 2.5-1.2B Q8_0 ~25-40 tok/s [1.2 GB]"
    echo "  nanbeige-3b             - Nanbeige4.2-3B Q4_K_M ~18-30 tok/s [2.4 GB]"
    echo "  nanbeige-3b-nothink     - Nanbeige4.2-3B (thinking OFF) resposta direta [2.4 GB]"
    echo "  minicpm5-1b             - MiniCPM5-1B Nemotron-DPO Q8_0 ~30-50 tok/s [1.1 GB]"
    echo "  minicpm5-1b-nothink     - MiniCPM5-1B (thinking OFF) ferramentas diretas [1.1 GB]"
    echo ""
    echo "FORMATOS DE IMAGEM: JPG, PNG, WEBP (PDF/DOCX não suportados)"
    echo "Caminho Windows: /mnt/c/Users/Nome/Pictures/foto.jpg (/mnt/c/Users/denil/...)"
    echo "Caminho Linux:   /home/user/llm-stack/foto.jpg (/home/denilsonbj/...)"
    echo "Uso: Digite /image com o caminho da imagem, depois o prompt do chat."
    echo ""
    echo "DICA: Certifique-se que o modelo está DESLIGADO no manage.sh"
    echo "DICA: Ao encerrar com /exit, o resumo de contexto/memória aparece no final"
    echo "================================================================="
}

if [ -z "$1" ]; then
    show_help
else
    run_chat "$1"
fi
