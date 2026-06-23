#!/usr/bin/env bash

BASE=~/llm-stack
CLI="$BASE/llama.cpp/build/bin/llama-cli"

# Configurações de Hardware (Idênticas ao manage.sh para consistência)
THREADS=8
CTX=4096
BATCH=256

function run_chat() {
    # Verificar se o servidor desse modelo já está rodando no TMUX para evitar crash de RAM
    if tmux ls 2>/dev/null | grep -q "$1"; then
        echo "⚠️ ERRO: O servidor '$1' está ativo no manage.sh!"
        echo "Pare o servidor primeiro com: ./manage.sh stop $1"
        exit 1
    fi

    case $1 in
        qwen-text)
            $CLI -m $BASE/models/text/qwen2.5-1.5b-instruct-q4_k_m.gguf \
                 -t $THREADS -c $CTX -b $BATCH -cnv --log-disable
            ;;
        gemma2)
            $CLI -m $BASE/models/text/gemma-2-2b-it-abliterated-q4_k_m.gguf \
                 -t $THREADS -c $CTX -b $BATCH -cnv --log-disable
            ;;
        qwen-coder)
            $CLI -m $BASE/models/code/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
                 -t $THREADS -c $CTX -b $BATCH -cnv --log-disable
            ;;
        ministral-agent)
            $CLI -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf \
                 -t $THREADS -c $CTX -cnv --log-disable
            ;;
        ministral-vision)
            $CLI -m $BASE/models/code/ministral-3-3b-instruct-2512-q4_k_m.gguf \
                 --mmproj $BASE/models/code/ministral-3-3b-instruct-2512-mmproj-f16.gguf \
                 -t $THREADS -c $CTX -cnv --log-disable
            ;;
        vision)
            # CLI de visão usa o mmproj (não usa o servidor Python)
            $CLI -m $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it-iq4_xs.gguf \
                 --mmproj $BASE/models/vision/qwen2.5-vl-3b-abliterated-caption-it.mmproj-Q8_0.gguf \
                 -t $THREADS -c $CTX -cnv --log-disable
            ;;
        *)
            show_help
            ;;
    esac
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