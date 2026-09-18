#!/usr/bin/env bash

BASE=~/llm-stack
SERVER="$BASE/llama.cpp/build/CPU/bin/llama-server"
PRESET="$BASE/external/webui-models.ini"
PORT=8080
SESSION="webui"
LOG="$BASE/logs/webui.log"

# ============================================================
# WebUI do llama.cpp — interface gráfica de chat (ChatGPT-like)
# Roda o llama-server em MODO ROUTER: um único processo serve
# TODOS os modelos do catálogo (external/webui-models.ini),
# carregando cada um sob demanda conforme selecionado na UI.
#
# Diferente do manage.sh (1 processo tmux por modelo, portas
# 80xx), aqui há 1 sessão única "webui" na porta 8080.
#
# IMPORTANTE: a RAM é compartilhada com os servidores do
# manage.sh. Antes de testar modelos pesados (Bonsai 27B),
# pare os servidores conflitantes: ./scripts/manage.sh stop-all
# ============================================================

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

mkdir -p "$BASE/logs"

start_webui() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "⚠️  A WebUI já está rodando na porta ${PORT}."
        echo "   Abra: http://localhost:${PORT}"
        echo "   Para reiniciar: ./scripts/webui.sh stop && ./scripts/webui.sh start"
        return 1
    fi

    if [ ! -x "$SERVER" ]; then
        echo "❌ llama-server não encontrado em: $SERVER"
        echo "   Compile o llama.cpp primeiro (ver README, seção 2.2)."
        return 1
    fi

    if [ ! -f "$PRESET" ]; then
        echo "❌ Catálogo de modelos não encontrado: $PRESET"
        return 1
    fi

    echo "🎨 Iniciando WebUI (llama-server modo router) na porta ${PORT}..."
    tmux new-session -d -s "$SESSION" \
        "$SERVER --models-preset $PRESET --models-max 2 --host 0.0.0.0 --port $PORT > $LOG 2>&1"

    # Aguarda o servidor abrir a porta
    for _ in $(seq 1 30); do
        if curl -s -o /dev/null --max-time 1 "http://localhost:${PORT}/models"; then
            echo "✅ WebUI ativa!"
            echo ""
            echo "   🌐 Interface : http://localhost:${PORT}"
            echo "      (no navegador do Windows: http://localhost:${PORT})"
            echo "   📋 Modelos   : catálogo em external/webui-models.ini"
            echo "   📝 Log       : $LOG  (ou: tmux attach -t $SESSION)"
            echo ""
            echo "   Dica: o modelo só é carregado na RAM quando você envia"
            echo "   a primeira mensagem com ele selecionado (autoload)."
            return 0
        fi
        sleep 0.5
    done

    echo "❌ O servidor não respondeu a tempo. Veja o log:"
    tail -20 "$LOG"
    tmux kill-session -t "$SESSION" 2>/dev/null
    return 1
}

stop_webui() {
    if tmux kill-session -t "$SESSION" 2>/dev/null; then
        echo "🛑 WebUI parada (porta ${PORT} liberada, RAM liberada)."
    else
        echo "⚠️  A WebUI não estava rodando."
    fi
}

status_webui() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "🎨 WebUI RODANDO na porta ${PORT} → http://localhost:${PORT}"
        echo ""
        echo "Modelos disponíveis no catálogo (status on-demand):"
        curl -s --max-time 2 "http://localhost:${PORT}/models" \
            | grep -o '"id":"[^"]*"' | sed 's/"id":"/  • /;s/"//' || echo "  (sem resposta do servidor)"
    else
        echo "💤 WebUI parada. Inicie com: ./scripts/webui.sh start"
    fi
}

logs_webui() {
    if [ -f "$LOG" ]; then
        tail -f "$LOG"
    else
        echo "⚠️  Sem log ainda. Inicie com: ./scripts/webui.sh start"
    fi
}

case "$1" in
    start)  start_webui ;;
    stop)   stop_webui ;;
    restart) stop_webui; start_webui ;;
    status) status_webui ;;
    logs)   logs_webui ;;
    *)
        echo "================================================================="
        echo "   🎨 LLM STACK — WEBUI DO LLAMA.CPP (interface gráfica) 🎨"
        echo "================================================================="
        echo "Uso: ./scripts/webui.sh [comando]"
        echo ""
        echo "COMANDOS:"
        echo "  start    - Inicia a WebUI (porta ${PORT}) em sessão tmux '$SESSION'"
        echo "  stop     - Para a WebUI e libera a RAM"
        echo "  restart  - Reinicia a WebUI"
        echo "  status   - Mostra se está ativa e lista os modelos do catálogo"
        echo "  logs     - Mostra o log do servidor em tempo real (Ctrl+C para sair)"
        echo ""
        echo "DEPOIS DE INICIAR:"
        echo "  Abra http://localhost:${PORT} no navegador (Chrome/Edge/Firefox)."
        echo "  Selecione o modelo no topo da interface e converse."
        echo ""
        echo "O QUE A INTERFACE TEM:"
        echo "  • Conversas com histórico salvo no navegador (sidebar)"
        echo "  • Seleção de modelo do catálogo (external/webui-models.ini)"
        echo "  • Upload de imagens nos modelos -vision (Gemma 4 / Qwen-VL)"
        echo "  • Anexos de texto/CSV, renderização Markdown e blocos de código"
        echo "  • Controles: temperatura, top_p, max_tokens, system prompt"
        echo "  • Modo de raciocínio (think) nos modelos que suportam"
        echo "  • JSON schema (respostas estruturadas) e grammar/EBNF"
        echo "  • Tema claro/escuro"
        echo ""
        echo "ATENÇÃO:"
        echo "  • A RAM é compartilhada com o manage.sh (portas 80xx)."
        echo "  • Bonsai 27B usa ~5-8 GB: feche os outros antes."
        echo "  • Sem rede: tudo roda local, nada sai da sua máquina."
        echo "================================================================="
        ;;
esac
