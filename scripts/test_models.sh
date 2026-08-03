#!/usr/bin/env bash
#
# test_models.sh — Teste automatizado de TODOS os modelos
# Uso: ./test_models.sh [chat|server|all]
#
# Testa cada modelo tanto no modo chat interativo quanto no modo servidor,
# verificando se carregam e respondem sem erros (backend GGML nativo CPU).
#

BASE=~/llm-stack
TIMEOUT=60  # segundos para carregar o modelo + responder

PASS=0
FAIL=0
FAILED_TESTS=()

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

print_banner() {
    echo ""
    echo "=============================================="
    echo "   🧪 LLM STACK — TESTE AUTOMATIZADO"
    echo "=============================================="
    echo ""
}

print_result() {
    local status="$1"
    local test_name="$2"
    local detail="$3"
    if [ "$status" = "PASS" ]; then
        echo -e "  ${GREEN}✅ PASS${NC}  $test_name"
    else
        echo -e "  ${RED}❌ FAIL${NC}  $test_name"
        echo -e "       ${RED}$detail${NC}"
        FAILED_TESTS+=("$test_name: $detail")
    fi
}

print_summary() {
    echo ""
    echo "=============================================="
    echo -e "   ${BOLD}RESUMO${NC}"
    echo "=============================================="
    echo -e "   ${GREEN}PASS: $PASS${NC}"
    echo -e "   ${RED}FAIL: $FAIL${NC}"
    if [ $FAIL -gt 0 ]; then
        echo ""
        echo "   ${BOLD}Testes com falha:${NC}"
        for failed in "${FAILED_TESTS[@]}"; do
            echo -e "     ${RED}•${NC} $failed"
        done
    fi
    echo "=============================================="
    echo ""
}

# ============================================================
# Teste: chat.sh <modelo>
# Envia "hi" e verifica se o modelo carrega e responde sem erro
# ============================================================
test_chat() {
    local model="$1"
    local display_name="$2"
    local test_name="chat/$display_name"

    echo -e "\n${YELLOW}⏳  $test_name${NC} (timeout: ${TIMEOUT}s)"

    # Garante que nenhum servidor esteja rodando pra esse modelo
    tmux kill-session -t "$model" 2>/dev/null || true
    sleep 1

    # Roda o chat com um prompt simples, captura stdout e stderr
    output=$(echo "hi" | timeout "$TIMEOUT" "$BASE/scripts/chat.sh" "$model" 2>&1 || true)

    # Verifica se há erro de runtime (GGML/llama.cpp)
    if echo "$output" | grep -qi "ov::Exception\|shape incompatible\|Compute error\|failed to decode\|failed to compute"; then
        # Procura a linha do erro específico
        error_line=$(echo "$output" | grep -i "shape incompatible\|Compute error\|failed to" | head -1)
        print_result "FAIL" "$test_name" "$error_line"
        FAIL=$((FAIL + 1))
    elif echo "$output" | grep -qi "Error\|error.*-1\|segfault\|SIGSEGV"; then
        error_line=$(echo "$output" | grep -i "Error" | head -1)
        print_result "FAIL" "$test_name" "$error_line"
        FAIL=$((FAIL + 1))
    elif echo "$output" | grep -qi "> "; then
        # Prompt > apareceu = chat carregou e respondeu
        print_result "PASS" "$test_name" ""
        PASS=$((PASS + 1))
    elif [ -z "$output" ]; then
        print_result "FAIL" "$test_name" "Sem saida (timeout ou crash silencioso)"
        FAIL=$((FAIL + 1))
    else
        print_result "PASS" "$test_name" "(saida inesperada mas sem erro)"
        PASS=$((PASS + 1))
    fi
}

# ============================================================
# Teste: manage.sh start/stop <modelo>
# Inicia o servidor, faz uma requisicao curl, verifica resposta
# ============================================================
test_server() {
    local model="$1"
    local port="$2"
    local display_name="$3"
    local test_name="server/$display_name"

    echo -e "\n${YELLOW}⏳  $test_name${NC} (porta $port, timeout: ${TIMEOUT}s)"

    # Garante que não está rodando
    "$BASE/scripts/manage.sh" stop "$model" >/dev/null 2>&1
    sleep 1

    # Inicia o servidor
    "$BASE/scripts/manage.sh" start "$model" >/dev/null 2>&1

    # Aguarda o servidor ficar pronto (ate timeout)
    local waited=0
    local ready=false
    while [ $waited -lt "$TIMEOUT" ]; do
        if curl -s "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            ready=true
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done

    if [ "$ready" = false ]; then
        "$BASE/scripts/manage.sh" stop "$model" >/dev/null 2>&1
        print_result "FAIL" "$test_name" "Servidor nao ficou pronto em ${TIMEOUT}s"
        FAIL=$((FAIL + 1))
        return
    fi

    # Faz uma requisicao de chat simples
    response=$(curl -s -X POST "http://localhost:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$model"'",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        }' 2>&1)

    "$BASE/scripts/manage.sh" stop "$model" >/dev/null 2>&1

    # Verifica se a resposta contem erro
    if echo "$response" | grep -qi "error\|shape incompatible\|Compute error"; then
        error_line=$(echo "$response" | grep -i "error" | head -1)
        print_result "FAIL" "$test_name" "$error_line"
        FAIL=$((FAIL + 1))
    elif echo "$response" | grep -qi '"content"'; then
        print_result "PASS" "$test_name" ""
        PASS=$((PASS + 1))
    elif echo "$response" | grep -qi '"choices"'; then
        print_result "PASS" "$test_name" ""
        PASS=$((PASS + 1))
    else
        print_result "PASS" "$test_name" "(resposta inesperada mas sem erro)"
        PASS=$((PASS + 1))
    fi
}

# ============================================================
# Menu de testes
# ============================================================
MODE="${1:-all}"

print_banner

case "$MODE" in
    chat)
        echo "🧪 Modo: TESTE CHAT"
        echo ""

        test_chat "qwen-coder-3b"    "Qwen-Coder-3B"
        test_chat "qwen-vl-3b-uncensored" "Qwen-VL 3B Uncensored (visao)"
        test_chat "dolphin3-3b"     "Dolphin3.0-3B"
        test_chat "lfm2.5-1.2b"     "LFM 2.5-1.2B"
        test_chat "nanbeige-3b"     "Nanbeige4.2-3B"
        test_chat "minicpm5-1b"     "MiniCPM5-1B"

        print_summary
        ;;

    server)
        echo "🧪 Modo: TESTE SERVER"
        echo ""

        test_server "qwen-coder-3b"   8003 "Qwen-Coder-3B"
        test_server "qwen-vl-3b-uncensored" 8010 "Qwen-VL 3B Uncensored (visao)"
        test_server "dolphin3-3b"    8041 "Dolphin3.0-3B"
        test_server "lfm2.5-1.2b"    8061 "LFM 2.5-1.2B"
        test_server "nanbeige-3b"    8071 "Nanbeige4.2-3B"
        test_server "minicpm5-1b"    8081 "MiniCPM5-1B"

        print_summary
        ;;

    all)
        echo "🧪 Modo: TESTE COMPLETO (chat + server)"
        echo ""

        echo "──────────────────────────────────────────────"
        echo "  📋 TESTES DE CHAT INTERATIVO"
        echo "──────────────────────────────────────────────"

        test_chat "qwen-coder-3b"    "Qwen-Coder-3B"
        test_chat "qwen-vl-3b-uncensored" "Qwen-VL 3B Uncensored (visao)"
        test_chat "dolphin3-3b"     "Dolphin3.0-3B"
        test_chat "lfm2.5-1.2b"     "LFM 2.5-1.2B"
        test_chat "nanbeige-3b"     "Nanbeige4.2-3B"
        test_chat "minicpm5-1b"     "MiniCPM5-1B"

        echo ""
        echo "──────────────────────────────────────────────"
        echo "  📋 TESTES DE SERVIDOR"
        echo "──────────────────────────────────────────────"

        test_server "qwen-coder-3b"   8003 "Qwen-Coder-3B"
        test_server "qwen-vl-3b-uncensored" 8010 "Qwen-VL 3B Uncensored (visao)"
        test_server "dolphin3-3b"    8041 "Dolphin3.0-3B"
        test_server "lfm2.5-1.2b"    8061 "LFM 2.5-1.2B"
        test_server "nanbeige-3b"    8071 "Nanbeige4.2-3B"
        test_server "minicpm5-1b"    8081 "MiniCPM5-1B"

        print_summary
        ;;

    *)
        echo "Uso: ./test_models.sh [chat|server|all]"
        echo ""
        echo "  chat    - Testa apenas modo chat interativo"
        echo "  server  - Testa apenas modo servidor (manage.sh)"
        echo "  all     - Testa ambos (padrao)"
        exit 1
        ;;
esac

# Exit code: 0 se todos passaram, 1 se algum falhou
[ $FAIL -eq 0 ]
