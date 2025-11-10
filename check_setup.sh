#!/bin/bash

# Script de verificação do ambiente Ollama
# Execute: bash check_setup.sh

echo "=================================="
echo "🔍 VERIFICAÇÃO DO AMBIENTE OLLAMA"
echo "=================================="
echo ""

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Contador de problemas
ISSUES=0

# 1. Verifica Ollama instalado
echo -n "1. Ollama instalado............ "
if command -v ollama &> /dev/null; then
    VERSION=$(ollama --version 2>&1 | head -n 1)
    echo -e "${GREEN}✅ $VERSION${NC}"
else
    echo -e "${RED}❌ Não encontrado${NC}"
    echo "   Instale: curl -fsSL https://ollama.com/install.sh | sh"
    ISSUES=$((ISSUES + 1))
fi

# 2. Verifica servidor Ollama
echo -n "2. Servidor rodando............ "
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo -e "${GREEN}✅ http://localhost:11434${NC}"
else
    echo -e "${RED}❌ Não está rodando${NC}"
    echo "   Execute em outro terminal: ollama serve"
    ISSUES=$((ISSUES + 1))
fi

# 3. Verifica modelo llama3
echo -n "3. Modelo llama3............... "
if ollama list 2>/dev/null | grep -q "llama3"; then
    SIZE=$(ollama list 2>/dev/null | grep "llama3" | awk '{print $2}')
    echo -e "${GREEN}✅ Instalado ($SIZE)${NC}"
else
    echo -e "${RED}❌ Não encontrado${NC}"
    echo "   Baixe: ollama pull llama3"
    ISSUES=$((ISSUES + 1))
fi

# 4. Verifica MongoDB
echo -n "4. MongoDB rodando............. "
if mongosh --eval "db.adminCommand('ping')" --quiet &> /dev/null || mongo --eval "db.adminCommand('ping')" --quiet &> /dev/null; then
    echo -e "${GREEN}✅ Conectado${NC}"
else
    echo -e "${RED}❌ Não está rodando${NC}"
    echo "   Execute em outro terminal: mongod"
    ISSUES=$((ISSUES + 1))
fi

# 5. Verifica Python
echo -n "5. Python 3.x.................. "
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✅ $PY_VERSION${NC}"
else
    echo -e "${RED}❌ Não encontrado${NC}"
    ISSUES=$((ISSUES + 1))
fi

# 6. Verifica dependências Python
echo -n "6. Dependências Python......... "
if python3 -c "import langchain_ollama" 2>/dev/null; then
    echo -e "${GREEN}✅ Instaladas${NC}"
else
    echo -e "${YELLOW}⚠️  Faltando${NC}"
    echo "   Instale: pip install -r requirements.txt"
    ISSUES=$((ISSUES + 1))
fi

# 7. Verifica .env
echo -n "7. Arquivo .env................ "
if [ -f .env ]; then
    echo -e "${GREEN}✅ Existe${NC}"
else
    echo -e "${YELLOW}⚠️  Não encontrado${NC}"
    echo "   Crie: cp .env.example .env"
fi

echo ""
echo "=================================="

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ TUDO PRONTO!${NC}"
    echo ""
    echo "Execute o projeto:"
    echo "  python3 exemplo_uso.py"
else
    echo -e "${RED}❌ $ISSUES problema(s) encontrado(s)${NC}"
    echo ""
    echo "Consulte: OLLAMA_SETUP.md"
fi

echo "=================================="
