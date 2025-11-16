import os
import json
from typing import Annotated, Sequence, TypedDict, Literal, List, Dict, Any
from operator import add
import math
from contextvars import ContextVar
from datetime import datetime, timezone
import re
import unicodedata

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dotenv import load_dotenv

# Bibliotecas para MongoDB
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Bibliotecas para busca na internet
import requests
from bs4 import BeautifulSoup
from googlesearch import search

# Biblioteca para embeddings especializados
from sentence_transformers import SentenceTransformer

# Carrega variáveis de ambiente
load_dotenv()

# =============== CONFIGURAÇÃO DO MONGODB ===============

# Conexão com MongoDB (local por padrão, pode ser Atlas)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "qa_system"
COLLECTION_NAME = "resumos_conhecimento"
LOG_COLLECTION_NAME = "acoes_colaborativas"
TOPICS_COLLECTION_NAME = "topicos_conversa"

# Identificadores auxiliares
AGENT_SYSTEM_ID = "agente_colaborativo"
DEFAULT_USER_ID = "anonimo"
DEFAULT_TOPICS = [
    {"name": "Conversa Geral", "chat_id": "geral", "descricao": "Canal livre para dúvidas gerais"},
    {"name": "Análise Técnica", "chat_id": "tecnico", "descricao": "Discussões sobre implementações"},
    {"name": "Brainstorm de Produto", "chat_id": "produto", "descricao": "Ideias de funcionalidades"},
]

SYSTEM_PROMPT = """Você é um assistente educacional colaborativo inteligente.

SEU FLUXO DE TRABALHO (ReACT):

1. Quando receber uma PERGUNTA:
    - PRIMEIRO: Use 'consultar_BD' para buscar resumos similares (busca por embeddings, retorna top 3)
    - LEIA CUIDADOSAMENTE O RESULTADO:
      * Se começar com "✅ ENCONTRADO NO BANCO": use os resumos encontrados e NÃO busque na internet
      * Se começar com "❌ SIMILARIDADE BAIXA" ou "❌ BANCO DE DADOS VAZIO": use 'buscar_referencias' imediatamente
    - SEMPRE responda na MESMA rodada após usar as ferramentas
    - NÃO diga que "vai buscar" sem realmente buscar

2. Quando o usuário pedir para GERAR RESUMO:
    - Use 'gerar_resumo' com o tema informado
    - Analise todo o histórico deste chat
    - Depois salve com 'atualizar_BD'

3. Interação Colaborativa:
    - Trabalhe com o usuário para esclarecer dúvidas
    - Não invente informação; use as ferramentas
    - Confie no threshold de similaridade de 0.6
"""

# Contexto thread-safe para saber quem está interagindo quando ferramentas são chamadas
current_user = ContextVar("current_user", default=DEFAULT_USER_ID)
current_chat = ContextVar("current_chat", default="geral")
_mongo_client: MongoClient | None = None


def _get_collection(collection_name: str):
    global _mongo_client
    try:
        if _mongo_client is None:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            _mongo_client = client
        db = _mongo_client[DATABASE_NAME]
        return db[collection_name]
    except ConnectionFailure as e:
        print(f"Erro ao conectar ao MongoDB ({collection_name}): {e}")
        return None


def get_mongo_collection():
    return _get_collection(COLLECTION_NAME)


def get_logs_collection():
    return _get_collection(LOG_COLLECTION_NAME)


def get_topics_collection():
    return _get_collection(TOPICS_COLLECTION_NAME)


def registrar_acao(acao: str, user_id: str, chat_id: str, conteudo: str, metadata: dict | None = None) -> None:
    """Registra ações colaborativas para auditoria e estatísticas."""
    collection = get_logs_collection()
    if collection is None:
        return

    try:
        collection.insert_one({
            "type": acao,
            "user_id": user_id or DEFAULT_USER_ID,
            "chat_id": chat_id or "geral",
            "content": conteudo,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc)
        })
    except Exception as e:
        print(f"Erro ao registrar ação '{acao}': {e}")


def _slugify_topic(name: str) -> str:
    """Gera um identificador seguro para o tópico."""
    if not name:
        return "topico"
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "topico"

def _normalize_identifier(name: str) -> str:
    """Normaliza o identificador para uso seguro."""
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def _align_tool_call_names(message: AIMessage) -> AIMessage:
    """Ajusta nomes de ferramentas da IA para corresponder aos registrados."""
    if not getattr(message, "tool_calls", None):
        return message

    for call in message.tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not name:
            continue
        if name in VALID_TOOL_NAMES:
            continue
        normalized = _normalize_identifier(name)
        mapped_name = NORMALIZED_TOOL_NAME_MAP.get(normalized)
        if mapped_name:
            call["name"] = mapped_name
    return message


def _parse_textual_tool_payload(content: str | None) -> Dict[str, Any] | None:
    if not content:
        return None
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not name:
        return None
    args = data.get("parameters") or data.get("args") or data.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    payload = {
        "name": name,
        "args": args,
        "id": data.get("id", "manual_tool_call")
    }
    return payload


def _execute_textual_tool_call(payload: Dict[str, Any], message_history: List[BaseMessage]) -> str:
    tool_name = payload["name"]
    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        return f"Ferramenta desconhecida: {tool_name}"
    try:
        tool_output = tool.invoke(payload["args"])
    except Exception as exc:
        tool_output = f"Erro ao executar a ferramenta {tool_name}: {exc}"

    message_history.append(
        ToolMessage(
            content=tool_output,
            name=tool_name,
            tool_call_id=payload.get("id", "manual_tool_call")
        )
    )
    return tool_output


def get_topicos_disponiveis() -> List[Dict[str, Any]]:
    """Retorna a lista de tópicos disponíveis combinando padrões e banco."""
    topicos = {topic["chat_id"]: {**topic, "created_by": "sistema"} for topic in DEFAULT_TOPICS}
    collection = get_topics_collection()
    if collection is None:
        return list(topicos.values())

    try:
        for doc in collection.find().sort("name", 1):
            chat_id = doc.get("chat_id")
            if not chat_id:
                continue
            topicos[chat_id] = {
                "name": doc.get("name", chat_id),
                "chat_id": chat_id,
                "descricao": doc.get("descricao", ""),
                "created_by": doc.get("created_by", DEFAULT_USER_ID),
                "timestamp": doc.get("timestamp")
            }
    except Exception as e:
        print(f"Erro ao buscar tópicos: {e}")

    return list(topicos.values())


def criar_topico(nome: str, user_id: str, descricao: str = "") -> Dict[str, Any]:
    """Cria um novo tópico persistido no MongoDB."""
    nome_limpo = (nome or "").strip()
    if not nome_limpo:
        return {"ok": False, "error": "O nome do tópico não pode ser vazio."}

    collection = get_topics_collection()
    if collection is None:
        return {"ok": False, "error": "Não foi possível conectar ao banco de dados para salvar o tópico."}

    existentes = {topic["chat_id"] for topic in get_topicos_disponiveis()}
    slug_base = _slugify_topic(nome_limpo)
    slug = slug_base
    contador = 2
    while slug in existentes:
        slug = f"{slug_base}-{contador}"
        contador += 1

    documento = {
        "name": nome_limpo,
        "chat_id": slug,
        "descricao": descricao.strip(),
        "created_by": user_id or DEFAULT_USER_ID,
        "timestamp": datetime.now(timezone.utc)
    }

    try:
        collection.insert_one(documento)
        registrar_acao("novo_topico", user_id or DEFAULT_USER_ID, slug, f"Criou o tópico '{nome_limpo}'", {"descricao": descricao})
        return {"ok": True, "topic": documento}
    except Exception as e:
        return {"ok": False, "error": f"Erro ao salvar o tópico: {e}"}

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"🔄 Carregando modelo de embeddings: {EMBEDDING_MODEL_NAME}")
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("✅ Modelo de embeddings pronto!")


def embed_text(text: str) -> List[float]:
    return _embedding_model.encode(text).tolist()

def calcular_similaridade_coseno(vec1: List[float], vec2: List[float]) -> float:
    """Calcula similaridade de cosseno entre dois vetores."""
    if not vec1 or not vec2:
        return 0.0

    limite = min(len(vec1), len(vec2))
    dot_product = sum(vec1[i] * vec2[i] for i in range(limite))
    norm1 = math.sqrt(sum(value * value for value in vec1[:limite]))
    norm2 = math.sqrt(sum(value * value for value in vec2[:limite]))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

# =============== DEFINIÇÃO DO STATE ===============

class AgentState(TypedDict, total=False):
    """Estado do agente contendo mensagens e metadados da sessão."""
    messages: Annotated[Sequence[BaseMessage], add]
    user_id: str
    chat_id: str

# =============== FERRAMENTAS (@tool) ===============

@tool
def consultar_BD(pergunta: str) -> str:
    """
    Consulta a base de dados MongoDB usando embeddings para encontrar os 3 resumos mais similares.
    
    Args:
        pergunta: A pergunta do usuário para buscar no banco
        
    Returns:
        String com os 3 resumos mais similares encontrados (k=3) OU mensagem indicando que não há informação relevante
    """
    user_id = current_user.get()
    chat_id = current_chat.get()
    metadata_base = {"pergunta": pergunta}

    try:
        collection = get_mongo_collection()
        if collection is None:
            mensagem = "Erro: Não foi possível conectar ao banco de dados."
            registrar_acao("ferramenta_consultar_BD", user_id, chat_id, mensagem, {**metadata_base, "status": "erro"})
            return mensagem
        
        # Gera embedding da pergunta
        pergunta_embedding = embed_text(pergunta)
        
        # Busca todos os documentos que têm embeddings
        documentos = list(collection.find({"embedding": {"$exists": True}}))
        
        if not documentos:
            mensagem = "❌ BANCO DE DADOS VAZIO: Não há resumos salvos. Use buscar_referencias para buscar na internet."
            registrar_acao("ferramenta_consultar_BD", user_id, chat_id, mensagem, {**metadata_base, "status": "vazio"})
            return mensagem
        
        # Calcula similaridade para cada documento
        similaridades = []
        for doc in documentos:
            similaridade = calcular_similaridade_coseno(
                pergunta_embedding, 
                doc['embedding']
            )
            similaridades.append((doc, similaridade))
        
        # Ordena por similaridade (maior para menor) e pega top 3
        similaridades.sort(key=lambda x: x[1], reverse=True)
        top_3 = similaridades[:3]
        
        # Verifica o melhor score
        melhor_score = top_3[0][1]
        
        # Threshold de similaridade para Sentence-BERT: 0.6 (60%)
        # Sentence-BERT gera scores bem separados:
        #   - Perguntas relacionadas: ~0.65-0.95 (alta confiança)
        #   - Perguntas não relacionadas: ~0.15-0.45 (baixa confiança)
        # Threshold de 0.6 separa perfeitamente os dois grupos!
        THRESHOLD = 0.6
        
        # Se o melhor resultado tem similaridade < threshold, não é relevante
        if melhor_score < THRESHOLD:
            resultado = f"❌ SIMILARIDADE BAIXA (melhor: {melhor_score:.3f} < {THRESHOLD}):\n\n"
            resultado += "Os resumos no banco NÃO são relevantes para esta pergunta:\n\n"
            
            for idx, (doc, score) in enumerate(top_3, 1):
                resultado += f"{idx}. [Score: {score:.3f}] Tema: {doc.get('tema', 'Sem tema')}\n"
            
            resultado += "\n⚠️ AÇÃO NECESSÁRIA: Use buscar_referencias para buscar informações na internet."
            registrar_acao(
                "ferramenta_consultar_BD",
                user_id,
                chat_id,
                resultado,
                {
                    **metadata_base,
                    "status": "sem_relevancia",
                    "melhor_score": melhor_score,
                    "temas": [doc.get('tema', 'Sem tema') for doc, score in top_3]
                }
            )
            return resultado
        
        # Se chegou aqui, tem resultados relevantes (score >= threshold)
        resultado = f"✅ ENCONTRADO NO BANCO (melhor similaridade: {melhor_score:.3f} >= {THRESHOLD}):\n\n"
        resultado += "Os seguintes resumos SÃO RELEVANTES para responder a pergunta:\n\n"
        
        for idx, (doc, score) in enumerate(top_3, 1):
            if score >= THRESHOLD:  # Só mostra os realmente relevantes
                resultado += f"{idx}. [Similaridade: {score:.3f}] ✅ RELEVANTE\n"
                resultado += f"   Tema: {doc.get('tema', 'Sem tema')}\n"
                resultado += f"   Resumo: {doc['resumo'][:300]}...\n"
                resultado += f"   Fontes: {doc.get('fontes', 'N/A')}\n"
                resultado += f"   {'-'*60}\n\n"
        
        resultado += "\n✅ AÇÃO: Use estas informações para responder ao usuário. NÃO busque na internet."

        registrar_acao(
            "ferramenta_consultar_BD",
            user_id,
            chat_id,
            resultado,
            {
                **metadata_base,
                "status": "relevante",
                "melhor_score": melhor_score,
                "temas": [doc.get('tema', 'Sem tema') for doc, score in top_3 if score >= THRESHOLD]
            }
        )
        return resultado
            
    except Exception as e:
        mensagem = f"Erro ao consultar banco de dados: {str(e)}"
        registrar_acao("ferramenta_consultar_BD", user_id, chat_id, mensagem, {**metadata_base, "status": "erro"})
        return mensagem


def _buscar_referencias_logic(query: str) -> str:
    """Implementação compartilhada da ferramenta de busca na internet."""
    user_id = current_user.get()
    chat_id = current_chat.get()
    metadata_base = {"query": query}

    try:
        search_results = list(search(query, num_results=1, lang="pt"))

        if not search_results:
            mensagem = "Nenhum resultado encontrado na busca."
            registrar_acao("ferramenta_buscar_referencias", user_id, chat_id, mensagem, {**metadata_base, "status": "sem_resultados"})
            return mensagem

        url = search_results[0]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()

        texto = soup.get_text()
        linhas = (line.strip() for line in texto.splitlines())
        chunks = (phrase.strip() for line in linhas for phrase in line.split("  "))
        texto_limpo = ' '.join(chunk for chunk in chunks if chunk)
        texto_resumido = texto_limpo[:2000]
        resultado = f"""🌐 Informações encontradas na internet:

URL: {url}
Conteúdo: {texto_resumido}..."""

        registrar_acao(
            "ferramenta_buscar_referencias",
            user_id,
            chat_id,
            resultado,
            {**metadata_base, "status": "ok", "url": url}
        )
        return resultado

    except Exception as e:
        mensagem = f"Erro ao buscar na internet: {str(e)}"
        registrar_acao("ferramenta_buscar_referencias", user_id, chat_id, mensagem, {**metadata_base, "status": "erro"})
        return mensagem


@tool
def buscar_referencias(query: str) -> str:
    """
    Busca referências na internet usando Google Search e retorna o conteúdo do primeiro link.
    """
    return _buscar_referencias_logic(query)


@tool("buscar_referências")
def buscar_referencias_acentos(query: str) -> str:
    """Alias acentuado para lidar com chamadas do LLM."""
    return _buscar_referencias_logic(query)


@tool
def atualizar_BD(tema: str, resumo: str, fontes: str) -> str:
    """
    Salva um resumo de conhecimento no banco de dados com seu embedding para busca futura.
    
    Args:
        tema: O tema/tópico principal do resumo
        resumo: O resumo consolidado da conversa
        fontes: As fontes de informação utilizadas (URLs, etc)
        
    Returns:
        Mensagem de confirmação
    """
    user_id = current_user.get()
    chat_id = current_chat.get()

    try:
        collection = get_mongo_collection()
        if collection is None:
            mensagem = "Erro: Não foi possível conectar ao banco de dados."
            registrar_acao("resumo", user_id, chat_id, mensagem, {"tema": tema, "status": "erro_conexao"})
            return mensagem
        
        # Gera embedding do resumo para busca futura por similaridade
        resumo_embedding = embed_text(resumo)
        
        # Insere documento com embedding
        documento = {
            "tema": tema,
            "resumo": resumo,
            "fontes": fontes,
            "embedding": resumo_embedding,
            "type": "resumo",
            "user_id": user_id,
            "chat_id": chat_id,
            "timestamp": datetime.now(timezone.utc)
        }
        
        resultado = collection.insert_one(documento)
        registrar_acao(
            "resumo",
            user_id,
            chat_id,
            resumo,
            {"tema": tema, "fontes": fontes, "registro_id": str(resultado.inserted_id)}
        )
        
        return f"✅ Resumo sobre '{tema}' salvo com sucesso! ID: {resultado.inserted_id}"
            
    except Exception as e:
        mensagem = f"Erro ao salvar no banco de dados: {str(e)}"
        registrar_acao("resumo", user_id, chat_id, mensagem, {"tema": tema, "status": "erro"})
        return mensagem


@tool
def gerar_resumo(tema: str) -> str:
    """
    Gera um resumo consolidado da conversa ATUAL usando o LLM.
    Esta ferramenta analisa o histórico de mensagens e cria um resumo inteligente.
    DEPOIS de gerar o resumo, use atualizar_BD para salvá-lo.
    
    Args:
        tema: O tema principal da conversa atual
        
    Returns:
        O resumo gerado pelo LLM
    """
    user_id = current_user.get()
    chat_id = current_chat.get()

    try:
        # Esta função retorna apenas uma flag indicando que deve processar
        # O agente irá pegar o histórico do estado e gerar o resumo
        mensagem = f"GERAR_RESUMO:{tema}"
        registrar_acao("ferramenta_gerar_resumo", user_id, chat_id, mensagem, {"tema": tema})
        return mensagem
        
    except Exception as e:
        mensagem = f"Erro ao gerar resumo: {str(e)}"
        registrar_acao("ferramenta_gerar_resumo", user_id, chat_id, mensagem, {"tema": tema, "status": "erro"})
        return mensagem


# =============== CONFIGURAÇÃO DO MODELO E TOOLS ===============

tools = [consultar_BD, buscar_referencias, buscar_referencias_acentos, atualizar_BD, gerar_resumo]
VALID_TOOL_NAMES = {tool.name for tool in tools}
NORMALIZED_TOOL_NAME_MAP: Dict[str, str] = {}
for tool in tools:
    normalized = _normalize_identifier(tool.name)
    if normalized and normalized not in NORMALIZED_TOOL_NAME_MAP:
        NORMALIZED_TOOL_NAME_MAP[normalized] = tool.name
# llama3.1 e llama3.2 suportam function calling (tools)
model = ChatOllama(model="llama3.1", temperature=0.7).bind_tools(tools)

# =============== NODES DO GRAFO ===============

def agent_node(state: AgentState) -> AgentState:
    """
    Nó principal do agente que processa mensagens e decide ações (padrão ReACT).
    """
    
    # Verifica se precisa gerar resumo da conversa
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and "GERAR_RESUMO:" in last_message.content:
        tema = last_message.content.replace("GERAR_RESUMO:", "")
        
        # Extrai histórico da conversa para resumir
        historico = ""
        fontes_usadas = []
        
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                historico += f"\n👤 Usuário: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                historico += f"\n🤖 Assistente: {msg.content}\n"
            elif isinstance(msg, ToolMessage):
                # Extrai fontes (URLs) das buscas
                if "URL:" in msg.content:
                    import re
                    urls = re.findall(r'URL: (https?://[^\s]+)', msg.content)
                    fontes_usadas.extend(urls)
        
        # Gera resumo com LLM
        resumo_prompt = f"""Analise a conversa abaixo sobre o tema "{tema}" e crie um resumo consolidado.

CONVERSA:
{historico}

Crie um resumo estruturado contendo:
1. Principais perguntas feitas
2. Respostas e conhecimentos adquiridos  
3. Conceitos-chave explicados
4. Conclusões importantes

Seja claro, objetivo e organize o conhecimento de forma útil para consultas futuras."""

        resumo_model = ChatOllama(model="llama3.1", temperature=0.3)
        resumo_response = resumo_model.invoke([HumanMessage(content=resumo_prompt)])
        resumo_gerado = resumo_response.content
        
        # Formata fontes
        fontes_str = ", ".join(set(fontes_usadas)) if fontes_usadas else "Conversa com LLM"
        
        # Retorna resposta com o resumo e instrução para salvar
        resposta_final = f"""📝 RESUMO GERADO DA CONVERSA:

{resumo_gerado}

---
Fontes utilizadas: {fontes_str}

Agora vou salvar este resumo no banco de dados para consultas futuras..."""
        
        # Cria mensagem AI com tool_call para salvar
        ai_msg = AIMessage(
            content=resposta_final,
            tool_calls=[{
                "name": "atualizar_BD",
                "args": {
                    "tema": tema,
                    "resumo": resumo_gerado,
                    "fontes": fontes_str
                },
                "id": "resumo_save",
                "type": "tool_call"
            }]
        )
        return {"messages": [ai_msg]}
    
    # Comportamento normal do agente
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    response = _align_tool_call_names(response)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Decide se deve chamar ferramentas ou finalizar.
    """
    last_message = state["messages"][-1]
    
    # Se a última mensagem tem chamadas de ferramentas, executá-las
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "end"


# =============== CONSTRUÇÃO DO GRAFO ===============

def create_agent_graph():
    """Cria e retorna o grafo compilado do agente ReACT."""
    
    # Cria o grafo
    workflow = StateGraph(AgentState)
    
    # Adiciona nós
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools=tools))
    
    # Define o ponto de entrada
    workflow.set_entry_point("agent")
    
    # Adiciona arestas condicionais
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Após executar ferramentas, volta para o agente
    workflow.add_edge("tools", "agent")
    
    # Compila o grafo
    return workflow.compile()


# Cria a instância do agente
agent = create_agent_graph()


# =============== FUNÇÕES AUXILIARES ===============


def _invoke_with_context(state: AgentState, user_id: str, chat_id: str):
    token_user = current_user.set(user_id)
    token_chat = current_chat.set(chat_id)
    try:
        return agent.invoke(state)
    finally:
        current_user.reset(token_user)
        current_chat.reset(token_chat)

def run_agent(
    user_input: str,
    conversation_history: list = None,
    user_id: str = DEFAULT_USER_ID,
    chat_id: str = "geral"
) -> dict:
    """Executa o agente registrando as interações por usuário e chat."""
    registrar_acao("pergunta", user_id, chat_id, user_input)

    if conversation_history is None:
        conversation_history = []

    conversation_history.append(HumanMessage(content=user_input))

    initial_state = {
        "messages": conversation_history,
        "user_id": user_id,
        "chat_id": chat_id
    }

    result = _invoke_with_context(initial_state, user_id, chat_id)

    conversation_history = list(result["messages"])
    resposta_final = conversation_history[-1]

    textual_tool = None
    if isinstance(resposta_final, AIMessage) and not getattr(resposta_final, "tool_calls", None):
        textual_tool = _parse_textual_tool_payload(resposta_final.content)

    if textual_tool:
        _execute_textual_tool_call(textual_tool, conversation_history)
        secondary_state = {
            "messages": conversation_history,
            "user_id": user_id,
            "chat_id": chat_id
        }

        result = _invoke_with_context(secondary_state, user_id, chat_id)

        conversation_history = list(result["messages"])
        resposta_final = conversation_history[-1]
    resposta_texto = resposta_final.content if hasattr(resposta_final, "content") else str(resposta_final)
    registrar_acao(
        "resposta",
        AGENT_SYSTEM_ID,
        chat_id,
        resposta_texto,
        {"destinatario": user_id}
    )
    
    return {
        "response": resposta_texto,
        "history": conversation_history
    }


# =============== FUNÇÕES DE ESTATÍSTICAS E HISTÓRICO ===============

def get_estatisticas_usuarios() -> dict:
    """Calcula estatísticas de contribuições por usuário a partir dos logs."""
    collection = get_logs_collection()
    if collection is None:
        return {}

    pipeline = [
        {
            "$match": {
                "user_id": {"$nin": [None, "", AGENT_SYSTEM_ID]}
            }
        },
        {
            "$group": {
                "_id": "$user_id",
                "perguntas": {
                    "$sum": {"$cond": [{"$eq": ["$type", "pergunta"]}, 1, 0]}
                },
                "resumos": {
                    "$sum": {"$cond": [{"$eq": ["$type", "resumo"]}, 1, 0]}
                },
                "ferramentas": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$type", [
                                "ferramenta_consultar_BD",
                                "ferramenta_buscar_referencias",
                                "ferramenta_gerar_resumo"
                            ]]},
                            1,
                            0
                        ]
                    }
                },
                "total": {"$sum": 1}
            }
        },
        {
            "$sort": {"total": -1}
        }
    ]

    stats = {}
    try:
        for doc in collection.aggregate(pipeline):
            stats[doc["_id"]] = {
                "perguntas": doc.get("perguntas", 0),
                "resumos": doc.get("resumos", 0),
                "ferramentas": doc.get("ferramentas", 0),
                "total": doc.get("total", 0)
            }
    except Exception as e:
        print(f"Erro ao buscar estatísticas: {e}")

    return stats


def get_historico_colaborativo(limit: int = 10) -> list:
    """Retorna as últimas ações registradas no sistema colaborativo."""
    collection = get_logs_collection()
    if collection is None:
        return []

    try:
        registros = []
        for doc in collection.find().sort("timestamp", -1).limit(limit):
            registro = {
                "id": str(doc.get("_id")),
                "type": doc.get("type"),
                "user_id": doc.get("user_id", DEFAULT_USER_ID),
                "chat_id": doc.get("chat_id", "geral"),
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "timestamp": doc.get("timestamp")
            }
            registros.append(registro)
        return registros
    except Exception as e:
        print(f"Erro ao buscar histórico: {e}")
        return []