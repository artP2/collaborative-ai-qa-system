import os
from typing import Annotated, Sequence, TypedDict, Literal, List
from operator import add
import numpy as np

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

def get_mongo_collection():
    """Retorna a coleção do MongoDB."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Testa a conexão
        client.admin.command('ping')
        db = client[DATABASE_NAME]
        return db[COLLECTION_NAME]
    except ConnectionFailure as e:
        print(f"Erro ao conectar ao MongoDB: {e}")
        return None

# =============== EMBEDDINGS ===============

# Classe wrapper para Sentence-BERT (compatível com a interface anterior)
class SentenceBertEmbeddings:
    """
    Wrapper para Sentence-BERT que fornece embeddings de alta qualidade.
    
    Vantagens sobre OllamaEmbeddings:
    - Scores de similaridade mais confiáveis (0.6-0.9 para relacionados)
    - Modelo especializado em embeddings (não generalista como llama3.1)
    - Mais rápido e menor (420MB vs 4.5GB)
    - Suporta português através do modelo multilingual
    """
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Inicializa o modelo de embeddings.
        
        Modelos recomendados:
        - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual, balanceado (420MB)
        - 'paraphrase-multilingual-mpnet-base-v2': Multilingual, mais preciso (1GB)
        - 'all-MiniLM-L6-v2': Inglês, mais rápido (80MB)
        """
        print(f"🔄 Carregando modelo de embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Modelo carregado com sucesso!")
    
    def embed_query(self, text: str) -> list:
        """Gera embedding para um texto (compatível com OllamaEmbeddings)."""
        return self.model.encode(text).tolist()

# Inicializa o modelo de embeddings especializado
# Nota: Primeira execução baixa o modelo (~420MB)
embeddings_model = SentenceBertEmbeddings()

def calcular_similaridade_coseno(vec1: List[float], vec2: List[float]) -> float:
    """Calcula similaridade de cosseno entre dois vetores."""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

# =============== DEFINIÇÃO DO STATE ===============

class AgentState(TypedDict):
    """Estado do agente contendo mensagens."""
    messages: Annotated[Sequence[BaseMessage], add]

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
    try:
        collection = get_mongo_collection()
        if collection is None:
            return "Erro: Não foi possível conectar ao banco de dados."
        
        # Gera embedding da pergunta
        pergunta_embedding = embeddings_model.embed_query(pergunta)
        
        # Busca todos os documentos que têm embeddings
        documentos = list(collection.find({"embedding": {"$exists": True}}))
        
        if not documentos:
            return "❌ BANCO DE DADOS VAZIO: Não há resumos salvos. Use buscar_referencias para buscar na internet."
        
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
        return resultado
            
    except Exception as e:
        return f"Erro ao consultar banco de dados: {str(e)}"


@tool
def buscar_referencias(query: str) -> str:
    """
    Busca referências na internet usando Google Search e retorna o conteúdo do primeiro link.
    
    Args:
        query: Consulta para buscar no Google
        
    Returns:
        String com o conteúdo extraído do primeiro resultado
    """
    try:
        # Busca no Google (pega apenas o primeiro resultado)
        search_results = list(search(query, num_results=1, lang="pt"))
        
        if not search_results:
            return "Nenhum resultado encontrado na busca."
        
        url = search_results[0]
        
        # Faz requisição para o primeiro link
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Extrai o texto da página
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts e styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Pega o texto
        texto = soup.get_text()
        
        # Limpa o texto (remove linhas vazias e espaços extras)
        linhas = (line.strip() for line in texto.splitlines())
        chunks = (phrase.strip() for line in linhas for phrase in line.split("  "))
        texto_limpo = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limita o tamanho (primeiros 2000 caracteres)
        texto_resumido = texto_limpo[:2000]
        
        return f"""🌐 Informações encontradas na internet:

URL: {url}
Conteúdo: {texto_resumido}..."""
        
    except Exception as e:
        return f"Erro ao buscar na internet: {str(e)}"


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
    try:
        collection = get_mongo_collection()
        if collection is None:
            return "Erro: Não foi possível conectar ao banco de dados."
        
        # Gera embedding do resumo para busca futura por similaridade
        resumo_embedding = embeddings_model.embed_query(resumo)
        
        # Insere documento com embedding
        documento = {
            "tema": tema,
            "resumo": resumo,
            "fontes": fontes,
            "embedding": resumo_embedding
        }
        
        resultado = collection.insert_one(documento)
        
        return f"✅ Resumo sobre '{tema}' salvo com sucesso! ID: {resultado.inserted_id}"
            
    except Exception as e:
        return f"Erro ao salvar no banco de dados: {str(e)}"


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
    try:
        # Esta função retorna apenas uma flag indicando que deve processar
        # O agente irá pegar o histórico do estado e gerar o resumo
        return f"GERAR_RESUMO:{tema}"
        
    except Exception as e:
        return f"Erro ao gerar resumo: {str(e)}"


# =============== CONFIGURAÇÃO DO MODELO E TOOLS ===============

tools = [consultar_BD, buscar_referencias, atualizar_BD, gerar_resumo]
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
    system_prompt = SystemMessage(content="""Você é um assistente educacional colaborativo inteligente.

SEU FLUXO DE TRABALHO (ReACT):

1. Quando receber uma PERGUNTA:
   - PRIMEIRO: Use 'consultar_BD' para buscar resumos similares (busca por embeddings, retorna top 3)
   
   - LEIA CUIDADOSAMENTE O RESULTADO:
     * Se começar com "✅ ENCONTRADO NO BANCO":
       → Os resumos SÃO RELEVANTES (similaridade >= 0.6)
       → RESPONDA IMEDIATAMENTE usando essas informações
       → Informe ao usuário que encontrou no banco de dados
       → NÃO busque na internet
     
     * Se começar com "❌ SIMILARIDADE BAIXA" ou "❌ BANCO DE DADOS VAZIO":
       → Os resumos NÃO são relevantes (similaridade < 0.6) ou não existem
       → Use 'buscar_referencias' IMEDIATAMENTE para buscar na internet
       → Após receber resultado, RESPONDA ao usuário
   
   - SEMPRE responda na MESMA rodada após usar as ferramentas
   - NÃO diga que "vai buscar" sem realmente buscar

2. Quando o usuário pedir para GERAR RESUMO:
   - Use a ferramenta 'gerar_resumo' informando o tema da conversa
   - O sistema irá analisar TODO o histórico desta conversa
   - Gerar um resumo consolidado com LLM
   - Automaticamente salvar no banco com embedding

3. Interação Colaborativa:
   - Trabalhe com o usuário para esclarecer dúvidas
   - Refine respostas conforme feedback
   - Mantenha conversas naturais e educativas

REGRAS CRÍTICAS:
- A ferramenta consultar_BD retorna ✅ quando encontra (score >= 0.6) ou ❌ quando não encontra (score < 0.6)
- CONFIE nos símbolos ✅ e ❌ - Sentence-BERT é MUITO preciso!
- Se viu ❌, você DEVE usar buscar_referencias
- Se viu ✅, você NÃO DEVE buscar na internet, apenas responda
- NUNCA diga "vou buscar" e depois não busque
- consultar_BD usa EMBEDDINGS (similaridade semântica, não busca exata de palavras)
- Não invente informações - use as ferramentas!""")
    
    response = model.invoke([system_prompt] + state["messages"])
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


# =============== FUNÇÃO AUXILIAR ===============

def run_agent(user_input: str, conversation_history: list = None) -> dict:
    """
    Executa o agente com uma entrada do usuário.
    
    Args:
        user_input: Mensagem do usuário
        conversation_history: Histórico de mensagens anteriores
        
    Returns:
        Dict com a resposta e o histórico atualizado
    """
    if conversation_history is None:
        conversation_history = []
    
    # Adiciona a mensagem do usuário
    conversation_history.append(HumanMessage(content=user_input))
    
    # Cria o estado inicial
    initial_state = {
        "messages": conversation_history
    }
    
    # Executa o agente
    result = agent.invoke(initial_state)
    
    # Retorna resultado e histórico atualizado
    return {
        "response": result["messages"][-1].content,
        "history": result["messages"]
    }