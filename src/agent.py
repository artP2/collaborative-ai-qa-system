import os
from typing import Annotated, Sequence, TypedDict, Callable, Optional

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from operator import add as add_messages
from dotenv import load_dotenv

@tool
def atualizar_BD():
    """Atualiza a base de dados com novos resumos gerados pelo LLM."""
    pass

@tool 
def consultar_BD(query: str) -> str:
    """Consulta a base de dados e retorna a resposta."""
    pass

@tool
def gerar_resumo(texto: str) -> str:
    """Gera um resumo da conversa com o LLM."""
    pass

@tool
def buscar_referencias(query: str) -> str:    
    """Busca referencias na internet relacionadas a uma consulta."""
    pass

tools = [atualizar_BD, consultar_BD, gerar_resumo, buscar_referencias]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    """Calls the LLM passing instructions"""
    msg = """
        Voce e um agente assistente de educacao para humanos, sua tarefa e auxiliar os usuarios com suas perguntas e duvidas.
        Responda com a melhor das suas habilidades, nao seja grosseiro.
        Em hiposetese de duvida, utilize as ferramentas disponiveis para buscar informacoes adicionais.
        Nao invente respostas, se nao souber a resposta, se nao souber responda que nao sabe.
    """
    system_prompt = SystemMessage(content=msg)
    # passamos a system message seguida do estado que vai conter as human messages
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages" : [response]}

def should_continue(state: AgentState) -> AgentState:
    """Check if we should call the model node again or end the graph"""
    last_message = state["messages"][-1]
    # - messages é uma subclasse de BaseMessage que contem informacao sobre o uso de tools
    #   isso é, quando o modelo decidiu usar uma das tools disponívels
    # - atributo .tool_calls que rastreia que tools foram chamadas como parte da geracao
    #   da resposta final
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)

tools_node = ToolNode(tools=tools)
graph.add_node("tools", tools_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
        {
            "continue": "tools",
            "end" : END
        }
)

graph.add_edge("tools", "agent")

agent = graph.compile()