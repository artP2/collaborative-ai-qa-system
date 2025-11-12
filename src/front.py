import streamlit as st
from agent import run_agent  # Importa a "ponte" para o cérebro
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuração da Página ---
st.set_page_config(
    page_title="Agente de Conhecimento",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente de Conhecimento Colaborativo")
st.caption("Eu aprendo com nossa conversa. Use o menu lateral para salvar um resumo.")

# --- Inicialização do Histórico (Memória do Streamlit) ---
# Usamos st.session_state para manter o histórico da conversa
# mesmo quando o Streamlit atualiza a página
if "lc_history" not in st.session_state:
    # 'lc_history' vai guardar os objetos de mensagem do LangChain
    st.session_state.lc_history = []

# --- Lógica da Barra Lateral (Para Gerar Resumo) ---
with st.sidebar:
    st.header("Opções")
    st.markdown("Quando terminarmos um tópico, gere um resumo aqui para eu aprender.")
    
    # Campo para o usuário definir o tema
    tema_resumo = st.text_input("Defina o tema principal da conversa:", key="tema_input")
    
    if st.button("Gerar e Salvar Resumo", type="primary"):
        if tema_resumo:
            with st.spinner("Analisando a conversa e gerando resumo..."):
                # 1. Prepara o input especial para acionar a ferramenta 'gerar_resumo'
                #    (Conforme o prompt do seu agente.py)
                prompt_resumo = f"Por favor, gere um resumo sobre o tema '{tema_resumo}'"
                
                # 2. Chama o agente com esse prompt
                #    O agente vai rodar o fluxo de resumo (gerar e salvar no BD)
                resultado = run_agent(prompt_resumo, st.session_state.lc_history)
                
                # 3. Atualiza o histórico com o resultado da ação de resumir
                st.session_state.lc_history = resultado["history"]
                
                st.success(f"Resumo sobre '{tema_resumo}' salvo no banco!")
                st.balloons()
                
                # Limpa o campo de tema
                st.session_state.tema_input = ""
        else:
            st.error("Por favor, defina um tema antes de gerar o resumo.")

# --- Lógica Principal do Chat ---

# 1. Exibe o histórico de mensagens
#    Este loop roda a cada interação, redesenhando o chat
for msg in st.session_state.lc_history:
    if isinstance(msg, HumanMessage):
        # Mensagem do Usuário
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        # Mensagem do Assistente
        # (Ignoramos mensagens de 'tool_calls' para não poluir o chat)
        if msg.content: 
            st.chat_message("assistant").write(msg.content)

# 2. Aguarda um novo input do usuário
if user_input := st.chat_input("Faça sua pergunta..."):
    
    # Adiciona a mensagem do usuário ao chat (apenas visual, por enquanto)
    st.chat_message("user").write(user_input)
    
    # Mostra um "spinner" enquanto o agente pensa
    with st.spinner("Pensando..."):
        # 3. Chama o "cérebro" (run_agent)
        #    Passamos o histórico atual e o novo input
        resultado = run_agent(user_input, st.session_state.lc_history)
        
        # 4. Atualiza o histórico no session_state com a conversa completa
        #    'resultado["history"]' contém a pergunta do usuário, 
        #    as chamadas de ferramentas e a resposta final do AI
        st.session_state.lc_history = resultado["history"]

    # 5. Exibe a resposta final do agente (apenas o conteúdo)
    st.chat_message("assistant").write(resultado["response"])

    # (O Streamlit vai "re-rodar" e redesenhar o chat com a nova 
    # mensagem do assistente que está no histórico)