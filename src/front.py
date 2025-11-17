import streamlit as st
from datetime import datetime
from agent import (
    run_agent,
    get_estatisticas_usuarios,
    AGENT_SYSTEM_ID,
    get_topicos_disponiveis,
    criar_topico,
    get_conversa_compartilhada,
)

# --- Configuração da Página ---
st.set_page_config(
    page_title="Agente de Conhecimento",
    page_icon="🤖",
    layout="wide"
)


def _format_timestamp(value):
    if isinstance(value, datetime):
        return value.astimezone().strftime("%d/%m %H:%M")
    return "--:--"


def _is_retry_placeholder(text: str) -> bool:
    if not text:
        return False
    normalized = text.strip().lower()
    retry_variants = {
        "desculpe pelo problema",
        "vou tentar novamente",
        "tentando novamente",
        "desculpe pelo problema. vou tentar novamente.",
        "desculpe pelo problema, vou tentar novamente.",
        "desculpe pelo erro",
    }
    return normalized in retry_variants


def _looks_like_tool_payload(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}") and '\"name\"' in stripped


def _should_show_ai_text(text: str) -> bool:
    if not text:
        return False
    if _is_retry_placeholder(text):
        return False
    if _looks_like_tool_payload(text):
        return False
    return True

# --- Sistema de Login/Identificação ---
# IMPORTANTE: Isso deve vir ANTES de qualquer outro conteúdo
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Se o usuário ainda não se identificou, mostra tela de login
if st.session_state.user_id is None:
    st.title("👥 Sistema Colaborativo de Q&A")
    st.markdown("### 🔐 Identifique-se para começar")
    st.info("💡 Este sistema registra as contribuições de cada usuário para construir conhecimento colaborativo.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        user_name = st.text_input(
            "Digite seu nome ou identificador:",
            placeholder="Ex: João Silva, Usuario1, etc.",
            key="login_input"
        )
        
        if st.button("Entrar", type="primary", use_container_width=True):
            if user_name and user_name.strip():
                st.session_state.user_id = user_name.strip()
                st.success(f"✅ Bem-vindo(a), {st.session_state.user_id}!")
                st.rerun()
            else:
                st.error("⚠️ Por favor, digite seu nome para continuar.")
    
    st.markdown("---")
    st.markdown("#### 📊 Estatísticas do Sistema")
    
    # Mostra estatísticas mesmo sem login
    stats = get_estatisticas_usuarios()
    if stats:
        st.markdown("**Usuários ativos:**")
        for user, info in list(stats.items())[:5]:  # Mostra top 5
            st.write(f"- {user}: {info['total']} contribuições ({info['perguntas']} perguntas, {info['resumos']} resumos)")
    else:
        st.write("Nenhuma contribuição ainda. Seja o primeiro!")
    
    st.stop()  # Para a execução aqui se não estiver logado

# Se chegou aqui, o usuário está logado
st.title("🤖 Agente de Conhecimento Colaborativo")
st.caption(f"👤 **Logado como:** {st.session_state.user_id} | Eu aprendo com nossa conversa. Use o menu lateral para salvar um resumo.")

# Inicializa o chat_id se não existir
if "chat_id" not in st.session_state:
    st.session_state.chat_id = "geral" # Tópico padrão

if "tema_value" not in st.session_state:
    st.session_state.tema_value = ""

# Lista dinâmica de tópicos
topicos_data = sorted(
    get_topicos_disponiveis(),
    key=lambda t: t["name"].lower()
)

if not topicos_data:
    topicos_data = [{"name": "Conversa Geral", "chat_id": "geral"}]

topicos_nomes = {topic["chat_id"]: topic["name"] for topic in topicos_data}
topicos_map = {topic["chat_id"]: topic for topic in topicos_data}

if st.session_state.chat_id not in topicos_nomes:
    st.session_state.chat_id = topicos_data[0]["chat_id"]

# --- Lógica da Barra Lateral (Para Gerar Resumo e Navegação) ---
with st.sidebar:
    st.header("⚙️ Opções")
    
    # Botão de logout
    if st.button("🚪 Sair", type="secondary"):
        st.session_state.user_id = None
        st.rerun()
    
    st.markdown("---")

    # Seletor de Tópico/Chat
    st.subheader("📚 Tópicos de Conversa")
    chat_options = [topic["chat_id"] for topic in topicos_data]
    try:
        default_index = chat_options.index(st.session_state.chat_id)
    except ValueError:
        default_index = 0
    chat_id_selecionado = st.selectbox(
        "Selecione o tópico para conversar:",
        options=chat_options,
        format_func=lambda x: topicos_nomes.get(x, "Desconhecido"),
        index=default_index,
        key="select_topic"
    )

    # Se o usuário mudar a seleção, atualizamos o chat_id e recarregamos a página
    if chat_id_selecionado != st.session_state.chat_id:
        st.session_state.chat_id = chat_id_selecionado
        st.rerun()

    if st.button("🔄 Atualizar conversa", use_container_width=True):
        st.rerun()

    # Exibe detalhes do tópico atual
    if st.session_state.chat_id in topicos_map:
        meta = topicos_map[st.session_state.chat_id]
        if meta.get("descricao"):
            st.caption(f"📝 {meta['descricao']}")
        origem = meta.get("created_by")
        if origem and origem != "sistema":
            st.caption(f"👤 Criado por {origem}")

    if st.session_state.get("reset_novo_topico_form"):
        st.session_state["novo_topico_nome"] = ""
        st.session_state["novo_topico_desc"] = ""
        st.session_state["reset_novo_topico_form"] = False

    st.markdown("**Crie novos tópicos para estudos específicos:**")
    novo_topico_nome = st.text_input("Nome do novo tópico", key="novo_topico_nome")
    novo_topico_desc = st.text_area("Descrição (opcional)", key="novo_topico_desc", height=80)
    if st.button("➕ Criar Tópico", type="primary", key="btn_criar_topico"):
        if novo_topico_nome.strip():
            resultado_topico = criar_topico(novo_topico_nome, st.session_state.user_id, novo_topico_desc)
            if resultado_topico.get("ok"):
                novo_chat_id = resultado_topico["topic"]["chat_id"]
                st.session_state.chat_id = novo_chat_id
                st.session_state["reset_novo_topico_form"] = True
                st.success(f"Tópico '{resultado_topico['topic']['name']}' criado com sucesso!")
                st.rerun()
            else:
                st.error(resultado_topico.get("error", "Não foi possível criar o tópico."))
        else:
            st.error("Informe um nome para o novo tópico.")

    st.markdown("---")
    
    # Estatísticas do usuário atual
    st.subheader(f"👤 {st.session_state.user_id}")
    stats = get_estatisticas_usuarios()
    if st.session_state.user_id in stats:
        user_stats = stats[st.session_state.user_id]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", user_stats['total'])
        col2.metric("Perguntas", user_stats['perguntas'])
        col3.metric("Resumos", user_stats['resumos'])
    else:
        st.info("Nenhuma contribuição ainda.")
    
    st.markdown("---")
    
    # Estatísticas gerais
    st.subheader("📊 Contribuições Gerais")
    if stats:
        for user, info in list(stats.items())[:5]:
            is_current = user == st.session_state.user_id
            prefix = "👉 " if is_current else "   "
            st.write(f"{prefix}**{user}**: {info['total']} contribuições")
    
    st.markdown("---")
    st.markdown("#### 📝 Gerar Resumo")
    nome_topico_atual = topicos_nomes.get(st.session_state.chat_id, st.session_state.chat_id)
    st.markdown(f"Gere um resumo do tópico **'{nome_topico_atual}'** para eu aprender.")
    
    # Campo para o usuário definir o tema
    tema_resumo = st.text_input("Defina o tema do resumo:", value=st.session_state.tema_value)
    
    if st.button("Gerar e Salvar Resumo", type="primary"):
        if tema_resumo:
            with st.spinner("Analisando a conversa e gerando resumo..."):
                # Prepara o input para acionar a ferramenta 'gerar_resumo'
                prompt_resumo = f"Por favor, gere um resumo sobre o tema '{tema_resumo}'"
                
                # Chama o agente com o user_id e o chat_id atual
                resultado = run_agent(
                    prompt_resumo,
                    None,
                    st.session_state.user_id,
                    st.session_state.chat_id
                )
                
                st.success(f"Resumo sobre '{tema_resumo}' salvo no banco!")
                st.balloons()
        else:
            st.error("Por favor, defina um tema antes de gerar o resumo.")

# --- Lógica Principal do Chat ---

conversa_docs = get_conversa_compartilhada(st.session_state.chat_id)

if conversa_docs:
    for registro in conversa_docs:
        role = registro.get("role", "user")
        content = registro.get("content", "")
        autor = registro.get("user_id") or "Usuário"
        timestamp = _format_timestamp(registro.get("timestamp"))

        if role == "assistant":
            if _should_show_ai_text(content):
                msg_box = st.chat_message("assistant")
                msg_box.write(content)
                if timestamp:
                    msg_box.caption(f"🕒 {timestamp}")
        else:
            msg_box = st.chat_message("user")
            msg_box.write(content)
            msg_box.caption(f"👤 {autor} · 🕒 {timestamp}")
else:
    st.info("Ainda não há mensagens neste tópico. Inicie a conversa!")

# 2. Aguarda um novo input do usuário
if user_input := st.chat_input(f"Sua mensagem em '{topicos_nomes.get(st.session_state.chat_id, st.session_state.chat_id)}'"):
    
    st.chat_message("user").write(user_input)
    
    with st.spinner("Pensando..."):
        # 3. Chama o "cérebro" (run_agent) com o user_id e o chat_id
        resultado = run_agent(
            user_input, 
            None,
            st.session_state.user_id,
            st.session_state.chat_id # Passa o ID do chat
        )

    # 5. Exibe a resposta final do agente
    if _should_show_ai_text(resultado["response"]):
        st.chat_message("assistant").write(resultado["response"])
    
    # Força o rerender para mostrar a resposta imediatamente
    st.rerun()