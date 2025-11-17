# Sistema Colaborativo com LangGraph, RAG e Streamlit

## 1\. Objetivo

Desenvolver um sistema colaborativo de Q\&A (Perguntas e Respostas) que constrói uma base de conhecimento dinâmica (RAG) a partir das interações dos usuários. O sistema utiliza **LangGraph** para orquestrar o fluxo de informação e **Streamlit** como interface.

## 2\. Cenário Colaborativo: Agente de Conhecimento

O cenário simula uma equipe de estudo ou um time de projeto construindo uma base de conhecimento compartilhada.

Múltiplos usuários, cada um identificado por um login (`user_id`), interagem com um agente de IA em **"Tópicos"** de conversa específicos.

O fluxo de trabalho colaborativo é o seguinte:

1.  **Perguntar (RAG):** Um usuário faz uma pergunta. O agente primeiro tenta responder usando o conhecimento existente no banco de dados (`consultar_BD`).
2.  **Buscar (RAG):** Se o conhecimento interno não for suficiente (similaridade abaixo de 0.6), o agente busca novas informações na web (`buscar_referencias`).
3.  **Consolidar (Colaboração):** Após uma discussão produtiva, qualquer usuário pode solicitar ao agente que **"Gere um Resumo"** da conversa atual.
4.  **Salvar (Colaboração):** O agente analisa o histórico do tópico, gera um resumo e o salva no banco de dados (`atualizar_BD`), incluindo seu *embedding*.
5.  **Aprender (Loop de RAG):** Esse novo resumo se torna parte da base de conhecimento. Na próxima vez que um usuário fizer uma pergunta similar, `consultar_BD` encontrará esse resumo e o utilizará na resposta, melhorando a inteligência do sistema coletivamente.

## 3\. Diagrama do Grafo (LangGraph)

<img width="534" height="333" alt="image" src="https://github.com/user-attachments/assets/9a753ab6-2a88-4ab3-babd-a7d9e63f9613" />


## 4\. Abordagem do Modelo 3C

O sistema implementa os três pilares do modelo 3C da seguinte forma:

### Comunicação

Os usuários interagem diretamente com o agente ao trocar perguntas e respostas para receber e melhorar as respostas desejadas. Ao contribuir para a estruturação da mensagem ideal do agente, o usuário indiretamente também interagem com outros usuários, os quais irão ter acesso e irão interagir com a resposta final por outros usuários

### Cooperação

Com cada usuário contribuindo para o desenvolvimento das respostas, há uma cooperação entre os usuários para chegar no objetivo desejado.

### Coordenação

Há uma coordenação entre os usuários para estruturar o desenvolvimento das respostas disponíveis do agente. Com essa coordenação, cada usuário contribui de forma colaborativa para essa estruturação, cada vez que interage com o agente.


## 5\. Instruções de Execução

Siga estes passos para rodar o projeto localmente.

### (Opcional) Configurar o Ambiente Python

Recomenda-se usar um ambiente virtual (`venv`) para isolar as dependências.

1.  Abra seu terminal na pasta do projeto.
2.  Crie o ambiente:
    ```bash
    python -m venv venv
    ```
3.  Ative o ambiente:
      * **Windows:** `.\venv\Scripts\activate`
      * **Linux:** `source venv/bin/activate`

### Passo 1: Instalar as Bibliotecas

Instale todas as dependências necessárias listadas no `agent.py` e `front.py`.

```bash
pip install streamlit langchain langgraph langchain-ollama pymongo requests beautifulsoup4 googlesearch-python sentence-transformers python-dotenv
```

### Passo 2: Configurar Serviços Externos

O projeto depende de dois serviços rodando em segundo plano: **Ollama** e **MongoDB**.

#### A. Ollama (Para o Modelo de IA)

O agente usa `ChatOllama(model="llama3.1")`.

1.  **Instale o Ollama:** Baixe e instale em [ollama.com](https://ollama.com/).
2.  **Baixe o Modelo:** Rode no seu terminal:
    ```bash
    ollama pull llama3.1
    ```
3.  **Deixe o Ollama Rodando:** O serviço precisa estar em execução.

#### B. MongoDB (Para o Banco de Dados)

O agente usa uma conexão local (`mongodb://localhost:27017/`) por padrão. A forma mais fácil de subir o banco é com Docker.

1.  Com o Docker Desktop instalado, rode:
    ```bash
    docker run -d -p 27017:27017 --name mongo-qa-system mongo
    ```
2.  Isso iniciará um contêiner MongoDB pronto para receber as conexões do `agent.py`.


### Passo 3: Executar a Aplicação

Com o ambiente ativado e os serviços rodando, inicie o Streamlit:

```bash
streamlit run front.py
```

O Streamlit abrirá a aplicação no seu navegador, pronta para ser usada. Na primeira execução, o `agent.py` baixará o modelo de embeddings, o que pode levar alguns minutos.

# Integrantes do grupo
- Albert Katayama Shoji - 13695358
- Leonardo Ishida - 12873424
- Arthur Pin - 12691964
- Diego Cabral Morales - 13672193

