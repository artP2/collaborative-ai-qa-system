## Objetivo

Desenvolver um sistema colaborativo baseado em Recuperação de Informação Aumentada por Geração (RAG), utilizando LangGraph e Streamlit, que simule um ambiente de trabalho em grupo, considerando os princípios do modelo 3C: Comunicação, Colaboração e Coordenação.

## Entregáveis

1. **Repositório Git aberto** com o código-fonte do projeto.
2. **Documentação** detalhada do projeto, pode ser em Markdown no próprio repositório:
   - Descrição do cenário colaborativo escolhido.
   - Diagrama do grafo do LangGraph utilizado.
   - Explicação de como o sistema aborda cada um dos 3C.
   - Instruções de execução do sistema.
3. **Aplicação funcional** (preferencialmente via Streamlit), permitindo a interação simulada de múltiplos usuários sobre até 5 documentos PDF.

## Requisitos Mínimos

- Utilizar LangGraph para modelar o fluxo de interação entre usuários e o agente LLM. (TODO)
- Implementar pelo menos um nó de ferramenta (ex: busca em PDF, sumarização, votação, etc.). (TODO)
- Permitir que múltiplos usuários interajam e que suas ações sejam registradas e atribuídas. ✅
- O sistema deve contemplar:
  - **Comunicação:** Troca de mensagens entre usuários e agente.
  - **Colaboração:** Construção coletiva de respostas, decisões ou documentos.
  - **Coordenação:** Gerenciamento do fluxo, atribuição de tarefas ou controle de etapas do processo.
- Diagrama do grafo do LangGraph, mostrando os nós, arestas e loops do sistema. 
- Interface via Streamlit para facilitar a interação. (TODO)

## Orientações

1. Modelar o grafo do LangGraph, detalhando os nós (ex: entrada de mensagem, busca, síntese, decisão, revisão).
2. Implementar o sistema, integrando LangGraph, LLM e Streamlit.
3. Documentar o projeto, incluindo o diagrama do grafo e explicações sobre o funcionamento.
4. Publicar o código em um repositório Git aberto e incluam um README com instruções de uso.

## Registro Colaborativo de Ações

- Cada interação feita na interface Streamlit exige identificação do usuário. As ações são persistidas na coleção `acoes_colaborativas`, incluindo perguntas, uso de ferramentas, resumos salvos e respostas do agente.
- O histórico consolidado aparece na tela principal em tempo real, exibindo **quem fez o quê**, em qual tópico de conversa e quando.
- O painel lateral mostra um placar de contribuições por usuário com contagens de perguntas, resumos e uso de ferramentas.
- Os resumos consolidados continuam sendo salvos na coleção `resumos_conhecimento`, agora também etiquetados com o autor, tópico e horário.

## Tópicos dinâmicos para estudos

- Na barra lateral do Streamlit é possível criar novos tópicos/salas de conversa informando um nome e, opcionalmente, uma descrição.
- Os tópicos ficam salvos na coleção `topicos_conversa`, permitindo que todos os participantes vejam e reutilizem esses espaços colaborativos.
- Cada mensagem enviada utiliza o `chat_id` do tópico selecionado, mantendo históricos separados por tema de estudo e simplificando a geração de resumos específicos.

## Avaliação
- Clareza e criatividade na modelagem do sistema colaborativo.
- Funcionamento correto da aplicação.
- Qualidade da documentação e do diagrama do grafo.
- Uso adequado dos conceitos de Comunicação, Colaboração e Coordenação.

--- 

## Lang Graph
- Arhur
- Leo

## Streamlit
- Albert
- Diego