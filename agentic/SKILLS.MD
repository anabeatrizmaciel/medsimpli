# MedSimpli Agentic Skills

Este módulo documenta as skills utilizadas na evolução do MedSimpli de um pipeline RAG clássico para uma arquitetura Agentic RAG.

A camada agentic permite que o sistema classifique a intenção do usuário, reformule consultas, recupere contextos, valide evidências, extraia termos técnicos, gere respostas em linguagem simples e acione fallback quando não houver informação suficiente na base documental.

## Fluxo geral

1. Receber entrada do usuário.
2. Classificar intenção.
3. Reformular consulta.
4. Recuperar contexto.
5. Validar evidência.
6. Extrair termos técnicos, quando aplicável.
7. Gerar resposta simples com fontes ou acionar fallback.

---

## 1. classify_intent_skill

### Objetivo
Classificar a intenção da entrada do usuário.

### Entradas
- `user_input`: pergunta ou texto enviado pelo usuário.

### Saídas
JSON com:
- `intent`
- `reason`

### Intenções possíveis
- `EXPLICAR_TERMO`
- `SIMPLIFICAR_TEXTO_MEDICO`
- `PERGUNTA_SAUDE`
- `PERGUNTA_SENSIVEL`
- `FORA_DO_DOMINIO`

---

## 2. rewrite_query_skill

### Objetivo
Reformular a pergunta do usuário para melhorar a recuperação de contexto.

### Exemplos
- "pressão alta" → "hipertensão arterial"
- "AIDS" → "HIV aids infecção transmissão prevenção"
- "dengue" → "arbovirose dengue sintomas transmissão prevenção"

---

## 3. retrieve_context_skill

### Objetivo
Recuperar contextos relevantes da base documental.

### Versão inicial
Nesta primeira versão, a skill pode usar contextos mockados em `mock_contexts.json`.

### Versão final
Após integração com o pipeline RAG, a skill deverá consultar o índice FAISS.

---

## 4. validate_context_skill

### Objetivo
Verificar se os contextos recuperados sustentam a resposta.

### Critérios
- Existem contextos recuperados?
- Os contextos estão relacionados à pergunta?
- A pergunta envolve prescrição, diagnóstico ou orientação clínica individual?
- A resposta deve ser gerada ou o fallback deve ser acionado?

---

## 5. extract_terms_skill

### Objetivo
Extrair termos médicos ou técnicos de um texto enviado pelo usuário.

### Uso
Principalmente em entradas do tipo `SIMPLIFICAR_TEXTO_MEDICO`.

---

## 6. generate_simple_answer_skill

### Objetivo
Gerar uma resposta em linguagem simples, usando apenas os contextos recuperados e validados.

### Regras
- Usar linguagem clara.
- Não diagnosticar.
- Não prescrever.
- Não inventar informação.
- Apresentar fontes utilizadas.
- Informar que a resposta não substitui orientação profissional.

---

## 7. fallback_skill

### Objetivo
Responder de forma segura quando a base não oferece evidência suficiente ou quando a pergunta está fora do escopo.

### Exemplo
"Não encontrei informação suficiente na base do MedSimpli para responder com segurança."