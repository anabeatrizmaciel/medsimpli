# 🩺 MedSimpli

### *Saúde em linguagem simples — IA aplicada à compreensão de termos médicos em português brasileiro*

---

## 📌 Visão Geral

O **MedSimpli** é um sistema de apoio à compreensão de linguagem médica em português brasileiro. A proposta é simples: o usuário faz uma pergunta sobre um termo, doença, sintoma ou orientação de saúde, e o sistema responde com uma explicação clara, acessível e baseada em fontes confiáveis.

O sistema utiliza a abordagem **RAG (Retrieval-Augmented Generation)**, que combina:

- **busca semântica** para recuperar os trechos mais relevantes da base documental;
- **geração com LLM** para transformar esses trechos em uma resposta em linguagem simples;
- **exibição da fonte** para que o usuário saiba de onde veio a informação.

O MedSimpli foi desenvolvido como protótipo acadêmico com foco em acessibilidade, letramento em saúde e contextualização cultural para o público brasileiro.

---

## 🎯 Objetivo

> **Transformar informações médicas complexas em linguagem acessível**, sem perder o significado original da informação.

O sistema busca reduzir barreiras cognitivas, aumentar a autonomia do paciente e apoiar atividades educacionais na área de saúde, com atenção especial ao contexto, regionalismos e ao modo como o brasileiro realmente fala sobre saúde.

---

## ⭐ Funcionalidades

- **Busca semântica** sobre base documental de saúde confiável
- **Respostas geradas por LLM** em linguagem simples e objetiva
- **Exibição das fontes** que sustentaram cada resposta
- **Regras de segurança** — o sistema não fornece diagnóstico, não prescreve tratamento e sinaliza quando não tem informação suficiente

---

## 🧠 Prompt Base

```
Você é um assistente do MedSimpli, um sistema de apoio à compreensão
de linguagem médica em português brasileiro.

O objetivo do MedSimpli é ajudar usuários a entender termos médicos,
doenças, sintomas, exames e orientações de saúde por meio de
explicações simples, claras e acessíveis, sempre com base em fontes
confiáveis recuperadas pelo sistema.

Contexto recuperado:
"""
{context}
"""

Pergunta do usuário: {query}

Sua tarefa é responder à pergunta usando apenas o contexto fornecido.

Siga estas regras:
- use apenas as informações presentes no contexto recuperado;
- não invente informações e não complemente com suposições;
- se o contexto não contiver informação suficiente, responda exatamente:
  "Não encontrei informações suficientes sobre esse tema na base do
  MedSimpli. Consulte um profissional de saúde.";
- escreva em português brasileiro claro e objetivo;
- evite jargões desnecessários;
- quando existir um termo popular equivalente ao termo técnico,
  mencione-o entre parênteses;
- quando útil, organize a resposta em tópicos curtos;
- não forneça diagnóstico;
- não prescreva tratamento;
- não substitua a avaliação de um profissional de saúde.

Formato esperado da resposta:
1. explicação simples;
2. pontos principais, se necessário;
3. aviso de limitação, quando aplicável.
```

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **Streamlit** — interface web

---

## ▶️ Como Executar

### 1) Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 2) Instalar dependências

```bash
pip install -r requirements.txt
```

### 3) Indexar a base documental

```bash
python scripts/ingest.py   # gera e valida os chunks
python scripts/embed.py    # cria o banco vetorial (roda uma vez)
```

### 4) Executar o app

```bash
python -m streamlit run app_streamlit.py
```

---

## 📌 Aviso

O MedSimpli é um protótipo acadêmico e **NÃO substitui avaliação médica profissional**. As informações fornecidas têm caráter educativo e são baseadas em fontes do Ministério da Saúde.

---

## 💙 Autoria

Desenvolvido por **Ana Beatriz Maciel Nunes e Marcelo Heitor de Almeida Lira**
Protótipo acadêmico para estudo de NLP e RAG aplicados à área de saúde.