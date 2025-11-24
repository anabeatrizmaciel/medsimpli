# 🩺 MedSimpli

### *Saúde em linguagem simples - IA aplicada à interpretação de termos médicos*

---

## 📌 Visão Geral

O **MedSimpli** é um **sistema de recomendação com sistema híbrido de recomendação (filtragem baseada em conteúdo + feedback dos usuários)** desenvolvido para **simplificar termos técnicos encontrados em bulas, laudos e documentos médicos**, facilitando a compreensão por parte de pacientes, estudantes e profissionais não especialistas.

O projeto utiliza técnicas de **Processamento de Linguagem Natural (NLP)**, especialmente **TF-IDF** e **t-SNE**, para:

- identificar termos complexos,
- oferecer explicações simplificadas,
- sugerir termos relacionados,
- e visualizar agrupamentos semânticos.

A **filtragem baseada em conteúdo** significa que o sistema recomenda itens com base nas características do **próprio conteúdo** (TF-IDF das descrições técnicas), e não com base no comportamento de outros usuários (como seria uma filtragem colaborativa).

No MedSimpli, agora em versão **híbrida**:

- cada termo técnico é transformado em um vetor TF-IDF;
- consultas do usuário também viram vetores;
- o sistema compara esses vetores usando cosine similarity;
- **e combina essas pontuações com o histórico de feedback útil/não útil**, fortalecendo recomendações bem avaliadas e reduzindo recomendações rejeitadas.

O MedSimpli foi desenvolvido como um protótipo acadêmico com foco em acessibilidade e educação em saúde.

---

## 🎯 Objetivo

> **Transformar informações médicas complexas em linguagem acessível**, sem perder precisão ou contexto.

O sistema busca reduzir barreiras cognitivas, aumentar a autonomia do paciente e apoiar atividades educacionais na área de saúde.

---

## ⭐ Funcionalidades Principais

### 🔍 1. **Busca de termos técnicos**

- Identificação de termos médicos digitados pelo usuário.
- Retorno de explicações simplificadas com base no dataset.
- Cálculo de **similaridade semântica** com TF-IDF.

---

### 🧠 2. **Detecção automática em laudos**

- O usuário cola um trecho de laudo.
- O sistema detecta automaticamente palavras difíceis.
- Exibe explicações simplificadas para cada termo encontrado.

---

### 📊 3. **Mapa Semântico Interativo (t-SNE + KMeans)**

- Visualização 2D interativa com zoom/hover.
- Agrupamento de termos por **proximidade semântica**.
- Cores baseadas em clusters automáticos.
- Hover exibe termo + explicação simples.
- Compatível com modo escuro.

---

### 🎧 4. **Leitura em voz alta (gTTS)**

- Converte a explicação simplificada em áudio.
- Melhora acessibilidade para pessoas com dificuldades de leitura.

---

### 🌙 5. **Modo Escuro Completo**

- CSS customizado para dark mode.
- Inputs, selects, cards e gráficos adaptados.
- Visual moderno e consistente.

---

### 📤 6. **Exportação de Relatórios**

- Exporta explicações encontradas em:
  - **HTML**
  - **TXT**

---

### 🕓 7. **Histórico de buscas**

- Lista de últimos termos pesquisados.
- Botão para repetir a consulta.

---

## 🧬 Arquitetura Técnica

### 🟦 **NLP**

- Representação vetorial com **TF-IDF (1-gram e 2-gram)**.
- Similaridade calculada com **cosine similarity**.

### 🟪 **Redução de dimensionalidade**

- **t-SNE** para visualizar estrutura semântica.
- Normalização com **StandardScaler**.

### 🟥 **Clusterização**

- **KMeans** para agrupamento automático de termos.
- Usado no mapa semântico interativo.

### 🟩 **Frontend**

- Desenvolvido em **Streamlit**, incluindo:
  - componentes customizados,
  - dark mode,
  - renderização responsiva,
  - visualizações interativas com **Plotly** e **Matplotlib**.

---

## 🗂️ Estrutura de Arquivos

A estrutura atual do projeto (conforme organização híbrida real utilizada) está assim:

```
MEDSIMPLI/
├── backend/
│   ├── feedback.csv
│   ├── main.py
│
├── app_streamlit.py   # frontend Streamlit
├── avaliacoes.csv     # avaliações manuais dos usuários
├── dados_saude_com_bulas.csv
│
├── README.md
├── render.yaml
└── requirements.txt
```

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **Streamlit**
- **Pandas**
- **scikit-learn**
  - TF-IDF
  - t-SNE
  - KMeans
- **Plotly**
- **Matplotlib**
- **gTTS**
- **NumPy**

---

## ▶️ Como Executar

### 1) Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 2) Instalar dependências

```bash
pip install -r requirements.txt
```

### 3) Executar o app

```bash
python -m streamlit run app_streamlit.py
```

---

## 📚 Dataset

O dataset contém três colunas obrigatórias e uma opcional:

- **termo**: palavra técnica original
- **tecnico**: definição ou frase técnica
- **simplificado**:  explicação em linguagem acessível
- **fonte**: de onde a explicação técnica foi tirada

---

## 📈 Resultados das Métricas do Sistema (Versão Híbrida)

### **📊 Resumo Geral (Baseado em 11 usuários)**

- **Precision:** 85,45%
- **Recall:** 100%
- **F1-Score:** 89,03%

### 🧩 **Interpretação das Métricas**

#### **🎯 Precision: 85,45%**

Indica que **a maioria das recomendações feitas pelo sistema realmente corresponde ao termo correto** ou à explicação apropriada. Ou seja:

> Quando o MedSimpli recomenda algo, ele acerta na maior parte das vezes.

Um valor acima de 80% demonstra **boa precisão**, com poucos falsos positivos.

---

#### **🔍 Recall: 100%**

O sistema conseguiu **recuperar todos os itens relevantes** durante a avaliação. Isso significa que:

> Sempre que havia uma explicação relevante no dataset, o sistema foi capaz de encontrá-la.

Um recall perfeito mostra que o modelo **não perde explicações importantes**.

---

#### **⚖️ F1-Score: 89,03%**

O F1 é o equilíbrio entre precision e recall. Um valor de \~89% mostra que:

- o sistema mantém **alta precisão**,
- sem abrir mão da **completude**,
- apresentando **boa performance geral**.

Esse valor é especialmente positivo para sistemas de recomendação em linguagem natural, onde equilíbrio é essencial.

---

### 📌 **Conclusão dos Resultados**

- O MedSimpli apresenta **excelente recall**, garantindo que o usuário sempre veja explicações relevantes.
- A precision alta indica que o sistema é **confiável** e acerta a maior parte das recomendações.
- O F1-Score reforça que o desempenho geral é **forte e consistente**.

Esses indicadores mostram que o protótipo é eficaz para sua finalidade acadêmica e demonstra potencial para ser expandido para datasets maiores e feedback em tempo real.

---

## 📌 Aviso

O MedSimpli é um protótipo acadêmico e **NÃO substitui avaliação médica profissional**.

---

## 💙 Autoria

Desenvolvido por **Ana Beatriz, Fernando Luiz, Luiz Daniel e Marcelo Heitor**
Protótipo acadêmico para estudo de NLP aplicado à área de saúde.