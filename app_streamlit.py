import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
import random
from datetime import datetime
from gtts import gTTS
import base64
import re
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =========================
# Título
# =========================
APP_TITLE = "MedSimpli"
APP_TAGLINE = "Saúde em linguagem simples - IA aplicada à interpretação médica"

# =========================
# API
# =========================
API_URL = "http://127.0.0.1:8000"

# =========================
# CSS
# =========================
def inject_base_css(dark=False):
    if not dark:
        # 🌞 Modo claro original
        st.markdown("""
            <style>
                .stApp { background-color: #f7fafc; font-family: 'Segoe UI', sans-serif; }
                h1, h2, h3 { color: #1e3a8a; font-weight: 600; }
                .card { background: #ffffff; padding: 1.2em 1.5em; border-radius: 12px;
                        box-shadow: 0 3px 8px rgba(0,0,0,0.05); margin-bottom: 1.0em; }
                div.stButton > button:first-child { background-color: #2563eb; color: #fff; border-radius: 10px; border: none;
                        font-size: 16px; font-weight: 600; padding: 0.6em 1.2em; transition: .2s; }
                div.stButton > button:first-child:hover { background-color: #1d4ed8; transform: scale(1.02); }
                .stSuccess { background-color: #e0f7eb !important; border-radius: 10px; padding: 1em !important; }
                [data-testid="stSidebar"] { background-color: #eef2ff; }
                .footer { text-align:center; color:#64748b; font-size:.85em; padding-top:1.5em; }
                .hist-chip { display:inline-block; background:#e0f2fe; color:#1e3a8a; padding:.35em .8em; border-radius:20px;
                             margin:.2em; font-size:.9em; font-weight:600; border:1px solid #bfdbfe; }
            </style>
        """, unsafe_allow_html=True)

    else:
        # 🌙 Modo escuro melhorado + correção dos textos pretos
        st.markdown("""
            <style>

                /* ======================
                   🎨 Fundo principal
                   ====================== */
                .stApp { 
                    background-color: #0f1624; 
                    color:#e5e7eb; 
                    font-family:'Segoe UI',sans-serif; 
                }

                /* ======================
                   🎨 Títulos
                   ====================== */
                h1, h2, h3 { 
                    color: #93c5fd; 
                    font-weight: 600; 
                }

                /* ======================
                   🎨 Cards
                   ====================== */
                .card {
                    background:#1a2433;
                    color:#e5e7eb;
                    padding:1.2em 1.5em;
                    border-radius:12px;
                    box-shadow: 0 3px 8px rgba(0,0,0,0.35);
                    border: 1px solid rgba(255,255,255,0.06);
                    margin-bottom:1.0em;
                }

                /* ======================
                   🎨 Botões
                   ====================== */
                div.stButton > button:first-child {
                    background-color:#2563eb; 
                    color:#fff; 
                    border-radius:10px; 
                    border:none;
                    font-size:16px; 
                    font-weight:600; 
                    padding:.6em 1.2em; 
                    transition:.2s;
                }
                div.stButton > button:first-child:hover { 
                    background-color:#1e40af; 
                    transform:scale(1.02); 
                }

                /* ======================
                   🎨 Destaques / sucesso
                   ====================== */
                .stSuccess { 
                    background-color:#163c34 !important; 
                    color:#d9faf2 !important; 
                    border-radius:10px; 
                    padding:1em !important; 
                }

                /* ======================
                   🎨 Sidebar
                   ====================== */
                [data-testid="stSidebar"] { 
                    background-color:#0c111b; 
                    color:#cbd5e1; 
                }

                /* ======================
                   🎨 Rodapé
                   ====================== */
                .footer { 
                    text-align:center; 
                    color:#94a3b8; 
                    font-size:.85em; 
                    padding-top:1.5em; 
                }

                /* ======================
                   🎨 Chips (histórico)
                   ====================== */
                .hist-chip { 
                    display:inline-block; 
                    background:#1e3a8a; 
                    color:#dbeafe; 
                    padding:.35em .8em; 
                    border-radius:20px;
                    margin:.2em; 
                    font-size:.9em; 
                    font-weight:600; 
                    border:1px solid #3b82f6;
                }

                /* ======================
                   ⚫ Correção geral: textos pretos
                   ====================== */

                input, textarea {
                    color: #e2e8f0 !important;
                    background-color: #1a2433 !important;
                    border: 1px solid #334155 !important;
                }

                ::placeholder {
                    color: #94a3b8 !important;
                    opacity: 1 !important;
                }

                label, .stTextInput label, .stTextArea label, .stSlider label {
                    color: #cbd5e1 !important;
                }

                .stSlider > div > div > div > input {
                    color: #e2e8f0 !important;
                }

                .stSelectbox select {
                    color: #e2e8f0 !important;
                    background-color: #1a2433 !important;
                    border: 1px solid #334155 !important;
                }

                [data-testid="stToggle-input"] + div {
                    color: #e2e8f0 !important;
                }

                .stCaption, caption, .stMarkdown p {
                    color: #cbd5e1 !important;
                }

            </style>
        """, unsafe_allow_html=True)

# =========================
# Funções auxiliares
# =========================
@st.cache_data
def load_default_data():
    return pd.read_csv("dados_saude_com_bulas.csv")

def make_report_html(query, results, chips, scores_chart_png_b64=None):
    rows = []
    for rank, termo, simplif, score in results:
        rows.append(f"""
        <div class="card">
          <h3>{rank}. {termo}</h3>
          <p><b>Similaridade:</b> {score:.1f}%</p>
          <div><b>Tradução:</b></div>
          <div style="background:#f1f5f9;border-radius:8px;padding:.8em;margin-top:.4em">{simplif}</div>
        </div>""")
    chips_html = f"<p><i>Termos que mais pesaram:</i> {chips}</p>" if chips else ""
    chart_html = f'<img src="data:image/png;base64,{scores_chart_png_b64}" style="max-width:100%"/>' if scores_chart_png_b64 else ""
    return f"""
    <html><head><meta charset="utf-8"><title>Relatório MedSimpli+</title></head>
    <body style="font-family:Segoe UI, sans-serif; padding:24px; background:#f8fafc">
      <h2>Relatório MedSimpli+</h2>
      <p><b>Consulta:</b> {query}</p>
      {chips_html}
      {''.join(rows)}
      {chart_html}
      <p style="color:#64748b;font-size:.85em">Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    </body></html>
    """

def mapa_semantico_interativo(df):
    termos = df["termo"].astype(str).tolist()
    explicacoes = df["simplificado"].astype(str).tolist()

    # Vetorização
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
        min_df=1
    )
    X = vectorizer.fit_transform(termos).toarray()

    # Redução de dimensionalidade - SOMENTE TSNE
    reducer = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate=150,
        max_iter=1500,
        random_state=42
    )
    coords = reducer.fit_transform(X)

    # Normalização para não ficar enorme
    coords = StandardScaler().fit_transform(coords)
    coords = np.round(coords, 2)   # hover mais bonito

    # Clustering automático (até 5 grupos)
    k = min(5, len(df))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(coords)

    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Termo": termos,
        "Explicação": explicacoes,
        "Cluster": clusters
    })

    return df_plot


# =========================
# Aplicação principal
# =========================
def main():
    # --- Sidebar ---
    st.sidebar.title("⚙️ Opções")
    dark_mode = st.sidebar.toggle("🌓 Modo escuro", value=False)
    inject_base_css(dark=dark_mode)

    usuario_id = st.sidebar.selectbox(
        "Usuário (ID):",
        list(range(1, 11)),  # 1 a 10
        index=0
    )
    top_k = st.sidebar.slider("Quantidade de resultados", 1, 10, 3)
    show_explain = st.sidebar.toggle("Mostrar termos que mais pesaram", True)
    detect_terms = st.sidebar.toggle("🧠 Detectar palavras difíceis", True)
    boost_strength = st.sidebar.slider("🎯 Ênfase no termo exato", 0.0, 1.0, 0.4, 0.1)

    st.sidebar.markdown("### 🗺️ Mapa Semântico")
    ativar_mapa = st.sidebar.toggle("Exibir mapa (t-SNE)", False)

    st.sidebar.markdown("### 📊 Métricas de Avaliação")
    exibir_metricas = st.sidebar.toggle("Exibir métricas (Precision/Recall/F1)", False)

    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Se nenhum CSV for enviado, o dataset padrão será usado.")

    # --- Header ---
    st.markdown(f"<h1>🩺 {APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4>{APP_TAGLINE}</h4>", unsafe_allow_html=True)

    # --- Sessão / Histórico ---
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_results" not in st.session_state:
        st.session_state.last_results = []  # [(rank, termo, simplif, score%), chips_str]
    
    # -- Inicializando variável de controle de busca ---
    if "buscar_resultado" not in st.session_state:
        st.session_state.buscar_resultado = False

    # --- Entrada principal ---
    st.markdown("### 🔍 Pesquise um termo médico")
    query = st.text_input("Digite um termo técnico ou trecho da bula:", "hipertensão arterial sistêmica")

    col_busca, col_pdf = st.columns([3,1])
    with col_busca:
        buscar = st.button("Buscar explicações 🩺")
    with col_pdf:
        exportar = st.button("📄 Exportar relatório")

    # =========================
    # Busca principal
    # =========================
    if buscar:
        st.session_state.buscar_resultado = True

    if st.session_state.buscar_resultado:
        if not query.strip():
            st.warning("Digite um termo ou texto para buscar.")
        else:
            if query not in st.session_state.history:
                st.session_state.history.insert(0, query)
                st.session_state.history = st.session_state.history[:6]

            termos_plot, scores_plot = [], []
            st.session_state.last_results = []
            chips_global = ""

            st.markdown("## 💬 Explicações encontradas")

            res = requests.get(
                f"{API_URL}/recomendar_simplificacoes",
                params={
                    "query": query,
                    "top_k": top_k,
                    "boost_strength": boost_strength
                }
            )

            if res.status_code == 200:
                st.session_state.recomendacoes = res.json()
            else:
                st.error("Erro ao gerar recomendações")

            # Agora renderiza SEM depender mais do 'buscar'
            recomendacoes = st.session_state.recomendacoes

            for rec in recomendacoes:
                st.markdown(f"""
                <div class="card">
                    <h4>{rec["rank"]}. {rec["termo"]}</h4>
                    <p><b>Similaridade:</b> {rec["score"]}%</p>
                    <p><b>🩺 Tradução sugerida:</b></p>
                    <div class="stSuccess">{rec["simplificado"]}</div>
                </div>
                """, unsafe_allow_html=True)

                col_u, col_nu = st.columns(2)

                # -----------------------------
                # BOTÃO ÚTIL
                # -----------------------------
                with col_u:
                    if st.button("👍 Útil", key=f"useful_{usuario_id}_{rec['index']}"):
                        requests.post(
                            f"{API_URL}/feedback",
                            json={
                                "user_id": usuario_id,
                                "index": rec["index"],
                                "termo": rec["termo"],
                                "simplificado": rec["simplificado"],
                                "useful": True
                            }
                        )
                        st.success("Obrigado pelo feedback!")

                # -----------------------------
                # BOTÃO NÃO ÚTIL
                # -----------------------------
                with col_nu:
                    if st.button("👎 Não útil", key=f"notuseful_{usuario_id}_{rec['index']}"):
                        requests.post(
                            f"{API_URL}/feedback",
                            json={
                                "user_id": usuario_id,
                                "index": rec["index"],
                                "termo": rec["termo"],
                                "simplificado": rec["simplificado"],
                                "useful": False
                            }
                        )
                        st.warning("Feedback registrado!")

                # Termos semelhantes
                res = requests.get(f"{API_URL}/termos_semelhantes", params={"index": rec["index"]})
                if res.status_code == 200:
                    similares = res.json()["similares"]
                    if similares:
                        st.caption("🔗 Termos relacionados: " + " • ".join(similares))

                # Explicabilidade
                res = requests.get(f"{API_URL}/explicabilidade", params={"query": query, "index": rec["index"]})
                if res.status_code == 200:
                    chips = res.json()["chips"]
                    st.caption(f"Contribuições de termos: {chips}")

                st.session_state.last_results.append(
                    (rec["rank"], rec["termo"], rec["simplificado"], rec["score"])
                )

            # --- Mapa de similaridade ---
            st.subheader("📊 Mapa de Similaridade")
            for rec in recomendacoes[::-1]:
                termos_plot.append(rec["termo"])
                scores_plot.append(rec["score"])

            fig, ax = plt.subplots()
            ax.barh(termos_plot[::], scores_plot[::],
                    color="#60a5fa" if not dark_mode else "#38bdf8")
            ax.set_xlabel("Similaridade (%)")
            ax.set_ylabel("Termos")
            ax.set_title("Ranking de proximidade semântica")
            st.pyplot(fig)


    # =========================
    # Exportar Relatório (HTML/TXT)
    # =========================
    if exportar:
        if not st.session_state.last_results:
            st.info("Faça uma busca para gerar o relatório.")
        else:
            # HTML
            html = make_report_html(query, st.session_state.last_results,
                                    chips=None, scores_chart_png_b64=None)
            html_bytes = html.encode("utf-8")
            st.download_button("⬇️ Baixar relatório (HTML)", data=html_bytes,
                               file_name=f"relatorio_medsimpli_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                               mime="text/html")
            # TXT simples
            buff = io.StringIO()
            buff.write(f"Relatório MedSimpli+ — {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            buff.write(f"Consulta: {query}\n\n")
            for rank, termo, simplif, score in st.session_state.last_results:
                buff.write(f"{rank}. {termo} — Similaridade: {score:.1f}%\n")
                buff.write(f"Tradução: {simplif}\n\n")
            st.download_button("⬇️ Baixar relatório (TXT)", data=buff.getvalue().encode("utf-8"),
                               file_name=f"relatorio_medsimpli_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                               mime="text/plain")

    # =========================
    # Histórico de buscas (chips)
    # =========================
    if st.session_state.history:
        st.markdown("### 🕓 Histórico de buscas")
        chip_cols = st.columns(min(3, len(st.session_state.history)))
        for i, q in enumerate(st.session_state.history):
            col = chip_cols[i % len(chip_cols)]
            with col:
                if st.button(f"🔁 {q}", key=f"hist_{i}"):
                    st.session_state["history"].insert(0, q)
                    st.session_state["history"] = list(dict.fromkeys(st.session_state["history"]))[:6]
                    st.session_state["last_results"] = []
                    st.session_state["query_restore"] = q
                    st.rerun()

    # =========================
    # Detecção de termos difíceis + Tradução de laudo completo
    # =========================
    if detect_terms:
        st.markdown("### 🧠 Palavras Difíceis Detectadas")
        text_input = st.text_area("Cole um trecho do laudo médico para análise e tradução:")
        col_analisar, col_traduzir = st.columns(2)
        with col_analisar:
            analisar = st.button("Analisar termos 🧩")
        with col_traduzir:
            traduzir = st.button("Traduzir laudo para linguagem simples ✍️")

        if analisar and text_input.strip():
            res = requests.get(f"{API_URL}/analisar_termos_dificeis", params={"text_input": text_input})
            if res.status_code == 200:
                matched = res.json()["matched"]
                if matched:
                    for termo, explic in matched:
                        st.markdown(f"<div class='card'><b>{termo}</b> → {explic}</div>", unsafe_allow_html=True)
                else:
                    st.info("Nenhum termo técnico reconhecido neste trecho.")

        if traduzir and text_input.strip():
            texto_out = text_input
            # substitui termos por versão simples (não sensível a maiúsc/minúsc)
            res = requests.get(f"{API_URL}/substituir_termos", params={"texto": text_input})
            if res.status_code == 200:
                texto_out = res.json()["texto_out"]
                st.markdown("#### 📝 Laudo em linguagem simples")
                st.success(texto_out)
            else:
                st.error("Erro ao substituir termos complexos")
    
    # =========================
    # Leitura em Voz Alta: Acessibilidade
    # =========================
    st.markdown("### 🔊 Leitura em Voz Alta")
    st.caption("Clique no botão abaixo para ouvir a explicação em voz natural.")

    if st.session_state.get("last_results"):
        ultimo_texto = st.session_state.last_results[0][2]  # pega a última explicação mostrada
        if st.button("🔊 Ler última explicação"):
            tts = gTTS(text=ultimo_texto, lang='pt', slow=False)
            tts.save("voz_temp.mp3")
            with open("voz_temp.mp3", "rb") as f:
                audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                audio_html = f"""
                <audio autoplay controls>
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.info("Realize uma busca primeiro para gerar explicações que possam ser lidas em voz alta.")
    
    # =========================
    # Mapa semântico dos termos médicos
    # =========================
    if ativar_mapa:
        st.markdown("### 🧭 Mapa Semântico (t-SNE)")

        with st.spinner("Gerando mapa…"):
            res = requests.get(f"{API_URL}/mapa_interativo")
            df_plot = res.json()
            df_plot = pd.DataFrame(df_plot)

        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            color="Cluster",
            hover_data=["Termo", "Explicação"],
            title="Mapa Semântico (t-SNE)",
            color_continuous_scale="Viridis"
        )

        # Dark mode compatível
        fig.update_layout(
            height=600,
            paper_bgcolor="#0f1624" if dark_mode else "white",
            plot_bgcolor="#0f1624" if dark_mode else "white",
            font_color="#e2e8f0" if dark_mode else "#1e293b"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Métricas de Avaliação 
    if exibir_metricas:
        st.markdown("### 📊 Métricas de Avaliação do Sistema")
        st.caption("Comparação das recomendações com o gabarito (avaliacoes.csv). Relevante = nota >= 4")
        
        # Botão
        if st.button("🔄 Calcular Métricas", key="btn_calcular_metricas"):
            with st.spinner("Calculando métricas"):
                try:
                    res = requests.get(f"{API_URL}/metricas")
                    
                    if res.status_code == 200:
                        dados = res.json()
                        media = dados["media"]
                        por_usuario = dados["por_usuario"]
                        
                        st.markdown("#### 📈 Resumo Geral (Média do Sistema)")
                        col_prec, col_rec, col_f1 = st.columns(3)
                        
                        with col_prec:
                            st.metric("Precision", f"{media['precision']:.2%}")
                        
                        with col_rec:
                            st.metric("Recall", f"{media['recall']:.2%}")
                        
                        with col_f1:
                            st.metric("F1-Score", f"{media['f1']:.2%}")
                        
                        st.caption(f"Baseado em {media['num_usuarios']} usuários")
                        
                        # Tabela resumida por usuário
                        st.markdown("#### 📋 Resumo por Usuário")
                        
                        # Cria DataFrame
                        df_metricas = pd.DataFrame(por_usuario)
                        df_metricas["Precision"] = df_metricas["precision"].apply(lambda x: f"{x:.2%}")
                        df_metricas["Recall"] = df_metricas["recall"].apply(lambda x: f"{x:.2%}")
                        df_metricas["F1-Score"] = df_metricas["f1"].apply(lambda x: f"{x:.2%}")
                        
                        # Seleciona colunas para exibição
                        df_display = df_metricas[["usuario_id", "Precision", "Recall", "F1-Score", "tp", "fp", "fn"]]
                        df_display.columns = ["Usuário", "Precision", "Recall", "F1-Score", "TP", "FP", "FN"]
                        
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                        
                        # Detalhamento visual por usuário
                        st.markdown("#### 🔍 Detalhamento: O que foi recomendado vs O que era relevante")
                        st.caption("Clique em cada usuário para ver os itens recomendados e relevantes")
                        
                        for usuario_data in por_usuario:
                            usuario_id = usuario_data["usuario_id"]
                            
                            with st.expander(f"👤 Usuário {usuario_id} | Precision: {usuario_data['precision']:.2%} | Recall: {usuario_data['recall']:.2%} | F1: {usuario_data['f1']:.2%}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**✅ O que o sistema RECOMENDOU:**")
                                    recomendados = usuario_data.get("itens_recomendados", [])
                                    if recomendados:
                                        for item in recomendados:
                                            # Verifica se é TP ou FP
                                            if item in usuario_data.get("itens_tp", []):
                                                st.success(f"✓ {item} (acerto - era relevante)")
                                            elif item in usuario_data.get("itens_fp", []):
                                                st.warning(f"✗ {item} (erro - não era relevante)")
                                            else:
                                                st.write(f"• {item}")
                                    else:
                                        st.info("Nenhum item recomendado")
                                
                                with col2:
                                    st.markdown("**⭐ O que o usuário achou RELEVANTE (nota >= 4):**")
                                    relevantes = usuario_data.get("itens_relevantes", [])
                                    if relevantes:
                                        for item in relevantes:
                                            # Verifica se foi recomendado ou não
                                            if item in usuario_data.get("itens_tp", []):
                                                st.success(f"✓ {item} (foi recomendado - acerto)")
                                            elif item in usuario_data.get("itens_fn", []):
                                                st.error(f"✗ {item} (NÃO foi recomendado - erro)")
                                            else:
                                                st.write(f"• {item}")
                                    else:
                                        st.info("Nenhum item relevante")
                                
                                # Resumo visual
                                st.markdown("---")
                                st.markdown("**📊 Resumo:**")
                                
                                col_tp, col_fp, col_fn = st.columns(3)
                                
                                with col_tp:
                                    st.metric("✅ Acertos (TP)", usuario_data["tp"])
                                    if usuario_data.get("itens_tp"):
                                        st.caption(", ".join(usuario_data["itens_tp"]))
                                
                                with col_fp:
                                    st.metric("⚠️ Falsos Positivos (FP)", usuario_data["fp"])
                                    if usuario_data.get("itens_fp"):
                                        st.caption(", ".join(usuario_data["itens_fp"]))
                                
                                with col_fn:
                                    st.metric("❌ Falsos Negativos (FN)", usuario_data["fn"])
                                    if usuario_data.get("itens_fn"):
                                        st.caption(", ".join(usuario_data["itens_fn"]))
                    
                    else:
                        st.error(f"Erro ao calcular métricas: {res.status_code}")
                        st.caption("Certifique-se de que o backend está rodando e o arquivo avaliacoes.csv existe.")
                
                except requests.exceptions.ConnectionError:
                    st.error("Não foi possível conectar ao backend. Certifique-se de que o servidor está rodando em http://127.0.0.1:8000")
                except Exception as e:
                    st.error(f"Erro ao calcular métricas: {str(e)}")

    st.markdown(
        "<div class='footer'>🧠 MedSimpli+ — IA aplicada à compreensão médica. Protótipo acadêmico sem fins diagnósticos.</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
