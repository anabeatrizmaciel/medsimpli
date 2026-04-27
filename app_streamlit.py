
import os
from datetime import datetime

import streamlit as st

from rag_test import run_rag_for_streamlit

APP_TITLE = "MedSimpli"
APP_TAGLINE = "Pergunte sobre condições, sintomas, exames e tratamentos. Respostas geradas com base em documentos recuperados da base."
DEFAULT_RAG_MODEL = os.getenv("MEDSIMPLI_MODEL", "qwen2.5:14b")
DEFAULT_EMBED_MODEL = os.getenv("MEDSIMPLI_EMBED_MODEL", "pucpr/biobertpt-all")
DEFAULT_FAISS_PATH = os.getenv("MEDSIMPLI_FAISS_PATH", "faiss_vectorstore")
DEFAULT_RAG_TEMPERATURE = float(os.getenv("MEDSIMPLI_RAG_TEMPERATURE", "0.2"))
DEFAULT_TOP_K = int(os.getenv("MEDSIMPLI_TOP_K", "5"))

SAMPLE_QUESTIONS = [
    "O que é hipertensão?",
    "Quais os sintomas da dengue?",
    "O que é lúpus?",
]

DISCLAIMER = (
    "As respostas não substituem orientação profissional. "
    "Em caso de dúvidas, procure um médico."
)


def inject_css():
    st.markdown(
        '''
        <style>
            :root {
                --bg: #f5f7fb;
                --panel: #ffffff;
                --text: #1e2a52;
                --muted: #7080a0;
                --border: #e6ebf5;
                --primary: #4a67f5;
                --shadow: 0 12px 36px rgba(58, 79, 143, 0.08);
            }
            .stApp {
                background: linear-gradient(180deg, #f7f9fc 0%, #f2f5fb 100%);
                color: var(--text);
            }
            [data-testid="stHeader"] {
                background: rgba(0,0,0,0);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
                border-right: 1px solid var(--border);
            }
            [data-testid="stSidebar"] > div:first-child {
                padding-top: 1.2rem;
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
                max-width: 1220px;
            }
            .brand-card {
                background: #fff;
                border: 1px solid #e6ebf5;
                border-radius: 20px;
                padding: 18px 18px 14px 18px;
                box-shadow: 0 12px 36px rgba(58, 79, 143, 0.08);
                margin-bottom: 18px;
            }
            .brand-row {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .brand-title {
                font-size: 2.95rem;
                line-height: 1;
                font-weight: 800;
                letter-spacing: -0.03em;
                color: #233572;
                margin: 0;
            }
            .brand-sub {
                color: #7080a0;
                font-size: 1rem;
                margin-top: 0.45rem;
                max-width: 780px;
            }
            .logo-wrap {
                width: 56px;
                height: 56px;
                border-radius: 16px;
                background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid #dfe8ff;
                flex-shrink: 0;
            }
            .pill {
                display: inline-block;
                background: #ede7ff;
                color: #6b4df5;
                border: 1px solid #ddd5ff;
                padding: 0.38rem 0.82rem;
                border-radius: 999px;
                font-size: 0.95rem;
                font-weight: 700;
                margin-left: 10px;
                vertical-align: middle;
            }
            .hero-grid {
                display: grid;
                grid-template-columns: 1fr 220px;
                gap: 18px;
                align-items: center;
            }
            .hero-illustration {
                height: 140px;
                background: radial-gradient(circle at 30% 30%, #f2fbff 0%, #edf3ff 58%, #f6f8ff 100%);
                border: 1px solid #e6ebf5;
                border-radius: 28px;
                position: relative;
                overflow: hidden;
            }
            .bubble {
                position: absolute;
                background: linear-gradient(135deg, #6fd4ba 0%, #7de0c7 100%);
                color: #fff;
                border-radius: 18px;
                padding: 10px 14px;
                font-size: 1.6rem;
                right: 22px;
                top: 34px;
                box-shadow: 0 10px 30px rgba(79, 161, 138, 0.28);
            }
            .stetho {
                position: absolute;
                left: 36px;
                top: 24px;
                font-size: 4.1rem;
                color: #6b80ff;
                opacity: 0.95;
            }
            .surface-card {
                background: #fff;
                border: 1px solid #e6ebf5;
                border-radius: 24px;
                box-shadow: 0 12px 36px rgba(58, 79, 143, 0.08);
                padding: 18px;
            }
            .section-title {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 1.45rem;
                font-weight: 800;
                color: #27396e;
                margin-bottom: 12px;
            }
            .response-card {
                background: linear-gradient(180deg, #f3fbf7 0%, #eef8f3 100%);
                border: 1px solid #bfe8d6;
                border-radius: 22px;
                padding: 20px;
                min-height: 220px;
            }
            .response-title {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 1.35rem;
                font-weight: 800;
                color: #244b3f;
                margin-bottom: 10px;
            }
            .response-text {
                color: #233244;
                line-height: 1.75;
                font-size: 1.05rem;
                white-space: pre-wrap;
            }
            .explain-card {
                background: #fcfefe;
                border: 1px solid #e8f1ee;
                border-radius: 22px;
                padding: 20px;
                height: 100%;
            }
            .explain-title {
                color: #2a4466;
                font-size: 1.1rem;
                font-weight: 800;
                margin-bottom: 10px;
            }
            .explain-text {
                color: #7483a2;
                line-height: 1.7;
            }
            .docs-header {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 1.9rem;
                font-weight: 800;
                color: #27396e;
                margin-top: 18px;
                margin-bottom: 12px;
            }
            .sidebar-section-title {
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: .06em;
                color: #93a0bd;
                font-weight: 800;
                margin: 1rem 0 0.4rem 0;
            }
            .side-nav-item {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px 14px;
                border-radius: 14px;
                margin-bottom: 8px;
                font-weight: 700;
                color: #4d5f87;
                background: transparent;
                border: 1px solid transparent;
            }
            .side-nav-item.active {
                background: #eef1ff;
                color: #4a67f5;
                border-color: #dfe5ff;
            }
            .important-card {
                background: linear-gradient(180deg, #f5f9ff 0%, #edf3ff 100%);
                border: 1px solid #dee8ff;
                border-radius: 18px;
                padding: 16px;
                color: #5a6d95;
                line-height: 1.7;
                margin-top: 14px;
            }
            .footer {
                text-align: center;
                color: #8191b0;
                font-size: 0.92rem;
                padding-top: 18px;
            }
            .stButton > button {
                background: linear-gradient(135deg, #4c69f7 0%, #3f5ce9 100%);
                color: white;
                border: 0;
                border-radius: 14px;
                padding: 0.8rem 1.2rem;
                font-weight: 700;
                font-size: 1rem;
                box-shadow: 0 10px 22px rgba(76, 105, 247, 0.22);
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #4460ef 0%, #3550da 100%);
                color: white;
            }
            div[data-baseweb="textarea"] textarea,
            div[data-baseweb="select"] > div,
            .stTextInput input {
                border-radius: 16px !important;
                border: 1px solid #dfe7f5 !important;
                background: #fcfdff !important;
            }
            .stTextArea textarea {
                min-height: 150px !important;
            }
        </style>
        ''',
        unsafe_allow_html=True,
    )


def init_session():
    if "query_text" not in st.session_state:
        st.session_state.query_text = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def render_sidebar():
    st.sidebar.markdown(
        '''
        <div class="brand-card">
            <div class="brand-row">
                <div class="logo-wrap">
                    <svg width="34" height="34" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M32 56C19.8 48.8 10 39.5 10 24.8C10 16.1 16.9 10 25.1 10C29.8 10 33.2 12.4 35 15.4C36.8 12.4 40.2 10 44.9 10C53.1 10 60 16.1 60 24.8C60 39.5 50.2 48.8 38 56L35 57.8L32 56Z" stroke="#63D1B4" stroke-width="4" fill="#F3FFFB"/>
                        <path d="M30 24H40V30H46V38H40V44H30V38H24V30H30V24Z" fill="#4B67F4"/>
                    </svg>
                </div>
                <div>
                    <div style="font-size:2rem;font-weight:800;color:#20346d;line-height:1">Med<span style="color:#63D1B4">Simpli</span></div>
                    <div style="font-size:0.93rem;color:#8190ad;line-height:1.35;margin-top:4px">Informação de saúde<br/>em linguagem simples</div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '''
        <div class="side-nav-item active">💬 <span>Nova consulta</span></div>
        <div class="side-nav-item">🕘 <span>Histórico</span></div>
        <div class="side-nav-item">ℹ️ <span>Sobre o projeto</span></div>
        ''',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="sidebar-section-title">Configurações</div>', unsafe_allow_html=True)


def render_hero():
    st.markdown(
        f'''
        <div class="brand-card">
            <div class="hero-grid">
                <div>
                    <div>
                        <span class="brand-title">{APP_TITLE}</span>
                        <span class="pill">✧ RAG-powered</span>
                    </div>
                    <div class="brand-sub">{APP_TAGLINE}</div>
                </div>
                <div class="hero-illustration">
                    <div class="stetho">🩺</div>
                    <div class="bubble">💬</div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_input_card():
    st.markdown(
        '''
        <div class="surface-card">
            <div class="section-title">💭 Faça sua pergunta</div>
        ''',
        unsafe_allow_html=True,
    )
    query = st.text_area(
        "Pergunta",
        label_visibility="collapsed",
        value=st.session_state.query_text,
        placeholder="Ex.: O que é hipertensão?",
        height=150,
    )
    st.session_state.query_text = query

    cols = st.columns(len(SAMPLE_QUESTIONS))
    for idx, question in enumerate(SAMPLE_QUESTIONS):
        with cols[idx]:
            if st.button(question, key=f"sample_{idx}"):
                st.session_state.query_text = question
                st.rerun()

    left, right = st.columns([1, 1])
    with left:
        gerar = st.button("✈️  Gerar resposta", use_container_width=True)
    with right:
        limpar = st.button("Limpar", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return gerar, limpar, st.session_state.query_text


def render_result(result, show_docs: bool, show_docs_limit: int, top_k: int, temperature: float):
    left, right = st.columns([2.2, 1])

    with left:
        st.markdown(
            f'''
            <div class="response-card">
                <div class="response-title">✅ Resposta gerada</div>
                <div class="response-text">{result["answer"]}</div>
                <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:16px;">
                    <div style="background:#f8fbff;border:1px solid #e6edf9;border-radius:999px;padding:.5rem .8rem;font-size:.9rem;font-weight:600;">🤖 Modelo: {result["model"]}</div>
                    <div style="background:#f8fbff;border:1px solid #e6edf9;border-radius:999px;padding:.5rem .8rem;font-size:.9rem;font-weight:600;">📚 Top-K: {top_k}</div>
                    <div style="background:#f8fbff;border:1px solid #e6edf9;border-radius:999px;padding:.5rem .8rem;font-size:.9rem;font-weight:600;">🌡️ Temperatura: {temperature:.1f}</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '''
            <div class="explain-card">
                <div class="explain-title">Como esta resposta foi construída</div>
                <div class="explain-text">
                    Utilizamos recuperação de documentos relevantes da base do MedSimpli
                    e um modelo de linguagem para gerar uma resposta em linguagem simples.
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    if show_docs:
        st.markdown('<div class="docs-header">📚 Documentos recuperados</div>', unsafe_allow_html=True)
        docs = result.get("source_documents", [])[:show_docs_limit]
        if not docs:
            st.info("Nenhum documento recuperado foi retornado.")
            return
        for i, doc in enumerate(docs, start=1):
            source_name = doc.get("source", "desconhecida")
            content = doc.get("content", "")
            with st.expander(f"{i}. {source_name}", expanded=(i == 1)):
                st.markdown(
                    f'<div style="line-height:1.8;color:#465579;background:#fbfdff;border:1px solid #e8eef8;border-radius:16px;padding:1rem;">{content}</div>',
                    unsafe_allow_html=True,
                )


def render_footer():
    st.markdown(
        f'''
        <div class="footer">
            {APP_TITLE} © 2025 • Protótipo acadêmico • Feito com 💜 e Streamlit
        </div>
        ''',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="MedSimpli",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session()
    inject_css()
    render_sidebar()

    options = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]
    rag_model_name = st.sidebar.selectbox(
        "Modelo",
        options=options,
        index=options.index(DEFAULT_RAG_MODEL) if DEFAULT_RAG_MODEL in options else 0,
    )
    rag_top_k = st.sidebar.slider("Top-K (documentos)", 1, 20, DEFAULT_TOP_K)
    rag_temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, DEFAULT_RAG_TEMPERATURE, 0.1)
    show_docs = st.sidebar.toggle("Exibir documentos", value=True)

    st.sidebar.markdown(
        f'''
        <div class="important-card">
            <div style="font-weight:800;color:#4e65a2;margin-bottom:8px;">🛡️ Importante</div>
            {DISCLAIMER}
        </div>
        ''',
        unsafe_allow_html=True,
    )

    render_hero()
    gerar, limpar, query = render_input_card()

    if limpar:
        st.session_state.query_text = ""
        st.session_state.last_result = None
        st.rerun()

    if gerar:
        if not query.strip():
            st.warning("Digite uma pergunta para continuar.")
        else:
            try:
                with st.spinner("Consultando a base e gerando resposta..."):
                    result = run_rag_for_streamlit(
                        query=query,
                        top_k=rag_top_k,
                        prev_model_name=rag_model_name,
                        embed_model_name=DEFAULT_EMBED_MODEL,
                        faiss_index_path=DEFAULT_FAISS_PATH,
                        temperature=rag_temperature,
                    )
                st.session_state.last_result = result
            except Exception as exc:
                st.error(f"Erro ao executar o pipeline RAG: {exc}")
                st.session_state.last_result = None

    if st.session_state.last_result:
        render_result(
            st.session_state.last_result,
            show_docs=show_docs,
            show_docs_limit=rag_top_k,
            top_k=rag_top_k,
            temperature=rag_temperature,
        )

    render_footer()


if __name__ == "__main__":
    main()
