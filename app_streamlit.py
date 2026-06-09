import os
import gc
from html import escape

import streamlit as st

from rag_test import ensure_or_build_faiss
from rag_response import respond_to_query
from agentic.agent import build_fallback_response, choose_query_to_use
from agentic.skills import (
    classify_intent_skill,
    rewrite_query_skill,
    retrieve_context_skill,
    validate_context_skill,
)

APP_TITLE = "MedSimpli"
APP_TAGLINE = (
    "Plataforma acadêmica para apoio à compreensão de informações médicas "
    "em linguagem simples, com RAG, FAISS e camada Agentic RAG."
)
DEFAULT_RAG_MODEL = os.getenv("MEDSIMPLI_MODEL", "qwen2.5:3b")
DEFAULT_EMBED_MODEL = os.getenv("MEDSIMPLI_EMBED_MODEL", "pucpr/biobertpt-all")
DEFAULT_FAISS_PATH = os.getenv("MEDSIMPLI_FAISS_PATH", "faiss_vectorstore")
DEFAULT_RAG_TEMPERATURE = float(os.getenv("MEDSIMPLI_RAG_TEMPERATURE", "0.2"))
DEFAULT_TOP_K = int(os.getenv("MEDSIMPLI_TOP_K", "1"))

SAMPLE_QUESTIONS = [
    "O que são hepatites virais?",
    "Quais os sintomas da dengue?",
    "Qual remédio devo tomar para dengue?",
]

DISCLAIMER = (
    "Este protótipo tem finalidade acadêmica e informativa. "
    "As respostas não substituem avaliação, diagnóstico ou orientação de profissionais de saúde."
)


def _safe(value) -> str:
    return escape(str(value or ""))


def clear_runtime_memory(clear_streamlit_cache: bool = False):
    """Libera memória do Python/Streamlit entre testes.

    Útil em notebooks com pouca RAM, porque o Streamlit reroda o script
    a cada interação e bibliotecas de embeddings/LLM podem manter objetos
    pesados vivos na sessão.
    """
    if clear_streamlit_cache:
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    gc.collect()


def build_light_answer_from_context(query: str, contexts: list[dict], validation: dict | None = None) -> str:
    """Resposta leve, sem chamar Ollama.

    Modo de emergência para demonstração em máquina com pouca RAM.
    Mantém o comportamento Agentic RAG: intenção -> recuperação -> validação -> resposta/fallback.
    """
    if not contexts:
        return (
            "Não encontrei contexto suficiente na base documental para responder com segurança. "
            "Tente reformular a pergunta ou consultar um profissional de saúde."
        )

    first = contexts[0]
    content = (first.get("content") or "").strip()
    source = first.get("source_file") or first.get("source") or "documento recuperado"
    page = first.get("page")

    # Mantém curto para não lotar a tela nem a memória.
    excerpt = content[:900].strip()
    if len(content) > 900:
        excerpt += "..."

    source_line = f"Fonte: {source}"
    if page is not None:
        source_line += f", página {page}"

    return (
        "Com base no trecho recuperado da base documental, a resposta é:\n\n"
        f"{excerpt}\n\n"
        f"{source_line}\n\n"
        "Observação: esta é uma resposta em modo leve, sem geração por LLM, usada para "
        "validar o fluxo Agentic RAG em ambiente com pouca memória."
    )


def _contexts_to_source_documents(contexts: list[dict]) -> list[dict]:
    source_documents = []
    for context in contexts:
        source_label = context.get("source") or context.get("source_file") or "Fonte não identificada"
        page = context.get("page")
        if page is not None:
            source_label = f"{source_label} — p. {page}"
        source_documents.append({"source": source_label, "content": context.get("content", "")})
    return source_documents


def run_agentic_rag_for_streamlit(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    prev_model_name: str = DEFAULT_RAG_MODEL,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    faiss_index_path: str = DEFAULT_FAISS_PATH,
    temperature: float = DEFAULT_RAG_TEMPERATURE,
    use_llm_generation: bool = False,
):
    """
    Fluxo usado pela interface.

    A camada agentic entra antes da geração final para:
    1. classificar a intenção;
    2. bloquear perguntas sensíveis ou fora do domínio;
    3. reescrever/selecionar a consulta;
    4. recuperar contexto no FAISS;
    5. validar se o contexto é suficiente;
    6. chamar o RAG + LLM quando a resposta for segura.
    """
    ensure_or_build_faiss(index_path=faiss_index_path, embed_model_name=embed_model_name)

    intent_result = classify_intent_skill(query)
    intent = intent_result.get("intent")
    intent_reason = intent_result.get("reason", "")

    if intent in ["FORA_DO_DOMINIO", "PERGUNTA_SENSIVEL"]:
        fallback = build_fallback_response(
            reason=intent_reason,
            intent=intent,
            intent_reason=intent_reason,
        )
        return {
            "query": query,
            "answer": fallback["answer"],
            "source_documents": [],
            "model": prev_model_name,
            "top_k": top_k,
            "temperature": temperature,
            "mode": "Agentic RAG",
            "agent_status": "fallback",
            "intent": intent,
            "intent_reason": intent_reason,
            "query_used": None,
            "validation": None,
            "agent_contexts": [],
        }

    queries = rewrite_query_skill(query)
    query_used = choose_query_to_use(query, queries)

    contexts = retrieve_context_skill(query=query_used, top_k=top_k, use_mock=False)
    validation = validate_context_skill(query, contexts)

    if validation.get("should_fallback"):
        fallback = build_fallback_response(
            reason=validation.get("reason", "Contexto insuficiente."),
            intent=intent,
            intent_reason=intent_reason,
            queries=queries,
            query_used=query_used,
            validation=validation,
            contexts=contexts,
        )
        return {
            "query": query,
            "answer": fallback["answer"],
            "source_documents": _contexts_to_source_documents(contexts),
            "model": prev_model_name,
            "top_k": top_k,
            "temperature": temperature,
            "mode": "Agentic RAG",
            "agent_status": "fallback",
            "intent": intent,
            "intent_reason": intent_reason,
            "query_used": query_used,
            "validation": validation,
            "agent_contexts": contexts,
        }

    if use_llm_generation:
        result = respond_to_query(
            embed_model_name=embed_model_name,
            prev_model_name=prev_model_name,
            top_k=top_k,
            query=query,
            faiss_index_path=faiss_index_path,
            temperature=temperature,
            verbose=False,
        )
        model_label = prev_model_name
        answer_mode = "Agentic RAG + LLM"
    else:
        result = {
            "query": query,
            "answer": build_light_answer_from_context(query, contexts, validation),
            "source_documents": _contexts_to_source_documents(contexts),
            "model": "Agentic RAG light",
            "top_k": top_k,
            "temperature": temperature,
        }
        model_label = "sem LLM local"
        answer_mode = "Agentic RAG light"

    result.update(
        {
            "mode": answer_mode,
            "agent_status": "answered",
            "intent": intent,
            "intent_reason": intent_reason,
            "query_used": query_used,
            "validation": validation,
            "agent_contexts": contexts,
            "model": model_label,
        }
    )

    clear_runtime_memory(clear_streamlit_cache=False)
    return result


@st.cache_data(show_spinner=False, ttl=1800)
def run_agentic_rag_cached(
    query: str,
    top_k: int,
    prev_model_name: str,
    embed_model_name: str,
    faiss_index_path: str,
    temperature: float,
    use_llm_generation: bool,
):
    return run_agentic_rag_for_streamlit(
        query=query,
        top_k=top_k,
        prev_model_name=prev_model_name,
        embed_model_name=embed_model_name,
        faiss_index_path=faiss_index_path,
        temperature=temperature,
        use_llm_generation=use_llm_generation,
    )


def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --bg: #f6f8fc;
                --panel: #ffffff;
                --panel-soft: #f9fbff;
                --ink: #172033;
                --muted: #64748b;
                --muted-2: #94a3b8;
                --line: #e2e8f0;
                --navy: #1e2a52;
                --blue: #3656f5;
                --blue-soft: #eef2ff;
                --teal: #14b8a6;
                --teal-soft: #ecfdf5;
                --amber-soft: #fffbeb;
                --rose-soft: #fff1f2;
                --shadow: 0 18px 55px rgba(31, 42, 82, 0.08);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(54, 86, 245, .12), transparent 30%),
                    radial-gradient(circle at top right, rgba(20, 184, 166, .12), transparent 28%),
                    linear-gradient(180deg, #f8fbff 0%, #f3f6fb 100%);
                color: var(--ink);
            }

            [data-testid="stHeader"] { background: rgba(0,0,0,0); }
            [data-testid="stSidebar"] {
                background: rgba(255,255,255,.86);
                border-right: 1px solid var(--line);
                backdrop-filter: blur(16px);
            }
            [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
            .block-container { padding-top: 1.2rem; max-width: 1240px; }

            .saas-card {
                background: rgba(255,255,255,.92);
                border: 1px solid rgba(226,232,240,.95);
                border-radius: 28px;
                box-shadow: var(--shadow);
            }

            .sidebar-logo {
                padding: 18px;
                margin-bottom: 14px;
            }
            .logo-row { display: flex; align-items: center; gap: 12px; }
            .logo-mark {
                width: 48px; height: 48px; border-radius: 16px;
                display: flex; align-items: center; justify-content: center;
                background: linear-gradient(135deg, #eef2ff 0%, #ecfeff 100%);
                border: 1px solid #dbeafe;
                font-size: 1.55rem;
            }
            .sidebar-title {
                font-size: 1.65rem; line-height: 1; font-weight: 900;
                color: var(--navy); letter-spacing: -.04em;
            }
            .sidebar-title span { color: var(--teal); }
            .sidebar-sub { color: var(--muted); font-size: .86rem; line-height: 1.35; margin-top: 4px; }

            .nav-item {
                display: flex; align-items: center; gap: 11px;
                border-radius: 16px; padding: 11px 13px; margin-bottom: 7px;
                color: #52627a; font-weight: 750; border: 1px solid transparent;
            }
            .nav-item.active { background: var(--blue-soft); color: var(--blue); border-color: #dbeafe; }
            .sidebar-label {
                margin: 18px 0 8px; color: var(--muted-2); text-transform: uppercase;
                font-size: .75rem; letter-spacing: .08em; font-weight: 850;
            }
            .disclaimer {
                background: linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
                border: 1px solid #dbeafe; border-radius: 20px; padding: 15px;
                color: #52627a; line-height: 1.62; font-size: .92rem;
            }

            .hero {
                padding: 28px 30px;
                margin-bottom: 18px;
                position: relative;
                overflow: hidden;
            }
            .hero:after {
                content: "";
                position: absolute; right: -90px; top: -90px; width: 280px; height: 280px;
                background: radial-gradient(circle, rgba(54,86,245,.14), rgba(20,184,166,.08), transparent 70%);
                border-radius: 999px;
            }
            .eyebrow {
                display: inline-flex; align-items: center; gap: 8px;
                background: #eef2ff; color: #3656f5; border: 1px solid #dbeafe;
                padding: 7px 12px; border-radius: 999px; font-size: .82rem; font-weight: 850;
                margin-bottom: 14px;
            }
            .hero-grid { display: grid; grid-template-columns: 1fr 290px; gap: 24px; align-items: center; position: relative; z-index: 1; }
            .hero-title { font-size: 3.45rem; line-height: .95; font-weight: 950; color: var(--navy); letter-spacing: -.055em; margin: 0; }
            .hero-title span { color: var(--teal); }
            .hero-sub { max-width: 760px; margin-top: 14px; color: var(--muted); font-size: 1.05rem; line-height: 1.65; }
            .hero-pills { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
            .pill {
                display: inline-flex; align-items: center; gap: 8px;
                border-radius: 999px; padding: 8px 13px;
                background: #f8fafc; border: 1px solid #e2e8f0;
                color: #334155; font-size: .88rem; font-weight: 800;
            }
            .pill.primary { background: #eef2ff; color: #3656f5; border-color: #dbeafe; }
            .hero-visual {
                border-radius: 30px; min-height: 178px; padding: 18px;
                background: linear-gradient(145deg, #172033 0%, #243b73 55%, #0f766e 130%);
                color: white; box-shadow: 0 22px 60px rgba(31, 42, 82, .20);
                position: relative; overflow: hidden;
            }
            .hero-visual:before {
                content: ""; position: absolute; inset: 0;
                background: radial-gradient(circle at 80% 10%, rgba(255,255,255,.22), transparent 30%);
            }
            .metric-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; position: relative; z-index: 1; }
            .metric {
                background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
                border-radius: 18px; padding: 13px;
            }
            .metric-value { font-size: 1.3rem; font-weight: 900; }
            .metric-label { font-size: .78rem; opacity: .84; margin-top: 3px; line-height: 1.25; }

            .panel { padding: 22px; margin-bottom: 18px; }
            .section-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
            .section-title { font-size: 1.32rem; font-weight: 900; color: var(--navy); letter-spacing: -.02em; }
            .section-caption { color: var(--muted); font-size: .92rem; }

            .workflow {
                display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 14px;
            }
            .step {
                background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 20px;
                padding: 14px; min-height: 105px;
            }
            .step-num { color: #3656f5; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 900; }
            .step-title { color: #1e293b; font-size: .98rem; font-weight: 900; margin-top: 6px; }
            .step-desc { color: var(--muted); font-size: .84rem; line-height: 1.45; margin-top: 4px; }

            .stTextArea textarea {
                min-height: 152px !important;
                border-radius: 20px !important;
                border: 1px solid #dbe3ef !important;
                background: #fbfdff !important;
                color: #172033 !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.8) !important;
            }
            div[data-baseweb="select"] > div,
            .stSlider [data-testid="stTickBar"] {
                border-radius: 16px !important;
            }
            .stButton > button {
                border-radius: 16px; border: 0; font-weight: 850;
                padding: .78rem 1rem;
            }
            .stButton > button[kind="primary"], .stButton > button:first-child {
                background: linear-gradient(135deg, #3656f5 0%, #2348d5 100%);
                color: white;
                box-shadow: 0 12px 28px rgba(54, 86, 245, .22);
            }

            .response-card {
                background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
                border: 1px solid #dbeafe;
                border-radius: 26px;
                padding: 22px;
                box-shadow: var(--shadow);
                min-height: 255px;
            }
            .response-header { display:flex; align-items:center; justify-content:space-between; gap: 12px; margin-bottom: 14px; }
            .response-title { color: var(--navy); font-size: 1.35rem; font-weight: 950; }
            .status-badge { border-radius: 999px; padding: 7px 11px; font-size: .82rem; font-weight: 900; background: var(--teal-soft); color: #047857; border: 1px solid #bbf7d0; }
            .status-badge.fallback { background: var(--amber-soft); color: #92400e; border-color: #fde68a; }
            .response-text { color: #253044; line-height: 1.78; font-size: 1.03rem; white-space: pre-wrap; }
            .chips { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 18px; }
            .chip { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 999px; padding: 7px 11px; color: #475569; font-size: .83rem; font-weight: 780; }

            .explain-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 26px; padding: 20px; box-shadow: var(--shadow); height: 100%; }
            .explain-title { color: var(--navy); font-weight: 950; font-size: 1.05rem; margin-bottom: 10px; }
            .explain-text { color: var(--muted); line-height: 1.65; font-size: .94rem; }
            .trace-list { margin-top: 14px; display:grid; gap: 8px; }
            .trace-item { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 9px 10px; color: #475569; font-size: .84rem; }

            .docs-title { font-size: 1.35rem; font-weight: 950; color: var(--navy); margin: 20px 0 10px; }
            .footer { text-align:center; color: var(--muted-2); font-size: .88rem; padding: 18px 0 8px; }

            @media (max-width: 920px) {
                .hero-grid { grid-template-columns: 1fr; }
                .workflow { grid-template-columns: 1fr 1fr; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session():
    if "query_text" not in st.session_state:
        st.session_state.query_text = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def render_sidebar():
    st.sidebar.markdown(
        """
        <div class="saas-card sidebar-logo">
            <div class="logo-row">
                <div class="logo-mark">🩺</div>
                <div>
                    <div class="sidebar-title">Med<span>Simpli</span></div>
                    <div class="sidebar-sub">Health information<br/>in plain Portuguese</div>
                </div>
            </div>
        </div>
        <div class="nav-item active">💬 <span>Consulta</span></div>
        <div class="nav-item">🧠 <span>Agentic RAG</span></div>
        <div class="nav-item">📚 <span>Base documental</span></div>
        <div class="nav-item">📊 <span>Avaliação</span></div>
        <div class="sidebar-label">Configurações do modelo</div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        f"""
        <div class="saas-card hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">🎓 Protótipo acadêmico · Micro-SaaS de saúde</div>
                    <h1 class="hero-title">Med<span>Simpli</span></h1>
                    <div class="hero-sub">{_safe(APP_TAGLINE)}</div>
                    <div class="hero-pills">
                        <span class="pill primary">✧ Agentic RAG</span>
                        <span class="pill">FAISS Vector Store</span>
                        <span class="pill">Ollama local</span>
                        <span class="pill">Fontes oficiais</span>
                    </div>
                </div>
                <div class="hero-visual">
                    <div style="position:relative;z-index:1;font-size:.82rem;font-weight:850;opacity:.85;margin-bottom:12px;">PAINEL DE CONFIABILIDADE</div>
                    <div class="metric-row">
                        <div class="metric"><div class="metric-value">RAG</div><div class="metric-label">recuperação com fonte</div></div>
                        <div class="metric"><div class="metric-value">FAISS</div><div class="metric-label">busca semântica local</div></div>
                        <div class="metric"><div class="metric-value">Agent</div><div class="metric-label">validação e fallback</div></div>
                        <div class="metric"><div class="metric-value">BR</div><div class="metric-label">português brasileiro</div></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow():
    st.markdown(
        """
        <div class="workflow">
            <div class="step"><div class="step-num">Etapa 01</div><div class="step-title">Classificação</div><div class="step-desc">O agente identifica intenção, domínio e risco da pergunta.</div></div>
            <div class="step"><div class="step-num">Etapa 02</div><div class="step-title">Recuperação</div><div class="step-desc">O FAISS busca trechos relevantes na base documental.</div></div>
            <div class="step"><div class="step-num">Etapa 03</div><div class="step-title">Validação</div><div class="step-desc">O contexto é verificado antes da geração da resposta.</div></div>
            <div class="step"><div class="step-num">Etapa 04</div><div class="step-title">Resposta ou fallback</div><div class="step-desc">O sistema responde em linguagem simples ou aciona fallback seguro.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_card():
    st.markdown(
        """
        <div class="saas-card panel">
            <div class="section-heading">
                <div>
                    <div class="section-title">Nova consulta</div>
                    <div class="section-caption">Digite uma pergunta sobre saúde para testar o fluxo Agentic RAG.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "Pergunta",
        label_visibility="collapsed",
        value=st.session_state.query_text,
        placeholder="Ex.: O que são hepatites virais?",
        height=150,
    )
    st.session_state.query_text = query

    cols = st.columns(len(SAMPLE_QUESTIONS))
    for idx, question in enumerate(SAMPLE_QUESTIONS):
        with cols[idx]:
            if st.button(question, key=f"sample_{idx}", use_container_width=True):
                st.session_state.query_text = question
                st.rerun()

    left, right = st.columns([1, 1])
    with left:
        gerar = st.button("Gerar resposta", use_container_width=True, type="primary")
    with right:
        limpar = st.button("Limpar", use_container_width=True)

    return gerar, limpar, st.session_state.query_text


def render_result(result, show_docs: bool, show_docs_limit: int, top_k: int, temperature: float):
    status = _safe(result.get("agent_status", "answered"))
    status_class = "fallback" if status == "fallback" else ""
    validation = result.get("validation") or {}
    validation_reason = validation.get("reason") or "Contexto validado para resposta."
    query_used = result.get("query_used") or "—"

    left, right = st.columns([2.15, 1])
    with left:
        st.markdown(
            f"""
            <div class="response-card">
                <div class="response-header">
                    <div class="response-title">Resposta gerada</div>
                    <div class="status-badge {status_class}">{status}</div>
                </div>
                <div class="response-text">{_safe(result.get("answer", ""))}</div>
                <div class="chips">
                    <span class="chip">🧠 {_safe(result.get("mode", "Agentic RAG"))}</span>
                    <span class="chip">🎯 Intenção: {_safe(result.get("intent", "—"))}</span>
                    <span class="chip">🤖 Modelo: {_safe(result.get("model", "—"))}</span>
                    <span class="chip">📚 Top-K: {top_k}</span>
                    <span class="chip">🌡️ Temp.: {temperature:.1f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="explain-card">
                <div class="explain-title">Rastreamento agentic</div>
                <div class="explain-text">
                    A pergunta passa por uma camada de controle antes da geração final, priorizando rastreabilidade e segurança.
                </div>
                <div class="trace-list">
                    <div class="trace-item"><b>Consulta usada:</b><br/>{_safe(query_used)}</div>
                    <div class="trace-item"><b>Motivo da intenção:</b><br/>{_safe(result.get("intent_reason", "—"))}</div>
                    <div class="trace-item"><b>Validação:</b><br/>{_safe(validation_reason)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if show_docs:
        st.markdown('<div class="docs-title">Documentos recuperados</div>', unsafe_allow_html=True)
        docs = result.get("source_documents", [])[:show_docs_limit]
        if not docs:
            st.info("Nenhum documento recuperado foi retornado. O sistema pode ter acionado fallback antes da recuperação.")
            return
        for i, doc in enumerate(docs, start=1):
            source_name = doc.get("source", "Fonte não identificada")
            content = doc.get("content", "")
            with st.expander(f"{i}. {source_name}", expanded=(i == 1)):
                st.markdown(_safe(content))


def render_footer():
    st.markdown(
        f"""
        <div class="footer">
            {APP_TITLE} · Protótipo acadêmico · RAG + Agentic RAG · Fontes oficiais do Ministério da Saúde
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="MedSimpli · Agentic RAG",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session()
    inject_css()
    render_sidebar()

    options = ["tinyllama", "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]
    rag_model_name = st.sidebar.selectbox(
        "Modelo",
        options=options,
        index=options.index(DEFAULT_RAG_MODEL) if DEFAULT_RAG_MODEL in options else 0,
    )
    rag_top_k = st.sidebar.slider("Top-K (documentos)", 1, 5, min(DEFAULT_TOP_K, 5))
    rag_temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, DEFAULT_RAG_TEMPERATURE, 0.1)
    show_docs = st.sidebar.toggle("Exibir documentos recuperados", value=True)
    use_llm_generation = st.sidebar.toggle("Gerar resposta com LLM local (pesado)", value=False)

    if st.sidebar.button("🧹 Limpar cache e memória", use_container_width=True):
        st.session_state.last_result = None
        clear_runtime_memory(clear_streamlit_cache=True)
        st.success("Cache limpo. Tente novamente com Top-K baixo.")
        st.rerun()

    st.sidebar.markdown(
        f"""
        <div class="disclaimer">
            <div style="font-weight:900;color:#1e2a52;margin-bottom:8px;">Aviso acadêmico</div>
            {_safe(DISCLAIMER)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_hero()
    render_workflow()
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
                with st.spinner("Executando Agentic RAG: classificando, recuperando, validando e gerando resposta..."):
                    result = run_agentic_rag_cached(
                        query=query,
                        top_k=rag_top_k,
                        prev_model_name=rag_model_name,
                        embed_model_name=DEFAULT_EMBED_MODEL,
                        faiss_index_path=DEFAULT_FAISS_PATH,
                        temperature=rag_temperature,
                        use_llm_generation=use_llm_generation,
                    )
                st.session_state.last_result = result
            except Exception as exc:
                clear_runtime_memory(clear_streamlit_cache=False)
                st.error(f"Erro ao executar o pipeline Agentic RAG: {exc}")
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
