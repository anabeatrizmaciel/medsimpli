import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MOCK_CONTEXTS_PATH = BASE_DIR / "mock_contexts.json"


def load_mock_contexts() -> list[dict]:
    with open(MOCK_CONTEXTS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def classify_intent_skill(user_input: str) -> dict:
    text = user_input.lower()

    sensitive_terms = [
        "qual remédio", "que remédio", "posso tomar", "devo tomar",
        "dose", "dosagem", "diagnóstico", "estou com", "o que eu tenho",
        "tratamento para mim"
    ]

    out_of_domain_terms = [
        "futebol", "jogo", "brasil ganhou", "capital da frança",
        "filme", "música", "programação em python"
    ]

    if any(term in text for term in sensitive_terms):
        return {
            "intent": "PERGUNTA_SENSIVEL",
            "reason": "A pergunta pode envolver diagnóstico, prescrição ou orientação clínica individual."
        }

    if any(term in text for term in out_of_domain_terms):
        return {
            "intent": "FORA_DO_DOMINIO",
            "reason": "A pergunta não está relacionada ao domínio de saúde do MedSimpli."
        }

    if "simplifique" in text or "explica esse laudo" in text or "laudo" in text or "bula" in text:
        return {
            "intent": "SIMPLIFICAR_TEXTO_MEDICO",
            "reason": "O usuário enviou ou mencionou um texto médico para simplificação."
        }

    if (
        "o que significa" in text
        or "significa" in text
        or "sigla" in text
        or re.match(r"^o que é [a-zá-úA-ZÁ-Ú0-9\- ]+\??$", user_input.strip())
    ):
        return {
            "intent": "EXPLICAR_TERMO",
            "reason": "O usuário está pedindo explicação de um termo, sigla ou conceito."
        }

    return {
        "intent": "PERGUNTA_SAUDE",
        "reason": "A pergunta parece estar relacionada a saúde pública, doença, prevenção ou vigilância."
    }


def rewrite_query_skill(user_input: str) -> list[str]:
    text = user_input.lower()
    queries = [user_input]

    expansions = {
        "pressão alta": ["hipertensão arterial", "pressão arterial elevada"],
        "aids": ["HIV aids infecção transmissão prevenção"],
        "hiv": ["infecção pelo HIV aids transmissão prevenção"],
        "dengue": ["arbovirose dengue sintomas transmissão prevenção"],
        "zika": ["arbovirose zika sintomas transmissão prevenção"],
        "chikungunya": ["arbovirose chikungunya sintomas transmissão prevenção"],
        "malária": ["malária transmissão sintomas prevenção vigilância"],
        "tuberculose": ["tuberculose transmissão sintomas diagnóstico vigilância"],
        "hanseníase": ["hanseníase transmissão sintomas diagnóstico vigilância"],
        "esquistossomose": ["esquistossomose transmissão sintomas prevenção vigilância"],
        "cid": ["classificação internacional de doenças CID"],
        "atenção básica": ["atenção básica saúde família SUS"]
    }

    for term, related_queries in expansions.items():
        if term in text:
            queries.extend(related_queries)

    return list(dict.fromkeys(queries))


def retrieve_context_skill(query: str, top_k: int = 5, use_mock: bool = True) -> list[dict]:
    if use_mock:
        contexts = load_mock_contexts()
        return _retrieve_from_mock(query, contexts, top_k)

    from rag_prep import retrieve_documents

    docs = retrieve_documents(
        query=query,
        top_k=top_k,
        index_path="faiss_vectorstore",
        embed_model_name="pucpr/biobertpt-all",
    )

    contexts = []

    for doc in docs:
        contexts.append(
            {
                "id": f"faiss_ctx_{doc.get('rank')}",
                "source": doc.get("source", "Fonte não identificada"),
                "source_file": doc.get("source_file"),
                "type": doc.get("type", "faiss_document"),
                "year": doc.get("year"),
                "page": doc.get("page"),
                "content": doc.get("content", ""),
                "score": doc.get("rank"),
                "metadata": doc.get("metadata", {}),
            }
        )

    return contexts


def _retrieve_from_mock(query: str, contexts: list[dict], top_k: int) -> list[dict]:
    query_terms = [
        term.strip(".,?!:;()[]{}").lower()
        for term in query.split()
        if len(term.strip(".,?!:;()[]{}")) > 2
    ]

    results = []

    for ctx in contexts:
        searchable_text = f"{ctx.get('content', '')} {ctx.get('source', '')} {ctx.get('type', '')}".lower()
        score = sum(1 for term in query_terms if term in searchable_text)

        if score > 0:
            item = ctx.copy()
            item["score"] = score
            results.append(item)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def validate_context_skill(question: str, contexts: list[dict]) -> dict:
    question_lower = question.lower()

    sensitive_terms = [
        "qual remédio", "que remédio", "posso tomar", "devo tomar",
        "dose", "dosagem", "diagnóstico", "o que eu tenho",
        "tratamento para mim"
    ]

    if any(term in question_lower for term in sensitive_terms):
        return {
            "supported": False,
            "confidence": "baixa",
            "should_fallback": True,
            "reason": "A pergunta envolve possível prescrição, diagnóstico ou orientação individual."
        }

    if not contexts:
        return {
            "supported": False,
            "confidence": "baixa",
            "should_fallback": True,
            "reason": "Nenhum contexto relevante foi recuperado."
        }

    best_score = contexts[0].get("score", 0)

    if best_score <= 0:
        return {
            "supported": False,
            "confidence": "baixa",
            "should_fallback": True,
            "reason": "Os contextos recuperados não parecem relacionados à pergunta."
        }

    return {
        "supported": True,
        "confidence": "media",
        "should_fallback": False,
        "reason": "Há contexto recuperado relacionado à pergunta."
    }


def extract_terms_skill(text: str) -> dict:
    known_terms = [
        "hipertensão arterial",
        "arbovirose",
        "dengue",
        "zika",
        "chikungunya",
        "malária",
        "tuberculose",
        "hanseníase",
        "esquistossomose",
        "tracoma",
        "HIV",
        "AIDS",
        "CID",
        "atenção básica",
        "vigilância epidemiológica"
    ]

    text_lower = text.lower()
    found_terms = []

    for term in known_terms:
        if term.lower() in text_lower:
            found_terms.append(term)

    return {
        "terms": found_terms
    }


def generate_simple_answer_skill(question: str, contexts: list[dict], terms: list[str] | None = None) -> dict:
    terms = terms or []

    sources = _format_sources(contexts)

    context_summary = " ".join(ctx["content"] for ctx in contexts[:2])

    answer = (
        "Resposta em linguagem simples:\n"
        f"Com base nos documentos recuperados, a pergunta está relacionada a: {context_summary}\n\n"
    )

    if terms:
        answer += "Termos importantes:\n"
        for term in terms:
            answer += f"- {term}\n"
        answer += "\n"
    else:
        answer += "Termos importantes:\n- Não foram extraídos termos técnicos específicos nesta etapa.\n\n"

    answer += "Fontes consultadas:\n"
    answer += sources
    answer += "\n\nObservação:\nEsta explicação tem finalidade informativa e não substitui orientação de um profissional de saúde."

    return {
        "status": "answered",
        "answer": answer,
        "sources": contexts[:5]
    }


def fallback_skill(reason: str) -> dict:
    return {
        "status": "fallback",
        "answer": (
            "Não encontrei informação suficiente na base do MedSimpli para responder com segurança. "
            "Essa explicação tem finalidade informativa e não substitui orientação de um profissional de saúde."
        ),
        "reason": reason,
        "sources": []
    }


def _format_sources(contexts: list[dict]) -> str:
    if not contexts:
        return "- Nenhuma fonte recuperada."

    lines = []

    for ctx in contexts[:5]:
        source = ctx.get("source", "Fonte não identificada")
        page = ctx.get("page", "página não informada")
        year = ctx.get("year", "ano não informado")
        lines.append(f"- {source}, {year}, p. {page}")

    return "\n".join(lines)