from agentic.skills import (
    classify_intent_skill,
    rewrite_query_skill,
    retrieve_context_skill,
    validate_context_skill,
    extract_terms_skill,
    generate_simple_answer_skill,
    fallback_skill,
)


def is_definition_question(user_input: str) -> bool:
    normalized = user_input.lower().strip()

    definition_starts = [
        "o que é",
        "o que e",
        "o que são",
        "o que sao",
        "defina",
        "explique o que é",
        "explique o que e",
        "explique o que são",
        "explique o que sao",
    ]

    return any(normalized.startswith(start) for start in definition_starts)


def choose_query_to_use(user_input: str, queries: list[str]) -> str:
    """
    Para perguntas de definição, usa a pergunta original.
    Isso evita transformar "O que é dengue?" em uma busca ampla como
    "dengue sintomas transmissão prevenção", que pode puxar chunks errados.

    Para perguntas mais abertas, usa a query reescrita mais específica.
    """
    if is_definition_question(user_input):
        return user_input

    if queries:
        return queries[-1]

    return user_input


def build_fallback_response(
    reason: str,
    intent: str,
    intent_reason: str,
    queries: list[str] | None = None,
    query_used: str | None = None,
    validation: dict | None = None,
    contexts: list[dict] | None = None,
) -> dict:
    response = fallback_skill(reason)

    return {
        **response,
        "status": "fallback",
        "intent": intent,
        "intent_reason": intent_reason,
        "queries": queries or [],
        "query_used": query_used,
        "validation": validation,
        "contexts": contexts or [],
        "sources": [],
    }


def run_medsimpli_agent(
    user_input: str,
    use_mock: bool = True,
    top_k: int = 2,
) -> dict:
    intent_result = classify_intent_skill(user_input)
    intent = intent_result["intent"]
    intent_reason = intent_result["reason"]

    if intent in ["FORA_DO_DOMINIO", "PERGUNTA_SENSIVEL"]:
        return build_fallback_response(
            reason=intent_reason,
            intent=intent,
            intent_reason=intent_reason,
        )

    queries = rewrite_query_skill(user_input)
    query_to_use = choose_query_to_use(user_input, queries)

    contexts = retrieve_context_skill(
        query=query_to_use,
        top_k=top_k,
        use_mock=use_mock,
    )

    validation = validate_context_skill(user_input, contexts)

    if validation["should_fallback"]:
        return build_fallback_response(
            reason=validation["reason"],
            intent=intent,
            intent_reason=intent_reason,
            queries=queries,
            query_used=query_to_use,
            validation=validation,
            contexts=contexts,
        )

    extracted_terms = []

    if intent in ["SIMPLIFICAR_TEXTO_MEDICO", "EXPLICAR_TERMO"]:
        extracted_terms = extract_terms_skill(user_input).get("terms", [])

    response = generate_simple_answer_skill(
        question=user_input,
        contexts=contexts,
        terms=extracted_terms,
    )

    return {
        **response,
        "status": "answered",
        "intent": intent,
        "intent_reason": intent_reason,
        "validation": validation,
        "queries": queries,
        "query_used": query_to_use,
        "contexts": contexts,
    }