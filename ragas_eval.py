from pathlib import Path
import traceback

import pandas as pd
from datasets import Dataset
from ragas import evaluate

from rag_response import respond_to_query
from agentic.agent import run_medsimpli_agent


QUESTIONS = [
    {
        "question": "O que são hepatites virais?",
        "ground_truth": (
            "Hepatites virais são infecções causadas pelos vírus A, B, C, D e E, "
            "que atingem as células do fígado e causam inflamação. Elas podem ser "
            "agudas ou crônicas, e algumas podem não apresentar sintomas por muito tempo."
        ),
    },
    {
        "question": "O que é hanseníase?",
        "ground_truth": (
            "Hanseníase é uma doença crônica causada pelo Mycobacterium leprae, "
            "também chamado bacilo de Hansen. Pode causar incapacidades físicas "
            "e estigma social, mas tem tratamento e cura."
        ),
    },
    {
        "question": "O que é dengue?",
        "ground_truth": (
            "Dengue é uma arbovirose urbana causada pelo vírus dengue e transmitida "
            "principalmente pela picada da fêmea do mosquito Aedes aegypti."
        ),
    },
    {
        "question": "O que é Zika?",
        "ground_truth": (
            "Zika é uma arbovirose causada pelo vírus Zika. Pode ser transmitida por "
            "mosquitos do gênero Aedes e também por outras formas, como transmissão "
            "sexual e vertical."
        ),
    },
]


EMBED_MODEL_NAME = "pucpr/biobertpt-all"
LLM_MODEL_NAME = "qwen2.5:7b"
TOP_K = 2
FAISS_INDEX_PATH = "faiss_vectorstore"

OUTPUT_DIR = Path("ragas_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_metrics():
    """
    Compatível com versões diferentes do RAGAS.
    Se der erro aqui, verificar versão instalada:
    python -c "import ragas; print(ragas.__version__)"
    """

    try:
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )

        return [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]

    except Exception:
        pass

    try:
        from ragas.metrics.collections import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        return [
            faithfulness() if callable(faithfulness) else faithfulness,
            answer_relevancy() if callable(answer_relevancy) else answer_relevancy,
            context_precision() if callable(context_precision) else context_precision,
            context_recall() if callable(context_recall) else context_recall,
        ]

    except Exception as error:
        raise ImportError(
            "Não foi possível carregar as métricas do RAGAS. "
            "Tente instalar uma versão estável: pip install ragas==0.1.21"
        ) from error


def extract_answer(result: dict) -> str:
    return (
        result.get("answer")
        or result.get("response")
        or result.get("output")
        or ""
    )


def extract_classic_contexts(result: dict) -> list[str]:
    contexts = []
    source_documents = result.get("source_documents", [])

    for doc in source_documents:
        if isinstance(doc, dict):
            content = doc.get("content", "")
        else:
            content = getattr(doc, "page_content", "")

        if content:
            contexts.append(content)

    return contexts


def extract_agentic_contexts(result: dict) -> list[str]:
    contexts = []

    for ctx in result.get("contexts", []):
        if isinstance(ctx, dict):
            content = ctx.get("content", "")
        else:
            content = str(ctx)

        if content:
            contexts.append(content)

    return contexts


def run_classic_rag(question: str) -> dict:
    result = respond_to_query(
        embed_model_name=EMBED_MODEL_NAME,
        prev_model_name=LLM_MODEL_NAME,
        top_k=TOP_K,
        query=question,
        faiss_index_path=FAISS_INDEX_PATH,
        temperature=0.2,
        verbose=False,
    )

    return {
        "answer": extract_answer(result),
        "contexts": extract_classic_contexts(result),
        "status": "answered",
        "query_used": question,
        "intent": None,
    }


def run_agentic_rag(question: str) -> dict:
    result = run_medsimpli_agent(
        user_input=question,
        use_mock=False,
        top_k=TOP_K,
    )

    return {
        "answer": extract_answer(result),
        "contexts": extract_agentic_contexts(result),
        "status": result.get("status"),
        "query_used": result.get("query_used"),
        "intent": result.get("intent"),
    }


def build_rows(pipeline_name: str) -> list[dict]:
    rows = []

    for item in QUESTIONS:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print("=" * 80)
        print(f"Pipeline: {pipeline_name}")
        print(f"Pergunta: {question}")

        if pipeline_name == "classic_rag":
            output = run_classic_rag(question)
        elif pipeline_name == "agentic_rag":
            output = run_agentic_rag(question)
        else:
            raise ValueError(f"Pipeline desconhecido: {pipeline_name}")

        answer = output["answer"]
        contexts = output["contexts"]

        print(f"Status: {output.get('status')}")
        print(f"Intent: {output.get('intent')}")
        print(f"Query usada: {output.get('query_used')}")
        print(f"Contextos recuperados: {len(contexts)}")
        print("Resposta:")
        print(answer[:500])

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "reference": ground_truth,
                "pipeline": pipeline_name,
                "status": output.get("status"),
                "intent": output.get("intent"),
                "query_used": output.get("query_used"),
            }
        )

    return rows


def dataset_from_rows(rows: list[dict]) -> Dataset:
    eval_rows = []

    for row in rows:
        eval_rows.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
                "reference": row["reference"],
            }
        )

    return Dataset.from_list(eval_rows)


def evaluate_pipeline(pipeline_name: str) -> pd.DataFrame:
    rows = build_rows(pipeline_name)

    raw_df = pd.DataFrame(rows)
    raw_path = OUTPUT_DIR / f"{pipeline_name}_raw_outputs.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"Saídas brutas salvas em: {raw_path}")

    dataset = dataset_from_rows(rows)
    metrics = get_metrics()

    print("Métricas carregadas:")
    for metric in metrics:
        print("-", type(metric).__name__)

    result = evaluate(
        dataset,
        metrics=metrics,
    )

    scores = result.to_pandas()
    scores["pipeline"] = pipeline_name

    score_path = OUTPUT_DIR / f"{pipeline_name}_ragas_scores.csv"
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")
    print(f"Scores salvos em: {score_path}")

    return scores


def main():
    all_scores = []

    for pipeline_name in ["classic_rag", "agentic_rag"]:
        print("\n" + "#" * 80)
        print(f"### Avaliando {pipeline_name} ###")
        print("#" * 80 + "\n")

        try:
            scores = evaluate_pipeline(pipeline_name)
            all_scores.append(scores)

        except Exception as error:
            print(f"Erro ao avaliar {pipeline_name}: {error}")
            traceback.print_exc()

    if not all_scores:
        print("Nenhum resultado RAGAS foi gerado.")
        return

    final_scores = pd.concat(all_scores, ignore_index=True)

    comparison_path = OUTPUT_DIR / "ragas_results_comparison.csv"
    final_scores.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print("\n=== RESULTADOS RAGAS ===")
    print(final_scores)
    print(f"\nResultados salvos em: {comparison_path}")

    summary = final_scores.groupby("pipeline").mean(numeric_only=True)

    summary_path = OUTPUT_DIR / "ragas_results_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")

    print("\n=== MÉDIA POR PIPELINE ===")
    print(summary)
    print(f"\nResumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()