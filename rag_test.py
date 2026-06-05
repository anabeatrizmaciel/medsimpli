import argparse
import time
from pathlib import Path

from rag_prep import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_INDEX_PATH,
    MODEL_NAME,
    restart_vectorstore,
    retrieve_documents,
)
from rag_response import respond_to_query


EMBED_MODEL_NAME = MODEL_NAME
PREV_MODEL_NAME = "qwen2.5:3b"
TOP_K = 2
QUERY = "O que são hepatites virais?"
FAISS_INDEX_PATH = DEFAULT_INDEX_PATH
TEMPERATURE = 0.2
CHUNK_SIZE = DEFAULT_CHUNK_SIZE
CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP


def log(message: str):
    current_time = time.strftime("%H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)


def faiss_exists(index_path: str = FAISS_INDEX_PATH) -> bool:
    index_dir = Path(index_path)
    return index_dir.exists()


def ensure_or_build_faiss(
    index_path: str = FAISS_INDEX_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_model_name: str = EMBED_MODEL_NAME,
):
    if faiss_exists(index_path):
        log("FAISS encontrado. Usando índice existente.")
        return

    log("FAISS não encontrado. Criando índice antes do teste.")
    restart_vectorstore(
        index_path=index_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=False,
    )


def build_index(
    force_rebuild: bool = False,
    index_path: str = FAISS_INDEX_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_model_name: str = EMBED_MODEL_NAME,
):
    log("=" * 80)
    log("MODO BUILD")
    log("Este modo cria o índice FAISS se ele não existir.")
    log("Se o índice já existir, ele será carregado e NÃO apagado.")
    log("Para apagar e recriar, use explicitamente --force-rebuild.")
    log(f"Force rebuild: {force_rebuild}")
    log("=" * 80)

    start = time.time()

    restart_vectorstore(
        index_path=index_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )

    log(f"Build finalizado em {time.time() - start:.2f} segundos.")


def run_rag_for_streamlit(
    query: str,
    top_k: int = TOP_K,
    prev_model_name: str = PREV_MODEL_NAME,
    embed_model_name: str = EMBED_MODEL_NAME,
    faiss_index_path: str = FAISS_INDEX_PATH,
    temperature: float = TEMPERATURE,
):
    """
    Função usada pelo Streamlit.
    Garante que o FAISS exista.
    Se não existir, cria.
    Se existir, usa o índice atual.
    """
    ensure_or_build_faiss(
        index_path=faiss_index_path,
        embed_model_name=embed_model_name,
    )

    return respond_to_query(
        embed_model_name=embed_model_name,
        prev_model_name=prev_model_name,
        top_k=top_k,
        query=query,
        faiss_index_path=faiss_index_path,
        temperature=temperature,
        verbose=False,
    )


def test_retrieval_only(
    query: str = QUERY,
    top_k: int = TOP_K,
    index_path: str = FAISS_INDEX_PATH,
    embed_model_name: str = EMBED_MODEL_NAME,
):
    ensure_or_build_faiss(
        index_path=index_path,
        embed_model_name=embed_model_name,
    )

    log("=" * 80)
    log("MODO RETRIEVAL")
    log("Este modo testa apenas a recuperação do FAISS, sem chamar LLM.")
    log(f"Pergunta: {query}")
    log(f"Top-k: {top_k}")
    log("=" * 80)

    start = time.time()

    docs = retrieve_documents(
        query=query,
        top_k=top_k,
        index_path=index_path,
        embed_model_name=embed_model_name,
    )

    log(f"Recuperação finalizada em {time.time() - start:.2f} segundos.")

    print("\n=== DOCUMENTOS RECUPERADOS ===\n")

    if not docs:
        print("Nenhum documento recuperado.")
        return

    for doc in docs:
        print("=" * 80)
        print(f"Rank: {doc['rank']}")
        print(f"Fonte: {doc.get('source')}")
        print(f"Arquivo: {doc.get('source_file')}")
        print(f"Página: {doc.get('page')}")
        print(f"Tipo: {doc.get('type')}")
        print(f"Ano: {doc.get('year')}")
        print("-" * 80)
        print(doc["content"][:1000])
        print()


def test_response_to_query(
    query: str = QUERY,
    top_k: int = TOP_K,
    prev_model_name: str = PREV_MODEL_NAME,
    embed_model_name: str = EMBED_MODEL_NAME,
    faiss_index_path: str = FAISS_INDEX_PATH,
    temperature: float = TEMPERATURE,
):
    ensure_or_build_faiss(
        index_path=faiss_index_path,
        embed_model_name=embed_model_name,
    )

    log("=" * 80)
    log("MODO RESPONSE")
    log("Este modo usa FAISS existente ou cria se não existir.")
    log("Ele NÃO apaga o índice existente.")
    log(f"Pergunta: {query}")
    log(f"Modelo LLM: {prev_model_name}")
    log(f"Embedding: {embed_model_name}")
    log(f"Top-k: {top_k}")
    log(f"FAISS path: {faiss_index_path}")
    log("=" * 80)

    start = time.time()

    result = respond_to_query(
        embed_model_name=embed_model_name,
        prev_model_name=prev_model_name,
        top_k=top_k,
        query=query,
        faiss_index_path=faiss_index_path,
        temperature=temperature,
        verbose=True,
    )

    log(f"Resposta finalizada em {time.time() - start:.2f} segundos.")

    return result


def print_result(result: dict):
    print("\n=== RESPOSTA FINAL ===\n")
    print(result.get("answer", "Nenhuma resposta retornada."))

    print("\n=== FONTES RECUPERADAS ===\n")

    source_documents = result.get("source_documents", [])

    if not source_documents:
        print("Nenhuma fonte recuperada.")
        return

    for index, doc in enumerate(source_documents, start=1):
        print("=" * 80)
        print(f"Fonte {index}")
        print("Source:", doc.get("source", "Fonte não identificada"))
        print("-" * 80)
        print(doc.get("content", "")[:1000])
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Testes do pipeline RAG do MedSimpli.")

    parser.add_argument(
        "--build",
        action="store_true",
        help="Cria o índice FAISS se ele não existir. Se já existir, usa o existente.",
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Apaga e recria o índice FAISS. Use com cuidado.",
    )

    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Testa apenas recuperação FAISS, sem LLM.",
    )

    parser.add_argument(
        "--response",
        action="store_true",
        help="Testa resposta completa com LLM usando FAISS existente ou criando se não existir.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=QUERY,
        help="Pergunta para testar.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Quantidade de documentos recuperados.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=PREV_MODEL_NAME,
        help="Modelo LLM usado pelo Ollama.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.force_rebuild:
        build_index(force_rebuild=True)
        return

    if args.build:
        build_index(force_rebuild=False)
        return

    if args.retrieval:
        test_retrieval_only(
            query=args.query,
            top_k=args.top_k,
        )
        return

    if args.response:
        result = test_response_to_query(
            query=args.query,
            top_k=args.top_k,
            prev_model_name=args.model,
        )
        print_result(result)
        return

    # Comportamento padrão:
    # Se FAISS existe, usa.
    # Se FAISS não existe, cria.
    # Depois testa resposta.
    log("Nenhum modo informado. Rodando comportamento padrão.")
    log("Se o FAISS existir, ele será usado.")
    log("Se não existir, ele será criado.")
    log("O índice existente NÃO será apagado.")

    result = test_response_to_query(
        query=args.query,
        top_k=args.top_k,
        prev_model_name=args.model,
    )
    print_result(result)


if __name__ == "__main__":
    main()