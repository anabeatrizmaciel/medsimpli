from rag_prep import restart_vectorstore
from rag_response import respond_to_query

EMBED_MODEL_NAME = "pucpr/biobertpt-all"
PREV_MODEL_NAME = "qwen2.5:14b"
TOP_K = 5
QUERY = "Como se trata a doença de Chagas?"
FAISS_INDEX_PATH = "faiss_vectorstore"
TEMPERATURE = 0.2

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
VECTORDB_PATH = "faiss_vectorstore"


def run_rag_for_streamlit(
    query: str,
    top_k: int = TOP_K,
    prev_model_name: str = PREV_MODEL_NAME,
    embed_model_name: str = EMBED_MODEL_NAME,
    faiss_index_path: str = FAISS_INDEX_PATH,
    temperature: float = TEMPERATURE,
):
    """Executa o pipeline RAG e retorna resposta + documentos para o app."""
    return respond_to_query(
        embed_model_name=embed_model_name,
        prev_model_name=prev_model_name,
        top_k=top_k,
        query=query,
        faiss_index_path=faiss_index_path,
        temperature=temperature,
        verbose=False,
    )


def test_response_to_query(
    query: str = QUERY,
    top_k: int = TOP_K,
    prev_model_name: str = PREV_MODEL_NAME,
    embed_model_name: str = EMBED_MODEL_NAME,
    faiss_index_path: str = FAISS_INDEX_PATH,
    temperature: float = TEMPERATURE,
):
    """Testa o pipeline usando um vectorstore já existente."""
    return respond_to_query(
        embed_model_name=embed_model_name,
        prev_model_name=prev_model_name,
        top_k=top_k,
        query=query,
        faiss_index_path=faiss_index_path,
        temperature=temperature,
        verbose=True,
    )


def test_rag_from_start(
    query: str = QUERY,
    top_k: int = TOP_K,
    prev_model_name: str = PREV_MODEL_NAME,
    embed_model_name: str = EMBED_MODEL_NAME,
    faiss_index_path: str = FAISS_INDEX_PATH,
    temperature: float = TEMPERATURE,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    vectordb_path: str = VECTORDB_PATH,
):
    """Recria o vectorstore e depois executa o pipeline."""
    restart_vectorstore(
        index_path=vectordb_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
    )

    return respond_to_query(
        embed_model_name=embed_model_name,
        prev_model_name=prev_model_name,
        top_k=top_k,
        query=query,
        faiss_index_path=faiss_index_path,
        temperature=temperature,
        verbose=True,
    )


def main():
    resultado = test_rag_from_start()
    print("\n=== RESPOSTA FINAL ===\n")
    print(resultado["answer"])


if __name__ == "__main__":
    main()
