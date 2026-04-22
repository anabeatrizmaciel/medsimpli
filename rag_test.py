from rag_prep import restart_vectorstore, PROMPT
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


"""
Testa a resposta do modelo sem mudar os parâmetros 
do vectorstore e do modelo de embedding.
"""
def test_response_to_query():
    respond_to_query(
        embed_model_name=EMBED_MODEL_NAME,
        prev_model_name=PREV_MODEL_NAME,
        top_k=TOP_K,
        query=QUERY,
        faiss_index_path=FAISS_INDEX_PATH,
        temperature=TEMPERATURE
    )
    
    return

"""
Reinicia o processo de RAG desde a criação do vectorstore
com parâmetros definidos nas constantes globais.

ATENÇÃO: para reiniciar o vectorstore, é necessário deletar 
a pasta "faiss_vectorstore" se ela existir.
"""
def test_rag_from_start():
    restart_vectorstore(
        index_path=VECTORDB_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embed_model_name=EMBED_MODEL_NAME
    )

    respond_to_query(
        embed_model_name=EMBED_MODEL_NAME,
        prev_model_name=PREV_MODEL_NAME,
        top_k=TOP_K,
        query=QUERY,
        faiss_index_path=FAISS_INDEX_PATH,
        temperature=TEMPERATURE
    )
    
    return


def main():
    test_response_to_query()

if __name__ == "__main__":
    main()