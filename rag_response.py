from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from rag_prep import calling_ollama_model, get_vectorstore, PROMPT

EMBED_MODEL_NAME = "pucpr/biobertpt-all"
PREV_MODEL_NAME = "qwen2.5:14b"
TOP_K = 5
QUERY = "Como se trata a doença de Chagas?"
FAISS_INDEX_PATH = "faiss_vectorstore"
TEMPERATURE = 0.2


def respond_to_query(embed_model_name=EMBED_MODEL_NAME, prev_model_name=PREV_MODEL_NAME, top_k=TOP_K, query=QUERY, faiss_index_path=FAISS_INDEX_PATH, temperature=TEMPERATURE):
    """
    Pipeline para chamada do modelo de geração de texto.
    Obs: usando modelo do HuggingFace. Caso outra plataforma seja
    usada, o pipeline muda um pouco, principalmente na função 'calling_model()'.
    """
    print("Carregando modelo de linguagem do Ollama...")

    llm = calling_ollama_model(prev_model_name, temperature)

    print("Carregando Embeddings do Hugging Face...")

    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Carregando VectorStore FAISS...")

    vectorstore = get_vectorstore(faiss_index_path, None, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )

    print("Calculando resposta para a pergunta...")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=multi_query_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    resposta = qa_chain.invoke(query)

    print("Documentos mais relevantes para a pergunta:")

    for doc in resposta["source_documents"]:
        print("\n------------------------------\n")
        print(f"Fonte: {doc.metadata['source']}")
        print(f"- {doc.page_content}")

    print("\n------------------------------\n")
    print(f"Resposta: {resposta['result']}")
    return

def main():
    respond_to_query()
    return

if __name__ == "__main__":
    main()