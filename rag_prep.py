from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
import os
import json

MODEL_NAME = "pucpr/biobertpt-all"
TOP_K = 5

PROMPT_TEMPLATE = """
Você é um assistente do MedSimpli, um sistema de apoio à compreensão
de linguagem médica em português brasileiro.

O objetivo do MedSimpli é ajudar usuários a entender termos médicos,
doenças, sintomas, exames e orientações de saúde por meio de
explicações simples, claras e acessíveis, sempre com base em fontes
confiáveis recuperadas pelo sistema.

Contexto recuperado:
{context}

Pergunta do usuário: {question}

Sua tarefa é responder à pergunta usando apenas o contexto fornecido.

Siga estas regras:
- use apenas as informações presentes no contexto recuperado;
- não invente informações e não complemente com suposições;
- se o contexto não contiver informação suficiente, responda exatamente:
  "Não encontrei informações suficientes sobre esse tema na base do
  MedSimpli. Consulte um profissional de saúde.";
- escreva em português brasileiro claro e objetivo;
- evite jargões desnecessários;
- quando existir um termo popular equivalente ao termo técnico,
  mencione-o entre parênteses;
- quando útil, organize a resposta em tópicos curtos;
- não forneça diagnóstico;
- não prescreva tratamento;
- não substitua a avaliação de um profissional de saúde.

Formato esperado da resposta:
1. explicação simples;
2. pontos principais, se necessário;
3. aviso de limitação, quando aplicável.

Resposta:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def load_docs(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo {path} não encontrado.")
    
    with open(path, "r", encoding="utf-8") as f:
        file_data = json.load(f)
        
    
    return {
        file_data["title"]: file_data["text"]
    }

def load_all_docs(directory: str):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Diretório {directory} não encontrado.")
    
    all_texts = {}
    for file in os.listdir(directory):
        if file.endswith(".json"):
            texts = load_docs(os.path.join(directory, file))
            all_texts.update(texts)
    
    return all_texts

def split_texts(texts: dict, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    documents = [
        Document(page_content=text, metadata={"source": f"doc_{title}"})
        for title, text in texts.items()
    ]
    return splitter.split_documents(documents)

def get_vectorstore(index_path, chunks, embeddings):
    if os.path.exists(index_path):
        print("Carregando índice de vetor existente...")
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Criando novo índice de vetor...")
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(index_path)
        print(f"Índice de vetor salvo em {index_path}.")
        return vectordb

def build_full_vectorstore(index_path: str, chunk_size: int, chunk_overlap: int, embeddings):
    texts = load_all_docs("data/cleaned")

    chunks = split_texts(texts, chunk_size, chunk_overlap)

    vectordb = get_vectorstore(index_path, chunks, embeddings)

    return vectordb

def restart_vectorstore(index_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_db = build_full_vectorstore(index_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embeddings=embeddings)
    return

def calling_hf_model(model_name: str):
    pipe = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def calling_ollama_model(model_name: str, temperature: float):
    llm = OllamaLLM(model=model_name, temperature=temperature)
    return llm

def test_chunk_sizes():
    chunk_size = [300, 400, 500, 600, 700]
    for file in os.listdir("data/cleaned/"):
        if file.endswith(".json"):
            file_name = file.split(".")[0]
            print(f"Processando arquivo: {file_name}")
        
        print("Carregando documentos...")
        texts = load_docs(f"data/cleaned/{file_name}.json")

        for c_size in chunk_size:
            print("Dividindo textos em chunks...")
            chunk_overlap = []
            chunk_overlap.append(int(c_size * 0.10))
            chunk_overlap.append(int(c_size * 0.15))
            chunk_overlap.append(int(c_size * 0.20))

            for c_overlap in chunk_overlap:
                print(f"Processando com chunk size {c_size} e chunk overlap {c_overlap}...")
                chunks = split_texts(texts, c_size, c_overlap)

                print("Criando embeddings...")
                embeddings = HuggingFaceEmbeddings(
                    model_name=MODEL_NAME,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )

                vectordb = get_vectorstore(chunks, embeddings)

                retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

                if file_name == "aedesaegypti":
                    query = "Como o Aedes aegypti se reproduz?"
                elif file_name == "arboviroses":
                    query = "O que são arboviroses?"
                elif file_name == "covid19":
                    query = "Quais os sintomas da COVID-19?"
                elif file_name == "dengue":
                    query = "O que é dengue?"
                elif file_name == "diabetes":
                    query = "Quais os riscos do diabetes?"
                elif file_name == "doencadechagas":
                    query = "Como se trata a doença de Chagas?"
                elif file_name == "elefantiase":
                    query = "O que é elefantíase?"
                elif file_name == "hipertensao":
                    query = "O que é hipertensão?"
                elif file_name == "esquistossomose":
                    query = "O que é esquistossomose?"
                elif file_name == "lupus":
                    query = "Quais os sintomas do lúpus?"
                elif file_name == "sarampo":
                    query = "Quais os sintomas do sarampo?"
                elif file_name == "malaria":
                    query = "Como se pega malária?"

                docs = retriever.invoke(query)

                with open(f"document_retrieval_test/documentos_relevantes_{file_name}_{c_size}_{c_overlap}.json", "w", encoding="utf-8") as f:
                    for i, doc in enumerate(docs):
                        doc_data = {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        json.dump(doc_data, f, ensure_ascii=False, indent=4)
                        f.write("\n")
    return

def main():
    # test_chunk_sizes()
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_db = build_full_vectorstore("faiss_vectorstore", chunk_size=500, chunk_overlap=100, embeddings=embeddings)

    respostas = vector_db.as_retriever(search_kwargs={"k": TOP_K}).invoke("Quais os sintomas do sarampo?")

    for resposta in respostas:
        print(resposta.metadata["source"] + "\n")
        print(resposta.page_content)
        print("\n------------------------------\n")


if __name__ == "__main__":
    main()