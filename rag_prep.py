from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
import os
import json

MODEL_NAME = "pucpr/biobertpt-all"
INDEX_PATH = "faiss_vectorstore"
TOP_K = 3

MAP_PROMPT = PromptTemplate(
    template="""
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
""",
    input_variables=["context", "question"]
)

COMBINE_PROMPT = PromptTemplate(
    template="""
Combine as respostas abaixo em uma única resposta clara e simplificada.

Respostas:
{summaries}

Resposta final:
""",
    input_variables=["summaries"]
)

def load_docs(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo {path} não encontrado.")
    
    with open(path, "r", encoding="utf-8") as f:
        file_data = json.load(f)
        
    
    return {
        file_data["title"]: file_data["text"]
    }

def split_texts(texts: dict):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    documents = [
        Document(page_content=text, metadata={"source": f"doc_{title}"})
        for title, text in texts.items()
    ]
    return splitter.split_documents(documents)

def get_vectorstore(chunks, embeddings):
    if os.path.exists(INDEX_PATH):
        print("Carregando índice de vetor existente...")
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Criando novo índice de vetor...")
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(INDEX_PATH)
        print(f"Índice de vetor salvo em {INDEX_PATH}.")
        return vectordb

# def calling_model():
#     print("Carregando modelo de linguagem do Hugging Face...")

#     pipe = pipeline(
#         "text-generation",
#         model=MODEL_NAME,
#         max_new_tokens=256
#     )

#     llm = HuggingFacePipeline(pipeline=pipe)
#     return llm

def main():
    print("Carregando documentos...")
    texts = load_docs("data/cleaned/covid19.json")

    print("Dividindo textos em chunks...")
    chunks = split_texts(texts)

    print("Criando embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = get_vectorstore(chunks, embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    query = "Quais são os sintomas da COVID-19?"

    docs = retriever.invoke(query)

    for i, doc in enumerate(docs):
        print(f"\n--- Documento {i} ---")
        print(doc.page_content)
        print(doc.metadata)

    # llm = calling_model()

    # print("Calculando resposta para a pergunta...")

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="map_reduce",
    #     chain_type_kwargs={
    #         "question_prompt": MAP_PROMPT,
    #         "combine_prompt": COMBINE_PROMPT
    #     }
    # )

    # resposta = qa_chain.invoke("Quais os sintomas da COVID-19?")

    # print(resposta)
    return

if __name__ == "__main__":
    main()