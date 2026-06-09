import json
import re
import shutil
import time
import unicodedata
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Outros nomes testados: pucpr/biobertpt-all, mixedbread-ai/mxbai-embed-large-v1, sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
TOP_K = 5
DATA_DIR = "data/cleaned"
DEFAULT_INDEX_PATH = "faiss_vectorstore"

# Mantido como no RAG atual
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


PROMPT_TEMPLATE = """
Você é um assistente do MedSimpli, um sistema de apoio à compreensão de linguagem médica em português brasileiro.

O objetivo do MedSimpli é ajudar usuários a entender termos médicos, doenças, sintomas, exames e orientações de saúde por meio de explicações simples, claras e acessíveis, sempre com base em fontes confiáveis recuperadas pelo sistema.

Contexto recuperado:
{context}

Pergunta do usuário:
{question}

Sua tarefa é responder à pergunta usando apenas o contexto fornecido.

Siga estas regras:
- use apenas as informações presentes no contexto recuperado;
- não invente informações e não complemente com suposições;
- se o contexto não contiver informação suficiente, responda exatamente:
"Não encontrei informações suficientes sobre esse tema na base do MedSimpli. Consulte um profissional de saúde.";
- escreva em português brasileiro claro e objetivo;
- evite jargões desnecessários;
- quando existir um termo popular equivalente ao termo técnico, mencione-o entre parênteses;
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
    input_variables=["context", "question"],
)


def log(message: str):
    current_time = time.strftime("%H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)


def remove_accents(text: str) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = remove_accents(text.lower())
    return " ".join(text.split())


def clean_extracted_text(text: str) -> str:
    """
    Limpeza leve do texto extraído do PDF.
    Não reescreve conteúdo; só reduz hifens quebrados, quebras excessivas e espaços repetidos.
    """
    if not text:
        return ""

    # "hepa-\ntite" -> "hepatite"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Remove quebras demais.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove espaços repetidos.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def is_noise_page(text: str) -> bool:
    """
    Remove páginas editoriais ou sem conteúdo útil para RAG:
    capa, ficha catalográfica, sumário, referências, bibliografia e página final de pesquisa.
    """
    normalized = normalize_text(text)

    if not normalized:
        return True

    # Muito curto geralmente é capa, divisória ou página final.
    if len(normalized) < 80:
        return True

    noise_markers = [
        "conte-nos o que pensa sobre esta publicacao",
        "clique aqui e responda a pesquisa",
        "clique aqui e responda a pesquisa",
        "responda a pesquisa",
        "ficha catalografica",
        "catalogacao na fonte",
        "catalogacao-na-fonte",
        "biblioteca virtual em saude",
        "bvsms.saude.gov.br",
        "isbn",
        "modo de acesso",
        "titulo para indexacao",
        "tiragem:",
        "elaboracao, distribuicao e informacoes",
        "ministra de estado da saude",
        "secretario de atencao primaria",
        "secretaria de vigilancia em saude",
        "coordenacao-geral",
        "coordenacao editorial",
        "projeto grafico",
        "diagramacao",
        "normalizacao:",
        "revisao de texto",
        "revisao tecnica",
        "colaboracao:",
        "sumario",
        "referencias",
        "bibliografia",
        "bibliografia consultada",
    ]

    if any(marker in normalized for marker in noise_markers):
        return True

    # Páginas praticamente só de referências bibliográficas.
    reference_like_terms = [
        "world health organization",
        "organizacao mundial da saude",
        "disponivel em:",
        "acesso em:",
        "et al.",
        "geneva:",
        "who,",
        "brasilia:",
    ]

    reference_hits = sum(1 for marker in reference_like_terms if marker in normalized)

    if reference_hits >= 3:
        return True

    return False


def extract_query_terms(query: str) -> list[str]:
    """
    Detecta termos centrais de doenças/agravos na pergunta.
    Isso ajuda a corrigir o problema do retrieval vetorial puro em perguntas curtas.
    """
    query_normalized = normalize_text(query)

    disease_terms = [
        "dengue",
        "zika",
        "chikungunya",
        "hanseniase",
        "esquistossomose",
        "malaria",
        "tuberculose",
        "hepatite",
        "hepatites",
        "hepatites virais",
        "raiva",
        "tracoma",
        "leishmaniose",
        "leishmanioses",
        "chagas",
        "doenca de chagas",
        "filariose",
        "filariose linfatica",
        "oncocercose",
        "geo-helmintiase",
        "geo-helmintiases",
        "helmintiase",
        "helmintiases",
        "doencas diarreicas",
        "diarreia",
        "arbovirose",
        "arboviroses",
    ]

    found_terms = []

    # Ordena por tamanho para termos compostos aparecerem primeiro.
    for term in sorted(disease_terms, key=len, reverse=True):
        if term in query_normalized:
            found_terms.append(term)

    # Remove duplicatas preservando ordem.
    return list(dict.fromkeys(found_terms))


def keyword_score(query_terms: list[str], content: str, metadata: dict) -> int:
    """
    Dá bônus para chunks que contêm literalmente a doença/termo da pergunta.
    O objetivo não é substituir FAISS, mas reranquear os candidatos iniciais.
    """
    text = normalize_text(content)
    source = normalize_text(str(metadata.get("source", "")))
    source_file = normalize_text(str(metadata.get("source_file", "")))
    doc_type = normalize_text(str(metadata.get("type", "")))

    score = 0

    for term in query_terms:
        normalized_term = normalize_text(term)

        if normalized_term in text:
            score += 20

        if normalized_term in source:
            score += 8

        if normalized_term in source_file:
            score += 6

        if normalized_term in doc_type:
            score += 4

    # Bônus para trechos que parecem seção de definição.
    definition_markers = [
        "o que e?",
        "o que e",
        "sao causadas",
        "e uma doenca",
        "e uma zoonose",
        "e causada",
        "e causado",
        "as hepatites virais sao",
        "a dengue destaca-se",
        "quanto ao virus zika",
        "a chikungunya",
    ]

    if any(marker in text for marker in definition_markers):
        score += 5

    # Penaliza chunks que são claramente não explicativos, mesmo se passaram pelo filtro.
    weak_markers = [
        "atencao!",
        "o que nao se deve fazer",
        "conte-nos",
        "referencias",
        "bibliografia",
    ]

    if any(marker in text for marker in weak_markers):
        score -= 5

    return score


def load_docs(path: str) -> dict:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    log(f"Lendo JSON: {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as file:
        file_data = json.load(file)

    title = file_data.get("title", file_path.stem)
    text = clean_extracted_text(file_data.get("text", ""))
    pages = file_data.get("pages", [])

    metadata = {
        "source": title,
        "source_file": file_data.get("source_file", file_path.name),
        "year": file_data.get("year"),
        "type": file_data.get("type"),
        "priority": file_data.get("priority"),
        "total_pages": file_data.get("total_pages"),
    }

    log(f"Documento carregado: {title}")
    log(f"Tamanho do texto completo: {len(text)} caracteres")
    log(f"Páginas estruturadas encontradas: {len(pages)}")

    return {
        title: {
            "text": text,
            "pages": pages,
            "metadata": metadata,
        }
    }


def load_all_docs(directory: str = DATA_DIR) -> dict:
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")

    log("=" * 80)
    log(f"Procurando arquivos JSON em: {directory_path}")

    json_files = sorted(directory_path.glob("*.json"))
    log(f"Arquivos JSON encontrados: {len(json_files)}")

    if not json_files:
        raise ValueError(f"Nenhum arquivo JSON encontrado em {directory_path}")

    all_docs = {}

    for index, file_path in enumerate(json_files, start=1):
        log("-" * 80)
        log(f"Arquivo {index}/{len(json_files)}")
        doc = load_docs(str(file_path))
        all_docs.update(doc)

    log("-" * 80)
    log(f"Total de documentos carregados: {len(all_docs)}")
    log("=" * 80)

    return all_docs


def split_texts(
    docs: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    log("=" * 80)
    log("Iniciando chunking")
    log(f"Chunk size: {chunk_size}")
    log(f"Chunk overlap: {chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []
    ignored_pages_total = 0

    for doc_index, (title, payload) in enumerate(docs.items(), start=1):
        metadata = payload["metadata"]
        pages = payload.get("pages", [])
        full_text = payload.get("text", "")

        log("-" * 80)
        log(f"Documento {doc_index}/{len(docs)}: {title}")

        if pages:
            log("Usando páginas estruturadas para preservar metadados por página.")

            before_count = len(all_chunks)
            ignored_pages = 0
            used_pages = 0

            for page_item in pages:
                page_number = page_item.get("page")
                page_text = clean_extracted_text(page_item.get("text", ""))

                if not page_text:
                    ignored_pages += 1
                    ignored_pages_total += 1
                    continue

                if is_noise_page(page_text):
                    ignored_pages += 1
                    ignored_pages_total += 1
                    log(
                        f"Página {page_number} ignorada por ruído editorial/referências."
                    )
                    continue

                page_document = Document(
                    page_content=page_text,
                    metadata={
                        **metadata,
                        "page": page_number,
                    },
                )

                page_chunks = splitter.split_documents([page_document])
                all_chunks.extend(page_chunks)
                used_pages += 1

            generated = len(all_chunks) - before_count

            log(f"Páginas usadas neste documento: {used_pages}")
            log(f"Páginas ignoradas neste documento: {ignored_pages}")
            log(f"Chunks deste documento: {generated}")
            log(f"Chunks acumulados: {len(all_chunks)}")

        else:
            log("JSON sem campo 'pages'. Usando campo 'text' completo.")

            if is_noise_page(full_text):
                log("Documento ignorado por parecer ruído editorial/referências.")
                continue

            document = Document(
                page_content=full_text,
                metadata={
                    **metadata,
                    "page": None,
                },
            )

            chunks = splitter.split_documents([document])
            all_chunks.extend(chunks)

            log(f"Chunks deste documento: {len(chunks)}")
            log(f"Chunks acumulados: {len(all_chunks)}")

    log("-" * 80)
    log(f"Total final de chunks: {len(all_chunks)}")
    log(f"Total de páginas ignoradas: {ignored_pages_total}")
    log("=" * 80)

    if not all_chunks:
        raise ValueError("Nenhum chunk foi gerado. Verifique os filtros de ruído.")

    return all_chunks


def get_embeddings(embed_model_name: str = MODEL_NAME):
    log("=" * 80)
    log(f"Carregando modelo de embeddings: {embed_model_name}")
    log("Essa etapa pode demorar na primeira execução.")

    start = time.time()

    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    log(f"Modelo de embeddings carregado em {time.time() - start:.2f} segundos.")
    log("=" * 80)

    return embeddings


def get_vectorstore(index_path: str, chunks=None, embeddings=None):
    index_path_obj = Path(index_path)

    if embeddings is None:
        embeddings = get_embeddings()

    if index_path_obj.exists():
        log("=" * 80)
        log(f"Carregando FAISS existente em: {index_path}")
        vectorstore = FAISS.load_local(
            str(index_path_obj),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        log("FAISS carregado com sucesso.")
        log("=" * 80)
        return vectorstore

    if chunks is None:
        raise ValueError(
            "FAISS não existe e nenhum chunk foi fornecido para criação do índice."
        )

    log("=" * 80)
    log("Criando novo índice FAISS.")
    log(f"Quantidade de chunks para indexar: {len(chunks)}")
    log("Essa é a etapa mais pesada: embeddings dos chunks + indexação.")
    log("Com muitos PDFs e pouca RAM, pode demorar bastante.")

    start = time.time()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    log(f"FAISS criado em {time.time() - start:.2f} segundos.")

    log(f"Salvando FAISS em: {index_path}")
    vectorstore.save_local(str(index_path_obj))
    log("FAISS salvo com sucesso.")
    log("=" * 80)

    return vectorstore


def build_full_vectorstore(
    index_path: str = DEFAULT_INDEX_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_model_name: str = MODEL_NAME,
    data_dir: str = DATA_DIR,
):
    log("=" * 80)
    log("BUILD FULL VECTORSTORE")
    log(f"Data dir: {data_dir}")
    log(f"Index path: {index_path}")
    log(f"Chunk size: {chunk_size}")
    log(f"Chunk overlap: {chunk_overlap}")
    log(f"Embedding model: {embed_model_name}")
    log("=" * 80)

    start = time.time()

    docs = load_all_docs(data_dir)
    chunks = split_texts(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embeddings = get_embeddings(embed_model_name)

    vectorstore = get_vectorstore(
        index_path=index_path,
        chunks=chunks,
        embeddings=embeddings,
    )

    log("=" * 80)
    log(f"Vectorstore finalizado em {time.time() - start:.2f} segundos.")
    log("=" * 80)

    return vectorstore


def restart_vectorstore(
    index_path: str = DEFAULT_INDEX_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_model_name: str = MODEL_NAME,
    data_dir: str = DATA_DIR,
    force_rebuild: bool = False,
):
    log("=" * 80)
    log("RESTART VECTORSTORE")
    log(f"Index path: {index_path}")
    log(f"Data dir: {data_dir}")
    log(f"Chunk size: {chunk_size}")
    log(f"Chunk overlap: {chunk_overlap}")
    log(f"Embedding model: {embed_model_name}")
    log(f"Force rebuild: {force_rebuild}")
    log("=" * 80)

    index_path_obj = Path(index_path)

    if index_path_obj.exists() and not force_rebuild:
        log("FAISS já existe e force_rebuild=False.")
        log("O índice existente será carregado. Nenhum novo índice será criado.")
        embeddings = get_embeddings(embed_model_name)
        return get_vectorstore(
            index_path=index_path,
            chunks=None,
            embeddings=embeddings,
        )

    if force_rebuild and index_path_obj.exists():
        log(f"Removendo FAISS antigo em: {index_path}")
        shutil.rmtree(index_path_obj)

    return build_full_vectorstore(
        index_path=index_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        data_dir=data_dir,
    )


def retrieve_documents(
    query: str,
    top_k: int = TOP_K,
    index_path: str = DEFAULT_INDEX_PATH,
    embed_model_name: str = MODEL_NAME,
):
    log("=" * 80)
    log("TESTE DE RECUPERAÇÃO SEM LLM")
    log(f"Pergunta: {query}")
    log(f"Top-k final: {top_k}")
    log("=" * 80)

    embeddings = get_embeddings(embed_model_name)
    vectorstore = get_vectorstore(
        index_path=index_path,
        chunks=None,
        embeddings=embeddings,
    )

    query_terms = extract_query_terms(query)
    fetch_k = max(20, top_k * 10)

    log(f"Busca vetorial inicial com fetch_k={fetch_k}")

    docs = vectorstore.similarity_search(query, k=fetch_k)

    if query_terms:
        log(f"Termos detectados na pergunta: {query_terms}")

        reranked_docs = []

        for doc in docs:
            score = keyword_score(
                query_terms=query_terms,
                content=doc.page_content,
                metadata=doc.metadata,
            )

            reranked_docs.append(
                {
                    "doc": doc,
                    "keyword_score": score,
                }
            )

        reranked_docs.sort(
            key=lambda item: item["keyword_score"],
            reverse=True,
        )

        if reranked_docs and reranked_docs[0]["keyword_score"] == 0:
            log(
                "Nenhum candidato continha diretamente os termos da pergunta. Mantendo ordem vetorial."
            )
            selected_docs = docs[:top_k]
        else:
            selected_docs = [item["doc"] for item in reranked_docs[:top_k]]

    else:
        log("Nenhum termo específico detectado. Mantendo ordem vetorial.")
        selected_docs = docs[:top_k]

    results = []

    for index, doc in enumerate(selected_docs, start=1):
        result = {
            "rank": index,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "source": doc.metadata.get("source"),
            "source_file": doc.metadata.get("source_file"),
            "page": doc.metadata.get("page"),
            "type": doc.metadata.get("type"),
            "year": doc.metadata.get("year"),
        }
        results.append(result)

    return results


def calling_ollama_model(model_name: str, temperature: float = 0.2):
    log(f"Carregando modelo Ollama: {model_name}")
    return OllamaLLM(model=model_name, temperature=temperature)


def main():
    restart_vectorstore(
        index_path=DEFAULT_INDEX_PATH,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        embed_model_name=MODEL_NAME,
        data_dir=DATA_DIR,
        force_rebuild=True,
    )

    generate_retrieval_report()


def generate_retrieval_report(
    index_path: str = DEFAULT_INDEX_PATH,
    embed_model_name: str = MODEL_NAME,
    data_dir: str = DATA_DIR,
    top_k: int = 3,
    output_dir: str = "embeddings_results",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    test_questions = [
        "O que é dengue?",
        "O que significa CID?",
        "Como a malária é transmitida?",
        "O que são hepatites virais?",
        "O que é esquistossomose?",
    ]

    log("=" * 80)
    log("Gerando relatório de recuperação")
    log(f"Modelo: {embed_model_name}")
    log(f"Índice: {index_path}")
    log(f"Top-k: {top_k}")
    log("=" * 80)

    results = {}

    for question in test_questions:
        log(f"Processando: {question}")
        docs = retrieve_documents(
            query=question,
            top_k=top_k,
            index_path=index_path,
            embed_model_name=embed_model_name,
        )
        results[question] = docs

    lines = []
    lines.append("# Relatório de Recuperação — MedSimpli\n")
    lines.append(f"_Modelo de embedding: `{embed_model_name}`_\n")
    lines.append(f"_Índice FAISS: `{index_path}`_\n")
    lines.append(f"_Gerado em: {time.strftime('%d/%m/%Y %H:%M')}_\n")
    lines.append("---\n")
    lines.append("## Resultados\n")

    for question, docs in results.items():
        lines.append(f'### Pergunta: "{question}"\n')

        if not docs:
            lines.append("_Nenhum documento recuperado._\n")
            continue

        for doc in docs:
            lines.append(
                f"- **Rank {doc['rank']}** — {doc['source']}, "
                f"p. {doc['page']}, {doc['year']}  \n"
            )
            snippet = doc["content"][:300].replace("\n", " ")
            lines.append(f"  ```\n  {snippet}\n  ```\n")

        lines.append("")

    report_path = output_path / f"retrieval_report_{MODEL_NAME.split('/')[0]}.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    log(f"Relatório salvo em: {report_path}")

    return str(report_path)


if __name__ == "__main__":
    main()
