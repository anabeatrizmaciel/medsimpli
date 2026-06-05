import json
from pathlib import Path

from pypdf import PdfReader


RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


DOCUMENT_METADATA = {
    "abcde-das-hepatites-2024_atualizado.pdf": {
        "title": "ABCDE das Hepatites Virais",
        "year": 2024,
        "type": "hepatites_virais",
        "priority": 1,
    },
    "caderno_tematico_pse_doencas_negligenciadas.pdf": {
        "title": "Caderno Temático do Programa Saúde na Escola - Prevenção de Doenças Negligenciadas",
        "year": 2022,
        "type": "doencas_negligenciadas",
        "priority": 1,
    },
    "guia_pratico_arboviroses_urbanas_aps.pdf": {
        "title": "Guia Prático de Arboviroses Urbanas - Atenção Primária à Saúde",
        "year": 2024,
        "type": "arboviroses",
        "priority": 1,
    },
}


def extract_pages_from_pdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []

    total_pages = len(reader.pages)
    print(f"Total de páginas: {total_pages}")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

        print(f"Página {page_number}/{total_pages} extraída")

    return pages


def build_full_text(pages: list[dict]) -> str:
    text_blocks = []

    for page in pages:
        page_number = page["page"]
        page_text = page["text"]

        text_blocks.append(f"\n\n[PÁGINA {page_number}]\n{page_text}")

    return "\n".join(text_blocks)


def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Nenhum PDF encontrado em data/raw.")
        return

    print(f"PDFs encontrados: {len(pdf_files)}")

    for pdf_path in pdf_files:
        metadata = DOCUMENT_METADATA.get(
            pdf_path.name,
            {
                "title": pdf_path.stem,
                "year": None,
                "type": "documento",
                "priority": 3,
            },
        )

        print("=" * 80)
        print(f"Extraindo texto de: {pdf_path.name}")

        pages = extract_pages_from_pdf(pdf_path)
        full_text = build_full_text(pages)

        output = {
            "title": metadata["title"],
            "year": metadata["year"],
            "type": metadata["type"],
            "priority": metadata["priority"],
            "source_file": pdf_path.name,
            "total_pages": len(pages),
            "pages": pages,
            "text": full_text,
        }

        output_path = CLEANED_DIR / f"{pdf_path.stem}.json"

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

        print(f"JSON salvo em: {output_path}")
        print(f"Páginas com texto extraído: {len(pages)}")
        print(f"Tamanho total do texto: {len(full_text)} caracteres")

    print("=" * 80)
    print("Conversão concluída.")


if __name__ == "__main__":
    main()