from pypdf import PdfReader


def extract_pages_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with open(pdf_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for idx, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append((idx, page_text.strip()))
    return pages


def extract_text_from_pdf(pdf_path: str) -> str:
    text_parts: list[str] = []
    for _, page_text in extract_pages_from_pdf(pdf_path):
        if page_text.strip():
            text_parts.append(page_text.strip())
    return "\n\n".join(text_parts).strip()