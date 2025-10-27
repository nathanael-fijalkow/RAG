from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict

from pypdf import PdfReader


@dataclass
class Document:
    content: str
    metadata: Dict[str, str]


def load_pdf(path: str | Path, doc_id: str | None = None) -> List[Document]:
    p = Path(path)
    reader = PdfReader(str(p))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(
            Document(
                content=text,
                metadata={
                    "source": str(p).split("/")[-1],
                    "page": str(i + 1),
                    "doc_id": doc_id or p.stem,
                    "type": "pdf",
                },
            )
        )
    return pages


def load_folder_pdfs(folder: str | Path) -> List[Document]:
    folder = Path(folder)
    docs: List[Document] = []
    for pdf in sorted(folder.glob("*.pdf")):
        docs.extend(load_pdf(pdf))
    return docs


def load_text(path: str | Path, doc_id: str | None = None) -> List[Document]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return [
        Document(
            content=text,
            metadata={
                "source": str(p).split("/")[-1],
                "page": "1",
                "doc_id": doc_id or p.stem,
                "type": "txt",
            },
        )
    ]


def load_folder(folder: str | Path) -> List[Document]:
    folder = Path(folder)
    docs: List[Document] = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() == ".pdf":
            docs.extend(load_pdf(p))
        elif p.suffix.lower() in {".txt", ".md"}:
            docs.extend(load_text(p))
    return docs
