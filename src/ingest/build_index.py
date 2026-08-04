from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import settings
from src.data.pdf_preprocessor import (
    export_processed_chunks,
    load_pdf_documents,
    load_web_json_documents,
    split_documents,
)


COLLECTION_NAME = "startup_support_docs"


def _document_id(doc: Document) -> str:
    metadata = doc.metadata or {}
    identity = "|".join(
        [
            str(metadata.get("source_file", "")),
            str(metadata.get("page_number", "")),
            str(metadata.get("source_url", "")),
            doc.page_content,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class IndexBuilder:
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        self.persist_dir = settings.vector_db_dir

    def build(self) -> tuple[int, str]:
        raw_docs = load_pdf_documents(settings.raw_docs_dir) + load_web_json_documents(
            settings.raw_docs_dir
        )
        if not raw_docs:
            raise ValueError("data/raw 폴더에 PDF 또는 JSON 데이터가 없습니다.")

        chunks = split_documents(raw_docs)
        preview_path = export_processed_chunks(chunks, settings.processed_docs_dir)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # A full build replaces the collection. Deterministic IDs make retries
        # idempotent and prevent duplicate chunks from accumulating.
        existing = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        try:
            existing.delete_collection()
        except ValueError:
            pass

        Chroma.from_documents(
            documents=chunks,
            ids=[_document_id(chunk) for chunk in chunks],
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=COLLECTION_NAME,
        )
        return len(chunks), preview_path

    def load_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )


def docs_to_tokenized_corpus(docs: List[Document]) -> List[List[str]]:
    return [doc.page_content.lower().split() for doc in docs]
