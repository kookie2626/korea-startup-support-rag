from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


REGION_KEYWORDS = [
    "서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종",
    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]
STAGE_KEYWORDS = ["예비", "초기", "도약", "재도전"]


def _extract_regions(text: str) -> str:
    return "|".join(region for region in REGION_KEYWORDS if region in text)


def _extract_stages(text: str) -> str:
    return "|".join(stage for stage in STAGE_KEYWORDS if stage in text)


def _extract_age_bucket(text: str) -> str:
    lowered = text.replace(" ", "")
    if "청년" in text or "만39세" in lowered or re.search(r"만\s*3[0-9]\s*세", text):
        return "youth"
    if "중장년" in text or "시니어" in text:
        return "senior"
    return "all"


def _extract_age_range(text: str) -> tuple[int | None, int | None]:
    age_min: int | None = None
    age_max: int | None = None

    for value, operator in re.findall(r"만\s*(\d{1,3})\s*세\s*(이상|초과)", text):
        candidate = int(value) + (1 if operator == "초과" else 0)
        age_min = candidate if age_min is None else max(age_min, candidate)

    for value, operator in re.findall(r"만\s*(\d{1,3})\s*세\s*(이하|미만)", text):
        candidate = int(value) - (1 if operator == "미만" else 0)
        age_max = candidate if age_max is None else min(age_max, candidate)

    return age_min, age_max


def _annotate_metadata(doc: Document) -> None:
    text = doc.page_content
    age_min, age_max = _extract_age_range(text)
    doc.metadata["regions"] = _extract_regions(text)
    doc.metadata["stages"] = _extract_stages(text)
    doc.metadata["age_bucket"] = _extract_age_bucket(text)
    # Chroma metadata cannot store None, so omit unknown bounds.
    doc.metadata.pop("age_min", None)
    doc.metadata.pop("age_max", None)
    if age_min is not None:
        doc.metadata["age_min"] = age_min
    if age_max is not None:
        doc.metadata["age_max"] = age_max


def load_pdf_documents(raw_docs_dir: str) -> List[Document]:
    raw_path = Path(raw_docs_dir)
    if not raw_path.exists():
        return []

    docs: List[Document] = []
    for pdf_path in sorted(raw_path.glob("*.pdf")):
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
        except Exception as exc:
            print(f"[경고] PDF 로딩 실패: {pdf_path.name}: {exc}")
            continue
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1
            page.metadata["source_kind"] = "pdf"
        docs.extend(pages)
    return docs


def load_web_json_documents(raw_docs_dir: str) -> List[Document]:
    raw_path = Path(raw_docs_dir)
    if not raw_path.exists():
        return []

    docs: List[Document] = []
    for json_path in sorted(raw_path.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[경고] JSON 로딩 실패: {json_path.name}: {exc}")
            continue
        if not isinstance(payload, list):
            print(f"[경고] JSON 최상위 값이 목록이 아님: {json_path.name}")
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            body = str(item.get("body", "")).strip()
            if not body:
                continue

            meta = {
                "source_file": json_path.name,
                "page_number": int(item.get("page_number", 0) or 0),
                "source_kind": "web",
                "source_site": item.get("source_site", "web"),
                "source_url": item.get("url", ""),
                "title": item.get("title", ""),
                "notice_id": item.get("notice_id", ""),
                "parent_url": item.get("parent_url", ""),
                "deadline": item.get("deadline", ""),
                "organization": item.get("organization", ""),
                "support_type": item.get("support_type", ""),
                "region": item.get("region", ""),
                "doc_type": "web_page",
            }
            docs.append(Document(page_content=body, metadata=meta))

            links = item.get("links", [])
            if not isinstance(links, list):
                continue
            for link in links:
                if not isinstance(link, dict):
                    continue
                link_title = str(link.get("title", "")).strip()
                link_url = str(link.get("url", "")).strip()
                notice_id = str(link.get("notice_id", "")).strip()
                if not link_title and not link_url and not notice_id:
                    continue

                link_text = "\n".join(
                    [
                        f"링크 제목: {link_title}",
                        f"링크 URL: {link_url}",
                        f"공고 ID: {notice_id}",
                        f"출처 사이트: {meta['source_site']}",
                    ]
                )
                link_meta = dict(meta)
                link_meta.update(
                    {
                        "doc_type": "web_link",
                        "title": link_title or meta["title"],
                        "link_title": link_title,
                        "link_url": link_url,
                        "source_url": link_url or meta["source_url"],
                        "notice_id": notice_id,
                    }
                )
                docs.append(Document(page_content=link_text, metadata=link_meta))
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    # Derive eligibility metadata from each chunk, not the whole source page.
    for chunk in chunks:
        _annotate_metadata(chunk)
    return chunks


def export_processed_chunks(chunks: List[Document], processed_docs_dir: str) -> str:
    processed_dir = Path(processed_docs_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    export_path = processed_dir / "chunks_preview.json"
    serialized = [
        {"id": index, "text": chunk.page_content, "metadata": chunk.metadata}
        for index, chunk in enumerate(chunks)
    ]
    export_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(export_path)
