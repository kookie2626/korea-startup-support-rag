from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings


SYSTEM_PROMPT = """
당신은 창업지원사업 전문 QA 어시스턴트입니다.
반드시 제공된 문서 컨텍스트 내부 정보만 사용해 답변하세요.
근거가 부족하면 추측하지 말고 확인할 수 없다고 답변하세요.
주요 주장 뒤에는 근거 문서 번호를 [문서1] 형식으로 표시하세요.
파일명, 페이지, URL로 된 출처 목록은 애플리케이션이 추가하므로 직접 만들지 마세요.
""".strip()


def _format_context(docs: List[Document]) -> str:
    lines = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source_file", "unknown")
        page = metadata.get("page_number", "?")
        title = metadata.get("title", "")
        source_url = metadata.get("source_url", "")
        source_kind = metadata.get("source_kind", "pdf")
        if source_kind == "web" or source_url:
            meta_line = f"{title or source} | {source_url}".strip(" |")
        else:
            meta_line = f"{source} p.{page}"
        lines.append(f"[문서{index}] ({meta_line})\n{doc.page_content}")
    return "\n\n".join(lines)


def _format_citations(docs: List[Document]) -> str:
    citations: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for doc in docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source_file", "unknown"))
        page = str(metadata.get("page_number", "?"))
        title = str(metadata.get("title", "")).strip()
        source_url = str(metadata.get("source_url", "")).strip()
        source_kind = str(metadata.get("source_kind", "pdf"))
        key = (source, page, source_url)
        if key in seen:
            continue
        seen.add(key)

        if source_kind == "web" or source_url:
            label = title or source
            citation = f"- {label}"
            if source_url:
                citation += f" | {source_url}"
        else:
            citation = f"- {source} p.{page}"
        citations.append(citation)
    return "\n".join(citations)


def answer_with_citations(query: str, docs: List[Document]) -> str:
    if not docs:
        return (
            "조건에 맞는 신뢰할 수 있는 근거 문서를 찾지 못했습니다. "
            "질문 조건을 완화하거나 데이터를 추가한 뒤 다시 시도해주세요."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "질문: {query}\n\n아래 컨텍스트만 사용해서 답변하세요:\n{context}\n\n"
                "정책과 지원 조건을 항목별로 정리하고 각 주장에 근거 문서 번호를 표시하세요.",
            ),
        ]
    )
    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    result = (prompt | llm).invoke({"query": query, "context": _format_context(docs)})
    answer = str(result.content).strip()
    return f"{answer}\n\n출처:\n{_format_citations(docs)}"
