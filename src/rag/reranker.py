from __future__ import annotations

import re
from typing import List, Tuple

from langchain_core.documents import Document


def _tokenize_ko_en(text: str) -> set[str]:
    words = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
    tokens: set[str] = set(words)
    for word in words:
        if re.fullmatch(r"[가-힣]+", word) and len(word) >= 2:
            tokens.update(word[index : index + 2] for index in range(len(word) - 1))
    return tokens


def _score_doc(query: str, doc: Document) -> float:
    query_tokens = _tokenize_ko_en(query)
    document_tokens = _tokenize_ko_en(doc.page_content)
    if not query_tokens or not document_tokens:
        return 0.0

    overlap = len(query_tokens & document_tokens) / len(query_tokens)
    metadata = doc.metadata or {}
    source_quality = 0.0
    if str(metadata.get("source_url", "")).strip():
        source_quality += 0.02
    if str(metadata.get("notice_id", "")).strip():
        source_quality += 0.01
    return float(overlap + source_quality)


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 6,
    threshold: float = 0.08,
) -> Tuple[List[Document], bool, float]:
    if not documents:
        return [], False, 0.0

    scored = sorted(
        [(_score_doc(query, doc), doc) for doc in documents],
        key=lambda item: item[0],
        reverse=True,
    )
    top_score = float(scored[0][0]) if scored else 0.0
    passed = [doc for score, doc in scored if score >= threshold]
    if not passed:
        return [], False, top_score
    return passed[:top_n], True, top_score
