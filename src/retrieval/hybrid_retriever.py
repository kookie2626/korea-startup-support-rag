from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import settings


REGIONS = [
    "서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종",
    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]
STAGES = ["예비", "초기", "도약", "재도전"]


@dataclass
class RetrievalResult:
    documents: List[Document]
    applied_filters: dict


@dataclass
class QueryProfile:
    regions: List[str]
    stages: List[str]
    age: int | None

    @property
    def has_constraints(self) -> bool:
        return bool(self.regions or self.stages or self.age is not None)


def _tokenize_ko_en(text: str) -> list[str]:
    """Whitespace tokens plus Korean character bigrams for lightweight BM25."""
    words = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
    tokens: list[str] = []
    for word in words:
        tokens.append(word)
        if re.fullmatch(r"[가-힣]+", word) and len(word) >= 2:
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


def _parse_query_profile(query: str) -> QueryProfile:
    regions = [region for region in REGIONS if region in query]
    stages = [stage for stage in STAGES if stage in query]
    age_match = re.search(r"(?:만\s*)?(\d{1,3})\s*세", query)
    age = int(age_match.group(1)) if age_match else None
    return QueryProfile(regions=regions, stages=stages, age=age)


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _is_doc_match(profile: QueryProfile, doc: Document) -> bool:
    metadata = doc.metadata or {}
    content = doc.page_content

    if profile.regions:
        region_text = f"{metadata.get('region', '')}|{metadata.get('regions', '')}"
        if not any(region in region_text or region in content for region in profile.regions):
            return False

    if profile.stages:
        stage_text = str(metadata.get("stages", ""))
        if not any(stage in stage_text or stage in content for stage in profile.stages):
            return False

    if profile.age is not None:
        age_min = _as_int(metadata.get("age_min"))
        age_max = _as_int(metadata.get("age_max"))
        if age_min is not None and profile.age < age_min:
            return False
        if age_max is not None and profile.age > age_max:
            return False

        # Backward-compatible fallback for indexes built before age_min/age_max.
        age_bucket = str(metadata.get("age_bucket", "all"))
        if age_min is None and age_max is None:
            if age_bucket == "youth" and profile.age > 39:
                return False
            if age_bucket == "senior" and profile.age < 40:
                return False

    return True


def _normalize_structured_filters(filters: dict | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in ("organization", "region", "support_type"):
        value = str((filters or {}).get(key, "")).strip()
        if value and value != "전체":
            normalized[key] = value
    return normalized


def _is_structured_match(doc: Document, filters: dict[str, str]) -> bool:
    metadata = doc.metadata or {}
    if organization := filters.get("organization"):
        if organization not in str(metadata.get("organization", "")):
            return False
    if region := filters.get("region"):
        region_text = f"{metadata.get('region', '')}|{metadata.get('regions', '')}"
        if region not in region_text:
            return False
    if support_type := filters.get("support_type"):
        if support_type not in str(metadata.get("support_type", "")):
            return False
    return True


def _doc_key(doc: Document) -> tuple[str, object, str, str]:
    metadata = doc.metadata or {}
    return (
        str(metadata.get("source_file", "unknown")),
        metadata.get("page_number", -1),
        str(metadata.get("source_url", "")),
        doc.page_content[:120],
    )


def _rrf_merge(rankings: Iterable[List[Document]], rrf_k: int = 60) -> List[Document]:
    scores: dict[tuple[str, object, str, str], float] = {}
    documents: dict[tuple[str, object, str, str], Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = _doc_key(doc)
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return [documents[key] for key in sorted(scores, key=scores.get, reverse=True)]


class HybridRetriever:
    def __init__(self, vectorstore, base_docs: List[Document]) -> None:
        self.vectorstore = vectorstore
        self.base_docs = base_docs
        self.corpus = [_tokenize_ko_en(doc.page_content) for doc in base_docs]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        candidates: List[Document] | None = None,
    ) -> List[Document]:
        docs = self.base_docs if candidates is None else candidates
        if not docs:
            return []

        bm25 = self.bm25
        if candidates is not None:
            bm25 = BM25Okapi([_tokenize_ko_en(doc.page_content) for doc in docs])
        if bm25 is None:
            return []

        scores = bm25.get_scores(_tokenize_ko_en(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[index] for index in top_indices]

    def _vector_search(
        self,
        query: str,
        top_k: int,
        profile: QueryProfile,
        structured_filters: dict[str, str],
    ) -> List[Document]:
        # Retrieve a broader pool before applying compound filters that Chroma
        # cannot express against pipe-delimited legacy metadata.
        candidate_k = max(top_k * 4, top_k)
        candidates = self.vectorstore.similarity_search(query=query, k=candidate_k)
        filtered = [
            doc
            for doc in candidates
            if _is_doc_match(profile, doc) and _is_structured_match(doc, structured_filters)
        ]
        return filtered[:top_k]

    def retrieve(self, query: str, structured_filters: dict | None = None) -> RetrievalResult:
        profile = _parse_query_profile(query)
        normalized_filters = _normalize_structured_filters(structured_filters)
        has_filters = profile.has_constraints or bool(normalized_filters)

        eligible_docs = [
            doc
            for doc in self.base_docs
            if _is_doc_match(profile, doc) and _is_structured_match(doc, normalized_filters)
        ]

        applied_filters = {
            "regions": profile.regions,
            "stages": profile.stages,
            "age": profile.age,
            "structured_filters": normalized_filters,
            "filter_applied": has_filters,
            "filter_hit": bool(eligible_docs) if has_filters else True,
        }

        # Eligibility filters are fail-closed: never silently answer from
        # documents that contradict explicit user conditions.
        if has_filters and not eligible_docs:
            applied_filters["filter_miss"] = True
            return RetrievalResult(documents=[], applied_filters=applied_filters)

        bm25_docs = self._bm25_search(
            query,
            settings.top_k_bm25,
            candidates=eligible_docs if has_filters else None,
        )
        vector_docs = self._vector_search(
            query,
            settings.top_k_vector,
            profile,
            normalized_filters,
        )
        merged_docs = _rrf_merge([bm25_docs, vector_docs])
        limit = settings.top_k_bm25 + settings.top_k_vector
        return RetrievalResult(
            documents=merged_docs[:limit],
            applied_filters=applied_filters,
        )
