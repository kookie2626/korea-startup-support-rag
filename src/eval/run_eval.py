from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from langchain_core.documents import Document

from src.config import settings
from src.ingest.build_index import IndexBuilder
from src.rag.qa_chain import answer_with_citations
from src.rag.reranker import rerank_documents
from src.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class EvalCase:
    question: str
    must_include: List[str]


EVAL_CASES = [
    EvalCase(
        question="만 39세 서울 거주 예비 창업자가 받을 수 있는 지원금은?",
        must_include=["출처", "서울", "예비"],
    ),
    EvalCase(
        question="예비창업패키지와 청년창업사관학교 지원 항목 차이점을 알려줘.",
        must_include=["출처", "차이", "지원"],
    ),
    EvalCase(
        question="업력 3년 제조업 기업의 정책자금 신청 가능 여부는?",
        must_include=["출처", "정책자금"],
    ),
]


def run_eval() -> list[dict]:
    builder = IndexBuilder()
    vectorstore = builder.load_vectorstore()
    stored = vectorstore.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(stored["documents"], stored["metadatas"])
    ]
    retriever = HybridRetriever(vectorstore=vectorstore, base_docs=docs)

    report: list[dict] = []
    pass_count = 0
    citation_count = 0
    refusal_count = 0
    for case in EVAL_CASES:
        retrieval_result = retriever.retrieve(case.question)
        reranked_docs, rerank_ok, rerank_top_score = rerank_documents(
            query=case.question,
            documents=retrieval_result.documents,
            top_n=settings.rerank_top_n,
            threshold=settings.rerank_threshold,
        )
        answer = answer_with_citations(case.question, reranked_docs if rerank_ok else [])
        keyword_hits = [keyword for keyword in case.must_include if keyword in answer]
        has_citation = bool(re.search(r"출처\s*:", answer))
        refused = not rerank_ok
        passed = rerank_ok and has_citation and len(keyword_hits) == len(case.must_include)
        pass_count += int(passed)
        citation_count += int(has_citation)
        refusal_count += int(refused)
        report.append(
            {
                "question": case.question,
                "passed": passed,
                "must_include": case.must_include,
                "keyword_hit_count": len(keyword_hits),
                "keyword_hit_ratio": round(len(keyword_hits) / len(case.must_include), 3),
                "has_citation": has_citation,
                "refused": refused,
                "retrieved_doc_count": len(reranked_docs),
                "rerank_ok": rerank_ok,
                "rerank_top_score": round(rerank_top_score, 4),
                "applied_filters": retrieval_result.applied_filters,
                "answer_preview": answer[:300],
            }
        )

    total_cases = len(EVAL_CASES)
    summary = {
        "total_cases": total_cases,
        "passed_cases": pass_count,
        "pass_rate": round(pass_count / total_cases, 3) if total_cases else 0,
        "citation_rate": round(citation_count / total_cases, 3) if total_cases else 0,
        "refusal_rate": round(refusal_count / total_cases, 3) if total_cases else 0,
    }
    return [{"summary": summary}, *report]
