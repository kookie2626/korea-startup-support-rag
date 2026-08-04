from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from src.config import settings
from src.data.web_collectors import collect_and_save_default
from src.eval.run_eval import run_eval
from src.ingest.build_index import IndexBuilder
from src.rag.qa_chain import answer_with_citations
from src.rag.reranker import rerank_documents
from src.retrieval.hybrid_retriever import HybridRetriever


def _load_base_docs(vectorstore) -> list[Document]:
    fetched = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(fetched["documents"], fetched["metadatas"])
    ]


def command_build_index() -> None:
    chunk_count, preview_path = IndexBuilder().build()
    print(f"[완료] 인덱스 생성: {chunk_count} chunks")
    print(f"[완료] 전처리 미리보기: {preview_path}")


def command_collect_web() -> None:
    output_path = str(Path(settings.raw_docs_dir) / "web_seed.json")
    count, saved_path = collect_and_save_default(output_path)
    print(f"[완료] 웹 데이터 수집: {count}건")
    print(f"[완료] 저장 파일: {saved_path}")


def command_ask(question: str) -> None:
    builder = IndexBuilder()
    vectorstore = builder.load_vectorstore()
    docs = _load_base_docs(vectorstore)
    retrieval_result = HybridRetriever(vectorstore=vectorstore, base_docs=docs).retrieve(question)
    reranked_docs, rerank_ok, rerank_top_score = rerank_documents(
        query=question,
        documents=retrieval_result.documents,
        top_n=settings.rerank_top_n,
        threshold=settings.rerank_threshold,
    )
    # Fail closed when no document reaches the configured relevance threshold.
    answer = answer_with_citations(question, reranked_docs if rerank_ok else [])

    print("\n=== 질문 ===")
    print(question)
    print("\n=== 적용 필터 ===")
    print(json.dumps(retrieval_result.applied_filters, ensure_ascii=False, indent=2))
    print("\n=== 리랭커 ===")
    print(
        json.dumps(
            {
                "rerank_ok": rerank_ok,
                "rerank_top_score": round(rerank_top_score, 4),
                "threshold": settings.rerank_threshold,
                "doc_count_after_rerank": len(reranked_docs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n=== 답변 ===")
    print(answer)


def command_eval() -> None:
    print(json.dumps(run_eval(), ensure_ascii=False, indent=2))


def command_run_all() -> None:
    print("[1/3] 웹 데이터 수집 시작")
    command_collect_web()
    print("[2/3] 인덱스 생성 시작")
    command_build_index()
    print("[3/3] 평가 실행 시작")
    command_eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="창업지원 문서 RAG 시스템")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build-index", help="PDF/웹 문서를 전처리하고 인덱스를 재생성")
    subparsers.add_parser("collect-web", help="정책 사이트 데이터를 수집해 JSON으로 저장")
    subparsers.add_parser("run-all", help="웹 수집, 인덱스 생성, 평가를 순차 실행")
    ask_parser = subparsers.add_parser("ask", help="질문에 대한 답변 생성")
    ask_parser.add_argument("question", type=str, help="사용자 질문")
    subparsers.add_parser("eval", help="기본 평가 시나리오 실행")
    args = parser.parse_args()

    commands = {
        "build-index": command_build_index,
        "collect-web": command_collect_web,
        "run-all": command_run_all,
        "eval": command_eval,
    }
    if args.command == "ask":
        command_ask(args.question)
    else:
        commands[args.command]()


if __name__ == "__main__":
    main()
