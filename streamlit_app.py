from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from src.eval.run_eval import run_eval
from src.ingest.build_index import IndexBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.qa_chain import answer_with_citations
from src.rag.reranker import rerank_documents
from src.config import settings
from src.data.web_collectors import collect_and_save_default


_ABOUT_TEXT = """
## 창업지원 문서 기반 RAG 질의응답 시스템

정부·지자체 창업지원 공고문·정책자금 가이드 등 문서를 벡터 DB에 적재한 뒤,
사용자 질문에 대해 **출처(파일명/페이지)를 명시**하며 답변하는 RAG(검색증강생성) 시스템입니다.

### 핵심 기능
- **하이브리드 검색**: BM25 키워드 검색 + 벡터(시맨틱) 검색 결합
- **메타데이터 필터링**: 지역 / 기관 / 지원분야 조건 사전 필터
- **리랭킹**: 임계값 기반 경량 리랭커로 검색 결과 정밀화
- **출처 강제 표기**: 답변 말미에 `파일명 p.페이지 | URL` 형식으로 근거 제시
- **환각 방지**: 컨텍스트 외 정보 생성 금지 정책 적용

### 기술 스택
- [LangChain](https://www.langchain.com/) · [Chroma](https://www.trychroma.com/) · OpenAI Embedding & Chat · BM25 (rank-bm25)

### 데이터 출처
K-Startup 공고문, 중소벤처기업부·소진공 정책자금 가이드, 지자체 창업지원 공고

### 사용 방법
1. 사이드바 **웹 데이터 수집** → **인덱스 생성** 순으로 실행
2. 지역 / 기관 / 지원분야 필터 선택 후 질문 입력
3. 답변과 함께 출처 미리보기 확인
"""

st.set_page_config(
    page_title="창업지원 RAG 데모",
    page_icon="📄",
    layout="wide",
    menu_items={
        "About": _ABOUT_TEXT,
    },
)
st.title("창업지원 문서 RAG 데모")
st.caption("웹 수집 → 인덱스 생성 → 질의응답까지 한 화면에서 실행")

with st.expander("ℹ️ 이 시스템에 대하여 (About)", expanded=False):
    st.markdown(_ABOUT_TEXT)


def _history_path() -> Path:
    return Path(settings.processed_docs_dir) / "query_history.jsonl"


def _append_query_history(row: dict) -> None:
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_recent_history(limit: int = 20) -> list[dict]:
    path = _history_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _history_to_csv(rows: list[dict]) -> str:
    header = [
        "timestamp",
        "question",
        "region",
        "organization",
        "support_type",
        "retrieved_doc_count",
        "rerank_ok",
        "rerank_top_score",
        "answer_preview",
    ]
    lines = [",".join(header)]

    for row in rows:
        filters = row.get("filters", {}) if isinstance(row.get("filters", {}), dict) else {}
        record = {
            "timestamp": str(row.get("timestamp", "")),
            "question": str(row.get("question", "")),
            "region": str(filters.get("region", "")),
            "organization": str(filters.get("organization", "")),
            "support_type": str(filters.get("support_type", "")),
            "retrieved_doc_count": str(row.get("retrieved_doc_count", "")),
            "rerank_ok": str(row.get("rerank_ok", "")),
            "rerank_top_score": str(row.get("rerank_top_score", "")),
            "answer_preview": str(row.get("answer_preview", "")),
        }

        escaped = []
        for col in header:
            value = record[col].replace('"', '""')
            escaped.append(f'"{value}"')
        lines.append(",".join(escaped))

    return "\n".join(lines)


def _extract_keywords(query: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", query)
    seen = set()
    uniq = []
    for tok in tokens:
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        uniq.append(tok)
    return uniq[:8]


def _highlight_text(text: str, keywords: list[str]) -> str:
    highlighted = text
    for kw in sorted(keywords, key=len, reverse=True):
        if len(kw) < 2:
            continue
        highlighted = re.sub(
            re.escape(kw),
            lambda m: f"<mark>{m.group(0)}</mark>",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted


def _load_base_docs(vectorstore) -> list[Document]:
    fetched = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(fetched["documents"], fetched["metadatas"])
    ]


@st.cache_resource(show_spinner=False)
def _load_retriever() -> HybridRetriever:
    builder = IndexBuilder()
    vectorstore = builder.load_vectorstore()
    docs = _load_base_docs(vectorstore)
    return HybridRetriever(vectorstore=vectorstore, base_docs=docs)


with st.sidebar:
    st.header("파이프라인 실행")

    st.markdown("---")
    st.subheader("검색 필터")
    selected_region = st.selectbox("지역", ["전체", "서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"])
    selected_org = st.selectbox(
        "기관",
        [
            "전체",
            "창업진흥원",
            "중소벤처기업부",
            "중소벤처기업진흥공단",
            "경기테크노파크",
            "울산과학기술원",
            "경기콘텐츠진흥원",
        ],
    )
    selected_support_type = st.selectbox("지원분야", ["전체", "사업화", "글로벌", "정책자금", "창업교육", "멘토링ㆍ컨설팅ㆍ교육", "인력"])

    st.markdown("---")
    st.subheader("질의 로그")
    recent_rows = _load_recent_history(limit=10)
    st.caption(f"최근 로그: {len(recent_rows)}건")
    if recent_rows:
        for row in reversed(recent_rows[-5:]):
            st.markdown(f"- {row.get('timestamp', '')} | {row.get('question', '')[:28]}")

        csv_blob = _history_to_csv(recent_rows)
        st.download_button(
            label="로그 CSV 다운로드",
            data=csv_blob,
            file_name="query_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        failure_rows = [r for r in recent_rows if int(r.get("retrieved_doc_count", 0) or 0) == 0]
        if failure_rows:
            st.caption(f"근거 0건 질문: {len(failure_rows)}건")
            for row in reversed(failure_rows[-3:]):
                st.markdown(f"- {row.get('question', '')[:40]}")

    if st.button("1) 웹 데이터 수집", use_container_width=True):
        try:
            output_path = f"{settings.raw_docs_dir}/web_seed.json"
            count, saved_path = collect_and_save_default(output_path)
            st.success(f"수집 완료: {count}건")
            st.write(saved_path)
        except Exception as exc:
            st.error(f"수집 실패: {exc}")

    if st.button("2) 인덱스 생성", use_container_width=True):
        try:
            builder = IndexBuilder()
            chunk_count, preview_path = builder.build()
            _load_retriever.clear()
            st.success(f"인덱스 생성 완료: {chunk_count} chunks")
            st.write(preview_path)
        except Exception as exc:
            st.error(f"인덱스 생성 실패: {exc}")

    if st.button("3) 평가 실행", use_container_width=True):
        try:
            report = run_eval()
            st.success("평가 완료")
            st.json(report)
        except Exception as exc:
            st.error(f"평가 실패: {exc}")

st.subheader("질의응답")
question = st.text_input(
    "질문 입력",
    value="만 39세 서울 거주 예비 창업자가 받을 수 있는 지원금은?",
)

if st.button("질문 실행", type="primary"):
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        try:
            retriever = _load_retriever()
            retrieval_result = retriever.retrieve(
                question,
                structured_filters={
                    "region": selected_region,
                    "organization": selected_org,
                    "support_type": selected_support_type,
                },
            )
            reranked_docs, rerank_ok, rerank_top_score = rerank_documents(
                query=question,
                documents=retrieval_result.documents,
                top_n=settings.rerank_top_n,
                threshold=settings.rerank_threshold,
            )
            final_docs = reranked_docs if rerank_ok else []
            answer = answer_with_citations(question, final_docs)

            if not final_docs:
                st.warning("검색된 근거 문서가 없어 안전 모드로 답변을 제한했습니다.")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 답변")
                st.write(answer)
            with col2:
                st.markdown("### 적용 필터")
                st.code(json.dumps(retrieval_result.applied_filters, ensure_ascii=False, indent=2), language="json")
                st.markdown("### 리랭커")
                st.code(
                    json.dumps(
                        {
                            "rerank_ok": rerank_ok,
                            "rerank_top_score": round(rerank_top_score, 4),
                            "threshold": settings.rerank_threshold,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    language="json",
                )
                st.markdown("### 검색 문서 수")
                st.write(len(final_docs))

            st.markdown("### 출처 미리보기")
            keywords = _extract_keywords(question)
            for idx, doc in enumerate(final_docs[:6], start=1):
                meta = doc.metadata
                source_file = meta.get("source_file", "unknown")
                title = meta.get("title", "")
                source_kind = meta.get("source_kind", "pdf")
                source_url = meta.get("source_url", "")
                notice_id = meta.get("notice_id", "")
                page_number = meta.get("page_number", "?")
                deadline = meta.get("deadline", "")
                organization = meta.get("organization", "")
                support_type = meta.get("support_type", "")
                region = meta.get("region", "")

                if source_kind == "web" or source_url:
                    line = f"{idx}. {title or source_file}"
                else:
                    line = f"{idx}. {source_file} p.{page_number}"
                if notice_id:
                    line += f" | notice_id={notice_id}"
                if organization:
                    line += f" | 기관={organization}"
                if support_type:
                    line += f" | 분야={support_type}"
                if region:
                    line += f" | 지역={region}"
                if deadline:
                    line += f" | 마감={deadline}"
                st.markdown(line)

                if source_url:
                    st.markdown(f"- 원문: [{source_url}]({source_url})")

                with st.expander(f"문서 {idx} 스니펫 보기"):
                    snippet = doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else "")
                    st.markdown(_highlight_text(snippet, keywords), unsafe_allow_html=True)
                    st.code(json.dumps(meta, ensure_ascii=False, indent=2), language="json")

            _append_query_history(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "question": question,
                    "filters": {
                        "region": selected_region,
                        "organization": selected_org,
                        "support_type": selected_support_type,
                    },
                    "retrieved_doc_count": len(final_docs),
                    "rerank_ok": rerank_ok,
                    "rerank_top_score": round(rerank_top_score, 4),
                    "answer_preview": answer[:220],
                    "source_urls": [
                        d.metadata.get("source_url", "")
                        for d in final_docs
                        if str(d.metadata.get("source_url", "")).strip()
                    ][:10],
                }
            )
        except Exception as exc:
            st.error(f"질의 실행 실패: {exc}")
