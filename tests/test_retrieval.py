from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever, _rrf_merge


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def similarity_search(self, query, k):
        return self.documents[:k]


def _doc(text, **metadata):
    return Document(page_content=text, metadata={"source_file": text, **metadata})


def test_structured_filter_is_fail_closed():
    docs = [_doc("부산 창업 지원", region="부산")]
    result = HybridRetriever(FakeVectorStore(docs), docs).retrieve(
        "창업 지원 알려줘",
        structured_filters={"region": "서울"},
    )
    assert result.documents == []
    assert result.applied_filters["filter_miss"] is True


def test_age_over_39_excludes_youth_only_documents():
    docs = [
        _doc("청년 전용 창업 지원", age_bucket="youth"),
        _doc("전 연령 창업 지원", age_bucket="all"),
    ]
    result = HybridRetriever(FakeVectorStore(docs), docs).retrieve("만 40세 창업 지원")
    assert result.documents
    assert all(doc.metadata.get("age_bucket") != "youth" for doc in result.documents)


def test_rrf_rewards_documents_found_by_both_retrievers():
    common = _doc("공통")
    lexical = _doc("키워드")
    semantic = _doc("벡터")
    merged = _rrf_merge([[lexical, common], [semantic, common]])
    assert merged[0] == common
