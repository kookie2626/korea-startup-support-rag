from langchain_core.documents import Document

from src.data.pdf_preprocessor import _extract_age_range, split_documents


def test_extracts_inclusive_age_range():
    assert _extract_age_range("신청일 기준 만 19세 이상 만 39세 이하") == (19, 39)


def test_extracts_exclusive_age_range():
    assert _extract_age_range("만 40세 초과 만 65세 미만") == (41, 64)


def test_metadata_is_derived_per_chunk():
    source = Document(
        page_content=("서울 예비창업 지원 안내 " * 80) + ("부산 도약기업 지원 안내 " * 80),
        metadata={"source_file": "guide.pdf", "page_number": 1, "source_kind": "pdf"},
    )
    chunks = split_documents([source])
    assert len(chunks) > 1
    assert any(chunk.metadata["regions"] == "서울" for chunk in chunks)
    assert any("부산" in chunk.metadata["regions"] for chunk in chunks)
