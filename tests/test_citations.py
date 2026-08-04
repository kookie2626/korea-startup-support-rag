from langchain_core.documents import Document

from src.rag.qa_chain import _format_citations


def test_pdf_and_web_citations_are_metadata_driven():
    docs = [
        Document(
            page_content="PDF 근거",
            metadata={"source_file": "notice.pdf", "page_number": 3, "source_kind": "pdf"},
        ),
        Document(
            page_content="웹 근거",
            metadata={
                "source_file": "web_seed.json",
                "title": "지원 공고",
                "source_url": "https://example.com/notice",
                "source_kind": "web",
            },
        ),
    ]
    citations = _format_citations(docs)
    assert "notice.pdf p.3" in citations
    assert "지원 공고 | https://example.com/notice" in citations
    assert "web_seed.json p." not in citations


def test_duplicate_citations_are_removed():
    doc = Document(
        page_content="근거",
        metadata={"source_file": "notice.pdf", "page_number": 1, "source_kind": "pdf"},
    )
    assert _format_citations([doc, doc]).count("notice.pdf") == 1
