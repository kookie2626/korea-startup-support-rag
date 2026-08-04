import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    vector_db_dir: str = os.getenv("VECTOR_DB_DIR", "./data/vectorstore/chroma")
    raw_docs_dir: str = os.getenv("RAW_DOCS_DIR", "./data/raw")
    processed_docs_dir: str = os.getenv("PROCESSED_DOCS_DIR", "./data/processed")
    top_k_vector: int = int(os.getenv("TOP_K_VECTOR", "8"))
    top_k_bm25: int = int(os.getenv("TOP_K_BM25", "8"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "6"))
    rerank_threshold: float = float(os.getenv("RERANK_THRESHOLD", "0.08"))


settings = Settings()
