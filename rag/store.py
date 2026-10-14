"""Chroma 벡터스토어 열기 + 간단한 검색 테스트.

기술 문서와 시장 문서는 컬렉션을 나눠, 각 에이전트가 자기 컬렉션만 검색하게 한다.
"""
import sys

from langchain_chroma import Chroma

from config import VECTORSTORE_DIR
from rag.embeddings import get_embeddings

COLLECTIONS = {"tech": "tech_docs", "market": "market_docs"}


def get_vectorstore(group: str, embeddings=None) -> Chroma:
    """group = 'tech' 또는 'market'."""
    return Chroma(
        collection_name=COLLECTIONS[group],
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},  # 거리 기준: 코사인
    )


if __name__ == "__main__":
    # 사용법: python -m rag.store tech "AI가 전력망 운영에 어떻게 쓰이나?"
    group, query = sys.argv[1], sys.argv[2]
    store = get_vectorstore(group)
    print(f"{COLLECTIONS[group]}: {store._collection.count()}개 청크")
    for rank, (doc, distance) in enumerate(store.similarity_search_with_score(query, k=5), 1):
        meta = doc.metadata
        print(f"\n[{rank}] {meta['doc_id']} p.{meta['page']}  거리={distance:.3f}")
        print(doc.page_content[:200].replace("\n", " "), "...")
