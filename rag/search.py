"""에이전트가 쓰는 문서 검색 창구 (Dense + Reranker, 3단계 평가로 결정).

기술 요약과 시장성 평가는 그래프에서 병렬로 실행된다. 두 스레드가 같은 임베딩·리랭커 모델을
동시에 처음 실행하면 프로그램이 죽을 수 있어서(팀 코드에서 겪은 문제) 한 번에 하나씩 실행한다.
"""
import threading

from langchain_core.documents import Document

from rag.retrievers import build_retriever

_LOCK = threading.Lock()


def search_docs(group: str, query: str, k: int = 5, filter: dict | None = None) -> list[Document]:
    with _LOCK:
        return build_retriever("dense_rerank", group, k=k, filter=filter).invoke(query)
