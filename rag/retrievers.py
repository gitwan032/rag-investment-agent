"""검색기(Retriever) 4종을 만든다. 3단계 평가와 이후 에이전트가 함께 쓴다.

  dense    : bge-m3 벡터 유사도 (의미 검색)
  bm25     : 키워드 검색 (한국어는 Kiwi 형태소 분석)
  ensemble : dense + bm25 결과를 RRF로 합침 (Hybrid, 수업 2-5-A)
  rerank   : ensemble 후보 20개를 Cross-Encoder로 다시 순위 매김 (수업 2-5-B)
  dense_rerank : dense 후보 20개만 Cross-Encoder로 다시 순위 매김 (BM25 제외)
"""
import re
from functools import lru_cache

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import RERANKER_MODEL
from rag.store import get_vectorstore

CANDIDATES = 20  # 리랭킹 전에 모아 둘 후보 수


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize(text: str) -> list[str]:
    """BM25용 토큰: 한국어 명사·동사 원형 + 영어 단어·숫자 (소문자)."""
    tokens = [t.form.lower() for t in _kiwi().tokenize(text) if t.tag.startswith(("N", "V"))]
    tokens += re.findall(r"[a-z0-9]+", text.lower())
    return tokens


def all_documents(group: str, embeddings=None) -> list[Document]:
    """벡터스토어에 저장된 청크 전체를 Document로 꺼낸다 (BM25 색인용)."""
    data = get_vectorstore(group, embeddings).get(include=["documents", "metadatas"])
    return [Document(page_content=text, metadata=meta, id=cid)
            for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"])]


@lru_cache(maxsize=1)
def _cross_encoder():
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    from rag.embeddings import pick_device

    return HuggingFaceCrossEncoder(model_name=RERANKER_MODEL, model_kwargs={"device": pick_device()})


def build_retriever(kind: str, group: str, k: int = 5, filter: dict | None = None, embeddings=None):
    """kind ∈ {dense, bm25, ensemble, rerank, dense_rerank}. filter는 Chroma 메타데이터 조건 (예: 국가·세그먼트)."""
    store = get_vectorstore(group, embeddings)
    if kind == "dense":
        return store.as_retriever(search_kwargs={"k": k, "filter": filter})
    if kind == "dense_rerank":
        dense = store.as_retriever(search_kwargs={"k": CANDIDATES, "filter": filter})
        reranker = CrossEncoderReranker(model=_cross_encoder(), top_n=k)
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=dense)

    docs = all_documents(group, embeddings)
    if filter:
        docs = [d for d in docs if _match(d.metadata, filter)]
    pool = CANDIDATES if kind in ("ensemble", "rerank") else k
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize, k=pool)
    if kind == "bm25":
        return bm25

    dense = store.as_retriever(search_kwargs={"k": pool, "filter": filter})
    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])  # 내부적으로 RRF
    if kind == "ensemble":
        return _TopK(ensemble, k)

    reranker = CrossEncoderReranker(model=_cross_encoder(), top_n=k)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble)


def _match(meta: dict, cond: dict) -> bool:
    """Chroma 필터 문법 중 이 프로젝트에서 쓰는 것($and, $in, 값 일치)만 파이썬으로 판정 (BM25용)."""
    if "$and" in cond:
        return all(_match(meta, c) for c in cond["$and"])
    key, value = next(iter(cond.items()))
    if isinstance(value, dict) and "$in" in value:
        return meta.get(key) in value["$in"]
    return meta.get(key) == value


class _TopK:
    """EnsembleRetriever는 두 검색기 결과를 합친 전체를 돌려주므로 상위 k개만 자른다."""

    def __init__(self, retriever, k):
        self.retriever, self.k = retriever, k

    def invoke(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)[: self.k]
