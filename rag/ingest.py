"""RAG 인덱스 만들기: PDF 로딩 → 정제 → 청킹 → 임베딩 → Chroma 저장.

실행:
    python -m rag.ingest            # 인덱스 생성 (이미 있으면 건너뜀)
    python -m rag.ingest --check    # 임베딩 없이 페이지·청크 수만 점검
    python -m rag.ingest --rebuild  # 기존 인덱스를 지우고 다시 생성
"""
import argparse
import re
from collections import Counter
from functools import lru_cache

import yaml
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CONFIG_DIR, EMBEDDING_MODEL, ROOT
from rag.store import COLLECTIONS, get_vectorstore

PAGE_LIMIT = 200                 # 과제 조건: 문서 총 200쪽 이내
# 청크 크기: 400 / 600 / 800토큰과 팀 방식을 비교한 결과 400이 가장 좋았다 (docs/EXPERIMENTS.md 1번)
CHUNK_TOKENS, OVERLAP_TOKENS = 400, 50
SEGMENTS = ("demand_forecasting", "generation_vpp", "ess_operation", "grid_management")


# ── 1. 문서 목록 읽기 ─────────────────────────────────────
def parse_pages(text: str) -> list[int]:
    """'6-20,39-51' → [6, 7, ..., 20, 39, ..., 51]"""
    pages = []
    for part in str(text).split(","):
        start, _, end = part.strip().partition("-")
        pages.extend(range(int(start), int(end or start) + 1))
    return pages


def load_registry() -> list[dict]:
    """config/docs.yaml을 읽고 200쪽 제한을 확인한다."""
    registry = yaml.safe_load((CONFIG_DIR / "docs.yaml").read_text(encoding="utf-8"))
    docs = []
    for group in COLLECTIONS:
        for doc in registry[group]:
            doc["group"] = group
            doc["original"] = parse_pages(doc["original_pages"])  # 발췌본 1쪽 = 원본 몇 쪽인지
            docs.append(doc)
    total = sum(len(doc["original"]) for doc in docs)
    print(f"사용 페이지: {total} / {PAGE_LIMIT}")
    if total > PAGE_LIMIT:
        raise SystemExit("200쪽 제한 초과: 인제스트를 중단합니다.")
    return docs


# ── 2. 로딩 + 정제 ───────────────────────────────────────
def remove_repeated_edges(pages: list[str]) -> list[str]:
    """여러 페이지 위·아래에 반복되는 머리말/꼬리말과 쪽번호 줄을 지운다 (수업 2-2 전처리 팁)."""
    split = [[line.strip() for line in page.splitlines() if line.strip()] for page in pages]
    edge_count = Counter(line for lines in split for line in set(lines[:2] + lines[-2:]))
    repeated = {line for line, n in edge_count.items() if n >= max(3, len(pages) // 2) and len(line) < 100}

    cleaned = []
    for lines in split:
        kept = [
            line for i, line in enumerate(lines)
            if not ((i < 2 or i >= len(lines) - 2) and (line in repeated or line.isdigit()))
        ]
        text = "\n".join(kept)
        text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)  # 영어 줄바꿈 하이픈 복원
        cleaned.append(text)
    return cleaned


def load_document(doc: dict) -> list[Document]:
    """PDF 1개 → 페이지별 Document 목록. 메타데이터는 검색·출처 표기에 필요한 것만 남긴다."""
    pages = PyMuPDFLoader(str(ROOT / doc["file"])).load()  # 페이지마다 Document 1개
    if len(pages) != len(doc["original"]):
        raise ValueError(f"{doc['id']}: PDF {len(pages)}쪽 ≠ docs.yaml {len(doc['original'])}쪽")

    texts = remove_repeated_edges([page.page_content for page in pages])
    result = []
    for index, text in enumerate(texts):
        if not text.strip():
            print(f"  [주의] 빈 페이지: {doc['id']} 발췌본 {index + 1}쪽")
            continue
        metadata = {
            "doc_id": doc["id"],
            "title": doc["title"],
            "year": str(doc["date"]),
            "page": doc["original"][index],  # 원본 보고서 기준 쪽수 (출처 표기·평가에 사용)
            "lang": doc["lang"],
        }
        if doc["group"] == "market":
            # Chroma 메타데이터는 리스트를 못 넣으므로 세그먼트마다 True/False 칸을 둔다
            metadata["country"] = doc["country"]
            for segment in SEGMENTS:
                metadata[f"seg_{segment}"] = segment in doc["segments"]
        result.append(Document(page_content=text, metadata=metadata))
    return result


# ── 3. 청킹 ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


@lru_cache(maxsize=8)
def get_splitter(size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> RecursiveCharacterTextSplitter:
    """문단 → 줄 → 문장 → 단어 순으로 자르되, 길이는 bge-m3 토크나이저의 실제 토큰 수로 잰다."""
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _tokenizer(),
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "다. ", " ", ""],
    )


def build_chunks() -> dict[str, list[Document]]:
    """그룹(tech/market)별 청크 목록. 페이지 단위로 자르므로 청크가 두 페이지에 걸치지 않는다."""
    chunks = {group: [] for group in COLLECTIONS}
    for doc in load_registry():
        pages = load_document(doc)
        for page in pages:
            for i, piece in enumerate(get_splitter().split_documents([page])):
                piece.id = f"{doc['id']}:p{piece.metadata['page']}:c{i}"  # 고정 ID → 재실행해도 중복 없음
                chunks[doc["group"]].append(piece)
        print(f"  {doc['id']}: {len(pages)}쪽")
    for group, items in chunks.items():
        print(f"{COLLECTIONS[group]}: 청크 {len(items)}개")
    return chunks


# ── 4. 임베딩 + 저장 ─────────────────────────────────────
def ingest(rebuild: bool = False, embeddings=None) -> None:
    chunks = build_chunks()
    for group, items in chunks.items():
        store = get_vectorstore(group, embeddings)
        if rebuild:
            store.delete_collection()
            store = get_vectorstore(group, embeddings)
        existing = store._collection.count()
        if existing == len(items):
            print(f"{COLLECTIONS[group]}: 이미 {existing}개 저장됨 → 건너뜀 (다시 만들려면 --rebuild)")
            continue
        for start in range(0, len(items), 32):  # 32개씩 나눠 진행 상황 표시
            batch = items[start:start + 32]
            store.add_documents(batch, ids=[chunk.id for chunk in batch])
            print(f"  {COLLECTIONS[group]}: {start + len(batch)}/{len(items)}", flush=True)
        print(f"{COLLECTIONS[group]}: 저장 완료 ({store._collection.count()}개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="임베딩 없이 페이지·청크 수만 점검")
    parser.add_argument("--rebuild", action="store_true", help="기존 인덱스를 지우고 다시 생성")
    args = parser.parse_args()
    if args.check:
        build_chunks()
    else:
        ingest(rebuild=args.rebuild)
