"""3단계: Retriever 성능 평가 (Hit Rate@K, MRR).

평가셋: data/eval/qa_tech.jsonl(20문항), qa_market.jsonl(30문항)
         각 줄 = {질문, 정답 doc_id, 정답 page}. 검색된 청크의 (doc_id, page)가 정답과 같으면 적중.

실행:
    python -m eval.retriever_eval                       # 검색기 5종 비교 (저장된 인덱스 사용)
    python -m eval.retriever_eval --chunking 400,600,800  # 청크 크기 비교 (Dense, 메모리에 임시 인덱스)
결과는 outputs/retriever_eval.md (검색기), outputs/chunking_eval.md (청킹) 에 저장된다.
"""
import argparse
import json
import time

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EVAL_DIR, OUTPUT_DIR
from rag.ingest import get_splitter, load_document, load_registry
from rag.retrievers import build_retriever

K = 5
KINDS = ["dense", "bm25", "ensemble", "rerank", "dense_rerank"]
LABELS = {
    "dense": "Dense (bge-m3)",
    "bm25": "BM25 (Kiwi)",
    "ensemble": "Ensemble (BM25 + Dense, RRF)",
    "rerank": "Ensemble + Reranker",
    "dense_rerank": "Dense + Reranker",
}


def load_questions(group: str) -> list[dict]:
    path = EVAL_DIR / f"qa_{group}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(retriever, questions: list[dict]) -> dict:
    """질문마다 상위 K개를 검색해 정답이 몇 등에 있는지 본다."""
    hits, reciprocal = 0, 0.0
    start = time.time()
    for q in questions:
        docs = retriever.invoke(q["question"])[:K]
        found = [(d.metadata["doc_id"], d.metadata["page"]) for d in docs]
        answer = (q["doc_id"], q["page"])
        if answer in found:
            hits += 1
            reciprocal += 1 / (found.index(answer) + 1)  # 1등이면 1, 2등이면 0.5 ...
    n = len(questions)
    return {"hit": hits / n, "mrr": reciprocal / n, "sec": (time.time() - start) / n}


def _row(label: str, r: dict, extra: str = "") -> str:
    return (f"| {label} | {r['tech']['hit']:.3f} | {r['tech']['mrr']:.3f} | "
            f"{r['market']['hit']:.3f} | {r['market']['mrr']:.3f} |{extra}")


def compare_retrievers(groups: dict, embeddings=None) -> None:
    lines = [
        "# Retriever 평가", "",
        f"평가셋: 기술 {len(groups['tech'])}문항, 시장 {len(groups['market'])}문항. "
        f"상위 {K}개 청크 중 (doc_id, page)가 정답과 같으면 적중.", "",
        f"| 검색기 | 기술 Hit@{K} | 기술 MRR | 시장 Hit@{K} | 시장 MRR | 질문당 시간(초) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in KINDS:
        r = {g: evaluate(build_retriever(kind, g, k=K, embeddings=embeddings), qs) for g, qs in groups.items()}
        sec = (r["tech"]["sec"] + r["market"]["sec"]) / 2
        lines.append(_row(LABELS[kind], r, f" {sec:.2f} |"))
        print(lines[-1], flush=True)
    _save("retriever_eval.md", lines)


def temp_store(group: str, splitter_for, embeddings=None) -> Chroma:
    """주어진 청킹 방식으로 임시 벡터스토어를 메모리에 만든다 (저장된 인덱스는 건드리지 않음)."""
    from rag.embeddings import get_embeddings

    chunks = []
    for doc in load_registry():
        if doc["group"] == group:
            chunks += splitter_for(doc).split_documents(load_document(doc))
    store = Chroma(collection_name=f"tmp_{group}", embedding_function=embeddings or get_embeddings(),
                   collection_metadata={"hnsw:space": "cosine"})  # persist_directory 없음 → 메모리
    store.add_documents(chunks)
    store.n_chunks = len(chunks)
    return store


def compare_chunking(groups: dict, sizes: list[int], embeddings=None) -> None:
    def team(doc):  # 팀 방식: 언어별 글자 수, 줄바꿈 기준
        size, overlap = (1200, 150) if doc["lang"] == "ko" else (3000, 375)
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=["\n", ""])

    settings = [("팀: 글자 수 (영 3,000 / 한 1,200자)", team)]
    for size in sizes:  # overlap은 크기의 1/8 (800 → 100)
        settings.append((f"토큰 {size} / overlap {size // 8}", lambda doc, s=size: get_splitter(s, s // 8)))

    lines = [
        "# 청킹 방식 비교 (Dense 검색 기준)", "",
        f"| 청킹 | 기술 Hit@{K} | 기술 MRR | 시장 Hit@{K} | 시장 MRR | 청크 수 (기술/시장) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, splitter_for in settings:
        r, counts = {}, []
        for g, qs in groups.items():
            store = temp_store(g, splitter_for, embeddings)
            r[g] = evaluate(store.as_retriever(search_kwargs={"k": K}), qs)
            counts.append(store.n_chunks)
            store.delete_collection()
        lines.append(_row(label, r, f" {counts[0]} / {counts[1]} |"))
        print(lines[-1], flush=True)
    _save("chunking_eval.md", lines)


def _save(name: str, lines: list[str]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {OUTPUT_DIR / name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunking", help="비교할 토큰 청크 크기들, 예: 400,600,800 (지정하면 청킹 비교만 실행)")
    args = parser.parse_args()
    groups = {g: load_questions(g) for g in ("tech", "market")}
    if args.chunking:
        compare_chunking(groups, [int(x) for x in args.chunking.split(",")])
    else:
        compare_retrievers(groups)
