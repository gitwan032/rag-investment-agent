"""Tavily 웹 검색 + 파일 캐시.

같은 검색어는 outputs/cache/search/에 저장한 결과를 재사용한다.
→ 재실행 시 비용이 들지 않고, 같은 입력이면 같은 결과로 다시 실행할 수 있다(재현성).
"""
import hashlib
import json

from langchain_tavily import TavilySearch

from config import OUTPUT_DIR

CACHE_DIR = OUTPUT_DIR / "cache" / "search"
USE_CACHE = True


def search(query: str, max_results: int = 5, topic: str = "general", time_range: str | None = None) -> list[dict]:
    """결과: [{title, url, content, published_date}] (없으면 빈 리스트)."""
    key = hashlib.sha1(json.dumps([query, max_results, topic, time_range]).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if USE_CACHE and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    tool = TavilySearch(max_results=max_results, topic=topic, time_range=time_range)
    try:
        raw = tool.invoke({"query": query})
    except Exception as exc:  # 결과 없음·일시적 오류는 빈 결과로 처리
        print(f"  [web_search] 실패: {query} ({type(exc).__name__})")
        return []
    if not isinstance(raw, dict):
        return []
    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", ""),
         "published_date": r.get("published_date")}
        for r in raw.get("results", []) if r.get("url")
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def format_results(results: list[dict], start: int = 1, max_chars: int = 600) -> str:
    """LLM에 넘길 근거 목록. 각 결과에 [번호]를 붙여 출처를 인용하게 한다."""
    return "\n\n".join(
        f"[{i}] {r['title']} ({r['url']})\n{r['content'][:max_chars]}"
        for i, r in enumerate(results, start)
    )
