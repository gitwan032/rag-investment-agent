"""REFERENCE(출처) 만들기와 교수님 표기 형식.

  기관 보고서 : 발행기관(YYYY). 보고서명. URL
  학술 논문   : 저자(YYYY). 논문제목. 학술지명, 권(호), 페이지.
  웹페이지    : 기관명 또는 작성자(YYYY-MM-DD). 제목. 사이트명, URL
"""
import hashlib
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib.parse import urlparse

import yaml

from config import CONFIG_DIR


@lru_cache(maxsize=1)
def _doc_registry() -> dict:
    data = yaml.safe_load((CONFIG_DIR / "docs.yaml").read_text(encoding="utf-8"))
    return {doc["id"]: doc for group in data.values() for doc in group}


def _ref_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def make_doc_ref(doc_id: str, startup: str, used_by: str) -> dict:
    """RAG 문서 → 출처. 문서 정보는 config/docs.yaml에서 가져온다."""
    doc = _doc_registry()[doc_id]
    return {
        "id": _ref_id(doc_id), "startup": startup, "used_by": used_by,
        "type": "paper" if doc["ref_type"] == "paper" else "report",
        "author": doc["author_org"], "date": str(doc["date"]), "title": doc["title"],
        "venue": doc.get("venue"), "volume_issue": doc.get("volume_issue"),
        "pages": doc.get("article_pages"), "url": doc.get("url"),
    }


def _parse_date(raw: str | None) -> str | None:
    """검색 결과의 게시일 ('2025-01-15...', 'Thu, 15 Jan 2026 ...') → 'YYYY-MM-DD'. 읽을 수 없으면 None."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:10]).strftime("%Y-%m-%d")
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(raw).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def make_web_ref(result: dict, startup: str, used_by: str, run_date: str) -> dict:
    """웹 검색 결과 → 출처. 게시일을 알 수 없으면 조회일(run_date)을 쓰고 accessed로 표시한다."""
    site = urlparse(result["url"]).netloc.removeprefix("www.")
    date = _parse_date(result.get("published_date"))
    return {
        "id": _ref_id(result["url"]), "startup": startup, "used_by": used_by, "type": "web",
        "author": site, "date": date or run_date, "accessed": date is None,
        "title": result.get("title") or site, "venue": site, "url": result["url"],
    }


def format_reference(ref: dict) -> str:
    year = ref["date"][:4]
    if ref["type"] == "report":
        return f"{ref['author']}({year}). {ref['title']}. {ref['url']}"
    if ref["type"] == "paper":
        detail = ", ".join(x for x in (ref.get("venue"), ref.get("volume_issue"), ref.get("pages")) if x)
        return f"{ref['author']}({year}). {ref['title']}. {detail}."
    return f"{ref['author']}({ref['date']}). {ref['title']}. {ref['venue']}, {ref['url']}"


def unique(refs: list[dict]) -> list[dict]:
    """같은 출처(id)는 한 번만 남긴다."""
    seen, result = set(), []
    for ref in refs:
        if ref["id"] not in seen:
            seen.add(ref["id"])
            result.append(ref)
    return result
