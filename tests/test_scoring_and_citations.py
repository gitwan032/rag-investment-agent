"""투자 판정 계산, 근거 번호 검증, REFERENCE 형식 테스트 (LLM·API 키 없이 실행)."""
from agents.investment_judge import compute, load_rubric
from tools.citations import NOT_FOUND, check_text, keep_cited, strip
from tools.references import format_reference, make_web_ref, unique


def _scores(value: int, **override) -> dict:
    scores = {i["id"]: {"score": value, "insufficient": False} for i in load_rubric()}
    for item_id, score in override.items():
        scores[item_id] = {"score": score, "insufficient": False}
    return scores


# ── 투자 판정 ────────────────────────────────────────────
def test_rubric_has_11_items_and_weights_sum_to_100():
    items = load_rubric()
    assert len(items) == 11 and sum(i["weight"] for i in items) == 100


def test_threshold_boundary_70_is_invest():
    total, decision, knockout = compute(_scores(7))
    assert total == 70 and decision == "투자" and knockout is None


def test_below_threshold_is_hold():
    total, decision, _ = compute(_scores(6))
    assert total == 60 and decision == "보류"


def test_knockout_team_score_3_holds_even_with_high_total():
    total, decision, knockout = compute(_scores(9, team=3))
    assert total >= 70 and decision == "보류" and "창업팀" in knockout


def test_insufficient_info_does_not_trigger_knockout():
    scores = _scores(8)
    scores["regulation"] = {"score": 5, "insufficient": True}
    _, decision, knockout = compute(scores)
    assert decision == "투자" and knockout is None


# ── 근거 번호 검증 ───────────────────────────────────────
def test_uncited_or_out_of_range_items_are_dropped():
    used = set()
    kept = keep_cited(["근거 있음 [1]", "근거 없음", "없는 번호 [9]"], n=3, used=used)
    assert kept == ["근거 있음 [1]"] and used == {1}


def test_uncited_paragraph_becomes_not_found():
    assert check_text("번호 없는 주장", n=3, used=set()) == NOT_FOUND


def test_strip_removes_citation_numbers():
    assert strip("상용화 단계다 [1][2].") == "상용화 단계다."


# ── REFERENCE ───────────────────────────────────────────
def test_web_reference_uses_published_date_in_professor_format():
    ref = make_web_ref({"url": "https://www.example.com/a", "title": "기사",
                        "published_date": "Thu, 15 Jan 2026 07:00:00 GMT"}, "A", "tech", "2026-10-14")
    assert format_reference(ref) == "example.com(2026-01-15). 기사. example.com, https://www.example.com/a"
    assert ref["accessed"] is False


def test_web_reference_without_date_falls_back_to_access_date():
    ref = make_web_ref({"url": "https://b.com/x", "title": "T"}, "A", "tech", "2026-10-14")
    assert ref["date"] == "2026-10-14" and ref["accessed"] is True


def test_duplicate_references_are_removed():
    ref = make_web_ref({"url": "https://b.com/x", "title": "T"}, "A", "tech", "2026-10-14")
    assert len(unique([ref, dict(ref)])) == 1
