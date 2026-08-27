from datetime import UTC, datetime

from app.core.timeutil import now_utc
from app.db.models import PostCommentORM
from app.schemas.community import preview_of


def test_parent_comment_id_is_indexed_for_cascade_delete() -> None:
    indexed = {
        tuple(c.name for c in idx.columns) for idx in PostCommentORM.__table__.indexes
    }
    assert ("parent_comment_id",) in indexed, indexed


def test_shared_now_helper_returns_aware_utc() -> None:
    now = now_utc()
    assert now.tzinfo is not None, "naive 로 저장하면 조회자가 기준 시계를 알 수 없다"
    assert now.utcoffset().total_seconds() == 0, "저장은 UTC 기준으로 고정한다"
    assert abs((now - datetime.now(UTC)).total_seconds()) < 5


def test_community_services_share_one_now_helper() -> None:
    from app.services import community_comment, community_post, community_reaction

    for mod in (community_post, community_comment, community_reaction):
        assert getattr(mod, "_now", None) is None, f"{mod.__name__} 에 _now 중복이 남아있다"
        assert mod.now_utc is now_utc


def test_preview_keeps_plain_text_intact() -> None:
    assert preview_of("가" * 150) == "가" * 100
    assert preview_of("짧은 글") == "짧은 글"


def test_preview_does_not_end_with_a_dangling_zwj() -> None:
    family = "\U0001f468‍\U0001f469‍\U0001f467"
    body = "가" * 98 + family

    cut = preview_of(body)

    assert not cut.endswith("‍"), repr(cut[-3:])
    assert len(cut) <= 100


def test_preview_does_not_leave_a_stray_combining_mark() -> None:
    body = "a" * 100 + "́"

    cut = preview_of("가" * 99 + "é")

    assert not cut or not _is_combining(cut[-1]), repr(cut[-2:])
    assert len(preview_of(body)) <= 100


def _is_combining(ch: str) -> bool:
    import unicodedata
    return unicodedata.combining(ch) != 0
