from datetime import UTC, datetime

from app.core.timeutil import utc_naive_now
from app.db.models import PostCommentORM
from app.schemas.community import preview_of


def test_parent_comment_id_is_indexed_for_cascade_delete() -> None:
    indexed = {
        tuple(c.name for c in idx.columns) for idx in PostCommentORM.__table__.indexes
    }
    assert ("parent_comment_id",) in indexed, indexed


def test_shared_now_helper_returns_naive_utc() -> None:
    now = utc_naive_now()
    assert now.tzinfo is None, "DB 컬럼이 naive DateTime 이라 tz 를 떼서 저장한다"
    assert abs((now - datetime.now(UTC).replace(tzinfo=None)).total_seconds()) < 5


def test_community_services_share_one_now_helper() -> None:
    from app.services import community_comment, community_post, community_reaction

    for mod in (community_post, community_comment, community_reaction):
        assert getattr(mod, "_now", None) is None, f"{mod.__name__} 에 _now 중복이 남아있다"
        assert mod.utc_naive_now is utc_naive_now


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
