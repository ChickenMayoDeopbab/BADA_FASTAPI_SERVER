import ast
import hashlib
from pathlib import Path

from app.services import example_audio, example_service

PINS = {"ns1": "a0c5e1bcbc17"}


def _fingerprint(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha1(ast.dump(tree).encode()).hexdigest()[:12]


def test_processing_change_requires_method_bump() -> None:
    got = _fingerprint(Path(example_audio.__file__))
    assert PINS.get(example_service._NS_METHOD) == got, (
        f"example_audio 처리 코드가 바뀌었다(지문 {got}). "
        "출력 바이트가 바뀌면 _NS_METHOD 를 올리고 H 를 다시 한다(켠 뒤면 프리셋 8개가 자동으로 다시 구워진다). "
        "바뀌지 않으면 보관 원본 79/79 바이트 동일을 확인하고 PINS 만 갱신한다"
    )


def test_fingerprint_ignores_comments_and_docstrings(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text('def f(x):\n    """설명"""\n    return x + 1  # 주석\n', encoding="utf-8")
    b.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    assert _fingerprint(a) == _fingerprint(b)
    b.write_text("def f(x):\n    return x + 2\n", encoding="utf-8")
    assert _fingerprint(a) != _fingerprint(b)
