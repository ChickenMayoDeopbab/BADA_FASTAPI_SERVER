import ast
import hashlib
import inspect
import sys
import textwrap
from pathlib import Path

from app.services import example_audio, example_service

PINS = {"ns1": "85fc1e5ff8fd"}


def _fingerprint(source: str) -> str:
    tree = ast.parse(source)
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
    dumped = ast.dump(tree, show_empty=True) if sys.version_info >= (3, 13) else ast.dump(tree)
    return hashlib.sha1(dumped.encode()).hexdigest()[:12]


def _processing_source() -> str:
    loop = textwrap.dedent(inspect.getsource(example_service._synthesize_qwen_ns))
    return Path(example_audio.__file__).read_text(encoding="utf-8") + "\n" + loop


def test_processing_change_requires_method_bump() -> None:
    got = _fingerprint(_processing_source())
    assert PINS.get(example_service._NS_METHOD) == got, (
        f"example_audio 처리 코드가 바뀌었다(지문 {got}). "
        "출력 바이트가 바뀌면 _NS_METHOD 를 올리고 H 를 다시 한다(켠 뒤면 프리셋 8개가 자동으로 다시 구워진다). "
        "바뀌지 않으면 보관 원본 79/79 바이트 동일을 확인하고 PINS 만 갱신한다"
    )


def test_fingerprint_ignores_comments_and_docstrings() -> None:
    a = 'def f(x):\n    """설명"""\n    return x + 1  # 주석\n'
    assert _fingerprint(a) == _fingerprint("def f(x):\n    return x + 1\n")
    assert _fingerprint(a) != _fingerprint("def f(x):\n    return x + 2\n")
