import ast
import ssl
from pathlib import Path

import httpx

from app.core.http import shared_ssl_context
from app.services.csm_tts import CsmRealtimeTTSClient
from app.services.qwen_tts import QwenRealtimeTTSClient, QwenTTSClient
from tests.unit.test_csm_tts import _settings as _csm_settings
from tests.unit.test_qwen_realtime_tts import _settings as _qwen_settings

APP = Path(__file__).resolve().parents[2] / "app"


def _new_contexts(monkeypatch) -> list:
    shared_ssl_context()
    made: list = []
    real = ssl.create_default_context

    def _spy(*args, **kwargs):
        made.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", _spy)
    return made


def _is_client(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr in ("AsyncClient", "Client") and (
        isinstance(func.value, ast.Name) and func.value.id == "httpx"
    )


def _aliases_httpx(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "httpx":
            return True
        if isinstance(node, ast.Import) and any(a.name == "httpx" and a.asname for a in node.names):
            return True
    return False


def _passes_shared(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "verify" and isinstance(kw.value, ast.Call):
            name = kw.value.func
            return getattr(name, "id", getattr(name, "attr", None)) == "shared_ssl_context"
    return False


def test_every_httpx_client_in_app_gets_the_shared_context() -> None:
    sites, missing, aliased = 0, [], []
    for path in sorted(APP.rglob("*.py")):
        tree = ast.parse(path.read_text())
        if _aliases_httpx(tree):
            aliased.append(str(path.relative_to(APP.parent)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_client(node):
                sites += 1
                if not _passes_shared(node):
                    missing.append(f"{path.relative_to(APP.parent)}:{node.lineno}")

    assert aliased == [], f"httpx 를 별칭으로 들여와 생성 지점을 셀 수 없다: {aliased}"
    assert sites >= 9
    assert missing == [], f"공유 SSL 컨텍스트 없이 만드는 곳: {missing}"


async def test_realtime_open_every_turn_reuses_the_shared_context(monkeypatch) -> None:
    made = _new_contexts(monkeypatch)
    client = QwenRealtimeTTSClient(_qwen_settings())

    for _ in range(2):
        session = await client.open()
        await session.aclose()

    assert made == [], "턴마다 CA 묶음을 다시 읽었다"


async def test_health_and_csm_clients_reuse_the_shared_context(monkeypatch) -> None:
    made = _new_contexts(monkeypatch)

    await QwenTTSClient(_qwen_settings())._client(1.0).aclose()
    await CsmRealtimeTTSClient(_csm_settings())._new_client().aclose()

    assert made == []


def test_shared_context_is_one_object_with_httpx_default_verification() -> None:
    ctx = shared_ssl_context()
    fresh = httpx.create_ssl_context()

    assert shared_ssl_context() is ctx
    assert ctx.verify_mode == fresh.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True and fresh.check_hostname is True
    assert ctx.cert_store_stats() == fresh.cert_store_stats()
    assert ctx.cert_store_stats()["x509_ca"] > 0
