import ast
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import boto3
import pytest

from app.core import aws
from app.services import scenario_image_service
from app.services.recording_storage import RecordingStorageService

ROOT = Path(__file__).resolve().parents[2]
_NETWORK = {"upload_pcm", "upload_wav", "download_pcm", "exists", "strict_exists", "copy", "put_marker", "delete"}


def _settings(**overrides) -> SimpleNamespace:
    base = {"aws_access_key": "AKIA", "aws_secret_key": "secret", "aws_region": "ap-northeast-2",
            "s3_bucket": "bucket", "recording_kms_key_id": "kms-key"}
    return SimpleNamespace(**(base | overrides))


@pytest.fixture(autouse=True)
def created(monkeypatch) -> list[SimpleNamespace]:
    made: list[SimpleNamespace] = []

    def _client(service: str, **kwargs) -> SimpleNamespace:
        made.append(SimpleNamespace(service=service, kwargs=kwargs))
        return made[-1]

    monkeypatch.setattr(boto3, "client", _client)
    aws._client.cache_clear()
    yield made
    aws._client.cache_clear()


def _direct_storage_calls(fn: ast.AsyncFunctionDef):
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _NETWORK
                and "storage" in ast.unparse(node.func.value).lower()):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def test_async_code_never_calls_s3_storage_on_the_event_loop() -> None:
    found = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(fn, ast.AsyncFunctionDef):
                found += [f"{path.relative_to(ROOT)}:{call.lineno} {ast.unparse(call.func)}"
                          for call in _direct_storage_calls(fn)]
    assert found == []


def test_only_the_shared_factory_imports_boto3() -> None:
    found = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        if path == ROOT / "app" / "core" / "aws.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "boto3" or name.startswith("boto3.") for name in names):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert found == []


def test_storages_and_thumbnails_share_one_client_per_service(created) -> None:
    first, second = RecordingStorageService(_settings()), RecordingStorageService(_settings())
    assert first._client is second._client
    assert first._kms_client() is second._kms_client()
    assert scenario_image_service._s3_client(_settings()) is first._client
    assert [c.service for c in created] == ["s3", "kms"]
    assert RecordingStorageService(_settings(s3_bucket=None))._client is None and len(created) == 2


def test_client_arguments_match_the_settings(created) -> None:
    aws.shared_boto_client("s3", _settings())
    aws.shared_boto_client("kms", _settings(aws_access_key=None, aws_secret_key=None))
    assert created[0].kwargs == {"aws_access_key_id": "AKIA", "aws_secret_access_key": "secret",
                                 "region_name": "ap-northeast-2"}
    assert created[1].kwargs == {"region_name": "ap-northeast-2"}


@pytest.mark.parametrize(
    "change",
    [{"aws_region": "us-east-1"}, {"aws_access_key": "AKIB"}, {"aws_secret_key": "other"},
     {"aws_access_key": None, "aws_secret_key": None}],
)
def test_other_region_or_credentials_get_their_own_client(created, change) -> None:
    assert aws.shared_boto_client("s3", _settings(**change)) is not aws.shared_boto_client("s3", _settings())
    assert len(created) == 2


def test_concurrent_first_use_creates_one_client(monkeypatch) -> None:
    made: list[str] = []
    start = threading.Barrier(8)
    got: list[object] = []

    def _slow_client(service: str, **_kwargs) -> object:
        made.append(service)
        time.sleep(0.05)
        return object()

    def _use() -> None:
        start.wait()
        got.append(aws.shared_boto_client("s3", _settings()))

    monkeypatch.setattr(boto3, "client", _slow_client)
    threads = [threading.Thread(target=_use) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert made == ["s3"] and len({id(client) for client in got}) == 1
