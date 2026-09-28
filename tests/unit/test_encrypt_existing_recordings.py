from app.services import recording_crypto
from app.services.recording_storage import RecordingStorageService
from scripts.encrypt_existing_recordings import migrate
from tests.unit.fake_kms import FakeKms


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _Paginator:
    def __init__(self, s3: "_S3") -> None:
        self._s3 = s3

    def paginate(self, *, Bucket, Prefix):  # noqa: N803
        keys = sorted(k for k in self._s3.objects if k.startswith(Prefix))
        return [{"Contents": [{"Key": k} for k in keys]}] if keys else [{}]


class _S3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.puts: list[str] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self)

    def get_object(self, *, Bucket, Key, Range=None):  # noqa: N803
        data = self.objects[Key]
        if Range:
            start, end = Range.removeprefix("bytes=").split("-")
            data = data[int(start) : int(end) + 1]
        return {"Body": _Body(data)}

    def put_object(self, *, Bucket, Key, Body, ContentType):  # noqa: N803
        self.puts.append(Key)
        self.objects[Key] = Body


_WAV = RecordingStorageService._to_wav(b"\x01\x00\x02\x00")


def _bucket(kms: FakeKms) -> _S3:
    locked = recording_crypto.encrypt(_WAV, s3_key="recordings/b/new.wav", kms=kms, key_id="k")
    return _S3(
        {
            "recordings/a/old.wav": _WAV,
            "recordings/b/new.wav": locked,
            "recordings/c/junk.wav": b"not audio",
            "examples/1.wav": _WAV,
        }
    )


def test_dry_run_counts_without_writing() -> None:
    kms = FakeKms()
    s3 = _bucket(kms)

    tally = migrate(s3, kms, bucket="b", key_id="k")

    assert tally.plain == ["recordings/a/old.wav"]
    assert tally.already_encrypted == 1
    assert tally.not_wav == ["recordings/c/junk.wav"]
    assert s3.puts == []


def test_apply_encrypts_only_plain_recordings_in_place() -> None:
    kms = FakeKms()
    s3 = _bucket(kms)

    tally = migrate(s3, kms, bucket="b", key_id="k", apply=True)

    assert tally.failed == []
    assert s3.puts == ["recordings/a/old.wav"]
    blob = s3.objects["recordings/a/old.wav"]
    assert recording_crypto.decrypt(blob, s3_key="recordings/a/old.wav", kms=kms) == _WAV
    assert s3.objects["examples/1.wav"] == _WAV


def test_second_run_has_nothing_left() -> None:
    kms = FakeKms()
    s3 = _bucket(kms)
    migrate(s3, kms, bucket="b", key_id="k", apply=True)

    tally = migrate(s3, kms, bucket="b", key_id="k", apply=True)

    assert tally.plain == []
    assert tally.already_encrypted == 2


def test_failure_leaves_original_untouched() -> None:
    class _BrokenKms(FakeKms):
        def generate_data_key(self, **kwargs):
            raise RuntimeError("AccessDenied")

    kms = FakeKms()
    s3 = _bucket(kms)

    tally = migrate(s3, _BrokenKms(), bucket="b", key_id="k", apply=True)

    assert tally.failed == ["recordings/a/old.wav: AccessDenied"]
    assert s3.objects["recordings/a/old.wav"] == _WAV
    assert s3.puts == []
