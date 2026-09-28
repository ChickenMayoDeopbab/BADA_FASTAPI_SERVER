import pytest
from cryptography.exceptions import InvalidTag

from app.services import recording_crypto
from tests.unit.fake_kms import FakeKms

_KEY = "recordings/sess-1/abc.wav"


def test_round_trip() -> None:
    kms = FakeKms()

    blob = recording_crypto.encrypt(b"RIFF-voice", s3_key=_KEY, kms=kms, key_id="alias/rec")

    assert recording_crypto.is_encrypted(blob)
    assert b"RIFF-voice" not in blob
    assert recording_crypto.decrypt(blob, s3_key=_KEY, kms=kms) == b"RIFF-voice"
    assert kms.generate_calls == [
        {
            "KeyId": "alias/rec",
            "KeySpec": "AES_256",
            "EncryptionContext": {"purpose": "bada-recording", "s3_key": _KEY},
        }
    ]


def test_each_file_gets_its_own_data_key_and_nonce() -> None:
    kms = FakeKms()

    a = recording_crypto.encrypt(b"same", s3_key=_KEY, kms=kms, key_id="k")
    b = recording_crypto.encrypt(b"same", s3_key=_KEY, kms=kms, key_id="k")

    assert a != b
    assert len(kms.generate_calls) == 2


def test_moved_file_does_not_open() -> None:
    """다른 S3 키 자리로 옮긴 파일은 KMS가 데이터 키를 풀어 주지 않는다."""
    kms = FakeKms()
    blob = recording_crypto.encrypt(b"voice", s3_key=_KEY, kms=kms, key_id="k")

    with pytest.raises(PermissionError):
        recording_crypto.decrypt(blob, s3_key="recordings/other/x.wav", kms=kms)


def test_tampered_body_is_rejected() -> None:
    kms = FakeKms()
    blob = bytearray(recording_crypto.encrypt(b"voice", s3_key=_KEY, kms=kms, key_id="k"))
    blob[-1] ^= 0x01

    with pytest.raises(InvalidTag):
        recording_crypto.decrypt(bytes(blob), s3_key=_KEY, kms=kms)


def test_plain_wav_is_not_treated_as_encrypted() -> None:
    assert not recording_crypto.is_encrypted(b"RIFF....WAVEfmt ")
    with pytest.raises(ValueError):
        recording_crypto.decrypt(b"RIFF....WAVEfmt ", s3_key=_KEY, kms=FakeKms())


def test_truncated_header_is_rejected() -> None:
    with pytest.raises(ValueError):
        recording_crypto.decrypt(recording_crypto.MAGIC + b"\x00\x05ab", s3_key=_KEY, kms=FakeKms())
