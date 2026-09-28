import os


class FakeKms:
    """KMS 흉내. 데이터 키를 만들 때의 컨텍스트와 다르면 풀어 주지 않는다."""

    def __init__(self) -> None:
        self._keys: dict[bytes, tuple[bytes, dict]] = {}
        self.generate_calls: list[dict] = []

    def generate_data_key(self, *, KeyId, KeySpec, EncryptionContext):  # noqa: N803
        self.generate_calls.append(
            {"KeyId": KeyId, "KeySpec": KeySpec, "EncryptionContext": EncryptionContext}
        )
        plaintext, blob = os.urandom(32), os.urandom(40)
        self._keys[blob] = (plaintext, dict(EncryptionContext))
        return {"Plaintext": plaintext, "CiphertextBlob": blob}

    def decrypt(self, *, CiphertextBlob, EncryptionContext):  # noqa: N803
        plaintext, context = self._keys[CiphertextBlob]
        if context != EncryptionContext:
            raise PermissionError("InvalidCiphertextException")
        return {"Plaintext": plaintext}
