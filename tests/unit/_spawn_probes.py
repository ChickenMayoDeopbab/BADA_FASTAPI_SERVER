import os

INHERITED = False


def probe() -> tuple[int, bool]:
    return os.getpid(), INHERITED


def boom() -> None:
    raise ValueError("자식 예외")


def die() -> None:
    os._exit(3)


def hold_gil(pcm: bytes, sr: int, semitones: float) -> bytes:
    sum(range(200_000_000))
    return pcm
