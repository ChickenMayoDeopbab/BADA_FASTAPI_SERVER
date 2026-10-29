import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_PROBE = textwrap.dedent(
    """
    import faulthandler
    import sys

    faulthandler.dump_traceback_later(60, exit=True)
    # numba 를 import 하는 순간 그 줄이 traceback 에 찍힌다
    sys.modules["numba"] = None

    import numpy as np

    import app.main  # noqa: F401
    from app.services.tremor import TremorAnalyzer, TremorConfig

    analyzer = TremorAnalyzer(TremorConfig(f0_chunk_sec=5.0))
    single_call, chunked = 3.0, 6.0
    for seconds in (single_call, chunked):
        t = np.arange(int(16000 * seconds)) / 16000
        pcm = (0.3 * 32767 * np.sin(2 * np.pi * 150 * t)).astype(np.int16).tobytes()
        assert analyzer.analyze(pcm).voiced_sec > 0

    heavy = ("numba", "llvmlite", "librosa.core", "librosa.feature", "librosa.util")
    print(sorted(m for m in heavy if sys.modules.get(m)))
    """
)


def test_server_process_analyzes_tremor_without_numba() -> None:
    # EAGER_IMPORT 가 켜져 있으면 lazy_loader 가 librosa 하위 모듈을 전부 올려 버린다
    env = {k: v for k, v in os.environ.items() if k != "EAGER_IMPORT"}
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"probe timed out:\n{exc.stderr}")

    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().splitlines()[-1] == "[]"
