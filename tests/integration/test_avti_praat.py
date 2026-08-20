"""벤더링한 tremor.praat 3.05 를 실제로 돌려보는 회귀 테스트.

저자가 동봉한 합성 테스트음(sounds/)과 그 기준 출력(results/)을 그대로 쓴다.
값이 어긋나면 Praat 버전이 바뀌었거나 우리 호출 방식이 틀어진 것이다.

README 가 경고하듯 어떤 Praat 버전(6.1.13)은 FTrC/FTrCIP 계열을 **틀린 값으로**
조용히 내놓는다. FTrCIP 는 AVTI 에서 비중이 가장 큰 변수라, 조용한 오류를
잡아내려면 기준값 대조가 필요하다.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.services.avti import (
    AvtiAnalyzer,
    AvtiConfig,
    AvtiStatus,
    compute_avti,
    parse_tremor_table,
)

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "tremor3.05"
SCRIPT = VENDOR / "tremor.praat"
TEST_SOUND = VENDOR / "sounds" / "test_F3_I10_envel_A5_I20.wav"
REFERENCE = VENDOR / "results" / "tremor_resCon.txt"

pytestmark = pytest.mark.skipif(
    shutil.which("praat") is None or not SCRIPT.is_file(),
    reason="praat 바이너리 또는 벤더링된 스크립트 없음",
)


@pytest.fixture(scope="module")
def analyzer() -> AvtiAnalyzer:
    return AvtiAnalyzer(praat_bin="praat", script_path=str(SCRIPT))


@pytest.fixture(scope="module")
def expected() -> dict[str, float | None]:
    table = parse_tremor_table(REFERENCE.read_text(encoding="utf-8"))
    return table[TEST_SOUND.stem]


def _run_on_test_sound(analyzer: AvtiAnalyzer, tmp_path: Path) -> dict:
    workdir = analyzer._prepare_workdir(tmp_path)
    shutil.copy2(TEST_SOUND, workdir / "sounds" / TEST_SOUND.name)
    return analyzer._run_praat(workdir / "sounds")


def test_vendored_script_is_available(analyzer: AvtiAnalyzer) -> None:
    assert analyzer.available


def test_reference_file_has_all_four_avti_inputs(expected) -> None:
    assert set(expected) == {"ftrcip", "atri", "fcohnr", "fcom"}
    assert all(v is not None for v in expected.values())


def test_reproduces_the_authors_reference_values(analyzer, expected, tmp_path) -> None:
    """전체 사슬(폼 인자·FULL-TRUST·작업 디렉터리·파서) 검증."""
    table = _run_on_test_sound(analyzer, tmp_path)

    got = table[TEST_SOUND.stem]
    for key, want in expected.items():
        assert got[key] == pytest.approx(want, rel=1e-6), key


def test_avti_from_reference_values_is_finite(expected) -> None:
    value = compute_avti(**{k: float(v) for k, v in expected.items()})
    assert value == pytest.approx(
        2.445
        + 0.467 * expected["ftrcip"]
        + 0.077 * expected["atri"]
        + 0.102 * expected["fcohnr"]
        - 1.405 * expected["fcom"]
    )


def test_runs_are_isolated_from_each_other(analyzer, tmp_path) -> None:
    """스크립트는 상대경로를 '스크립트 디렉터리' 기준으로 푼다.

    작업 디렉터리를 복사하지 않으면 동시 세션이 같은 결과 파일에 행을 덧쌓는다.
    """
    first = _run_on_test_sound(analyzer, tmp_path / "a")
    second = _run_on_test_sound(analyzer, tmp_path / "b")

    assert len(first) == 1 and len(second) == 1
    assert first == second
    # 원본 벤더 디렉터리는 건드리지 않는다
    assert (VENDOR / "results" / "tremor_resCon.txt").read_text(encoding="utf-8").count("\n") <= 2
    assert not (VENDOR / "temp").exists()


def test_end_to_end_produces_an_avti_score(analyzer, tmp_path, monkeypatch) -> None:
    """analyze() 로 들어가는 전체 경로. 합성 테스트음을 통째로 한 파트로 준다."""
    import wave

    with wave.open(str(TEST_SOUND), "rb") as wav:
        frames, rate = wav.getnframes(), wav.getframerate()
        pcm = wav.readframes(frames)
    duration = frames / rate

    monkeypatch.setattr(analyzer, "config", AvtiConfig(sample_rate=rate))
    results = analyzer.analyze(pcm, [(0.0, duration)], [(0.0, duration)])

    assert results[0].status == AvtiStatus.OK
    assert results[0].avti is not None
    assert results[0].fcom is not None and 0.0 <= results[0].fcom <= 1.0
