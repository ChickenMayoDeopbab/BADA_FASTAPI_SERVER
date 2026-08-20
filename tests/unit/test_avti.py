"""AVTI 측정 단위 테스트.

핵심 계약 두 가지를 지킨다.
  - 못 잰 값은 절대 0이 되지 않는다 (논문이 직접 경고한 부분)
  - 3초 지속발성이 없으면 실패가 아니라 NO_SUSTAINED 라는 정상 결과다
"""
import sys

import numpy as np
import pytest

from app.services.avti import (
    AvtiAnalyzer,
    AvtiConfig,
    AvtiStatus,
    compute_avti,
    parse_tremor_table,
)
from app.services.tremor import TremorAnalyzer

SAMPLE_RATE = 16000


def _pcm(duration_s: float, *, silent: bool = False) -> bytes:
    """지속 모음 비슷한 PCM. silent=True 면 무음."""
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    if silent:
        return np.zeros(len(t), dtype=np.int16).tobytes()
    f0 = 140 * (1 + 0.05 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    y = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)
    return (y / np.abs(y).max() * 0.6 * 32767).astype(np.int16).tobytes()


def stub_script(tmp_path) -> str:
    """available 판정을 통과하는 최소 스크립트 트리(tremor.praat + procedures/)."""
    script = tmp_path / "tremor.praat"
    script.write_text("# stub")
    (tmp_path / "procedures").mkdir(exist_ok=True)
    return str(script)


def _analyzer(tmp_path, **kwargs) -> AvtiAnalyzer:
    """_run_praat 을 가짜로 바꿔 쓰는 테스트용. praat 설치 여부를 타지 않는다.

    available 판정이 실행 파일 존재를 보므로, 어디에나 있는 파이썬 경로를
    praat 자리에 넣는다. 실제로 실행되지는 않는다.
    """
    return AvtiAnalyzer(
        praat_bin=sys.executable,
        script_path=stub_script(tmp_path),
        config=AvtiConfig(**kwargs) if kwargs else None,
    )


# --- 회귀식 -------------------------------------------------------------


def test_regression_matches_paper_coefficients() -> None:
    """AVTI = 2.445 + 0.467×FTrCIP + 0.077×ATrI + 0.102×FCoHNR - 1.405×FCoM"""
    got = compute_avti(ftrcip=1.0, atri=1.0, fcohnr=1.0, fcom=1.0)
    assert got == pytest.approx(2.445 + 0.467 + 0.077 + 0.102 - 1.405)


def test_higher_fcom_lowers_avti() -> None:
    """FCoM 은 계수가 음수인 유일한 변수다(떨림이 심할수록 낮아짐)."""
    low = compute_avti(ftrcip=2.0, atri=5.0, fcohnr=3.0, fcom=0.2)
    high = compute_avti(ftrcip=2.0, atri=5.0, fcohnr=3.0, fcom=0.9)
    assert high < low


# --- 출력 표 파싱 --------------------------------------------------------


def test_parser_finds_columns_by_name() -> None:
    table = parse_tremor_table(
        "file\tFCoM\tFTrF\tFTrCIP\tATrI\tFCoHNR\n"
        "part000\t0.81\t5.2\t1.4\t9.7\t3.3\n"
    )
    assert table["part000"] == {"fcom": 0.81, "ftrcip": 1.4, "atri": 9.7, "fcohnr": 3.3}


def test_parser_survives_column_reorder() -> None:
    """스크립트가 컬럼 순서를 바꿔도 깨지지 않아야 한다."""
    table = parse_tremor_table(
        "ATrI\tfile\tFCoHNR\tFTrCIP\tFCoM\n"
        "9.7\tpart000\t3.3\t1.4\t0.81\n"
    )
    assert table["part000"]["atri"] == 9.7
    assert table["part000"]["fcom"] == 0.81


def test_parser_maps_undefined_to_none_not_zero() -> None:
    table = parse_tremor_table(
        "file\tFCoM\tFTrCIP\tATrI\tFCoHNR\n"
        "part000\t0.81\t--undefined--\t9.7\t3.3\n"
    )
    assert table["part000"]["ftrcip"] is None
    assert table["part000"]["ftrcip"] != 0


def test_parser_skips_banner_before_header() -> None:
    table = parse_tremor_table(
        "Tremor 3.05\n"
        "analysing folder...\n"
        "file\tFCoM\tFTrCIP\tATrI\tFCoHNR\n"
        "part001\t0.5\t1.0\t2.0\t3.0\n"
    )
    assert list(table) == ["part001"]


# --- 구간 고르기 ---------------------------------------------------------


def test_no_window_when_sustained_shorter_than_three_seconds(tmp_path) -> None:
    analyzer = _analyzer(tmp_path)
    assert analyzer.pick_window((0.0, 10.0), [(1.0, 3.5)]) is None


def test_window_is_centered_on_longest_run(tmp_path) -> None:
    """논문처럼 앞뒤 상승·하강을 피해 가운데 3초를 쓴다."""
    analyzer = _analyzer(tmp_path)
    window = analyzer.pick_window((0.0, 30.0), [(1.0, 2.0), (10.0, 20.0)])
    assert window == (13.5, 16.5)  # 10~20 의 중심 15 에서 ±1.5


def test_sustained_outside_part_is_ignored(tmp_path) -> None:
    analyzer = _analyzer(tmp_path)
    # 런은 길지만 파트와 겹치는 부분이 2초뿐
    assert analyzer.pick_window((0.0, 2.0), [(0.0, 30.0)]) is None


def test_window_length_follows_config(tmp_path) -> None:
    analyzer = _analyzer(tmp_path, window_sec=5.0, min_sustained_sec=5.0)
    assert analyzer.pick_window((0.0, 30.0), [(10.0, 20.0)]) == (12.5, 17.5)


# --- 분석 흐름 -----------------------------------------------------------


def test_missing_script_marks_every_part() -> None:
    analyzer = AvtiAnalyzer(praat_bin="praat", script_path=None)
    results = analyzer.analyze(_pcm(5), [(0.0, 5.0), (0.0, 2.0)], [(0.0, 5.0)])
    assert [r.status for r in results] == [AvtiStatus.NO_SCRIPT] * 2
    assert all(r.avti is None for r in results)


def test_praat_not_called_when_no_sustained_segment(tmp_path, monkeypatch) -> None:
    analyzer = _analyzer(tmp_path)
    called = []
    monkeypatch.setattr(analyzer, "_run_praat", lambda d: called.append(d) or {})

    results = analyzer.analyze(_pcm(5), [(0.0, 5.0)], [(0.0, 1.0)])

    assert not called
    assert results[0].status == AvtiStatus.NO_SUSTAINED
    assert results[0].avti is None


def test_successful_measurement_computes_avti(tmp_path, monkeypatch) -> None:
    analyzer = _analyzer(tmp_path)
    monkeypatch.setattr(
        analyzer, "_run_praat",
        lambda d: {"part000": {"ftrcip": 1.0, "atri": 2.0, "fcohnr": 3.0, "fcom": 0.5}},
    )

    results = analyzer.analyze(_pcm(6), [(0.0, 6.0)], [(0.0, 6.0)])

    assert results[0].status == AvtiStatus.OK
    assert results[0].avti == pytest.approx(
        compute_avti(ftrcip=1.0, atri=2.0, fcohnr=3.0, fcom=0.5), abs=1e-4
    )
    assert results[0].ftrcip == 1.0


def test_one_undefined_variable_blocks_avti_but_keeps_the_rest(tmp_path, monkeypatch) -> None:
    analyzer = _analyzer(tmp_path)
    monkeypatch.setattr(
        analyzer, "_run_praat",
        lambda d: {"part000": {"ftrcip": None, "atri": 2.0, "fcohnr": 3.0, "fcom": 0.5}},
    )

    results = analyzer.analyze(_pcm(6), [(0.0, 6.0)], [(0.0, 6.0)])

    assert results[0].status == AvtiStatus.UNDEFINED
    assert results[0].avti is None
    assert results[0].atri == 2.0  # 잰 건 버리지 않는다


def test_praat_failure_does_not_raise(tmp_path, monkeypatch) -> None:
    analyzer = _analyzer(tmp_path)

    def boom(_):
        raise RuntimeError("praat exit=1")

    monkeypatch.setattr(analyzer, "_run_praat", boom)
    results = analyzer.analyze(_pcm(6), [(0.0, 6.0)], [(0.0, 6.0)])

    assert results[0].status == AvtiStatus.ERROR
    assert results[0].avti is None


def test_results_keep_part_order(tmp_path, monkeypatch) -> None:
    analyzer = _analyzer(tmp_path)
    monkeypatch.setattr(analyzer, "_run_praat", lambda d: {})
    parts = [(0.0, 20.0), (0.0, 1.0), (2.0, 12.0), (13.0, 14.0)]

    results = analyzer.analyze(_pcm(20), parts, [(0.0, 20.0)])

    assert [r.part_index for r in results] == [0, 1, 2, 3]


def test_every_part_gets_a_row(tmp_path, monkeypatch) -> None:
    """못 잰 파트도 행이 남아야 '얼마나 자주 실패하나'를 셀 수 있다."""
    analyzer = _analyzer(tmp_path)
    monkeypatch.setattr(analyzer, "_run_praat", lambda d: {})

    results = analyzer.analyze(_pcm(20), [(0.0, 20.0), (1.0, 5.0), (6.0, 18.0)], [])

    assert len(results) == 3
    assert all(r.status == AvtiStatus.NO_SUSTAINED for r in results)


# --- TremorAnalyzer 연동 -------------------------------------------------


def test_tremor_result_carries_sustained_spans() -> None:
    result = TremorAnalyzer().analyze(_pcm(4))
    assert result.sustained_spans
    assert all(e > s for s, e in result.sustained_spans)


def test_silence_yields_no_sustained_spans() -> None:
    result = TremorAnalyzer().analyze(_pcm(3, silent=True))
    assert result.sustained_spans == []


# --- Praat 버전 대응 -----------------------------------------------------


def test_full_trust_detected_from_help(monkeypatch) -> None:
    """Praat 7 은 --FULL-TRUST 가 필요하고 6.x 는 그 플래그를 모른다."""
    import subprocess

    from app.services import avti as avti_mod

    class _Proc:
        def __init__(self, out: str) -> None:
            self.stdout, self.stderr = out, ""

    avti_mod.supports_full_trust.cache_clear()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc("--FULL-TRUST  trust"))
    assert avti_mod.supports_full_trust("praat7") is True

    avti_mod.supports_full_trust.cache_clear()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc("--run  --version"))
    assert avti_mod.supports_full_trust("praat63") is False


def test_missing_binary_is_not_full_trust(monkeypatch) -> None:
    import subprocess

    from app.services import avti as avti_mod

    def _boom(*a, **k):
        raise FileNotFoundError("praat 없음")

    avti_mod.supports_full_trust.cache_clear()
    monkeypatch.setattr(subprocess, "run", _boom)
    assert avti_mod.supports_full_trust("nope") is False
    avti_mod.supports_full_trust.cache_clear()


def test_unavailable_when_praat_binary_missing(tmp_path) -> None:
    """스크립트가 있어도 praat 가 없으면 NO_SCRIPT 여야 한다."""
    analyzer = AvtiAnalyzer(praat_bin="praat-that-does-not-exist", script_path=stub_script(tmp_path))
    assert analyzer.available is False
    results = analyzer.analyze(_pcm(5), [(0.0, 5.0)], [(0.0, 5.0)])
    assert results[0].status == AvtiStatus.NO_SCRIPT
