import json

from scripts.stt_compare import cer, load_runs, normalize, summarize_condition


def test_normalize_drops_space_and_punctuation() -> None:
    assert normalize("안녕하세요, 예약 변경하려고 전화드렸는데요.") == "안녕하세요예약변경하려고전화드렸는데요"
    assert normalize("010-2345-6789예요.") == "01023456789예요"


def test_cer_is_char_edit_distance_over_reference_length() -> None:
    assert cer("가나다라", "가나다라") == 0.0
    assert cer("가나다라", "가나라") == 0.25
    assert cer("가나다라", "가나다라마") == 0.25
    assert cer("가나다라", "") == 1.0


def test_summarize_condition_counts_finals_and_mismatches() -> None:
    manifest = {"a.wav": "안녕하세요", "b.wav": "감사합니다"}
    turns = [
        {"audio": "a.wav", "stt_final_ms": 900.0, "transcript": "안녕하세요."},
        {"audio": "a.wav", "stt_final_ms": 1100.0, "transcript": "안녕하세용"},
        {"audio": "b.wav", "stt_final_ms": None, "transcript": None, "terminal": "TIMEOUT"},
        {"audio": "b.wav", "stt_final_ms": 1000.0, "transcript": "감사합니다"},
    ]
    s = summarize_condition(turns, manifest)
    assert s["n"] == 4
    assert s["finals"] == 3
    assert 0.3 < s["final_rate_lo"] < 0.75 == s["final_rate"] < s["final_rate_hi"] < 1.0
    assert s["stt_final_p50"] == 1000.0
    assert s["mismatches"] == 1
    assert s["mean_cer"] > 0


def test_load_runs_reads_condition_from_json(tmp_path) -> None:
    p = tmp_path / "gemini_live-800.json"
    p.write_text(json.dumps({
        "session_id": "s", "engine": "gemini_live", "tail_silence_ms": 800,
        "turns": [{"audio": "a.wav", "stt_final_ms": 1.0, "transcript": "x"}],
    }), encoding="utf-8")
    runs = load_runs([str(p)])
    assert list(runs) == [("gemini_live", 800)]
    assert runs[("gemini_live", 800)][0]["audio"] == "a.wav"


def test_session_closed_turns_are_not_stt_attempts() -> None:
    manifest = {"a.wav": "가"}
    turns = [
        {"audio": "a.wav", "stt_final_ms": 500.0, "transcript": "가"},
        {"audio": "a.wav", "stt_final_ms": None, "transcript": None, "terminal": "TIMEOUT"},
        {"audio": "a.wav", "stt_final_ms": None, "transcript": None, "terminal": "error"},
        {"audio": "a.wav", "stt_final_ms": None, "transcript": None, "terminal": "CLOSED"},
    ]
    s = summarize_condition(turns, manifest)
    assert s["n"] == 2
    assert s["finals"] == 1
    assert s["excluded"] == 2
