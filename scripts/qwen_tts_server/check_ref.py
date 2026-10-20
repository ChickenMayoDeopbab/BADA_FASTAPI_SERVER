"""새 ref 오디오 검사 + voices.json 항목 출력 (torch 없이, 맥·학교 어디서든)

  python check_ref.py <wav> --voice <보이스 이름> --text "<ref 대본>" [--refs-dir refs] [--date YYYYMMDD]

- 실패(exit 1): 모노 아님, 길이 3~20초 밖
- 경고: 24000Hz 아님, 길이 10.7~19.5초(지금 근거 범위) 밖, 앞이나 뒤 무음(-50dBFS·20ms) 1초 넘음
- 보고: 잡음 바닥, 피크, 클리핑, float 형식
- 통과하면 새 파일명 refs/<보이스>-<날짜>-<지문>.wav 와 voices.json 항목을 출력한다.
  같은 경로가 이미 있으면 거부(exit 2) — ref 파일은 덮어쓰지 않는다(워커가 기동 때 스냅숏, 롤링으로만 바꿈)
지문은 server.py ref_fingerprint 와 같은 식이다(sha1(파일 바이트 + ref_text)[:8]).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys

import numpy as np
import soundfile as sf

_FRAME_S, _SILENCE_DBFS = 0.02, -50.0
_LEN_FAIL, _LEN_WARN = (3.0, 20.0), (10.7, 19.5)


def ref_fingerprint(data: bytes, text: str) -> str:
    return hashlib.sha1(data + text.encode("utf-8")).hexdigest()[:8]


def inspect(path: pathlib.Path) -> dict:
    info = sf.info(str(path))
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = x[:, 0] if x.shape[1] else np.zeros(0, dtype=np.float32)
    n = int(sr * _FRAME_S)
    k = len(mono) // n
    frames = mono[: k * n].reshape(k, n).astype(np.float64) if k else np.zeros((0, n))
    db = 10 * np.log10((frames**2).mean(axis=1) + 1e-12) if k else np.zeros(0)
    loud = np.nonzero(db >= _SILENCE_DBFS)[0]
    lead = float(loud[0] * _FRAME_S) if len(loud) else float(k * _FRAME_S)
    trail = float((k - 1 - loud[-1]) * _FRAME_S) if len(loud) else float(k * _FRAME_S)
    peak = float(np.abs(mono).max()) if len(mono) else 0.0
    return {
        "channels": info.channels,
        "sample_rate": sr,
        "subtype": info.subtype,
        "duration_s": round(len(mono) / sr, 2) if sr else 0.0,
        "lead_silence_s": round(lead, 2),
        "trail_silence_s": round(trail, 2),
        "noise_floor_dbfs": round(float(np.percentile(db, 10)), 1) if k else None,
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1),
        "clipped_samples": int((np.abs(mono) >= 0.999).sum()),
    }


def judge(report: dict) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    if report["channels"] != 1:
        fails.append(f"모노가 아님(channels={report['channels']})")
    if not _LEN_FAIL[0] <= report["duration_s"] <= _LEN_FAIL[1]:
        fails.append(f"길이 {report['duration_s']}s 가 {_LEN_FAIL[0]}~{_LEN_FAIL[1]}s 밖")
    elif not _LEN_WARN[0] <= report["duration_s"] <= _LEN_WARN[1]:
        warns.append(f"길이 {report['duration_s']}s 가 지금 근거 범위 {_LEN_WARN[0]}~{_LEN_WARN[1]}s 밖")
    if report["sample_rate"] != 24000:
        warns.append(f"샘플레이트 {report['sample_rate']}Hz(24000 아님, 포크가 리샘플)")
    if report["lead_silence_s"] > 1.0 or report["trail_silence_s"] > 1.0:
        warns.append(f"앞뒤 무음 {report['lead_silence_s']}/{report['trail_silence_s']}s 가 1초 넘음")
    if report["clipped_samples"]:
        warns.append(f"클리핑 {report['clipped_samples']}샘플")
    if report["subtype"].startswith("FLOAT") or report["subtype"] == "DOUBLE":
        warns.append(f"형식 {report['subtype']}")
    return fails, warns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--refs-dir", default="refs")
    ap.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    a = ap.parse_args(argv)
    path = pathlib.Path(a.wav)
    report = inspect(path)
    fails, warns = judge(report)
    print(json.dumps(report, ensure_ascii=False))
    for w in warns:
        print(f"경고: {w}")
    for f in fails:
        print(f"실패: {f}")
    if fails:
        return 1
    fp = ref_fingerprint(path.read_bytes(), a.text)
    target = pathlib.Path(a.refs_dir) / f"{a.voice}-{a.date}-{fp}.wav"
    if target.exists():
        print(f"거부: {target} 가 이미 있다 — ref 파일은 덮어쓰지 않는다")
        return 2
    print(f"지문 {fp} → 이 파일을 {target} 로 복사")
    print(json.dumps({a.voice: {"ref_audio": str(target), "ref_text": a.text}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
