"""
TremorAnalyzer 수동 검증 — "말하면 shake_count가 올라가는지" 직접 확인한다.

전제: 분석 의존성 설치 필요 (librosa/numpy/scipy). 마이크 모드는 sounddevice 추가.
    uv 사용 시:   uv sync
    pip 사용 시:  pip install librosa scipy numpy sounddevice

실행 (저장소 루트에서, 모듈로):
    # 1) 합성 신호 자체검증 — 마이크 없이 카운터가 도는지 즉시 확인 (기본값)
    python -m tests.tremor.manual_tremor_check
    python -m tests.tremor.manual_tremor_check synth

    # 2) 마이크로 직접 말하기 (기본 6초)
    python -m tests.tremor.manual_tremor_check mic
    python -m tests.tremor.manual_tremor_check mic 8        # 8초 녹음

    # 3) wav 파일 분석
    python -m tests.tremor.manual_tremor_check file 내목소리.wav

떨림(shake_count)은 '지속 모음 + 4~8Hz 음높이 떨림'에서만 잡힌다.
평상시 말투로는 0이 정상이고, "아~~~"를 떨리는 목소리로 길게 내야 카운트된다.
합성(synth) 모드는 네 목소리/마이크와 무관하게 카운터 자체가 도는지 증명한다.
"""
from __future__ import annotations

import pathlib
import sys
import time

# 저장소 루트를 import 경로에 추가 (파일을 직접 실행해도 app.* 가 잡히도록)
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    from app.services.tremor import TremorAnalyzer, TremorConfig
except ModuleNotFoundError as e:
    sys.exit(
        f"의존성 누락: {e.name}\n"
        f"먼저 설치하세요 →  uv sync   (또는  pip install librosa scipy numpy)"
    )

CFG = TremorConfig()
SR = CFG.sample_rate  # 16000


# ---------------------------------------------------------------- 공통 유틸

def float_to_pcm16(x: np.ndarray) -> bytes:
    """float 파형([-1,1]) → int16 PCM bytes (analyze()가 받는 형식)."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def report(label: str, pcm: bytes) -> int:
    """analyze() 결과를 보기 좋게 출력하고 shake_count를 반환한다."""
    res = TremorAnalyzer().analyze(pcm)
    dbg = res.debug_windows
    on = [d for d in dbg if d["on"]]
    max_amp = max((d["amp"] for d in dbg), default=0.0)
    max_conc = max((d["conc"] for d in dbg), default=0.0)

    print(f"\n===== {label} =====")
    print(f"  shake_count : {res.shake_count}")
    print(f"  voiced_sec  : {res.voiced_sec:.2f}   sustained_sec: {res.sustained_sec:.2f}")
    print(f"  창(window)  : 전체 {len(dbg)}개 중 on {len(on)}개")
    print(
        f"  관측 최대   : amp={max_amp:.1f} cents (임계 {CFG.amp_thresh_cents})"
        f"   conc={max_conc:.2f} (임계 {CFG.concentration_min})"
    )
    if res.episodes:
        print("  에피소드 (시작s, 끝s, 평균peak Hz, 평균amp cents):")
        for ep in res.episodes:
            print(f"    - {ep}")
    if on:
        print("  on 윈도우 (t, amp, conc, fpk):")
        for d in on[:12]:
            print(f"    t={d['t']:>6}  amp={d['amp']:>5}  conc={d['conc']:>4}  fpk={d['fpk']:>4}")
        if len(on) > 12:
            print(f"    ... (+{len(on) - 12}개)")
    if res.shake_count == 0:
        print("  ⓘ 0입니다. 떨림이 더 길고/분명해야 잡혀요. 위 '관측 최대 amp'가")
        print(f"    임계({CFG.amp_thresh_cents})에 얼마나 가까운지 보면 감이 옵니다.")
    return res.shake_count


# ---------------------------------------------------------------- 합성 신호

def synth_pcm(bursts, seconds, f0=150.0, tremor_hz=6.0, depth_cents=50.0, sr=SR) -> bytes:
    """
    f0(Hz) 캐리어를 만들되, bursts=[(시작s,끝s), ...] 구간에서만
    음높이를 tremor_hz로 ±depth_cents 떨리게 한 합성 발성음(int16 PCM)을 만든다.
    """
    n = int(seconds * sr)
    t = np.arange(n) / sr

    cents = np.zeros(n)  # 시간별 음높이 흔들림(cents)
    for start, end in bursts:
        seg = (t >= start) & (t < end)
        cents[seg] = depth_cents * np.sin(2 * np.pi * tremor_hz * t[seg])

    f_inst = f0 * (2.0 ** (cents / 1200.0))        # 순간 주파수(Hz)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr     # 위상으로 적분
    # 배음을 섞어 librosa.yin이 피치를 안정적으로 추적하게 함
    y = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    y = 0.9 * y / np.max(np.abs(y))
    return float_to_pcm16(y)


def run_synth():
    print("합성 신호로 카운터 자체를 검증합니다 (마이크·목소리 불필요).")

    bursts = [(1.0, 3.0), (4.5, 6.5), (8.0, 10.0)]  # 떨림 3구간
    c1 = report("떨림 3구간 합성음 (6Hz, ±50cents)", synth_pcm(bursts, seconds=11.0))
    verdict = "✅ 검출 OK" if c1 >= 1 else "❌ 미검출 — 알고리즘/임계값 점검 필요"
    print(f"  → 떨림 구간 3개 주입 / {c1}개 카운트  ({verdict})")

    c2 = report("평탄음 (떨림 없음)", synth_pcm([], seconds=4.0))
    verdict = "✅ 오검출 없음" if c2 == 0 else "❌ 오검출 발생"
    print(f"  → 떨림 0 주입 / {c2}개 카운트  ({verdict})")


# ---------------------------------------------------------------- 마이크 / 파일

def run_mic(seconds: float):
    try:
        import sounddevice as sd
    except ModuleNotFoundError:
        sys.exit("마이크 모드는 sounddevice가 필요합니다 →  pip install sounddevice")

    print(f"{seconds:.0f}초 동안 녹음합니다.")
    print("  떨림을 보려면: '아~~~'를 일부러 떨리는(흔들리는) 목소리로 길게 내보세요.")
    print("  (평상시 말투로는 0이 정상입니다.)")
    for k in (3, 2, 1):
        print(f"  {k}...", flush=True)
        time.sleep(1)
    print("● 녹음 시작!", flush=True)
    rec = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    print("■ 녹음 끝. 분석 중...", flush=True)
    report("내 목소리 (마이크)", rec.reshape(-1).tobytes())


def run_file(path: str):
    try:
        import librosa
    except ModuleNotFoundError:
        sys.exit("librosa가 필요합니다 →  uv sync  (또는 pip install librosa)")
    y, _ = librosa.load(path, sr=SR, mono=True)  # float32 [-1,1]
    report(f"파일: {path}", float_to_pcm16(y))


# ---------------------------------------------------------------- 진입점

def main(argv: list[str]) -> None:
    mode = argv[0] if argv else "synth"
    if mode == "synth":
        run_synth()
    elif mode == "mic":
        run_mic(float(argv[1]) if len(argv) > 1 else 6.0)
    elif mode == "file":
        if len(argv) < 2:
            sys.exit("사용법:  python -m tests.tremor.manual_tremor_check file <경로.wav>")
        run_file(argv[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
