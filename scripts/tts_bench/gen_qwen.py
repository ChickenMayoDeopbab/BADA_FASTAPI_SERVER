"""Qwen3-TTS 를 운영 워커와 같은 설정으로 돌려 벤치 문장을 합성한다 (집 PC, 포크 venv).

  <포크 venv>/bin/python gen_qwen.py --sentences sentences.tsv --voices /mnt/c/qwen-out/voices.json --only ai \
      --ref 'new=/mnt/c/qwen-out/ref.wav:녹음에서 말한 그대로의 전사' --out /mnt/c/qwen-out/bench/out
워커 server.py 의 _generate 인자·max_frames_for·24k→16k 리샘플·_pcm16 과 FastAPI 가드(runaway_cap_ms)를 따른다.
운영처럼 문장 끝에 공백 하나를 붙여 보낸다. 결과: out/qwen_<이름>/<id>.wav(16kHz PCM16) + gen.jsonl.
같은 GPU 에 qwen-tts 서비스가 떠 있으면 먼저 내릴 것(8GB 에 모델 두 벌은 안 들어간다).
"""

import argparse
import json
import math
import os
import pathlib
import time
import wave

import numpy as np

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OUT_SR = 16000
EMIT = 2
FRAME_S = 0.08
CAP_BASE_S = 5.0
CAP_PER_CHAR_S = 0.4
GUARD_FLOOR_MS = 3000
GUARD_MS_PER_CHAR = 300


def max_frames_for(text: str) -> int:
    cap = math.ceil((CAP_BASE_S + CAP_PER_CHAR_S * len(text)) / FRAME_S)
    return cap + (-cap) % EMIT


def hit_cap(n_samples: int, sr: int | None, cap: int) -> bool:
    return sr is not None and n_samples > (cap - 0.5) * FRAME_S * sr


def guard_cap_ms(text: str) -> int:
    return GUARD_FLOOR_MS + len(text.strip()) * GUARD_MS_PER_CHAR


def gen_kwargs(text: str, ref: str, ref_text: str) -> dict:
    return dict(text=text, language="Korean", ref_audio=ref, ref_text=ref_text,
                emit_every_frames=EMIT, temperature=0.7, top_k=20, max_frames=max_frames_for(text))


def to_pcm16(w: np.ndarray) -> bytes:
    return (np.clip(w, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def load_tsv(path: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                key, value = line.rstrip("\n").split("\t", 1)
                rows[key] = value
    return rows


def parse_voices(voices_file: str | None, only: str | None, refs: list[str]) -> dict[str, tuple[str, str]]:
    voices: dict[str, tuple[str, str]] = {}
    if voices_file:
        with open(os.path.expanduser(voices_file), encoding="utf-8") as f:
            for name, v in json.load(f).items():
                if not only or name in only.split(","):
                    voices[name] = (os.path.expanduser(v["ref_audio"]), v["ref_text"])
    for spec in refs:
        name, rest = spec.split("=", 1)
        wav, text = rest.split(":", 1)
        voices[name] = (os.path.expanduser(wav), text.strip())
    return voices


def synth(model, resample, ref: str, ref_text: str, sentence: str) -> tuple[bytes, dict]:
    """워커 _synth 와 같은 생성·리샘플, 그 뒤 FastAPI 가드 상한에서 자른다"""
    text = sentence + " "
    kw = gen_kwargs(text, ref, ref_text)
    parts, sr = [], None
    for pcm, chunk_sr in model.stream_generate_voice_clone(**kw):
        parts.append(np.asarray(pcm, dtype=np.float32))
        sr = chunk_sr
    w = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    eos = not hit_cap(len(w), sr, kw["max_frames"])
    pcm16 = to_pcm16(resample(w, sr)) if len(w) else b""
    limit = guard_cap_ms(text) * OUT_SR // 1000 * 2
    guard = len(pcm16) > limit
    pcm16 = pcm16[:limit]
    frames = round(len(w) / sr / FRAME_S) if sr else 0
    return pcm16, dict(eos=eos, frames=frames, guard=guard, audio_s=round(len(pcm16) / 2 / OUT_SR, 3))


def write_wav(path: pathlib.Path, pcm: bytes) -> None:
    tmp = path.with_suffix(".part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_SR)
        w.writeframes(pcm)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="운영 워커 설정으로 벤치 문장 합성")
    ap.add_argument("--sentences", required=True, help="id<TAB>문장 (레포 scripts/tts_bench/sentences.tsv)")
    ap.add_argument("--voices", help="운영 voices.json")
    ap.add_argument("--only", help="voices.json 에서 이 이름들만(쉼표)")
    ap.add_argument("--ref", action="append", default=[], help="이름=wav경로:전사")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    sentences = load_tsv(a.sentences)
    voices = parse_voices(a.voices, a.only, a.ref)
    assert voices, "--voices 또는 --ref 가 필요하다"
    for name, (wav_path, text) in voices.items():
        assert os.path.isfile(wav_path), (name, wav_path)
        print(f"참조 {name}: {wav_path} · {text[:40]}", flush=True)
    out = pathlib.Path(os.path.expanduser(a.out))

    import torch
    import torchaudio
    from qwen_tts import Qwen3TTSModel

    def resample(w: np.ndarray, sr: int) -> np.ndarray:
        return torchaudio.functional.resample(torch.from_numpy(w), sr, OUT_SR).numpy()

    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=a.device, dtype=torch.bfloat16,
                                          attn_implementation="flash_attention_2")
    model.enable_streaming_optimizations()
    for name, (ref, ref_text) in voices.items():
        d = out / f"qwen_{name}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "ref.json").write_text(json.dumps({"ref_audio": ref, "ref_text": ref_text}, ensure_ascii=False),
                                    encoding="utf-8")
        t = time.perf_counter()
        synth(model, resample, ref, ref_text, "네, 안녕하세요.")
        print(f"[{name}] 워밍업·컴파일 {time.perf_counter() - t:.0f}s", flush=True)
        for sid, sentence in sentences.items():
            path = d / f"{sid}.wav"
            if path.exists():
                continue
            t = time.perf_counter()
            pcm, info = synth(model, resample, ref, ref_text, sentence)
            write_wav(path, pcm)
            with open(d / "gen.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": sid, **info}, ensure_ascii=False) + "\n")
            flag = "" if info["eos"] and not info["guard"] else f" ⚠ eos={info['eos']} guard={info['guard']}"
            print(f"[{name}] {sid} {info['audio_s']:5.2f}s ({time.perf_counter() - t:.1f}s){flag} · {sentence[:30]}",
                  flush=True)
    print(f"끝 → {out} · 보이스 {len(voices)} × 문장 {len(sentences)}", flush=True)


if __name__ == "__main__":
    main()
