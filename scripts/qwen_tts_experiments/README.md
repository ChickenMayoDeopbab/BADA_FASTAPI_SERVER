# Qwen3-TTS 실험 스크립트 (집 PC 에서 회수, 2026-09-02)

계획 0043 의 근거가 된 실측(첫 청크 181ms / RTF 0.77)을 만든 일회성 스크립트들이다.
지금까지 집 PC 한 대에만 있었고 버전 관리가 안 됐다 (계획 0045).

**원본을 그대로 둔다 — lint 를 위해 고치지 않는다.** 코드가 바뀌면 그 스크립트가 낸 숫자가
아니게 되어 회수한 목적이 사라진다. 그래서 `pyproject.toml` 의 ruff 에서 이 디렉터리를 제외했다.

실행 환경은 포크 쪽이다: `dffdeeq/Qwen3-TTS-streaming` @ `13ef823` (Apache-2.0),
Python 3.12 / torch 2.9.1 / transformers 4.57.3 / flash-attn 2.8.3 — 자세한 건 DECISIONS.md 2026-09-02.

배포되는 서버 본체는 여기가 아니라 `scripts/qwen_tts_server/server.py` 다.
