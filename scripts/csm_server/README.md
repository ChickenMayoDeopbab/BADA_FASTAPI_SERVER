# CSM 실시간 TTS 워커 (`scripts/csm_server/`, 2026-09-28)

한국어 문맥 조건 음성 모델(`sesame/csm-1b` 파인튜닝 b1, 계획 `.harness/plans/13`·`15`)을 통화 파이프라인에 붙이는 워커다. 설계 `.harness/plans/16-CSM-워커-설계.md`, 구현 계획 `16a`. Qwen 워커(`scripts/qwen_tts_server/`)와 **같은 자리, 같은 계약**(HTTP, 통화당 워커 1개, chunked PCM 16 kHz int16, 장애는 `QwenTTSUnavailableError` 로 ElevenLabs 폴백)이고, 다른 점은 셋: 워커가 통화 세션(KV 캐시 = 참조 음성 + 지금까지의 턴)을 들고 있고, 턴 시작에 사용자 발화(PCM + 전사)를 문맥에 넣고, AI 가 말한 오디오를 이득 뒤 재인코딩해 문맥에 넣는다(레벨 드리프트 방지).

**원본은 스터디 레포 `labs/worker/`(학교 서버에서 실제로 돈 파일).** 이 폴더의 복사본은 `csm_worker.py` 의 `sys.path` 한 줄과 `test_worker.py` 의 `level` 임포트 경로 두 곳만 B 배치(`../csm_experiments`)에 맞게 바꿨다. `scripts/csm_experiments/` 처럼 ruff `extend-exclude` 에 넣었다 — 아래 숫자를 낸 코드를 lint 로 고치지 않는다. **AIHub 파생물(전사·오디오·토큰)은 없다**; 참조 음성 파일은 `voices.example.json` 꼴로 서버에만 둔다(운영은 사용자 본인 목소리 15~20 s + 전사).

| 파일 | 하는 일 |
|---|---|
| `csm_worker.py` | 전송과 분리된 세 클래스. `Session`(정적 루프 `StaticStack` 에 위치 명시 프리필로 턴을 이어 붙임, 예산 2048−250−32 넘으면 참조 + 최근 턴만 남기고 `rebase`) · `Codec`(Mimi 인코드 — `level.py` 로 −26 dBFS 정규화 뒤 — 와 세그먼트 프리픽스 재디코드) · `Generator`(프레임 루프 백본 1 + depth 32, EOS·상한 250프레임·취소, 2프레임 청크, 24 k→16 k Kaiser 385탭, 세그먼트 끝에 이득 준 파형을 재인코딩해 캐시에 확정, 적응 이득) |
| `server.py` | FastAPI. `GET /health` · `POST /v1/session/{open,user,context,context_codes,speak,cancel,close}`. `user` 는 JSON `{session_id, text, pcm_b64}`(헤더는 ASCII 라 한글 불가) · `speak` 는 chunked raw PCM 16 kHz(`X-Sample-Rate: 16000`), 클라이언트가 끊으면 프레임 사이에서 멈추고 낸 만큼만 확정. 프로세스 = 세션 1개(lock). 환경변수 `CSM_WEIGHTS`(기본 `~/CSM/runs/b1/epoch_1`) · `CSM_REPO` · `VOICES_FILE` · `CSM_DEVICE` · `CSM_COMPILE` · `CSM_TARGET_DB`(−26) · `CSM_GAIN_DB`(3) |
| `launch_csm_workers.sh` | GPU 별 기동/정지(Qwen `launch_workers.sh` 와 같은 꼴). 기본 GPU "0,3", 포트 8020부터. `CSM_DIR`(server.py 가 있는 폴더)·`CSM_PYTHON`·`VOICES_FILE`(필수)·`CC/CXX`(torch.compile, `~/gcc-env`). 끝에 앱 설정용 `CSM_TTS_URLS` 를 찍어 준다 |
| `test_worker.py` · `test_server.py` | W1a~W1d 단위 시험(맥 CPU, 실물 모델·Mimi). B 의 `.venv` 에는 torch 가 없어 여기서는 돌리지 않는다 — 스터디 레포에서 돈 결과가 아래 |
| `replay_g1b.py` | W2 — G1b 100문장을 `/context_codes` 로 문맥을 넣고 API 로 재생(`--drift K` 는 자기 턴 되먹임 K턴). 세트는 AIHub 파생이라 레포 밖 |
| `bench.py` | W3 — 사용자 턴 프리필·첫 청크·RTF 지연 |
| `voices.example.json` | `VOICES_FILE` 의 꼴(참조 음성 경로 + 전사) |

## 앱 쪽 (W4, 이 브랜치)

- `app/core/config.py`: `csm_tts_urls`(쉼표 목록) · `csm_tts_realtime_enabled` · `csm_tts_voice`(기본 `ai`) · `csm_tts_health_timeout`.
- `app/services/csm_tts.py`: `try_acquire_realtime_csm`(통화당 워커 1개, `/health` 의 `ready`) · `CsmRealtimeTTSClient.open()`(첫 턴에만 `/v1/session/open`) · `CsmRealtimeTTSSession.begin(emotion, user_turn=(pcm16k, text))`(→ `/v1/session/user`; 실패해도 턴은 계속) · `stream()`(문장 버퍼 → `/v1/session/speak`) · `aclose()` · `release_slot()`(→ `/v1/session/close` 비동기 + 풀 반납). 장애 = `QwenTTSUnavailableError`.
- `app/services/pipeline.py` 세 곳: `_init_qwen_tts` 가 CSM(켜져 있으면) → Qwen → ElevenLabs 순으로 고름 · `ensure_tts` → `_begin_tts` 가 세션에 `accepts_user_turn` 이 있으면 이번 사용자 발화(`_tremor_buf` 의 `_user_turn_intervals[-1]` 구간, 16 kHz int16)와 전사를 넘김 · `tts_engine` 라벨은 클라이언트의 `engine_name`(`csm`/`qwen`/`eleven`).
- 시험: `tests/unit/test_csm_tts.py`(9) · `tests/unit/test_pipeline_csm_engine.py`(5). 전체 1,138 통과, ruff 통과.
- 켜기: `.env` 에 `CSM_TTS_REALTIME_ENABLED=true` · `CSM_TTS_URLS=http://127.0.0.1:8020`(Tailscale 이면 그 주소). 꺼져 있으면 기존 Qwen/ElevenLabs 경로 그대로.

## 실측 (스터디 레포 `labs/worker/`, 학교 서버 3090 GPU 0, b1 epoch_1, 2026-09-28)

- **W1** 단위(맥 CPU): 이어 프리필 = 통째 프리필(정적 루프) · 인코드/정규화 왕복 −40.3 → −26.0 dBFS · **창 디코드 기각**(창 6~96 에서 전체 디코드와 0.3~1 % 차, Mimi 디코더 sliding_window 250) → 세그먼트 프리픽스 재디코드(인과 4e-8, 꼬리 1e-7) · EOS/상한/취소 · HTTP 층 open/user/speak/끊기/close.
- **W2** 재생 100턴(문맥 120 s, 같은 100 id, 맥 채점 `g1_score.py --norm -26`, 심판 Gemini transcribe): **CER 7.22 %** [5.1, 9.6](합산 6.82) · 붕괴(CER>50) 0 · SIM 0.936(사람 0.940) · UTMOS 2.20 · 길이 비 1.00 · 끝남 실패 0 · 출력 레벨 −25.6 dBFS [−29.3, −22.9](목표 −26). 짝 비교: worker − b1_ctx120(E-B, 같은 모델을 오프라인으로) **+0.87 [−1.52, +3.28]**(나쁨 26/좋음 24/같음 50) · worker − mimi(코덱 천장 5.94) **+1.28 [−1.46, +3.76]**(판정선 +2 %p 안) · worker − human +3.48 [+1.26, +5.89]. 채점기의 사람 원본 레벨 −35 dBFS 기준 부분집합: 보통(30) worker 6.18 vs b1 5.33 · 조용(70) 7.67 vs 6.79 — 두 쪽 모두 짝 +0.9 로 같다(조용 채널 붕괴 없음).
- **지연**(W2 100턴, 문맥 프리필 별도): 첫 청크 p50 **79 ms** · p90 233 · p95 245 · 최대 251, RTF p50 0.360 · p95 0.418. 18개가 228~251 ms 에 몰려 있고 그 18개는 문맥 프레임(평균 1,184 vs 1,124)·글자(790 vs 721)·턴 수(15.2 vs 12.9)·목표 글(33.6 vs 28.2자)이 모두 많다 → **추정**: KV 예산(1,766위치)을 넘겨 `Session.rebase`(재프리필 ≈ +165 ms). 확인은 `speak` 에 rebase 횟수를 기록해서(다음 할 일).
- **W3** 벤치: 8 s 사용자 턴 프리필 21 ms · 첫 청크 75 ms · RTF 0.37.
- **드리프트**(자기 턴 되먹임 5턴 × 10세션, 워커 출력 레벨 p50): −26.4 / −26.1 / −25.2 / −26.7 / −25.1(최소 −30.6, 최대 −22.7). 같은 모델의 코드 되먹임(E-D)은 −28.4 → −34.6 이었다 → 이득 뒤 재인코딩이 드리프트를 막는다. 이득 전 원 출력 레벨은 재생 로그에 없다(확인 필요).

## 수정 이력

- **2026-09-28 끊김 확정 버그(PR 리뷰 지적)**: `speak()` 가 클라이언트 끊김(`GeneratorExit`) 때 `g.run(cancel)` 을 **새로** 만들어 비웠기 때문에 `frames` 가 비어 **0프레임**이 확정됐다(이미 낸 오디오가 문맥에 안 남음). 고침: `Generator.run` 의 확정(재인코딩 + `commit_audio`)을 `finally` 로 옮겨 EOS·상한·cancel·`GeneratorExit`·예외 어느 쪽으로 끝나든 낸 프레임을 확정하고(`finally` 안에서는 yield 없음, 꼬리 출력은 정상 종료 때만), `speak()` 는 멈춰 있던 **같은** 생성기를 `close()` 한다. 검증(맥 CPU 실물 모델): `test_server.py` 에 응답 `body_iterator` 를 첫 청크 뒤 닫는 시험 추가 → **확정 2프레임(0 아님, 상한 50 미만) · 위치 146 → 166(글 17 + 프레임 2 + eos 1) · cancelled=True · 락 해제**; TestClient 경유 끊기는 생성이 다 끝난 뒤에야 닫혀(CPU) 락 해제·확정만 본다. `test_worker.py` W1a~W1c 재통과(20프레임·청크 11·취소 플래그 2프레임). W2/W3 숫자는 정상 경로(끊김 없음)라 그대로. **서버 배포 번들도 다시 올려야 한다**(스터디 레포 `labs/worker/dist/csm_worker_bundle.tgz`, sha256 `b8460ea0…`).

- **2026-09-28 세션 변경 핸들러 락(PR 리뷰 3회차)**: `open`/`user`/`context`/`context_codes`/`close` 가 `_busy()`(락 상태만 확인)만 보고 세션(`s.pos`·`s.turns`·`S["session"]`)을 바꿨다. 동기 핸들러라 스레드풀에서 겹치면 경쟁 조건. 고침: `_locked` 데코레이터가 `speak` 와 같은 락을 0.5 s 안에 쥐고 핸들러를 돌리며 못 쥐면 409(`cancel` 은 생성 중 불리는 것이라 락을 안 쥔다). 검증(맥 CPU 실물): 락 점유 중 `user` → 409 · `append_turn` 호출 순간 락 보유 · 끝나면 해제, 나머지 시험 전부 재통과. 부작용: `/health` 의 `busy` 가 사용자 턴 프리필(≈ 20 ms) 동안도 true. 번들 sha256 `c9b4d15a398f…`.

- **2026-09-28 죽은 코드 정리(PR 리뷰 4회차)**: `Session.rebase()` 의 `i, s = self._ids_spans(...) if False else (None, None)` 한 줄(항상 거짓 분기, 디버깅 잔재) 삭제. `_ids_spans` 는 `append_turn` 에서 계속 쓴다. 동작 변화 없음(py_compile + `test_worker.py` 재실행). 번들 sha256 `01c1523d976f…`.

- **2026-09-28 rebase 기록(머지 뒤)**: `Session.rebases` 카운터를 두고 `speak` 로그에 `rebase %d`(그 speak 동안의 재프리필 횟수)를 남긴다. W2 의 첫 청크 p95 245 ms 군집(18개)이 재프리필 때문인지 서버 로그로 확정하기 위한 것. 이득 전 원 출력 레벨은 이미 같은 로그의 `level` 이다. 동작 변화 없음. 번들 sha256 `1988351fa7af…` — **서버에 올릴 최종 번들**.

- **2026-09-28 밤: 전화 대역 모드(`CSM_BAND=phone`, 기본) + 세그먼트 내 AGC — 첫 실제 통화의 붕괴 대응.** 실제 통화 5턴(맥 앱 → 워커, 사용자 광대역 마이크 + Kspon 페르소나)에서 3턴부터 대역이 전화 대역으로 좁아지고(4~7 kHz 비 −36 → −50 dB) 목소리 유사도가 다른 화자 수준으로 떨어졌으며(x-vector 0.939 → 0.866), 사용자 청취 "3턴부터 먹먹하고 품질이 크게 떨어짐". 같은 워커의 전화 대역 문맥 되먹임 세트(10 × 5턴)는 SIM 0.965 → 0.951·대역 −50 일정으로 안 무너짐 → **원인 = 광대역 다중 턴 문맥이 학습 분포 밖**(대화 문맥 학습 재료가 전부 8 kHz). 대응: `fir_bandpass`(Kaiser 513탭 @24 k, 300~3,400 Hz, 통과대역 0 dB·200 Hz −44·4 kHz −115) 로 페르소나·사용자 턴 입력(`Codec(band="phone")`)과 출력·재인코딩(`Generator(band="phone")`, 프리픽스 conv + holdback 256) 을 전부 전화 대역으로. **AGC**: 첫 청크는 앞 세그먼트 이득, 둘째 청크부터 지금까지 레벨로 청크당 ≤ 2 dB 이동, 피크 −2 dBFS 상한, 버퍼는 무클리핑(클리핑은 int16 출력 때만); 서버의 다음 세그먼트 시작 이득 = 이번 세그먼트 최종 이득. 실측(맥 CPU): 대역 필터 스트리밍 = 전체(3.6e-7), 전화 대역 인코드 −55 dB, phone+AGC 출력 대역비 −73 dB·원 −24.5 → 출력 −25.2(이득 0 → −1.5)·재인코딩 턴 −26.3, 취소 경로 유지. `/health` 에 `band`. 번들 sha256 `87157529551e…`. **판정은 같은 5턴 통화를 다시 돌려 사용자 귀로** — 그래도 Qwen 보다 못하면 CSM 보류.

- **2026-09-28 밤, AGC 말소리 게이트(PR 리뷰 지적)**: AGC 목표 이득을 `speech_rms_db` 만으로 정하면 묵음·잡음뿐인 시작부에서 상대 게이트가 잡음을 말소리로 읽어 +12 dB 까지 밀어 올릴 수 있었다 → `agc_step` 이 `level.analyze` 의 `weak`(말소리 < 0.3 s 또는 SNR < 10 dB)면 이득을 안 움직인다. 모델 없는 시험 `test_agc_gate`: 잡음만 → 불변, −40 dBFS 버즈 → 0 → +12(걸음 ≤ 2), −16 dBFS → −6, 앞 묵음 0.5 s 는 안 올리고 말소리 뒤 +2.

## 다음

W4 실제 통화 시험(`scripts/ws_listen.py ws`, 워커 켠 상태) · `speak` 에 rebase·원 레벨 기록 · 16 kHz 마이크 문맥(E-E) · 페르소나 = 사용자 본인 목소리 · WebSocket 전송(2차) · 끼어들기.
