# 소음 환경 STT 계측 (계획 0075, #280)

같은 TTS 발화 8문장(`.harness/stt-compare-2026-09/audio/u01~u08.wav`)에 합성 잡음을 SNR 단계별로 섞어
chirp 를 **운영 설정 그대로** 와 **denoiser 켜기** 로 나란히 잰다. 결과는 상대 비교용이다(아래 '한계').

| 파일 | 하는 일 | 돈 |
|---|---|---|
| `mix_noise.py` | 깨끗한 발화 + 잡음(백색·분홍·웅성거림) → 조건별 wav 와 `manifest.json`, 잡음만 클립, 예상 비용 | 없음 |
| `run_chirp.py` | 클립을 실시간 속도로 chirp 에 흘려 모든 FINAL·SPEECH 이벤트를 JSON 으로 | **chirp** |
| `score.py` | (잡음, SNR, denoiser)별 채점표. ws_bench 결과도 받는다 | 없음 |
| `tests/perf/ws_bench.py` | 로컬 서버 전체 경로(`--noise-manifest`) | **chirp·LLM·TTS** |

결과는 전부 `.harness/stt-noise-2026-10/`(gitignore). 워크트리의 `.harness` 는 메인 체크아웃으로 가는 심링크라 같은 곳이다.

## 1. 오디오 만들기 (네트워크 없음)

```
.venv/bin/python scripts/stt_noise/mix_noise.py
```

- 기본: SNR 20·10·5·0 dB × 잡음 3종 × 잡음 변형 3개 + clean, 앞 잡음 1초 · 발화 · 뒤 잡음 5초, 잡음만 클립 10초 × 12 × 변형 3,
  반복 3회 기준 예상 비용.
- **SNR 정의**: 깨끗한 발화의 말소리 프레임(20ms, −45dBFS 이상, `tts_bench/prep_ref.py` 와 같은 판정) 평균 전력 ÷ 같은 프레임 위
  잡음 평균 전력. 발화는 먼저 말소리 레벨 −26dBFS 로 맞추고(clean 도), 피크가 −1dBFS 를 넘으면 클립 전체를 줄인다(SNR 불변,
  `clip_gain_db` 에 기록). 시험이 int16 로 쓴 뒤 다시 재서 ±0.5dB 를 확인한다.
- 분홍 잡음은 20Hz 아래를 뺐다(안 빼면 들리지 않는 저역이 전력의 절반 가까이를 먹어 SNR 이 실제보다 나빠 보인다).
- 웅성거림 = 대상 발화를 뺀 나머지 7문장을 `--babble-talkers` 겹(기본 6) 순서·시작점을 달리해 겹친 것(잡음만 클립은 8문장 전부). 같은 TTS 목소리라 실제 군중보다 균질하다.
- 잡음 실현은 (잡음, 발화, 변형)마다 seed 로 고정 — 같은 변형의 SNR 단계끼리는 같은 잡음을 크기만 바꾼 것이라 단계 비교가 깨끗하다.
  변형(`--variants`, 기본 3)끼리는 다른 잡음이고, `run_chirp` 의 반복 r 은 변형 (r−1) mod K + 1 을 쓴다 — 같은 오디오를 세 번
  보내면 chirp 결과가 사실상 같아(9/09 원자료) 반복이 표본을 늘리지 못하기 때문이다. clean 은 하나뿐이라 반복해도 같은 오디오다.
  경로: `<조건>/v<변형>/<발화>.wav`, `noise_only/v<변형>/<조건>.wav`, `clean/<발화>.wav`.
- `manifest.json` 클립 항목: `file`·`utt`·`text`·`noise`·`snr_db`·`variant`(clean 은 null)·`noise_only`·`lead_s`·`speech_start_s`·
  `speech_end_s`(마지막 말소리 프레임 끝)·`utt_end_s`(원본 파일 끝)·`dur_s`·`speech_level_db`·`noise_rms_db`·`clip_gain_db`.
  최상위 `variants`.

## 2. 마른 실행 → 실측 (chirp 비용, 사용자 승인 뒤)

```
.venv/bin/python scripts/stt_noise/run_chirp.py --dry-run                         # 작업 수·예상 비용만, STT 호출 없음
.venv/bin/python scripts/stt_noise/run_chirp.py                                   # 전부(운영·denoiser × 3회), 약 100분
.venv/bin/python scripts/stt_noise/run_chirp.py --noise clean pink --snr 10 --reps 1 --concurrency 4   # 일부만
```

- 기본 전부 = 작업 696개(발화 624 + 잡음만 72), 오디오 상한 약 103분 ≈ **$1.65**($0.016/분, `app/core/pricing.py`).
  발화 클립은 턴 FINAL 에서 멈추므로 실제는 그보다 적다. `--max-usd`(기본 5)를 넘으면 시작하지 않는다.
- 실시간 속도라 걸리는 시간 ≈ 보내는 오디오 길이 ÷ `--concurrency`. 결과 JSON 은 클립마다 다시 써서 중간에 끊겨도 남는다.
- 자격 증명은 운영과 같은 ADC(`gcloud auth application-default login`), 설정은 레포 `.env` 를 `get_settings()` 로 읽는다.
  `STT_ENGINE` 이 chirp 가 아니면 거절한다.

결과 JSON(`runs/chirp-<시각>.json`): `git`(commit·dirty) · `stt`(엔진·리전·모델·언어 + 조건별 실제 config 요청 전체) · `chunk_ms` ·
`reps` · `warmup` · `clips[]`(조건 필드 + `denoise`·`denoise_snr_threshold`·`chunk_ms`·`rep`·`stopped`·`audio_sent_s`·`streams`·
`interims`·`events[]`) ·
`audio_sent_s`(마이크 기준) · `stt_audio_s`(클라이언트가 실제로 gRPC 에 실은 초) · `est_usd`.
`events[]` = FINAL(빈 것 포함, `text`·`end_offset_s`)·SPEECH_BEGIN·SPEECH_END·error, 각각 `t`(클립 시작부터 초)·`stream`(몇 번째
스트림)·`stream_open_s`(그 스트림이 열린 오디오 위치)·`after_eos`. INTERIM 은 개수와 첫 시각만.

### 러너가 운영을 흉내 내는 방식 (`app/services/pipeline.py`)

- 첫 오디오 청크가 올 때 스트림을 연다(`_consume_one_stream`). **FINAL 이 오면 빈 것이든 아니든 닫고**(`final_closes`)
  큐를 새로 바꾼 뒤(`_half_close`) 다음 청크에서 다시 연다. 닫히는 사이 옛 큐에 남은 청크는 운영처럼 버려진다.
- 턴 = 첫 비어 있지 않은 FINAL(`_handle_stt_event`). 앞 잡음에서 나온 빈 FINAL 에서 측정이 끝나지 않는다.
- 발화 클립은 **덮은 오디오가 말소리 끝 − 0.3초를 넘는 비어 있지 않은 FINAL** 에서 멈춘다. 위치는 도착 시각이 아니라
  `stream_open_s + end_offset_s`(FINAL 의 `result_end_offset` 은 그 스트림 첫 오디오 기준)로 본다(`score.final_end_s`) — 말소리 끝 뒤에
  도착한 앞 조각 FINAL 에서 멈추지 않으려고. 그 전의 비어 있지 않은 FINAL 은 기록하고 계속한다. `end_offset_s` 가 0 이면(값 없음)
  도착 시각으로 대신한다. `--full-tail` 이면 턴 뒤에도 끝까지 보낸다. 잡음만 클립은 늘 끝까지.
- 청크는 마이크처럼 그 길이만큼 시간이 지난 뒤 보낸다(기본 100ms, 앱 실측 웹 256·iOS 64 → `--chunk-ms`). 꼬리 잡음을 보내는
  동안에도 받는다.
- 오디오가 다 나가면 EOS 를 넣고 `--drain-s`(3초) 동안 기다린다. 그 뒤에 온 FINAL 은 끝점이 아니라 반쯤 닫기로 나온 것이라
  `after_eos` 로 표시하고 채점에서 뺀다(운영은 마이크가 계속 흐르니 이런 FINAL 이 없다).
- 조건(운영·denoiser)마다 먼저 clean 클립 하나를 흘려 버린다(`--warmup`) — 9/09 측정에서 chirp 실패 2건이 둘 다 세션 첫 턴의 콜드
  스트림이었다(`.harness/stt-compare-2026-09/report.md`). **워밍업에서 스트림 에러가 하나라도 나면 본 측정을 시작하지 않고 멈춘다**
  (권한·쿼터, chirp_3 가 denoiser_config 를 거절하는 경우 등 — 그대로 돌리면 전 작업이 에러로 끝난다). 에러 문구는 결과 JSON `warmup` 에.
- 스트림 에러(STTError)는 기록하고 다음 청크에서 바로 다시 연다(클립당 3번이면 그 클립 포기, `stopped=error`). 운영의 재오픈 백오프는
  흉내 내지 않는다. 에러가 난 클립은 채점에서 빠지고 `err` 로 따로 보인다.
- 작업 순서는 반복 → 클립 → (운영, denoiser) 로 섞어 시간대 차이가 한 조건에 몰리지 않게 한다.

## 3. 채점

```
.venv/bin/python scripts/stt_noise/score.py .harness/stt-noise-2026-10/runs/*.json [--details] [--json-out 요약.json]
```

- 조건 = (잡음, SNR, denoiser, 청크 ms, denoiser snr_threshold). 청크·임계값이 다른 실행을 같이 넣어도 섞이지 않는다.
- 스트림 에러가 난 클립(`stopped=error` 이거나 error 이벤트)은 n 에서 빼고 `err` 열로 센다. `drain_timeout`(클립 끝까지 턴 FINAL
  없이 EOS 뒤에도 스트림이 안 닫힘)은 턴 없음으로 센다.
- 비어 있지 않은 FINAL 은 덮은 오디오 끝 위치(위 `final_end_s`)로 나눈다: **noise**(말소리 시작 전에 끝났거나, 말소리가 끝난 뒤 열린
  스트림 = 앞뒤 잡음) · **fragment**(말소리 끝 − 0.3초 전에 끝남 = 발화 조각) · **complete**(그 뒤).
- 발화 클립: n · err · aud(서로 다른 오디오 수) · FINAL%(턴이 나온 비율, Wilson 95%) · p50/p95(말소리 끝 → 턴 FINAL 도착 ms, complete
  턴만) · CER(턴이 있는 클립) · CER*(턴 없음 = 1) · mis(정규화 후 정답과 다른 턴) · noiseT(턴이 잡음 FINAL) · early(턴이 발화 조각 —
  운영에선 이 조각이 턴이 되고 나머지는 버려진다) · split(발화가 비어 있지 않은 FINAL 2개 이상으로 쪼개져 둘 다 보인 클립) ·
  클립당 빈 FINAL·SPEECH_BEGIN. `--json-out` 에는 `turn_end_gap_p50`(complete 턴의 끝 위치 − 말소리 끝, ms)도 있다.
- **95% CI 의 표본 수는 클립 수가 아니라 서로 다른 오디오 파일 수(aud)** 다. 같은 파일을 반복하면 결과가 사실상 같아 독립 시행이
  아니기 때문이다(비율은 전체 클립으로 내고 구간만 aud 로). 잡음 조건은 반복마다 변형이 달라 8발화 × 3변형 = 24, clean 은 8.
- 잡음만 클립: 거짓 FINAL(비어 있지 않은 FINAL) 수·분당·나온 클립 수, 빈 FINAL, SPEECH_BEGIN 분당(에러 클립 제외, `err`).
- 정규화·CER·Wilson·백분위는 `scripts/stt_compare.py` 그대로. "세 시"→"3시" 같은 표기 차이도 CER 에 들어간다(u03·u05, 9/09 리포트 §4)
  — 조건 사이 비교엔 영향이 같지만 절대치는 `--details` 로 손으로 본다.
- 지연 기준점이 ws_bench(원본 파일 끝)와 다르다: 여기는 마지막 말소리 프레임 끝이라 TTS 파일 끝 무음(약 0.2초)만큼 크게 나온다.

## 4. 로컬 서버 전체 경로 (chirp·LLM·TTS 비용, 사용자 승인 뒤)

`tests/perf/ws_bench.py --noise-manifest <manifest> --noise <clean|white|pink|babble> [--snr <dB>]` 를 주면 `--audio` 의 각 발화를
그 조건의 클립(앞 잡음 + 발화)으로 바꾸고, 발화 뒤에는 **응답(`speaking_end`)이 올 때까지** 그 잡음을 실시간으로 계속 보낸다
(클립 꼬리 5초 → 같은 조건의 잡음만 클립을 변형 순서대로 이어 돌림) — 운영 마이크처럼. 그래서 잡음 모드에선
`--tail-silence-ms` 를 쓰지 않고 JSON 의 `tail_silence_ms` 는 null 이다. 턴 간격(`--turn-gap-ms`)도 그 잡음으로 채운다.
발화를 한 바퀴 돌 때마다 다음 잡음 변형을 쓴다(턴 결과의 `variant`). 클립 끝까지는 오프라인 벤치와 바이트 단위로 같고, 그 뒤는
오프라인 러너가 EOS 를 넣는 대신 잡음을 계속 보낸다. `--noise-manifest` 를 안 주면 예전과 똑같다(디지털 0 꼬리 `--tail-silence-ms`).
JSON 에 `noise`·`snr_db`·`lead_ms`·`chunk_ms`·`turn_gap_ms` 가 더해진다(없으면 null/0).

- 격리 자원으로만: 임시 DB · 별도 Redis 인덱스(예 15) · 별도 포트(예 8020), `scripts/seed_local_session.py` 로 조건마다 새 세션.
  GPU 워커와 겹치지 않게 `CSM_TTS_REALTIME_ENABLED=false`·`QWEN_TTS_REALTIME_ENABLED=false` 둘 다(엔진은 CSM → Qwen → EL 순으로
  고른다, `pipeline._init_qwen_tts`). 통화 종료 때 Spring 콜백이 로컬 8080 으로 가지 않게 닫힌 포트로 돌린다. 턴마다 LLM·TTS 비용이 붙는다.
- 앞 잡음에서 비어 있지 않은 FINAL 이 나오면 서버는 그걸 턴으로 받는다 — ws_bench 는 발화를 다 보낸 뒤에야 받기 시작하므로
  그 전사가 `stt_final_ms ≈ 0` 과 엉뚱한 `transcript` 로 찍힌다. 운영에서도 실제로 벌어지는 실패라 그대로 둔다.
- 채점: `score.py --manifest .harness/stt-compare-2026-09/audio/manifest.json <ws_bench json>...` → (engine, 잡음, SNR, 꼬리, 청크,
  앞 잡음)별로 `stt_compare.summarize_condition` 표. `noise` 가 없는 결과(9/09 JSON·잡음 모드 아닌 실행)는 `raw` 로 따로 묶인다
  (레벨을 맞추고 앞 1초를 붙인 `clean` 과 다른 오디오다). 꼬리 `reply` = 잡음 모드.

## 한계

- 합성 잡음엔 기기의 잡음 억제·에코 제거·자동 이득과 롬바드 효과(시끄러우면 크게 말하기)가 없다 → **비관적**, 상대 비교로만.
- 발화가 TTS 낭독이라 더듬거림·발음 오류가 없다. 웅성거림도 같은 목소리.
- chirp_3 가 denoiser 를 실제로 쓰는지는 문서상 미확인("May not be supported for all models and may have no effect").
  운영과 같은 결과면 효과 없음일 수 있다. 거절(에러)하면 워밍업에서 멈춘다.
- FINAL 분류·멈춤은 chirp_3 의 `result_end_offset` 이 '그 결과의 말소리 끝'이라는 전제다. 끝점 판정 위치(도착 시각 쪽)라면 조각이
  complete 로 보여 early·split 이 적게 나온다. 첫 실측에서 clean 조건 `turn_end_gap_p50` 이 0 근처(수백 ms 안)인지 먼저 본다.
- 잡음은 STT 밖에서도 떨림·턴 크기·말 속도 분석(`_tremor_buf`)과 silence_total(SPEECH_BEGIN 이 `_bank_silence`)에 들어간다 —
  이번엔 STT 지표만 잰다.
- 러너 비용은 `usage_event` 에 남지 않는다(청구서와 `usage_report` 가 몇 달러 어긋난다). 스트림마다의 반올림·최소 과금은
  예상 비용에 반영 안 됨 — 잡음이 빈 FINAL 을 많이 내면 스트림 수가 늘어난다(`streams` 로 확인).
