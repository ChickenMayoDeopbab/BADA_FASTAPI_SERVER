# 소음 환경 STT 계측 (계획 0075, #280)

같은 TTS 발화 8문장(`.harness/stt-compare-2026-09/audio/u01~u08.wav`)에 합성 잡음을 SNR 단계별로 섞어
chirp 를 **운영 설정 그대로** 와 **denoiser 켜기** 로 나란히 잰다. 결과는 상대 비교용이다(아래 '한계').
실제 사람 목소리로 같은 비교를 하는 키트(§5)와 운영 로그로 켜기 전후를 보는 집계기(§6)는 계획 0077.

| 파일 | 하는 일 | 돈 |
|---|---|---|
| `mix_noise.py` | 깨끗한 발화 + 잡음(백색·분홍·웅성거림) → 조건별 wav 와 `manifest.json`, 잡음만 클립, 예상 비용 | 없음 |
| `run_chirp.py` | 클립을 실시간 속도로 chirp 에 흘려 모든 FINAL·SPEECH 이벤트를 JSON 으로 | **chirp** |
| `score.py` | (잡음, SNR, denoiser)별 채점표. ws_bench 결과도 받는다 | 없음 |
| `tests/perf/ws_bench.py` | 로컬 서버 전체 경로(`--noise-manifest`) | **chirp·LLM·TTS** |
| `real_sentences.tsv` | 실발화 녹음 문장표(u01~u14) | — |
| `real_manifest.py` | 폰 녹음 폴더 → 16k mono wav 와 run_chirp·score 가 그대로 읽는 `manifest.json`, 클립별 말소리 구간·추정 SNR·경고, quiet 테이크와 구간 길이 대조, 손으로 고친 구간(`--spans`) | 없음 |
| `prod_compare.py` | 운영 텍스트 로그 → 세션을 `stt_config denoise=` 로 묶어 final_dropped·ERROR 종료·지연 | 없음 |

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
- 운영 스위치 `GOOGLE_STT_DENOISE`(기본 꺼짐)를 `.env` 에서 켜 둬도 `off`(결과 JSON `operational`)는 denoiser 를 끄고, `on` 은
  `--denoise-snr-threshold` 로 켠다(`GOOGLE_STT_DENOISE_SNR_THRESHOLD` 는 안 읽는다).

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

## 5. 실발화 확인 (계획 0077 F130) — 폰 녹음으로 끔·켬 비교

합성 잡음 + TTS 한 목소리 결과(§1~3)를 실제 사람 목소리·실제 잡음으로 확인한다. 순서: 녹음 → `real_manifest.py`(네트워크 없음)
→ `run_chirp.py --manifest … --denoise both`(chirp 비용) → `score.py`.

### 5-1. 문장표 `real_sentences.tsv` (`id<TAB>정답`)

- u01~u08 은 9/09 TTS 세트(`.harness/stt-compare-2026-09/audio/manifest.json`)와 같은 문장, u09~u14 는 숫자·고유명사가 많은 문장
  (시각·인원·주문번호·이름·전화·날짜·장소·호수). 이름·번호는 전부 지어낸 값이다.
- **정답 숫자는 아라비아 숫자로 쓴다.** chirp 는 숫자를 숫자로 적는다 — 10/20 `runs/chirp-main.json` clean 6/6 이 "세 시"→"3시",
  "열 시 반"→"10시 반", "1998년 3월 15일", "010-2345-6789". 채점 정규화(`stt_compare.normalize`)는 공백·구두점·하이픈만 지우고
  한글 수사↔숫자는 안 바꾸므로, 정답을 한글 수사로 두면 끔·켬 모두 매번 mismatch·CER 로 잡혀 진짜 오인식이 묻힌다.
  그래서 u03·u05 는 TTS manifest("세 시"·"열 시 반")와 **표기만** 다르다(읽는 소리는 같다).
- u09~u14 를 chirp 가 어떤 표기로 쓰는지는 아직 안 봤다("4명"·"73052819"·"1203호" 등). 첫 실측 뒤 `score.py --details` 에서
  끔·켬 둘 다 같은 자리가 같게 다르면 오인식이 아니라 표기 차이다 → 문장표를 고치고 manifest 를 다시 만든다(채점은 결과 JSON 의
  `text` 를 쓰므로 이미 돌린 결과는 손으로 본다).
- 읽을 때: 숫자는 평소 말하듯(3시=세 시, 4명=네 명, 2번 출구=이 번 출구), 전화·주문번호는 한 자리씩.

### 5-2. 녹음 안내

- 폰 기본 녹음 앱(아이폰 '음성 메모', 갤럭시 '음성 녹음'). 앱의 잡음 제거·'녹음 향상' 같은 기능은 끈다(원본 그대로).
- **통화하듯** 폰을 귀에 대고 입과의 거리를 일정하게(스피커폰처럼 멀리 들지 않는다). 문장마다 **파일 하나**:
  녹음 시작 → **1초** 쉼 → 문장 → **2초** 쉼 → 정지. 뒤 여유가 짧으면 chirp 가 끝점을 내기 전에 오디오가 끝나 '턴 없음'으로 잡힌다
  (EOS 뒤 FINAL 은 채점에서 뺀다, §2). `real_manifest` 가 앞 0.5초·뒤 1.5초 미만이면 경고한다.
- 환경 **3종 이상**, 환경 안에서는 위치·음량을 그대로:
  - `quiet` — 조용한 방. **문장마다 꼭 녹음한다** — 다른 환경의 말소리 구간을 이 길이와 견준다(§5-3).
  - `cafe`(또는 `rain`) — 유튜브 '카페 소음'·빗소리처럼 **알아들을 대사가 없는** 앰비언스 영상을 1~2m 떨어진 스피커로,
    대화가 조금 방해될 정도로
  - `fan`(또는 `street`) — 선풍기·환풍기 가까이, 또는 거리
  - TV 드라마·뉴스처럼 **한 사람 대사가 또렷한 소리는 피한다.** 말소리 구간을 소리 크기로 추정하는데 대사도 말소리라,
    문장 앞뒤 0.8초 안의 대사가 구간에 붙어 끝이 최대 2초 늦게 잡힌다(합성 시험: 대사 SNR 10~20dB 에서 절반 이상이 0.3초 넘게
    틀림). quiet 길이 대조가 대부분 걸러 내지만 빠진 만큼 표본이 준다. 꼭 TV 로 하려면 대사가 적은 장면·음악 방송으로.
- (선택) 환경마다 말 없이 15~20초 `noise.m4a` — 잡음만 클립(거짓 FINAL). 문장 녹음과 같은 자리·음량. quiet 엔 필요 없다.
- **파일 이름**: 폴더 = 환경 라벨(영문 소문자 권장, score 표에 그대로 나온다), 파일 = `<문장 id>.<확장자>`, 같은 문장을 다시 녹음하면
  `<id>_2.<확장자>`(테이크). 음성 메모는 녹음 직후 제목을 `u01` 처럼 바꾸면 내보낼 때 그 이름이 된다.

```
bada-rec/
  quiet/u01.m4a … u14.m4a
  cafe/u01.m4a … u14.m4a  noise.m4a
  fan/u01.m4a … u14.m4a  noise.m4a
```

- **맥으로 옮기기**: 아이폰은 음성 메모 → 공유 → AirDrop(맥 '다운로드'로 온다), 또는 맥 '음성 메모' 앱(같은 iCloud 면 동기화)에서
  녹음을 Finder 로 끌어다 놓는다. 갤럭시는 공유 → Quick Share·구글 드라이브, 또는 USB. 녹음 폴더는 레포 밖(예 `~/Desktop/bada-rec`)에
  두고 변환 결과는 `.harness`(gitignore) 아래로 — 본인 목소리라 커밋하지 않는다.

### 5-3. manifest → 실측 → 채점

```
.venv/bin/python scripts/stt_noise/real_manifest.py --rec-dir ~/Desktop/bada-rec          # 네트워크 없음, 표 확인
.venv/bin/python scripts/stt_noise/real_manifest.py --rec-dir ~/Desktop/bada-rec --spans ~/Desktop/bada-spans.tsv  # (선택) 고친 구간
M=.harness/stt-noise-2026-10/real/audio/manifest.json
.venv/bin/python scripts/stt_noise/run_chirp.py --manifest $M --denoise both --reps 1 --dry-run
.venv/bin/python scripts/stt_noise/run_chirp.py --manifest $M --denoise both --reps 1 --concurrency 4 \
    --out .harness/stt-noise-2026-10/real/runs/chirp-$(date +%m%d-%H%M).json                 # chirp 비용
.venv/bin/python scripts/stt_noise/score.py .harness/stt-noise-2026-10/real/runs/*.json --details
```

- **변환**: wav 는 soundfile, 그 밖(m4a 등)은 맥 내장 `afconvert`(없으면 `ffmpeg`, 둘 다 없으면 안내하고 멈춘다)로 PCM 만 꺼낸 뒤
  스테레오는 채널 평균, 다른 샘플레이트는 `resample_poly` 로 16kHz → 16k mono PCM16 `<out>/<환경>/<파일 이름>.wav`. 레벨은 안 맞춘다.
- **말소리 구간**: 20ms 프레임 전력을 100ms 로 고른 뒤 잡음 바닥(하위 10%) 위로 4~12dB(바닥~상위 2% 폭의 절반) 문턱을 넘는
  곳을 말소리 핵으로 잡는다. 0.15초 미만 튐은 버리고 0.8초 안 틈은 잇는다. **가장 긴 덩어리가 문장**이다. 0.8초 넘게 떨어진
  덩어리는 0.4초 이상·에너지가 문장의 20% 이상이고 문장보다 6dB 넘게 크지 않을 때만 붙인다(`말소리 덩어리 N개`). 그보다 크면
  `구간 모호(뺌)`, 짧거나 작으면 `말소리 밖 소리`로만 센다. 그다음 0.8초 안의 짧은 run(원래 프레임 60ms 이상, 약한 음절)을
  흡수하고, 가장자리는 **고른 레벨이 잡음 중앙값 + max(3dB, 1.3 × (중앙값 − 하위 10%) + 2dB) 이상인 동안** 바깥으로 넓힌다
  (0.3초 이하 틈은 잇고 탁 소리는 밟지 않는다). 끝으로 고르기 전 프레임으로 가장자리를 다시 자른다. 잡음 중앙값은 핵과 튐 밖의
  프레임에서 잰다.
- **정확도(로컬 확인, CI 밖)**: §1 의 TTS 클립(8문장 × 변형 3)을 실녹음처럼 앞 1초 + 발화 + 뒤 2초로 잘라 mix_noise 정답과 견줬다.
  끝 오차 0.3초 초과는 백색·분홍 10·20dB 0/24, 웅성거림 20dB 0/24·10dB 2/24(최대 −0.32초), 5dB 6~9/24, 0dB 대부분이다.
  10dB 이상 끝 오차 90%는 백색·분홍 ±0.05초, 웅성거림 10dB ±0.17초 안이다. 문턱 하나로 자르던 이전 방식은 백색·분홍 10dB 에서
  15/24 가 0.3초 넘게 일렀다(최악 −1.66초). `runs/chirp-main.json` 을 추정 구간으로 다시 채점해도 차이 방향이 정답 구간과 같다.
  백색 10dB p50 운영/denoiser 는 정답 1536/1620ms, 추정 1562/1660ms 다. 분홍 10dB 는 1552/1642 대 1592/1685, 백색 5dB early
  운영 3 → 3. 단 웅성거림 5dB 는 1973/1905 대 2245/2318 로 뒤집힌다 — 추정 SNR 9dB 미만 경고가 뜨는 조건이다.
  잡음에 묻힌 끝 음절(잡음보다 약한 만큼)은 구간에서 빠져 끝이 그만큼 이르게 잡히고, 지연은 그만큼 크게 나온다.
- **추정 SNR** = (말소리 프레임 평균 전력 − 잡음 평균 전력) ÷ 잡음 평균 전력. 말소리 프레임 = 구간 안에서 고른 레벨이 잡음 평균
  +3dB 이상인 프레임, 잡음 = 말소리 구간 앞뒤 0.1초 밖(다른 소리·튐 제외). 정의는 §1 과 같은 전력 비지만, 잡음에 묻힌 약한
  음절이 빠지는 만큼 **높게 나온다**. 같은 TTS 클립에서 편향은 실제 20dB +0.0, 10dB +0.4~0.5, 5dB +1.3~1.6, 0dB +3.3~3.9dB 였다
  (실제 5dB → 추정 5.5~8.2, 10dB → 9.2~11.8). 합성 조건과는 10dB 이상에서만 대략 견줄 수 있다. 그래서 **추정 9dB 미만은
  `구간 추정 불확실` 경고**를 낸다. 이 데이터에서 추정 9 이상 144클립 중 끝 오차 0.3초 초과는 2개였다.
- **quiet 길이 대조**: `--clean-label`(기본 `quiet`) 테이크가 있는 문장은 다른 환경 클립의 말소리 길이(끝 − 시작)를 그 문장 quiet
  테이크들의 중앙값과 견준다. 0.5초와 15% 중 큰 값보다 다르면 `구간 길이가 quiet 보다 +1.2초 — 잡음에 끌림?(뺌)`으로 뺀다.
  `--keep-mismatch` 면 경고만 하고 넣는다. 차이는 `span_vs_quiet_s` 에 남는다. TV 한 사람 대사를 잡음으로 넣은 합성 시험에서
  0.3초 넘게 틀린 80클립이 전부 걸렸고, 정상 잡음 288클립 중 잘못 뺀 것은 없었다. 같은 사람의 테이크 간 길이 차(느리게 읽기 등)는
  0.5초 안이면 통과한다. quiet 테이크가 없는 문장은 대조하지 않는다(표 끝에 몇 문장을 대조했는지 찍힌다).
- **구간 손으로 고치기 `--spans`**: `파일<TAB>시작초<TAB>끝초` TSV(`#` 줄·빈 줄 무시). 파일은 표의 파일 열(`cafe/u03.wav`)이나
  원본 경로(`cafe/u03.m4a`)다. 시각은 원본 녹음 그대로라(앞을 자르지 않는다) 아무 플레이어로 들으며 적으면 된다. 적힌 클립은 추정
  대신 그 구간을 쓰고(`manual_span`, 경고 `구간 수동(spans)`), 잡음·추정 SNR 을 그 구간으로 다시 잰다. quiet 길이 대조는 건너뛴다.
  끝이 녹음보다 길면 뺀다. 녹음에 없는 파일 이름은 표 아래에 `spans 에 있지만 녹음에 없는 파일` 로 찍힌다. `구간 모호`·`잡음에 끌림`
  으로 뺀 클립도 16k wav 는 `--out` 에 남는다.
- **표**: 파일 · 문장 · 길이 · 말소리 구간 · 잡음 바닥(하위 10%) · 잡음(말소리 밖 평균) · 추정 SNR · 경고. `✗` = manifest 에서 뺌
  (말소리 못 찾음 · 문장표에 없는 id · 같은 이름 중복 · 구간 모호 · quiet 길이와 다름 · spans 가 녹음 밖 · 문장 녹음이 없어 SNR 을
  못 잰 환경). 경고: 앞/뒤 여유 부족 · 녹음 처음 0.5초·끝 1초 안 소리(대사 같은 잡음이면 구간이 끌린다) · 클리핑(|x|≥0.99 3샘플
  이상) · 말소리 덩어리 2개 이상(0.8초 넘는 쉼 — 구간 확인) · 말소리 밖 소리 · 추정 SNR 9dB 미만 · 구간 수동. 끝에 환경별 요약·
  quiet 길이 대조 범위·예상 비용.
- **조건 묶음은 환경 라벨 단위**: `noise` = 환경 라벨, `--clean-label`(기본 `quiet`)은 `clean` 으로 — run_chirp 가 clean 클립을
  워밍업으로 쓰고 score 표 첫 줄에 온다. `snr_db` = 그 환경의 추정 SNR 중앙값(정수, clean 은 null) — run_chirp 진행 줄이
  `<라벨>_snr<값>` 으로 찍히고 score 가 환경마다 한 줄로 묶는다. 클립별 추정은 `snr_est_db`(manifest 에만). manifest 를 다시 만들면
  중앙값이 바뀔 수 있어 그 전 실행과 따로 묶인다 — 끔·켬은 같은 manifest 로 같이 돌린다. 그 밖 필드(`file`·`utt`(문장 id)·`text`·
  `variant`(null)·`noise_only`·`lead_s`·`speech_start_s`·`speech_end_s`·`utt_end_s`(= 말소리 끝)·`dur_s`)는 mix_noise manifest 와 같은 뜻.
  run_chirp·score 는 고치지 않았다.
- **`--reps 1`**: `variant` 가 null 이라 반복마다 같은 파일을 다시 보낸다 — 같은 오디오 반복은 결과가 사실상 같아(9/09) 돈만 든다.
  테이크를 여러 개 녹음하면 각각 한 번씩 간다(CI 표본 = 서로 다른 파일 수 `aud`). 일부만은 `--noise cafe --utt u02 u06`.
- 합성 벤치 결과(`runs/`)와 폴더를 나눈다 — 둘 다 `clean` 을 쓰므로 score 에 같이 넣으면 섞인다.
- **비용**: 14문장 × 3환경 × 테이크 1 = 42클립, 클립 ≈ 6초(1 + 문장 3 + 2)라 끝까지 보내도 조건당 약 4.2분 → 끔·켬 2조건 + 워밍업 ≈
  8.6분 × $0.016/분 ≈ **$0.14**(턴 FINAL 에서 멈추니 실제는 더 적다). 잡음만 20초 × 2환경을 넣으면 +1.3분 ≈ $0.02. 정확한 상한은
  `--dry-run`. 실시간 속도라 걸리는 시간 ≈ 오디오 길이 ÷ `--concurrency`(4 면 3분 안팎).

### 5-4. 한계

- **음성 메모는 통화 모드 신호 처리를 안 거친다.** 앱(BADA_APP develop d815b33 `hooks/useAudio.ts`·`useTrainPlayback.ts`)은 웹이
  getUserMedia `noiseSuppression`·`echoCancellation`·`autoGainControl` 을 켜고, iOS 는 오디오 세션 `playAndRecord` + `voiceChat`
  (애플 음성 처리), 안드로이드는 `audioSource: 6`(VOICE_RECOGNITION)이다. 실제 통화 오디오는 기기에서 잡음이 이미 일부 줄어든 것이라
  음성 메모 결과는 잡음 쪽으로 비관적이다 — 기기 억제 뒤 남은 잡음에 chirp denoiser 가 더 할 일이 있는지는 이 키트로 못 잰다.
- 한 사람·한 폰·낭독. 같은 환경 라벨 안에서도 영상 내용·위치에 따라 잡음이 다르다(`snr_est_db` 로 본다).
- 말소리 구간은 소리 크기로 한 추정이다. 잡음보다 약한 끝 음절은 빠져 끝이 이르게 잡히고(지연이 그만큼 크게, 5dB 안팎에선 0.3초
  넘게), 문장 0.8초 안의 대사·말소리 같은 잡음은 구간에 붙어 끝이 늦게 잡힌다(정상 턴이 early 로, 꼬리 대사 FINAL 이 split 으로
  보인다). 앞쪽은 quiet 길이 대조와 `추정 SNR` 경고, 뒤쪽은 quiet 길이 대조로 거르지만 둘 다 완전하지 않다. 경고가 붙은 클립은 원본을
  들어 보고 `--spans` 로 고친다.
- 추정 SNR 은 말소리 밑 잡음이 앞뒤 잡음과 같다는 전제다. 출렁이는 잡음(TV·거리)에선 어긋난다(합성 시험 ±2dB).

## 6. 운영 로그로 켜기 전후 보기 (계획 0077, 사용자가 받아 맥에서)

F128 이 들어간 이미지는 세션마다 `metric=stt_config session_id=… engine=chirp model=chirp_3 location=… denoise=True|False
snr_threshold=…` 한 줄을 남긴다. `prod_compare.py` 는 세션을 이 값으로 묶는다: `off`(False) · `on`(True) · `none`(stt_config 없음 =
스위치 이전 이미지) · `gemini`(engine=gemini_live) · `mixed`(한 세션에 값 둘).

순서: F128 이미지(기본 꺼짐)로 며칠 = off → **켜기 직전 로그 받기** → (실발화 확인 뒤, 사용자) EC2 `.env` `GOOGLE_STT_DENOISE=true` +
컨테이너 재생성 → 같은 길이만큼 운영 → 로그 받기 → 맥에서 집계.

EC2에서 — 컨테이너를 재생성하기 **전에**(docker logs 는 재생성하면 사라진다):

```
docker logs bada-app-1 2>&1 \
  | grep -E 'metric=(stt_config|voice_turn|final_dropped|session_usage) |STT 무요청|STT 스트리밍 실패|STT 스트림|STT 하드' \
  > ~/stt-$(date +%m%d-%H%M).log
wc -l ~/stt-*.log
```

- 사용자 발화 글자가 든 줄(`STT FINAL 수신 … text=…`)은 이 패턴에 안 걸린다. session_id·user_id·scenario_id 는 들어 있으니 받은 파일은
  `.harness` 아래(gitignore)에만 둔다.
- 이미 재생성돼 사라진 기간은 매시 백업(`scripts/backup_logs.sh` → S3 `logs/app/<YYYY>/<MM>/<DD>/*.log.gz`)을 `zcat` 해 같은 grep.

로컬 맥에서:

```
mkdir -p .harness/stt-noise-2026-10/prod
scp '<EC2 호스트>:~/stt-*.log' .harness/stt-noise-2026-10/prod/
.venv/bin/python scripts/stt_noise/prod_compare.py .harness/stt-noise-2026-10/prod/*.log
.venv/bin/python scripts/stt_noise/prod_compare.py .harness/stt-noise-2026-10/prod/*.log --since 2026-10-27 --until 2026-11-03 \
    --json-out .harness/stt-noise-2026-10/prod/after.json
```

- 묶음마다: 세션 수 · voice_turn 수 · `final_dropped` 사유별(empty·state·time_up·closing) 건수·100턴당·하나라도 있던 세션 비율
  (Wilson 95%), state 중 글자>0 · state 중 턴 FINAL 뒤 3초 안(`since_final_ms ≤ 3000`) · state 를 받은 상태(thinking·speaking) ·
  `session_usage reason` 분포와 ERROR 종료 세션 비율(Wilson) · voice_turn `stt_ms`·`response_ms`·`turn_total_ms` p50/p90.
  묶음이 둘 이상이면 비교 한 표. 끝에 세션에 못 붙이는 STT 줄 수(무요청 ABORTED · 스트리밍 실패 · 끊김 재오픈 · 백오프 재오픈 ·
  복구 포기 · 하드 재활용).
- `--since`(포함)·`--until`(제외)는 **세션 첫 줄의 로그 시각**으로 세션을 고르고(경계에 걸친 세션이 반으로 잘리지 않게), 세션 없는
  줄은 그 줄 시각으로 거른다. 시각은 로그에 찍힌 그대로 비교한다 — Dockerfile·compose 에 TZ 가 없어 UTC 로 보인다(미확인).
  그래서 **시간대 없이** 적는다(`2026-10-21`, `'2026-10-21 09:00'`). `Z`·`+09:00` 이 붙으면 어느 쪽으로 맞출지 모르니 거절한다
  (docker `-t` 의 `…Z` 를 그대로 붙여 넣지 말고 `Z` 를 뺀다).
- 여러 파일을 같이 넣어도 된다: 시각 있는 같은 줄은 파일마다 센 개수의 최댓값만 센다(백업과 docker logs 가 겹쳐도 한 번).
- 텍스트 로그(운영은 ENV=dev 라 텍스트)와 JSON 로그 둘 다 읽는다. `docker logs -t` 의 앞 시각이 있으면 그 시각을 쓴다.

읽을 때 주의:

- **켜기 전후는 다른 시간대다** — 요일·시간대·사용자 층(시연·새 테스터)·시나리오 구성·앱 버전이 달라 묶음 차이를 denoiser 탓으로
  단정하지 않는다. 기간 길이를 맞추고 세션 수가 적으면 CI 부터 본다.
- 한 세션의 턴끼리는 독립이 아니다 — 100턴당 값보다 '하나라도 있던 세션' 비율(세션 수가 표본)의 CI 가 정직하다.
- `empty` 는 잡음이 STT 를 깨워 빈 FINAL 이 나온 횟수에 가깝다(denoiser 로 줄길 기대). `state` 3초 안은 끊긴 FINAL 의 나머지가
  AI 차례에 온 신호(10/20 벤치의 u02 숫자 중간 끊김). 지금 코드에선 빈 FINAL 이 먼저 `empty` 로 분류돼 `state` 는 늘 글자>0 이다
  (`pipeline._handle_stt_event`) — 'state 중 글자>0' 줄은 state 와 같게 나온다.
- `stt_ms` 는 '마지막 오디오 청크 → FINAL' 이라 마이크가 계속 흐르는 운영에선 청크 간격 정도다. 말끝→FINAL(denoiser 지연 +110ms 가
  보일 곳)은 운영 로그에 없고, `response_ms` 는 STT 뒤라 denoiser 와 무관하다 — 지연은 벤치(§2·§5)로 본다.
- ERROR 종료는 STT 말고도(LLM·TTS·백그라운드 예외) 나온다. 'STT 스트리밍 실패' 줄 수와 같이 보고, 그 줄은 세션에 안 붙으니
  켜기 전후를 `--since`/`--until` 로 나눠 각각 돌린다.
