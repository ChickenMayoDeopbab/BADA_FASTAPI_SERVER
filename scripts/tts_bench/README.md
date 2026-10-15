# TTS 벤치 — ElevenLabs 대 Qwen 음질 비교 (계획 0067)

같은 문장을 두 엔진에 운영 설정 그대로 넣어 뽑고, 같은 채점기로 비교한다. **지연은 재지 않는다**(집 4060 과 운영 3090, EL 의 인터넷 왕복이 섞여 공정하지 않다).

| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `sentences.tsv` | — | s01~s52 = 운영 로그 대사 44(`qwen-tests/merge_test/split.txt`, 로그의 `\n\n` 제거) + 감탄사·단문 8(`guard_fp/lines.txt`) |
| `turns.tsv` | — | t01~t16 = 같은 로그의 16턴(문장 id 순서). 운영 `_SentenceBuffer` 가 턴을 정확히 이 문장들로 자른다(시험) |
| `gen_el.py` | 맥, 레포 venv | 운영 `ElevenLabsTTSClient` + 레포 `.env`, 감정 NEUTRAL. 문장마다 세션 → `el/`, 턴마다 세션(문장을 차례로) → `el_turns/` |
| `gen_qwen.py` | 집 PC, 포크 venv | 워커 `server.py` 의 생성 인자·`max_frames_for`·24k→16k 리샘플·`_pcm16` + FastAPI 가드 상한(시험으로 고정). 문장 끝 공백 하나 → `qwen_<이름>/` |
| `post.py` | 맥, 레포 venv | Qwen 턴 = 문장 오디오를 틈 없이 잇기, g1_score 세트(`set/`), 바다 지표, 블라인드 청감(`listen_blind.html`), `bench.md` |

채점은 `scripts/csm_experiments/g1_score.py`(= 맥 `labs/eval/g1_score.py`, Gemini 심판 CER·UTMOS·SIM)를 고치지 않고 쓴다.

## 순서

결과 폴더는 레포 밖에 둔다(복제한 목소리가 레포에 들어가지 않게). 아래 `$OUT` = `~/school/narsha/2026/qwen-tests/el_vs_qwen/out`.

1. 맥 — ElevenLabs(약 1,800자): `.venv/bin/python scripts/tts_bench/gen_el.py --out $OUT`
2. 맥 → 집 PC: `scp scripts/tts_bench/gen_qwen.py scripts/tts_bench/sentences.tsv homepc:C:/qwen-out/`
3. 집 PC WSL — Qwen. 같은 GPU 에 `qwen-tts` 서비스가 떠 있으면 먼저 내린다(`nvidia-smi` 로 확인, 8GB 에 모델 두 벌은 안 들어간다).
   `/home/uhihi/Qwen3-TTS-streaming/.venv/bin/python /mnt/c/qwen-out/gen_qwen.py --sentences /mnt/c/qwen-out/sentences.tsv --voices /mnt/c/qwen-out/voices.json --only ai --ref 'new=/mnt/c/qwen-out/<ref>.wav:<녹음에서 말한 그대로>' --out /mnt/c/qwen-out/bench_out`
4. 집 PC → 맥: `scp -r homepc:C:/qwen-out/bench_out/qwen_ai homepc:C:/qwen-out/bench_out/qwen_new $OUT/` 와 ref wav 도 맥으로.
5. 맥 — 묶기: `.venv/bin/python scripts/tts_bench/post.py --out $OUT --ref <맥의 ref.wav> --ref-text '<전사>'`
6. 맥 — 채점(labs venv, 키는 환경변수로만):
   `source ~/School/narsha/2026/LLM-STT-멀티모달/labs/env.sh && python ~/School/narsha/2026/LLM-STT-멀티모달/labs/eval/g1_score.py --set $OUT/set --cond el=$OUT/el --cond qwen_ai=$OUT/qwen_ai --cond qwen_new=$OUT/qwen_new --out $OUT/score --sim xvector`
7. 맥 — 5 를 다시 돌리면 `bench.md` 에 CER·UTMOS·SIM 과 짝 비교가 합쳐진다. 귀로는 `listen_blind.html`.

모든 단계는 이미 만든 파일을 건너뛰고(g1_score 는 캐시) 몇 번이고 다시 돌려도 된다.

## 읽을 때
- EL 은 프리셋 시나리오 보이스(`.env` `ELEVENLABS_VOICE_ID`)다. 커스텀 시나리오는 `pick_voice_id` 풀의 다른 보이스를 쓴다(`--voice-id`).
- Qwen 은 운영처럼 문장마다 따로 부르고, EL 턴은 운영처럼 한 세션에 문장을 흘려 넣는다 → 턴 비교가 통화에서 실제로 듣는 소리다.
- 운영 `QWEN_TTS_SPEED` 가 1.0 이 아니면 `post.py --qwen-speed <값>` 으로 같은 속도 열(`qwen_*_x<값>`)을 더한다(운영 `PcmStretcher`, 턴 첫 소리 180ms 원속도).
- SIM 은 새 ref 기준이다: 그 ref 를 복제한 열만 복제 충실도이고 나머지는 '다른 사람' 바닥값이다(대체 모델 `wavlm-base-plus-sv`, 상대 비교만). UTMOS 는 영어 학습 모델이라 상대값만 본다.
- 요청 끝 '뚝'·앞 무음은 `qwen-tests/merge_test/count_cuts.py`·`guard_fp/measure.py` 와 같은 계산이다.
