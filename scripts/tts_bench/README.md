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

## 화자 12칸 만들기 (계획 0069, #267)

칸 = 성별(female/male) × 나이(young/middle/old) × 톤(soft/rough), 이름 `{gender}_{age}_{tone}` = 운영 `voices.json` 키. 처음엔 Qwen3-TTS VoiceDesign 합성 참조로 시작했으나 10/15 시범 청취에서 탈락 → 실제 사람 목소리는 아래 'AIHub 71631' 절.

| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `design_voices.py` | GPU(집 PC·학교), 포크 venv | VoiceDesign 으로 칸마다 후보 N개(seed 고정) → `<칸>__c<i>.wav` · `candidates.json`(gen_qwen `--voices` 형식) · `listen.html`(칸별로 나란히). 대본 음절 수에 비해 말소리가 너무 짧거나(끝 잘림) 길면(EOS 누락) 후보에서 뺀다 |
| `audition.tsv` | — | `sentences.tsv` 에서 고른 13문장(감탄사·단문·긴 문장) — 1차 오디션용 |
| `prep_ref.py` | 맥, 레포 venv | 고른 후보 → 24kHz mono PCM16, 앞뒤 무음 0.5초, 말소리 레벨 = 운영 `ref_ai.wav`, 길이·잡음 바닥(이득 반영) 경고 + `voices.part.json`(절대경로) + `set/`(참조 전사 대조용 g1_score 세트) |

결과 폴더는 레포 밖. 아래 `$VD` = 집 PC `/mnt/c/qwen-out/voice_design`, `$PY` = `/home/uhihi/Qwen3-TTS-streaming/.venv/bin/python`.
학교 박스에서 돌릴 땐 `$VD` = `~/voice_design`, `$PY` = `~/bada-qwen3-tts/fork/.venv/bin/python`, 그리고 `nvidia-smi` 로 빈 GPU 를 골라
`CUDA_VISIBLE_DEVICES=<번호>`(운영 워커는 GPU 1·2)를 앞에 붙인다. gen_qwen.py 는 torch.compile 을 써서 학교에선 C 컴파일러도 붙여야 한다:
`CC=$HOME/gcc-env/bin/x86_64-conda-linux-gnu-gcc CXX=$HOME/gcc-env/bin/x86_64-conda-linux-gnu-g++`(boot.sh 와 같은 값). 결과는 JupyterHub 로 내려받는다.

1. 맥 → 집 PC: `scp scripts/tts_bench/design_voices.py scripts/tts_bench/gen_qwen.py scripts/tts_bench/audition.tsv scripts/tts_bench/sentences.tsv homepc:C:/qwen-out/`
   학교 박스는 같은 4개를 JupyterHub 로 올린다. `~/bada-qwen3-tts/repo` 는 운영 boot.sh 를 주는 사본이라 checkout 으로 바꾸지 말 것
   (브랜치를 푸시했다면 `git -C ~/bada-qwen3-tts/repo fetch origin feature/#267-qwen-12-voices && git -C ~/bada-qwen3-tts/repo show FETCH_HEAD:scripts/tts_bench/design_voices.py > ~/design_voices.py` 처럼 파일만 꺼낸다).
   VoiceDesign 모델은 첫 실행 때 약 4.5GB 를 받는다. 집 PC 는 먼저 `sudo systemctl stop qwen-tts`(8GB 에 모델 두 벌 불가), 다 끝나면 `start`.
2. 시범 3칸: `$PY /mnt/c/qwen-out/design_voices.py --out $VD --n 5 --slots male_old_rough,female_old_soft,female_middle_rough`
   → 맥으로 `scp -r homepc:C:/qwen-out/voice_design <맥 폴더>` 후 `listen.html` 청취.
   약한 칸은 `--instruct '<칸>=<영어 설명>'` 으로 다시 돌린다 — 바꾼 설명은 이름에 꼬리표가 붙어(`<칸>__<4자>c<i>`) 같은 seed 1~N 으로 새로 만들어지고, 페이지에 옛 후보와 나란히 나온다. 같은 설명으로 더 뽑으려면 `--n` 을 늘린다(이미 만든 후보는 건너뜀).
3. 12칸 전부: `--slots` 를 빼고 같은 명령(바꾼 설명이 있으면 그 `--instruct` 도 그대로). 지금 `ai` 참조를 같이 비교하려면 `--ref 'female_young_soft__ai=/mnt/c/qwen-out/<ai 참조>.wav:<그 전사>'`.
   칸마다 1~2개 고른다(성별·나이·톤, 자연스러움, 끝음절). **UTMOS 로 고르지 않는다.**
4. 운영 설정 그대로 1차 — 한 프로세스에 참조 6개 이하(dynamo 재컴파일 한도 8):
   `$PY /mnt/c/qwen-out/gen_qwen.py --sentences /mnt/c/qwen-out/audition.tsv --voices $VD/candidates.json --only <후보1>,<후보2>,… --out $VD/audition`
   → 맥에서 `.venv/bin/python scripts/tts_bench/post.py --out <audition 폴더> --sentences scripts/tts_bench/audition.tsv` 로 앞 무음·말 속도·eos·guard 확인. 칸마다 1개로 좁힌다.
5. 맥 — 게이트용 정규화(운영에 올라갈 것과 같은 오디오):
   `.venv/bin/python scripts/tts_bench/prep_ref.py --candidates <맥 폴더>/candidates.json --pick <칸>__c<i> …(12개) --dest-dir /mnt/c/qwen-out/voice_design/refs_v1 --tag v1 --level-ref ~/school/narsha/2026/qwen-tests/ref_ai.wav --out <맥 폴더>/refs_v1`
   → 경고(길이 10~15초 밖·잡음 바닥 > -60dBFS·피크 제한)가 없는지 본다. 참조 발화가 대본과 같은지는 출력 끝에 찍히는 g1_score 명령(labs venv)으로 CER 확인 — 0 이 아니면 그 후보는 쓰지 않는다(전사가 어긋나면 참조 문장이 출력에 샌다).
6. 최종 게이트(12개): `scp -r <맥 폴더>/refs_v1 homepc:C:/qwen-out/voice_design/` 후
   `$PY /mnt/c/qwen-out/gen_qwen.py --sentences /mnt/c/qwen-out/sentences.tsv --voices $VD/refs_v1/voices.part.json --only <칸1>,…(6개씩 두 번) --out $VD/gate`
   → `post.py --out <gate 폴더>`(턴은 `turns.tsv` 문장으로 맥에서 잇는다): eos 100%, guard 0, 앞 무음 최대 < 1.3초, 말 속도 ≥ 약 4음절/s, 끝음절 안 잘림.
7. 운영 경로로 다시: 5 와 같은 `--pick`·`--tag` 로 `--dest-dir /home/<학교 사용자>/bada-qwen3-tts/refs --out <맥 폴더>/refs_v1_school` (오디오는 같고 경로만 바뀐다).
   `*.wav` 를 워커 `refs/` 에 **새 이름 그대로** 올리고 `voices.part.json` 을 워커 `voices.json` 에 합친다(`ai`·`user` 유지). 참조를 바꿀 땐 `--tag v2` — 같은 이름 덮어쓰기 금지.
8. `prep.json`(후보 이름·seed·instruct) 과 `refs_v1/` 은 레포 밖에 백업 — 칸별 출처 기록이다.

### 실제 사람 목소리(AIHub 71631)로 바꿀 때 — 10/15 VoiceDesign 청취 탈락 뒤

`pick_aihub_voices.py`(집 PC, numpy 만): 71631 감정 자유대화(성인, 16kHz 스테레오, 대부분 스마트폰 녹음)의 라벨·원천 zip 에서, **wav 가 있는 세션만** 화자마다 9~14초 연속 구간을 찾는다(발화 사이 1.5초 이하, 경계 앞뒤에 같은 화자의 쉼 0.3초 이상, 상대 발화는 앞뒤 0.5초 밖, 전사에 마스킹·주석 기호 없음, 마스크 착용·목잠김 등 이상 상태 화자 제외). 구간 ±0.15초로 바짝 잘라 앞뒤를 디지털 무음으로 0.5초 채운다(전사 밖 말이 섞이지 않게). 말소리 − 잡음 바닥 30dB 미만·채널 섞임·내 채널에 60ms 넘게 들리는 상대 소리는 빼고, 칸(성별 × 20~30대/40~50대/60대 이상)마다 깨끗한 순으로 `--per-cell` 명. 39dB 미만이면 페이지에 'prep_ref 잡음 경고 예상' 표시.
맥에 있는 3세션(`~/Downloads/aihub71631/wav_in`)으로 돌려 보면 남성 젊음 2명만 통과한다(여성은 잡음 바닥 -44dBFS) — 실제 데이터는 시끄러운 편이고, 라벨 3,331개 기준 60대 이상은 여성 29명·남성 18명뿐이라 받은 wav 조각에 따라 노년 칸이 빌 수 있다.

1. 집 PC 에서 zip 위치 확인: `find ~/aihub -maxdepth 5 -name '*.zip' | grep -E '/(TL|VL|TS|VS)_'`
2. 맥 → 집 PC: `scp -o ClearAllForwardings=yes scripts/tts_bench/pick_aihub_voices.py homepc:C:/qwen-out/`
3. 집 PC WSL(GPU 안 씀, qwen-tts 를 내릴 필요 없음):
   `/home/uhihi/Qwen3-TTS-streaming/.venv/bin/python /mnt/c/qwen-out/pick_aihub_voices.py --label <라벨 zip 들> --wav <원천 zip 들> --out /mnt/c/qwen-out/aihub_voices --per-cell 10`
   → '칸별 화자 수(wav 있는 세션 기준)'·'2단계 탈락 이유' 를 본다. 노년 칸이 비면 71631 원천 조각을 더 받거나 다른 데이터셋으로.
4. 맥으로 `scp -o ClearAllForwardings=yes -r homepc:C:/qwen-out/aihub_voices ~/school/narsha/2026/qwen-tests/` 후 `aihub_voices/listen.html` 청취.
   목소리를 고른 뒤 같은 화자의 다른 구간을 보려면 `--speakers <화자 ID,…> --windows-per-speaker 3 --out …/aihub_voices2` 로 다시 돌린다(구간은 문장 끝 부호로 끝난다 — 10/15 게이트에서 문장 중간에서 끝난 참조는 클론 끝 '뚝' 69~92%·끝맺음 실패를 냈다). 칸마다 부드러운 것·거친 것을 하나씩 고른다('거친 후보' 는 감정 라벨 힌트일 뿐).
5. 운영 설정 그대로 1차(집 PC, qwen-tts 를 내리고): `$PY /mnt/c/qwen-out/gen_qwen.py --sentences /mnt/c/qwen-out/audition.tsv --voices /mnt/c/qwen-out/aihub_voices/candidates.json --only <후보1>,…(6개 이하) --out /mnt/c/qwen-out/aihub_voices/audition` → 맥 `post.py --out <그 폴더> --sentences scripts/tts_bench/audition.tsv`.
6. 맥 — 게이트용 정규화(칸을 직접 붙인다, 태그는 VoiceDesign 과 겹치지 않게 `h1`):
   `.venv/bin/python scripts/tts_bench/prep_ref.py --candidates ~/school/narsha/2026/qwen-tests/aihub_voices/candidates.json --pick female_young_soft=<후보 이름> --pick male_old_rough=<후보 이름> …(12개) --dest-dir /mnt/c/qwen-out/aihub_voices/refs_h1 --tag h1 --level-ref ~/school/narsha/2026/qwen-tests/ref_ai.wav --out ~/school/narsha/2026/qwen-tests/aihub_voices/refs_h1`
   → 출력 끝의 g1_score 명령으로 전사 CER 확인(라벨 전사가 틀린 후보는 쓰지 않는다). 16kHz 원본을 24kHz 로 올리므로 고음은 늘지 않는다.
7. 최종 게이트·운영 반영은 위 6~8단계와 같다 — 폴더만 `aihub_voices/refs_h1`, 학교용은 같은 `--pick`·`--tag h1` 로 `--dest-dir /home/<학교 사용자>/bada-qwen3-tts/refs`.

AIHub 원본은 이용 정책상 'AI 학습용' 이라 실제 사람 목소리를 서비스 클론 참조로 쓰는 것은 회색 지대다 — 쓰기 전에 AIHub 에 확인할 것(계획 0069 10절).

## 프리셋 예시 굽기 비교 (계획 0070)

프리셋 예시 대화를 지금 방식(plain, 턴마다 한 요청)과 운영 새 방식(clean — 꼬리말 ` ... 네.` 붙여 합성 → 마지막 쉼에서 잘라내기 → 끝 '뚝' 이면 그 턴만 다시, 앞 무음 100ms)으로 구워 비교한다. clean 은 운영 `app/services/example_audio.py` 를 그대로 쓴다. **배포 전에 듣고 승인하는 용도.**

| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `bake_presets.py export` | 맥, 레포 venv | 레포 프리셋 8개(대본·칸) → `presets.json` |
| `bake_presets.py gen` | 집 PC, 포크 venv | 프리셋 × {plain, clean} 굽기 → `report.md`(방식·보이스별 끝 잘림·시도 수·앞 무음) · `index.html`(대화 단위 청취) · `set/`(g1_score 받아쓰기 — 꼬리말 '네' 가 남거나 말이 잘렸는지) |

1. 맥: `.venv/bin/python scripts/tts_bench/bake_presets.py export --out /tmp/presets.json`
2. 맥 → 집 PC: `scp -o ClearAllForwardings=yes scripts/tts_bench/bake_presets.py scripts/tts_bench/gen_qwen.py app/services/example_audio.py /tmp/presets.json homepc:C:/qwen-out/`
3. 집 PC WSL(qwen-tts 를 내리고): `$PY /mnt/c/qwen-out/bake_presets.py gen --presets /mnt/c/qwen-out/presets.json --voices /mnt/c/qwen-out/voices.json --voices /mnt/c/qwen-out/aihub_voices2/refs_h3/voices.part.json --ai-voice slot --out /mnt/c/qwen-out/presets_slot`
   — 지금 운영(ai 목소리)으로 보려면 `--ai-voice ai --out /mnt/c/qwen-out/presets_ai`. 한두 개만 먼저 보려면 `--only 1,7`.
   — 'clean 의 꼬리말 포기(plain 대체)' 가 많으면 Qwen 이 '...' 에서 150ms 넘게 쉬지 않는다는 뜻이다. `--methods clean --tail-phrase ' ...... 네.'`(또는 `'. 네.'`) 로 꼬리말만 바꿔 다른 `--out` 에 굽고 비교한다.
4. 맥으로 `scp -o ClearAllForwardings=yes -r homepc:C:/qwen-out/presets_slot ~/school/narsha/2026/qwen-tests/` 후 `presets_slot/index.html` 청취, `report.md` 확인.
   받아쓰기: `OUT=~/school/narsha/2026/qwen-tests/presets_slot; source ~/School/narsha/2026/LLM-STT-멀티모달/labs/env.sh && python ~/School/narsha/2026/LLM-STT-멀티모달/labs/eval/g1_score.py --set $OUT/set --cond plain=$OUT/turns/plain --cond clean=$OUT/turns/clean --out $OUT/score --metrics wer`
   승인 기준(제안): clean 끝 잘림 ≤ 1/80 · 앞 무음 최대 < 0.3초 · 받아쓰기에 꼬리말 '네' 가 남은 턴 0.

## 읽을 때
- EL 은 프리셋 시나리오 보이스(`.env` `ELEVENLABS_VOICE_ID`)다. 커스텀 시나리오는 `pick_voice_id` 풀의 다른 보이스를 쓴다(`--voice-id`).
- Qwen 은 운영처럼 문장마다 따로 부르고, EL 턴은 운영처럼 한 세션에 문장을 흘려 넣는다 → 턴 비교가 통화에서 실제로 듣는 소리다.
- 운영 `QWEN_TTS_SPEED` 가 1.0 이 아니면 `post.py --qwen-speed <값>` 으로 같은 속도 열(`qwen_*_x<값>`)을 더한다(운영 `PcmStretcher`, 턴 첫 소리 180ms 원속도).
- SIM 은 새 ref 기준이다: 그 ref 를 복제한 열만 복제 충실도이고 나머지는 '다른 사람' 바닥값이다(대체 모델 `wavlm-base-plus-sv`, 상대 비교만). UTMOS 는 영어 학습 모델이라 상대값만 본다.
- 요청 끝 '뚝'·앞 무음은 `qwen-tests/merge_test/count_cuts.py`·`guard_fp/measure.py` 와 같은 계산이다.
