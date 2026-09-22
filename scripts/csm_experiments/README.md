# CSM-1B 실험 스크립트 (스터디 레포에서 이관, 2026-09-21~22)

설계 `.harness/plans/DRAFT-대화형-음성-모델-한국어-CSM-설계.md` 의 **G0 속도 게이트 통과(RTF 0.277)** · **KsponSpeech 토큰화** · **A단계 학습** · **G1 평가**를 만든 스크립트들이다.

**원본을 그대로 둔다 — lint 를 위해 고치지 않는다.** `scripts/qwen_tts_experiments/` 와 같은 원칙이다(계획 0045): 코드가 바뀌면 그 스크립트가 낸 숫자가 아니게 된다. 그래서 `pyproject.toml` 의 ruff `extend-exclude` 에 이 디렉터리를 넣었다. 예외는 **실행 불가 버그**(9/22 홀수 바이트 .pcm)뿐이며, 그때도 이미 낸 출력이 바뀌지 않음을 확인하고 기록한다(DECISIONS 2026-09-22).
g0 3종과 `tok_kspon.py` 는 학교 3090·집 PC 4060 에서 실제로 돈 파일과 바이트 단위로 같다(원본: 스터디 레포 `~/School/narsha/2026/LLM-STT-멀티모달/labs/`, `labs/data/`, `labs/train/`, `labs/eval/`).

| 파일 | 하는 일 | 낸 숫자 (2026-09-21) |
|---|---|---|
| `g0_csm_speed.py` | HF 기본 `generate()` 경로의 G0 — 프리필 / 백본 1스텝 / depth 31스텝 / Mimi 분해 + 학습 1스텝 메모리 | RTF **1.651 (실패)** · 첫 프레임 251 ms · 학습 fwd+bwd peak 7.02 GB → 8-bit Adam 전체 FT 환산 16.3 GB · Mimi 인코딩 실시간의 594배 |
| `g0b_cards.py` | 2차 — `direct`(GenerationMixin 없이 HF forward 호출) · `static`(HF 정적 캐시 자동 컴파일) · `check` | direct 1.457 · direct 16코드북 0.787 · static 0.92 · check: CPU 에서 HF 와 256/256 토큰 |
| `g0c_static_loop.py` | 3차 — **정적 KV + `torch.compile(fullgraph, reduce-overhead)` 스텝. 실시간 워커 생성 루프의 뼈대** | RTF **0.277 (통과)** · 첫 프레임 172 ms · GPU fp32 탐욕 디코딩에서 HF 와 **320/320 토큰 일치** |
| `tok_kspon.py` | KsponSpeech zip → Mimi 32코드북 토큰(npz 샤드 + jsonl 매니페스트). zip 을 풀지 않는다. **9/22 패치**: 홀수 바이트 .pcm 은 끝 1 B 를 버린다(eval zip 6,000개 전부 2N+1 — DECISIONS 2026-09-22) | 4060 에서 실시간의 128배 · **완료: 625묶음 · 628,545발화 · 982.4 h · check_tokens 이상 0** |
| `check_tokens.py` | `tok_kspon.py` 출력의 무결성 검사(offsets·매니페스트 줄 수·코드 범위·짝·남은 임시 파일) | 가짜 묶음 4개(정상·offsets 어긋남·깨진 파일·임시 파일)로 동작 확인 |
| `csm_data.py` | 미리 뽑은 토큰 → CSM 학습 배치(글 ids · 오디오 임베딩 · 3D 레이블 · ratio). HF 가 오디오에서 만드는 것을 토큰에서 똑같이 만든다 | HF 경로와 input_ids·inputs_embeds(차이 0.0)·labels 동일 |
| `verify_inputs.py` | 위가 HF 경로(오디오 → `CsmProcessor` → `_merge_input_ids_with_input_values`)와 같은지 확인. CPU fp32 | loss **7.31607 = 7.31607** · 패딩 방향이 다른 2발화 배치도 동일 |
| `train_a.py` | **A단계 학습 루프** — 전체 FT · fp32 가중치 + bf16 autocast · 8-bit AdamW · 체크포인팅 · 검증 loss · 이어받기 | CPU 스모크: depth loss 4.11 → 1.48(6 업데이트), 이어받기 1.48 → 1.42. **GPU 미실행** |
| `g1_pick.py` | **G1 시험셋**(집 PC) — 학습 제외 `eval_clean` 에서 목표 100 + 프롬프트 100 을 고정 seed 로 짝지어 seed-tts-eval 형식 `meta.lst` + 대조군 음성(사람 원본·Mimi 재합성) + 프롬프트 Mimi 코드. 9/22 패치: 홀수 바이트 .pcm 처리 + 토큰화 길이 대조 | **실제 eval_clean 실행: 3,000 → 조건 통과 320 → 100 + 100** (seed 0). 맥 채점: human CER 6.56 · mimi 8.49 |
| `g1_generate.py` | **G1 생성**(학교 서버) — 프롬프트 있음/없음 × 정적 루프/HF, TTFA·RTF·끝남 기록, 이어받기, `--check` 로 HF 와 토큰 대조 | CPU fp32 탐욕: 내 입력 구성 + 정적 루프 = HF `generate` **96/96**(프롬프트 있음·없음), 프롬프트 임베딩 = HF merge 차이 **0.0**. **GPU 미실행** |
| `g1_score.py` | **G1 채점**(맥) — 심판 `gemini-3.5-transcribe-live`(B `GeminiLiveSTTClient` 와 같은 호출) CER/WER + UTMOS(`utmos22_strong`) + SIM(WavLM-large SV, seed-tts-eval 과 같은 모델) + `gen.jsonl` 의 TTFA·RTF → `result.md`·`listen.html`. 전 측정 캐시 | 편집거리·집계·리샘플·캐시 31항목 통과 · UTMOS 실모델 동작(사용자 녹음 3.09, Mimi 재합성 2.76) · **진짜 심판 호출·SIM 공식 모델 미실행**(키·체크포인트 없음) |
| `make_fake_tokens.py` | `train_a.py` 의 GPU 경로를 진짜 토큰 없이 보려고 만든 난수 토큰 셋(tok_kspon 출력 모양, loss 값은 무의미) | 서버 3090 20업데이트: peak 17.7 GB · 3,850 위치/s · 에러 없음 |
| `test_g1.py` | 위 두 파일의 순수 함수 시험(모델·네트워크 없음) | 31/31 |
| `test_judge_flow.py` | 심판의 비동기 흐름을 가짜 세션으로 시험 — 턴 단위 끊김·늦은 FINAL·송신 실패 | 4/4. **송신 실패 시 수신이 영원히 기다리던 결함**을 잡아 고쳤다 |

## 실행 환경
- **학교 GPU 서버**: GPU 0(3090, 비어 있었음 — 1·2 는 운영 Qwen 워커) · venv `~/csm-venv` = Python 3.12 / torch 2.9.1+cu126 / transformers 5.17.0 · 파일 `~/CSM/` · 모델은 `sesame/csm-1b` 의 HF 형식 파일만(7.1 GB), 학교망에선 `HF_HUB_DISABLE_XET=1` 로 받았다 · `HF_HOME=~/.cache/hf`
- torch.compile 은 런타임에 C 컴파일러를 부른다 → `CC/CXX=~/gcc-env/bin/x86_64-conda-linux-gnu-{gcc,g++}` (Qwen 워커 `boot.sh` 와 같은 방식)
- **집 PC(4060)**: 새 환경을 만들지 않고 `~/Qwen3-TTS-streaming/.venv/bin/python`(torch 2.9.1 / transformers 4.57.3)으로 실행 · `kyutai/mimi` 385 MB 만 받는다
- g0 세 파일은 같은 폴더에 있어야 한다(`g0b`·`g0c` 가 `g0_csm_speed` 를 import).
- 명령 전체와 실측 원문: `.harness/records/2026-09-21-csm-g0/`

## A단계 학습 (2026-09-21 추가)
- 세 파일(`csm_data.py`·`verify_inputs.py`·`train_a.py`)은 같은 폴더에 둔다. 입력은 `tok_kspon.py` 의 출력 폴더이고 오디오·Mimi 는 필요 없다.
- 서버에서 처음 돌릴 때: `uv pip install bitsandbytes` → `--max-updates 20` 으로 메모리(추정 17~18 GB)·속도·loss 하강을 먼저 본다. 명령·기본값·이유는 `.harness/records/2026-09-21-csm-g0/train-README.md`.
- 이 디렉터리에 둔 이유: 학습 run 의 숫자를 낸 파일을 그대로 남기기 위해서다(서버에서 도는 파일과 바이트 단위로 같다). 계획 0058 에서 워커·파이프라인으로 굳힐 때는 lint 를 맞춘 별도 모듈로 옮긴다.

## G1 평가 (2026-09-22 추가)
- 2026-09-22 사용자 결정: **TTFA · RTF · UTMOS · WER · SIM 을 seed-tts-eval 방식으로**, 심판 STT 는 **`gemini-3.5-transcribe-live`**. 원본 채점 코드를 읽고 확인한 것 — seed-tts-eval 은 WER·SIM 두 개만 잰다(WER = 발화별 (S+D+I)/N 의 **산술평균**, 중국어는 글자 단위 · SIM = 합성음 ↔ **프롬프트 음성** WavLM-large SV 코사인). UTMOS 는 F5-TTS 가 같은 셋에 붙여 쓰는 방식, TTFA·RTF 는 이 레포의 Qwen 측정(`scripts/qwen_tts_experiments/gate6.py`)과 같은 정의. 한국어 심판이 seed-tts-eval 에 없어 **공개 표와 직접 비교는 안 된다.**
- 세 기계에 세 파일: `g1_pick.py`(집 PC) → `g1_generate.py`(학교 서버, `g0c_static_loop.py` 옆에) → `g1_score.py`(맥, `GEMINI_API_KEY` 환경변수). 조건 하나 = `<문장 id>.wav` 폴더 하나라서 Qwen·ElevenLabs 폴더는 나중에 `--cond` 로 붙이면 된다. 명령·규칙·검증 표는 `.harness/records/2026-09-21-csm-g0/eval-README.md`.
- 심판 호출은 `app/services/stt.py` 의 `GeminiLiveSTTClient` 와 같은 설정(`google-genai==2.22.0`, `VERBATIM`, 16 kHz PCM, `audio_stream_end`)이지만 **코드를 import 하지 않고 따로 짰다**(스터디 레포 맥에서 돌고, 클립 하나 = 세션 하나라 파이프라인용 재활용 로직이 필요 없다). B 의 9/9 실측(F76)대로 Gemini 는 chirp 보다 오인식이 있으므로 `human` 행을 기준선으로 읽는다.
- `test_*.py` 두 개는 `testpaths = ["tests"]` 라 CI 의 pytest 에 걸리지 않는다. 돌리려면 스터디 레포 venv(numpy·torch·transformers·google-genai·torchaudio)에서 `python test_g1.py`.

## 주의
- **`tok_kspon.py` 를 같은 출력 폴더에 동시에 두 개 돌리지 말 것.** 끝난 묶음은 건너뛰지만 잠금이 없어, 같은 미완료 묶음을 두 프로세스가 만들면 임시 파일 이름이 같아 서로 덮어쓴다(2026-09-21 에 실수로 겹쳐 띄운 적이 있다). 겹쳤다면 `pgrep -af tok_kspon.py` 로 **파이썬 PID** 를 찾아 죽이고(작업 번호·감싼 셸의 PID 는 자식을 안 죽인다) 끝난 뒤 `check_tokens.py` 를 돌린다.
- `g0_csm_speed.load()` 는 로드 직후 **오디오 임베딩을 수동으로 묶고 assert 2개**를 건다. 학습·추론 스크립트 어디서든 빼면 안 된다(DECISIONS 2026-09-21).
- `g0c --mode check` 는 **fp32 로** 볼 것. bf16 에서는 동률 근처 argmax 가 뒤집혀 불일치가 나오는 게 정상이다.
- `.harness/records/2026-09-17-voice-naturalness/scripts/` 의 `g0_csm_speed.py`·`mimi_tokenize_bench.py` 는 미실행 초안이었고 이 디렉터리가 대체한다.
