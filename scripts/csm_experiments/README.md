# CSM-1B 실험 스크립트 (스터디 레포에서 이관, 2026-09-21)

설계 `.harness/plans/DRAFT-대화형-음성-모델-한국어-CSM-설계.md` 의 **G0 속도 게이트 통과(RTF 0.277)** · **KsponSpeech 토큰화** · **A단계 학습**을 만든 스크립트들이다.

**원본을 그대로 둔다 — lint 를 위해 고치지 않는다.** `scripts/qwen_tts_experiments/` 와 같은 원칙이다(계획 0045): 코드가 바뀌면 그 스크립트가 낸 숫자가 아니게 된다. 그래서 `pyproject.toml` 의 ruff `extend-exclude` 에 이 디렉터리를 넣었다.
g0 3종과 `tok_kspon.py` 는 학교 3090·집 PC 4060 에서 실제로 돈 파일과 바이트 단위로 같다(원본: 스터디 레포 `~/School/narsha/2026/LLM-STT-멀티모달/labs/`, `labs/data/`).

| 파일 | 하는 일 | 낸 숫자 (2026-09-21) |
|---|---|---|
| `g0_csm_speed.py` | HF 기본 `generate()` 경로의 G0 — 프리필 / 백본 1스텝 / depth 31스텝 / Mimi 분해 + 학습 1스텝 메모리 | RTF **1.651 (실패)** · 첫 프레임 251 ms · 학습 fwd+bwd peak 7.02 GB → 8-bit Adam 전체 FT 환산 16.3 GB · Mimi 인코딩 실시간의 594배 |
| `g0b_cards.py` | 2차 — `direct`(GenerationMixin 없이 HF forward 호출) · `static`(HF 정적 캐시 자동 컴파일) · `check` | direct 1.457 · direct 16코드북 0.787 · static 0.92 · check: CPU 에서 HF 와 256/256 토큰 |
| `g0c_static_loop.py` | 3차 — **정적 KV + `torch.compile(fullgraph, reduce-overhead)` 스텝. 실시간 워커 생성 루프의 뼈대** | RTF **0.277 (통과)** · 첫 프레임 172 ms · GPU fp32 탐욕 디코딩에서 HF 와 **320/320 토큰 일치** |
| `tok_kspon.py` | KsponSpeech zip → Mimi 32코드북 토큰(npz 샤드 + jsonl 매니페스트). zip 을 풀지 않는다 | 4060 에서 실시간의 ~124배(전체 ≈ 8 h) · 전사 628,545개 전부 매칭 |
| `check_tokens.py` | `tok_kspon.py` 출력의 무결성 검사(offsets·매니페스트 줄 수·코드 범위·짝·남은 임시 파일) | 가짜 묶음 4개(정상·offsets 어긋남·깨진 파일·임시 파일)로 동작 확인 |
| `csm_data.py` | 미리 뽑은 토큰 → CSM 학습 배치(글 ids · 오디오 임베딩 · 3D 레이블 · ratio). HF 가 오디오에서 만드는 것을 토큰에서 똑같이 만든다 | HF 경로와 input_ids·inputs_embeds(차이 0.0)·labels 동일 |
| `verify_inputs.py` | 위가 HF 경로(오디오 → `CsmProcessor` → `_merge_input_ids_with_input_values`)와 같은지 확인. CPU fp32 | loss **7.31607 = 7.31607** · 패딩 방향이 다른 2발화 배치도 동일 |
| `train_a.py` | **A단계 학습 루프** — 전체 FT · fp32 가중치 + bf16 autocast · 8-bit AdamW · 체크포인팅 · 검증 loss · 이어받기 | CPU 스모크: depth loss 4.11 → 1.48(6 업데이트), 이어받기 1.48 → 1.42. **GPU 미실행** |

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

## 주의
- **`tok_kspon.py` 를 같은 출력 폴더에 동시에 두 개 돌리지 말 것.** 끝난 묶음은 건너뛰지만 잠금이 없어, 같은 미완료 묶음을 두 프로세스가 만들면 임시 파일 이름이 같아 서로 덮어쓴다(2026-09-21 에 실수로 겹쳐 띄운 적이 있다). 겹쳤다면 `pgrep -af tok_kspon.py` 로 **파이썬 PID** 를 찾아 죽이고(작업 번호·감싼 셸의 PID 는 자식을 안 죽인다) 끝난 뒤 `check_tokens.py` 를 돌린다.
- `g0_csm_speed.load()` 는 로드 직후 **오디오 임베딩을 수동으로 묶고 assert 2개**를 건다. 학습·추론 스크립트 어디서든 빼면 안 된다(DECISIONS 2026-09-21).
- `g0c --mode check` 는 **fp32 로** 볼 것. bf16 에서는 동률 근처 argmax 가 뒤집혀 불일치가 나오는 게 정상이다.
- `.harness/records/2026-09-17-voice-naturalness/scripts/` 의 `g0_csm_speed.py`·`mimi_tokenize_bench.py` 는 미실행 초안이었고 이 디렉터리가 대체한다.
