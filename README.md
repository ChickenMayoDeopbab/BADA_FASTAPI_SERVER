# 바다
콜포비아 극복을 위한 AI 기반 전화 훈련 서비스

<img width="80" height="80" alt="image" src="https://github.com/user-attachments/assets/e357fde4-ffd1-42fd-b12a-59e2b1658f89" />

<br>
<br>

# 문제정의
최근 MZ세대를 중심으로 메신저 기반 소통을 선호하는 문화가 확산되면서, 단순한 소통 방식의 변화에 그치지 않고 **전화 통화 자체에 대한 불안과 두려움**을 느끼는 사람들이 꾸준히 증가하고 있습니다.

이러한 현상을 콜포비아(Call Phobia) 또는 전화공포증이라고 하며, 전화를 걸거나 받을 때 극심한 긴장감과 불안으로 인해 통화를 기피하는 심리적 현상을 의미합니다.

특히 취업 준비생은 지원·면접 관련 전화를 미루거나 포기하고, 사회초년생은 업무 전화를 두려워해 실수를 걱정하며, 대학생은 병원 예약이나 문의 전화와 같은 일상적인 통화조차 부담을 느끼는 등 다양한 어려움을 겪고 있습니다. 실제로 MZ세대의 콜포비아 경험 비율은 29.9%에서 35.6%로 증가했으며, 신입사원의 39.4%가 전화벨이 울리는 순간을 가장 두려운 상황 중 하나로 꼽았습니다. 이는 구직 기회를 놓치거나 업무 적응에 어려움을 겪는 등 다양한 사회적 문제로 이어질 수 있습니다.

하지만 실제 전화 상황을 부담 없이 반복 연습하며 전화 불안을 극복할 수 있는 서비스는 없습니다. 대부분의 AI 대화 서비스는 외국어 회화나 면접 연습에 초점을 맞추고 있어, 전화 상황 자체에 익숙해질 수 있는 훈련 환경이 부족합니다.

### MZ세대 콜포비아 현황 (알바천국, MZ세대 1,496명)
[출처: 전화 통화 시 긴장, 불안, 두려움 느껴…알바 구직도 문자, 채팅으로! 알바천국, 콜 포비아 MZ세대 늘었다](https://m.alba.co.kr/story/MediaReportView.asp?page=2&idx=3713)
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/2a73e08d-b102-4850-a6cd-6ccc170876e2" />
- 콜포비아 증상 경험 비율 **29.9% → 35.6%** (1년 만에 **5.7%p 증가**)
- 텍스트 기반 소통 선호 **61.4% → 70.7%**
- 전화가 가장 부담스러운 상황
    - 구직 관련 전화 (72.8%)
    - 업무상 전화 (60.4%)
    - 문의 전화 (44.5%)
    - 예약·취소 전화 (39.2%)
    - 배달 주문 전화 (34.3%)
### 신입사원이 가장 두려워하는 상황 (잡코리아)
[출처: 연합 뉴스 신입사원이 가장 두려운 것…선배호출 그리고 전화벨](https://www.yna.co.kr/view/AKR20161017060500003)
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/907d7fe6-359f-4fbc-a46b-a004a6638ded" />
- 전화벨이 울릴 때 긴장한다 **39.4%**
- 업무 중 전화 실수 경험 **26.8%**
- 기업이 원하는 역량 1위
    - **협업 및 커뮤니케이션 능력 (77.7%)**
### 콜포비아 시장 확대 (중앙일보)
[출처: 중앙일보 전화하면 문자로 "부장님, 왜?"…콜포비아 과외 1시간 10만원](https://www.joongang.co.kr/article/25135411)
- 콜포비아 극복을 위한 비즈니스 스피치 과외가 시간당 3만~9만 9천 원 수준으로 운영될 만큼 관련 수요가 증가하고 있습니다.

<br>

# 아키텍처
<img width="1091" height="511" alt="image" src="https://github.com/user-attachments/assets/2cb8ecf1-4087-463d-9169-46d361654839" />

<br>

# 사용 스택
### FrontEnd
| 분야 | 기술 |
| --- | --- |
| Framework | React Native (Expo) |
| Language | TypeScript |
| Audio | expo-av, react-native-audio-record |
| Network | Axios, WebSocket |
| Form | React Hook Form |

React Native(Expo)와 TypeScript를 기반으로 모바일 애플리케이션을 개발했습니다. Axios를 이용한 서버 통신과 WebSocket 기반 실시간 음성 스트리밍을 구현했으며, React Hook Form을 활용해 사용자 입력 검증과 인증 흐름을 구성했습니다.

### BackEnd
| 분야 | 기술 |
| --- | --- |
| Backend | Spring Boot, FastAPI |
| Language | Java, Python |
| Database | PostgreSQL, Redis |
| AI | Gemini, Claude |
| Speech | Google Cloud Speech-to-Text, ElevenLabs |
| Infra | Docker, GitHub Actions, AWS EC2, AWS S3 |

Spring Boot는 회원 관리, 시나리오, 훈련 기록 등 핵심 비즈니스 로직을 담당하며, FastAPI는 AI 음성 처리 파이프라인을 담당합니다.
사용자의 음성은 Google Cloud Speech-to-Text를 통해 텍스트로 변환되고, Gemini와 Claude가 상황에 맞는 응답을 생성한 뒤 ElevenLabs TTS를 통해 자연스러운 음성으로 출력됩니다.

<br>

# 실행방법
### OneStore
[앱 다운로드 링크](https://onesto.re/0001007890) <br>
설치 후 실행

<br>

# AI 사용 내역
| 구분 | 활용 내용 |
| --- | --- |
| 실시간 대화 | 사용자 발화에 맞는 전화 응답 생성 |
| 시나리오 생성 | 사용자 맞춤형 전화 상황 생성 |
| 음성 인식 | 사용자의 음성을 텍스트로 변환(STT) |
| 음성 합성 | AI 응답을 자연스러운 음성으로 출력(TTS) |
| 통화 피드백 | 목소리 떨림, 침묵 분석 결과를 출력 |

<br>

# 사용한 AI 모델
| 구분 | 모델 | 용도 |
| --- | --- | --- |
| LLM (실시간 대화) | Gemini 2.5 Flash | 전화 대화 응답 스트리밍, 워밍업, 칭찬 문장 생성 |
| LLM (분석/생성) | Claude Sonnet 4 | 커스텀 시나리오 생성 |
| STT | Google Cloud Speech-to-Text(chirp_3) | 음성 인식 |
| TTS | ElevenLabs Flash v2.5 | 음성 합성 |

<br>

# Lisence
This project is licensed under the MIT License.<br>
See the LICENSE file for details.

<br>

# 오픈소스 패키지
**FastAPI 서버**
| 구분 | **패키지** |
| --- | --- |
| 웹 | fastapi, uvicorn[standard], pydantic, pydantic-settings |
| 인프라 | redis[hiredis] |
| DB | sqlalchemy[asyncio], asyncpg |
| 통신 | httpx, websockets, boto3 |
| AI SDK | google-cloud-speech, google-genai, anthropic, elevenlabs |
| 내부 알고리즘 설계 | librosa, scipy, numpy |
| 인증·로깅 | PyJWT, python-json-logger |
| 개발 | pytest, pytest-asyncio, ruff |

<br>

# 외부 자문
서비스의 신뢰성과 실효성을 높이기 위해 미디어 심리학자 조연주 대표님으로부터 프로젝트 전반에 대한 자문을 받았습니다.

### 주요 자문 내용
- **자가진단 문항 검토**
    - 자가진단 문항이 콜포비아의 심리적 기제를 적절히 측정하는지 신뢰도와 타당성을 검토
- **훈련 솔루션 제언**
    - 전화에 대한 불안뿐 아니라 감정을 스스로 인식하고 주체적으로 대처할 수 있도록 심리적 훈련 설계에 대한 의견 제공
- **지속적인 피드백**
    - 프로젝트 진행 과정에서 심리학적 관점과 사용자 경험(UX)에 대한 지속적인 자문 제공

<br>

# 저장소 구조
**Fast API** - 현재 저장소<br>
바다의 AI 서버로 실시간 음성 인식(STT), LLM 기반 대화 생성, 음성 합성(TTS)과 같은 AI 전화 훈련 파이프라인을 담당합니다.<br>
https://github.com/ChickenMayoDeopbab/BADA_FASTAPI

**App**<br>
React Native(Expo) 기반 모바일 애플리케이션입니다.<br>
https://github.com/ChickenMayoDeopbab/BADA_APP

**Spring Server**<br>
백엔드 서버로 핵심 비즈니스 로직과 API를 담당합니다.<br>
https://github.com/ChickenMayoDeopbab/BADA_SPRING_SERVER
