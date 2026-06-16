import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Final

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech as cs

logger = logging.getLogger(__name__)

AUDIO_EOS: Final = None

# 5분
STREAM_LIMIT_SECONDS: Final[int] = 5 * 60

class STTEventType(str, Enum):
    INTERIM = "interim"
    FINAL = "final"
    SPEECH_BEGIN = "speech_begin"
    SPEECH_END = "speech_end"

@dataclass(frozen=True)
class STTEvent:
    """오케스트레이터/websocket 프레임과의 계약 객체."""
    type: STTEventType
    text: str = ""
    end_offset_sec: float | None = None

class GoogleSTTClient:
    def __init__(
        self,
        project_id: str,
        location: str,
        model: str,
        language: str,
        recognizer: str = "_",
        sample_rate_hertz: int = 16000,
    ) -> None:
        self._model = model
        self._language = language
        self._sample_rate_hertz = sample_rate_hertz
        self._recognizer_path = (
            f"projects/{project_id}/locations/{location}/recognizers/{recognizer}"
        )

        if location == "global":
            client_options = None
        else:
            client_options = ClientOptions(
                api_endpoint=f"{location}-speech.googleapis.com"
            )

        self._client = SpeechAsyncClient(client_options=client_options)

    def _build_config_request(self) -> cs.StreamingRecognizeRequest:
        recognition_config = cs.RecognitionConfig(
            explicit_decoding_config=cs.ExplicitDecodingConfig(
                encoding=cs.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self._sample_rate_hertz,
                audio_channel_count=1
            ),
            language_codes=[self._language],
            model=self._model,
            # TODO(2차): features에 자동 구두점 추가 고민
            #   features=cs.RecognitionFeatures(enable_automatic_punctuation=True)
        )
        streaming_config = cs.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cs.StreamingRecognitionFeatures(
                interim_results=True,
                enable_voice_activity_events=True,
            )
        )
        return cs.StreamingRecognizeRequest(
            recognizer=self._recognizer_path,
            streaming_config=streaming_config,
        )

    async def _request_generator(
            self,
            audio_queue: "asyncio.Queue[bytes | None]"
    ) -> AsyncIterator[cs.StreamingRecognizeRequest]:
        """streaming_recognize에 넘길 요청 스트림"""
        yield self._build_config_request()

        while True:
            chunk = await audio_queue.get()
            if chunk is AUDIO_EOS:
                logger.debug("STT 요청 스트림 종료 신호 수신")
                return
            yield cs.StreamingRecognizeRequest(audio=chunk)

    def _parse_response(
            self,
            response: cs.StreamingRecognizeRequest,
    ) -> list[STTEvent]:
        """단일 응답에서 STTEvent들을 추출."""
        events: list[STTEvent] = []

        event_type = response.speech_event_type
        begin = cs.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
        end = cs.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END

        if event_type == begin:
            events.append(STTEvent(type=STTEventType.SPEECH_BEGIN))
        elif event_type == end:
            events.append(STTEvent(type=STTEventType.SPEECH_END))

        for result in response.results:
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if result.is_final:
                events.append(
                    STTEvent(
                        type=STTEventType.FINAL,
                        text=transcript,
                        end_offset_sec=_to_seconds(result.result_end_offset),
                    )
                )
            else:
                events.append(
                    STTEvent(type=STTEventType.INTERIM, text=transcript)
                )
        return events

    async def stream(
            self,
            audio_queue: "asyncio.Queue[bytes | None]",
    ) -> AsyncIterator[STTEvent]:
        """메인 진입점"""
        try:
            responses = await self._client.streaming_recognize(
                requests=self._request_generator(audio_queue)
            )
            async for response in responses:
                for event in self._parse_response(response):
                    yield event
        except GoogleAPICallError:
            logger.exception("STT 스트리밍 API 에러")
            raise
        except asyncio.CancelledError:
            logger.debug("STT 스트림 취소")
            raise
        finally:
            logger.debug("STT 스트림 종료")

def _to_seconds(duration) -> float | None:
    """시간 객체 초로 바꾸는 함수"""
    if duration is None:
        return None
    if hasattr(duration, "total_seconds"):
        return duration.total_seconds()
    if hasattr(duration, "seconds"):
        nanos = getattr(duration, "nanos", 0)
        return duration.seconds + nanos / 1e9
    return None
