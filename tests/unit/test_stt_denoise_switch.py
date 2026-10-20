import logging
from types import SimpleNamespace

import pytest
from google.cloud.speech_v2.types import cloud_speech as cs

from app.core.config import Settings, get_settings
from app.services import pipeline as pipeline_module
from app.services.pipeline import VoicePipeline
from app.services.stt import GeminiLiveSTTClient, GoogleSTTClient, build_stt_client, stt_config_fields
from scripts.stt_noise import run_chirp

_RECOGNIZER = "projects/p/locations/asia-northeast1/recognizers/_"


@pytest.fixture(autouse=True)
def _no_grpc(monkeypatch) -> None:
    monkeypatch.setattr("app.services.stt.SpeechAsyncClient", lambda **kw: object())


def _client(**kw) -> GoogleSTTClient:
    return GoogleSTTClient(project_id="p", location="asia-northeast1", model="chirp_3", language="ko-KR", **kw)


def _pre_switch_request() -> cs.StreamingRecognizeRequest:
    """스위치 전 _build_config_request 와 같은 요청"""
    return cs.StreamingRecognizeRequest(
        recognizer=_RECOGNIZER,
        streaming_config=cs.StreamingRecognitionConfig(
            config=cs.RecognitionConfig(
                explicit_decoding_config=cs.ExplicitDecodingConfig(
                    encoding=cs.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    audio_channel_count=1,
                ),
                language_codes=["ko-KR"],
                model="chirp_3",
            ),
            streaming_features=cs.StreamingRecognitionFeatures(
                interim_results=True,
                enable_voice_activity_events=True,
            ),
        ),
    )


def _bytes(request: cs.StreamingRecognizeRequest) -> bytes:
    return cs.StreamingRecognizeRequest.serialize(request)


def _config_pb(client: GoogleSTTClient):
    return cs.StreamingRecognizeRequest.pb(client._build_config_request()).streaming_config.config


def _without_denoiser(client: GoogleSTTClient) -> bytes:
    pb = cs.StreamingRecognizeRequest.pb(client._build_config_request())
    pb.streaming_config.config.ClearField("denoiser_config")
    return pb.SerializeToString()


def _settings(**update) -> Settings:
    base = {"stt_engine": "chirp", "google_project_id": "p", "google_stt_location": "asia-northeast1",
            "google_stt_model": "chirp_3", "google_stt_language": "ko-KR"}
    return get_settings().model_copy(update={**base, **update})


def test_switch_defaults_off() -> None:
    fields = Settings.model_fields
    assert fields["google_stt_denoise"].default is False
    assert fields["google_stt_denoise_snr_threshold"].default == 0.0


def test_default_off_request_is_byte_identical_to_pre_switch() -> None:
    expected = _bytes(_pre_switch_request())
    assert _bytes(_client()._build_config_request()) == expected
    assert _bytes(_client(denoise=False, denoise_snr_threshold=9.0)._build_config_request()) == expected
    assert not _config_pb(_client()).HasField("denoiser_config")


def test_denoise_on_adds_only_denoiser_config_with_threshold() -> None:
    on = _client(denoise=True, denoise_snr_threshold=5.0)
    pb = _config_pb(on)
    assert pb.HasField("denoiser_config")
    assert pb.denoiser_config.denoise_audio is True
    assert pb.denoiser_config.snr_threshold == pytest.approx(5.0)
    assert _without_denoiser(on) == _bytes(_pre_switch_request())


def test_factory_passes_switch_from_settings() -> None:
    on = build_stt_client(_settings(google_stt_denoise=True, google_stt_denoise_snr_threshold=5.0))
    assert isinstance(on, GoogleSTTClient)
    assert _config_pb(on).denoiser_config.snr_threshold == pytest.approx(5.0)
    assert _without_denoiser(on) == _bytes(_pre_switch_request())
    off = build_stt_client(_settings(google_stt_denoise=False, google_stt_denoise_snr_threshold=5.0))
    assert _bytes(off._build_config_request()) == _bytes(_pre_switch_request())


def test_gemini_live_ignores_switch() -> None:
    on = build_stt_client(_settings(stt_engine="gemini_live", gemini_stt_model="m-x", google_stt_denoise=True))
    off = build_stt_client(_settings(stt_engine="gemini_live", gemini_stt_model="m-x", google_stt_denoise=False))
    assert isinstance(on, GeminiLiveSTTClient)
    assert on._build_config().model_dump() == off._build_config().model_dump()
    assert stt_config_fields(on) == stt_config_fields(off) == {"engine": "gemini_live", "model": "m-x"}


def test_config_fields_describe_chirp_switch() -> None:
    assert stt_config_fields(_client(denoise=True, denoise_snr_threshold=5.0)) == {
        "engine": "chirp", "model": "chirp_3", "location": "asia-northeast1", "denoise": True, "snr_threshold": 5.0,
    }
    assert stt_config_fields(_client(denoise_snr_threshold=5.0)) == {
        "engine": "chirp", "model": "chirp_3", "location": "asia-northeast1", "denoise": False, "snr_threshold": None,
    }
    other = GoogleSTTClient(project_id="p", location="us", model="chirp_2", language="ko-KR", denoise=True)
    assert stt_config_fields(other) == {
        "engine": "chirp", "model": "chirp_2", "location": "us", "denoise": True, "snr_threshold": 0.0,
    }
    assert stt_config_fields(object()) == {}


def _pipeline(monkeypatch, settings) -> VoicePipeline:
    monkeypatch.setattr(pipeline_module, "LLMClient", lambda: object())
    monkeypatch.setattr(pipeline_module, "RecordingStorageService", lambda s: object())
    return VoicePipeline(ws=object(), session_id="s-1", session={}, settings=settings, spring=object())


def _stt_config_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "app.metrics" and r.metric == "stt_config"]


@pytest.mark.parametrize(
    ("denoise", "tail"),
    [(True, "denoise=True snr_threshold=5.0"), (False, "denoise=False snr_threshold=None")],
)
def test_pipeline_logs_one_stt_config_line_per_session(monkeypatch, caplog, denoise, tail) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    settings = _settings(google_stt_location="us", google_stt_model="chirp_2",
                         google_stt_denoise=denoise, google_stt_denoise_snr_threshold=5.0)
    _pipeline(monkeypatch, settings)
    assert _stt_config_lines(caplog) == [
        f"metric=stt_config session_id=s-1 engine=chirp model=chirp_2 location=us {tail}"
    ]


def test_pipeline_logs_gemini_live_engine_and_model(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    _pipeline(monkeypatch, _settings(stt_engine="gemini_live", gemini_stt_model="m-x", google_stt_denoise=True))
    assert _stt_config_lines(caplog) == ["metric=stt_config session_id=s-1 engine=gemini_live model=m-x"]


def test_pipeline_log_survives_unknown_stt_client(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    monkeypatch.setattr(pipeline_module, "build_stt_client", lambda settings: object())
    _pipeline(monkeypatch, _settings())
    assert [line.strip() for line in _stt_config_lines(caplog)] == ["metric=stt_config session_id=s-1"]


def _bench_settings(**update) -> SimpleNamespace:
    base = {"stt_engine": "chirp", "google_project_id": "p", "google_stt_location": "asia-northeast1",
            "google_stt_model": "chirp_3", "google_stt_language": "ko-KR",
            "google_stt_denoise": True, "google_stt_denoise_snr_threshold": 7.0}
    return SimpleNamespace(**{**base, **update})


@pytest.mark.parametrize("make", [_bench_settings, lambda: _settings(google_stt_denoise=True,
                                                                      google_stt_denoise_snr_threshold=7.0)])
def test_bench_off_ignores_switch_and_on_uses_bench_threshold(make) -> None:
    settings = make()
    clients = run_chirp.make_clients(settings, [False, True], 3.0)
    off, on = clients[False], clients[True]
    assert type(off) is GoogleSTTClient
    assert type(on) is run_chirp.DenoiseSTTClient
    assert _bytes(off._build_config_request()) == _bytes(_pre_switch_request())
    sent = _config_pb(on).denoiser_config.snr_threshold
    assert on.snr_threshold == sent == pytest.approx(3.0)
    job = run_chirp.Job(clip={}, denoise=True, rep=1)
    assert run_chirp._clip_row(job, 100, on)["denoise_snr_threshold"] == sent
    assert _without_denoiser(on) == _bytes(_pre_switch_request())
    assert (settings.google_stt_denoise, settings.google_stt_denoise_snr_threshold) == (True, 7.0)


def test_bench_on_matches_operational_switch_on() -> None:
    on = run_chirp.make_clients(_bench_settings(), [True], 3.0)[True]
    prod = build_stt_client(_settings(google_stt_denoise=True, google_stt_denoise_snr_threshold=3.0))
    assert _bytes(on._build_config_request()) == _bytes(prod._build_config_request())
