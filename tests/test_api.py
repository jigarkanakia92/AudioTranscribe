from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import SOAPNote
from app.services.audio import AudioError
from app.services.nvidia import ASRResult


class FakeASR:
    def __init__(self, result=None, error=None):
        self.result = result or ASRResult(
            transcript="Patient: No chest pain. Doctor: Blood pressure 120/80 mm Hg."
        )
        self.error = error
        self.calls = []

    async def transcribe(self, audio, language):
        self.calls.append((audio, language))
        if self.error:
            raise self.error
        return self.result


class FakeChat:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def generate_note(self, transcript, entities, specialty):
        self.calls.append((transcript, entities, specialty))
        if self.error:
            raise self.error
        return SOAPNote(
            subjective="Patient denies chest pain.",
            objective="Blood pressure 120/80 mm Hg.",
            assessment="",
            plan="",
            follow_up=[],
        )


def make_test_client(*, settings=None, asr=None, chat=None, normalizer=None):
    runtime_settings = settings or Settings(
        asr_base_url="http://asr.test/v1",
        llm_base_url="http://llm.test/v1",
        max_audio_bytes=10,
        max_normalized_audio_bytes=20,
        max_audio_duration_seconds=60,
        max_transcript_chars=200,
        phi_redaction_enabled=True,
    )
    asr_client = asr or FakeASR()
    chat_client = chat or FakeChat()

    def fake_normalizer(data, *, max_output_bytes, max_duration_seconds):
        if normalizer:
            return normalizer(data, max_output_bytes, max_duration_seconds)
        return b"normalized wav"

    app = create_app(
        runtime_settings,
        asr_client=asr_client,
        chat_client=chat_client,
        audio_normalizer=fake_normalizer,
    )
    return TestClient(app), asr_client, chat_client


def test_health_and_static_app_have_privacy_headers():
    client, _, _ = make_test_client()

    health = client.get("/api/health")
    page = client.get("/")

    assert health.status_code == 200
    assert health.json()["providers"] == {
        "asr_configured": True,
        "llm_configured": True,
    }
    assert health.json()["server_side_storage"] is False
    assert health.headers["cache-control"] == "no-store"
    assert page.status_code == 200
    assert "SOAP note draft" in page.text
    assert page.headers["content-security-policy"].startswith("default-src 'self'")
    assert client.get("/static/app.js").status_code == 200


def test_audio_request_runs_normalization_asr_extraction_and_drafting():
    asr = FakeASR()
    chat = FakeChat()
    client, _, _ = make_test_client(asr=asr, chat=chat)

    response = client.post(
        "/api/process",
        data={
            "language": "en-US",
            "specialty": "Family medicine",
            "privacy_confirmed": "true",
        },
        files={"file": ("visit.wav", b"fake audio", "audio/wav")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["source"] == "audio"
    assert payload["transcript"].startswith("Patient:")
    assert payload["note"]["subjective"] == "Patient denies chest pain."
    assert payload["speaker_labels_available"] is True
    assert len(payload["entities"]) == 2
    assert payload["entities"][0]["negated"] is True
    assert len(payload["warnings"]) >= 3
    assert asr.calls == [(b"normalized wav", "en-US")]
    assert chat.calls[0][2] == "Family medicine"
    assert "Blood pressure 120/80 mm Hg." in chat.calls[0][0]


def test_audio_request_requires_privacy_confirmation_before_processing():
    asr = FakeASR()
    client, _, _ = make_test_client(asr=asr)

    response = client.post(
        "/api/process",
        data={
            "language": "en-US",
            "specialty": "Primary care",
            "privacy_confirmed": "false",
        },
        files={"file": ("visit.wav", b"fake audio", "audio/wav")},
    )

    assert response.status_code == 400
    assert "approved" in response.json()["detail"]
    assert asr.calls == []


def test_audio_request_rejects_unsupported_extension_and_oversized_content():
    client, _, _ = make_test_client()
    fields = {
        "language": "en-US",
        "specialty": "Primary care",
        "privacy_confirmed": "true",
    }

    unsupported = client.post(
        "/api/process",
        data=fields,
        files={"file": ("visit.txt", b"audio", "text/plain")},
    )
    oversized = client.post(
        "/api/process",
        data=fields,
        files={"file": ("visit.wav", b"01234567890", "audio/wav")},
    )

    assert unsupported.status_code == 415
    assert oversized.status_code == 413


def test_audio_conversion_error_is_a_client_error_without_calling_asr():
    asr = FakeASR()

    def fail_conversion(_data, _max_output, _max_duration):
        raise AudioError("Audio conversion failed safely.")

    client, _, _ = make_test_client(asr=asr, normalizer=fail_conversion)
    response = client.post(
        "/api/process",
        data={
            "language": "en-US",
            "specialty": "Primary care",
            "privacy_confirmed": "true",
        },
        files={"file": ("visit.wav", b"fake audio", "audio/wav")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Audio conversion failed safely."
    assert asr.calls == []


def test_transcript_endpoint_masks_identifiers_before_llm_and_returns_note():
    chat = FakeChat()
    client, _, _ = make_test_client(chat=chat)
    transcript = "Patient: Contact alex@example.com or 212-555-0119. I have a cough."

    response = client.post(
        "/api/notes/generate",
        json={
            "transcript": transcript,
            "specialty": "Primary care",
            "language": "en-US",
            "privacy_confirmed": True,
        },
    )

    assert response.status_code == 200, response.text
    assert "alex@example.com" not in chat.calls[0][0]
    assert "212-555-0119" not in chat.calls[0][0]
    assert "[REDACTED]" in chat.calls[0][0]
    assert response.json()["source"] == "transcript"
    assert response.json()["note"]["follow_up"] == []


def test_transcript_request_rejects_blank_unconfirmed_and_invalid_language():
    client, _, chat = make_test_client()
    base = {"specialty": "Primary care", "language": "en-US", "privacy_confirmed": True}

    blank = client.post("/api/notes/generate", json={**base, "transcript": "   "})
    unconfirmed = client.post(
        "/api/notes/generate",
        json={**base, "transcript": "Patient: cough", "privacy_confirmed": False},
    )
    invalid_language = client.post(
        "/api/notes/generate",
        json={**base, "transcript": "Patient: cough", "language": "not a language"},
    )

    assert blank.status_code == 422
    assert unconfirmed.status_code == 400
    assert invalid_language.status_code == 422
    assert chat.calls == []


def test_validation_errors_do_not_echo_large_transcript_values():
    settings = Settings(
        asr_base_url="http://asr.test/v1",
        llm_base_url="http://llm.test/v1",
        max_transcript_chars=50_000,
    )
    client, _, _ = make_test_client(settings=settings)
    marker = "SENSITIVE-TRANSCRIPT-MARKER"
    transcript = marker + (" x" * 25_100)

    response = client.post(
        "/api/notes/generate",
        json={"transcript": transcript, "privacy_confirmed": True},
    )

    assert response.status_code == 422
    assert marker not in response.text


def test_missing_hosted_api_key_is_reported_without_calling_provider():
    settings = Settings(
        asr_base_url="https://integrate.api.nvidia.com/v1",
        llm_base_url="http://llm.test/v1",
        max_audio_bytes=10,
        max_normalized_audio_bytes=20,
        max_audio_duration_seconds=60,
    )
    # Use the real ASR client to exercise the hosted-key configuration check.
    app = create_app(
        settings,
        chat_client=FakeChat(),
        audio_normalizer=lambda *args, **kwargs: b"wav",
    )
    client = TestClient(app)

    response = client.post(
        "/api/process",
        data={
            "language": "en-US",
            "specialty": "Primary care",
            "privacy_confirmed": "true",
        },
        files={"file": ("visit.wav", b"fake audio", "audio/wav")},
    )

    assert response.status_code == 503
    assert "ASR_API_KEY" in response.json()["detail"]
