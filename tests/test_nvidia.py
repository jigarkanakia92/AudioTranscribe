import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.services.nvidia import (
    NVIDIAASRClient,
    NVIDIAChatClient,
    ProviderConfigurationError,
    ProviderError,
    _api_url,
)


def mock_client_factory(handler):
    transport = httpx.MockTransport(handler)
    return lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs)


def test_api_url_adds_or_preserves_v1_prefix():
    assert _api_url("https://example.test", "chat/completions") == (
        "https://example.test/v1/chat/completions"
    )
    assert _api_url("https://example.test/v1/", "/audio/transcriptions") == (
        "https://example.test/v1/audio/transcriptions"
    )


def test_asr_client_posts_normalized_audio_and_preserves_diarization_segments():
    async def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "https://asr.test/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-asr-key"
        body = await request.aread()
        assert b'name="language"' in body
        assert b"en-US" in body
        assert b"audio.wav" in body
        return httpx.Response(
            200,
            json={
                "text": "Hello there.",
                "segments": [
                    {"start": 0, "end": 0.7, "text": "Hello", "speaker_tag": 0},
                    {"start": 0.7, "end": 1.2, "text": "there.", "speaker_tag": 0},
                ],
            },
        )

    settings = Settings(
        asr_base_url="https://asr.test/v1",
        llm_base_url="http://llm.test/v1",
        asr_api_key="test-asr-key",
    )
    client = NVIDIAASRClient(settings, client_factory=mock_client_factory(handler))

    result = asyncio.run(client.transcribe(b"RIFF fake wav", "en-US"))

    assert result.transcript == "Hello there."
    assert result.segments[0].speaker == "Speaker 1"
    assert result.segments[1].start == 0.7


def test_asr_client_accepts_plain_text_and_requires_key_for_hosted_api():
    async def handler(_request):
        return httpx.Response(200, text="A plain transcript.")

    settings = Settings(
        asr_base_url="https://integrate.api.nvidia.com/v1",
        asr_api_key="hosted-key",
    )
    client = NVIDIAASRClient(settings, client_factory=mock_client_factory(handler))
    result = asyncio.run(client.transcribe(b"normalized audio", "en-US"))
    assert result.transcript == "A plain transcript."

    unconfigured = NVIDIAASRClient(
        Settings(asr_base_url="https://integrate.api.nvidia.com/v1")
    )
    with pytest.raises(ProviderConfigurationError):
        asyncio.run(unconfigured.transcribe(b"audio", "en-US"))


def test_asr_upstream_error_does_not_return_response_body():
    async def handler(_request):
        return httpx.Response(429, text="secret transcript must not be surfaced")

    client = NVIDIAASRClient(
        Settings(asr_base_url="http://asr.test/v1"),
        client_factory=mock_client_factory(handler),
    )
    with pytest.raises(ProviderError) as error:
        asyncio.run(client.transcribe(b"audio", "en-US"))
    assert "429" in str(error.value)
    assert "secret transcript" not in str(error.value)


def test_chat_client_sends_grounded_json_prompt_and_validates_soap_schema():
    expected_note = {
        "subjective": "Patient reports a cough.",
        "objective": "",
        "assessment": "",
        "plan": "",
        "follow_up": [],
    }

    async def handler(request):
        assert str(request.url) == "https://llm.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-llm-key"
        request_body = json.loads(request.content)
        assert request_body["response_format"] == {"type": "json_object"}
        assert "Do not infer missing facts" in request_body["messages"][0]["content"]
        assert "Patient: I have a cough." in request_body["messages"][1]["content"]
        assert request_body["model"] == "meta/llama-3.3-70b-instruct"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(expected_note)}}]},
        )

    settings = Settings(
        asr_base_url="http://asr.test/v1",
        llm_base_url="https://llm.test/v1",
        llm_model="meta/llama-3.3-70b-instruct",
        llm_api_key="test-llm-key",
    )
    client = NVIDIAChatClient(settings, client_factory=mock_client_factory(handler))
    note = asyncio.run(
        client.generate_note("Patient: I have a cough.", [], "Primary care")
    )

    assert note.subjective == expected_note["subjective"]
    assert note.follow_up == []


def test_chat_client_rejects_non_soap_json():
    async def handler(_request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"diagnosis":"pneumonia"}'}}]},
        )

    client = NVIDIAChatClient(
        Settings(llm_base_url="http://llm.test/v1"),
        client_factory=mock_client_factory(handler),
    )
    with pytest.raises(ProviderError, match="SOAP schema"):
        asyncio.run(client.generate_note("Cough", [], "Primary care"))
