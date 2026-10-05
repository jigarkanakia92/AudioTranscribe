"""NVIDIA Speech NIM and OpenAI-compatible chat-completions clients."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.schemas import ClinicalEntity, SOAPNote, TranscriptSegment
from app.services.prompts import build_note_messages


class ProviderError(RuntimeError):
    """A safe-to-display upstream failure (does not include request content)."""


class ProviderConfigurationError(ProviderError):
    """A model endpoint or credential has not been configured."""


@dataclass(slots=True)
class ASRResult:
    transcript: str
    segments: list[TranscriptSegment] = field(default_factory=list)


ClientFactory = Callable[..., httpx.AsyncClient]


class NVIDIAASRClient:
    """Client for an NVIDIA ASR NIM HTTP endpoint.

    Works with the hosted NIM-compatible API or a self-hosted Speech NIM by
    setting ASR_BASE_URL. Standard NIM HTTP responses are text-only; richer
    compatible endpoints may also return timestamped speaker segments, which
    are preserved when present.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: ClientFactory = httpx.AsyncClient,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory

    async def transcribe(self, audio_wav: bytes, language: str) -> ASRResult:
        if not self.settings.asr_is_configured():
            raise ProviderConfigurationError(
                "Configure ASR_API_KEY or NVIDIA_API_KEY for the hosted NVIDIA endpoint, "
                "or set ASR_BASE_URL to a reachable self-hosted NIM."
            )
        if not audio_wav:
            raise ProviderError("No normalized audio was available for transcription.")

        headers = _auth_headers(self.settings.asr_api_key)
        data = {
            "model": self.settings.asr_model,
            "language": language,
            "response_format": "json",
        }
        try:
            async with self._client_factory(
                timeout=self.settings.request_timeout_seconds
            ) as client:
                response = await client.post(
                    _api_url(self.settings.asr_base_url, "audio/transcriptions"),
                    headers=headers,
                    data=data,
                    files={"file": ("audio.wav", audio_wav, "audio/wav")},
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("The speech transcription service timed out.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Could not connect to the speech transcription service."
            ) from exc

        if response.is_error:
            raise ProviderError(
                f"The speech transcription service returned HTTP {response.status_code}."
            )
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text

        if isinstance(payload, str):
            transcript = payload.strip()
            segments: list[TranscriptSegment] = []
        elif isinstance(payload, dict):
            transcript = str(
                payload.get("text") or payload.get("transcript") or ""
            ).strip()
            segments = _parse_segments(payload)
        else:
            transcript = ""
            segments = []

        if not transcript and segments:
            transcript = " ".join(segment.text for segment in segments).strip()
        if not transcript:
            raise ProviderError(
                "The speech transcription service returned no transcript."
            )
        return ASRResult(transcript=transcript, segments=segments)


class NVIDIAChatClient:
    """Client for NVIDIA NIM's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: ClientFactory = httpx.AsyncClient,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory

    async def generate_note(
        self,
        transcript: str,
        entities: list[ClinicalEntity],
        specialty: str,
    ) -> SOAPNote:
        if not self.settings.llm_is_configured():
            raise ProviderConfigurationError(
                "Configure LLM_API_KEY or NVIDIA_API_KEY for the hosted NVIDIA endpoint, "
                "or set LLM_BASE_URL to a reachable self-hosted NIM."
            )

        body = {
            "model": self.settings.llm_model,
            "messages": build_note_messages(transcript, entities, specialty),
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"},
        }
        try:
            async with self._client_factory(
                timeout=self.settings.request_timeout_seconds
            ) as client:
                response = await client.post(
                    _api_url(self.settings.llm_base_url, "chat/completions"),
                    headers={
                        **_auth_headers(self.settings.llm_api_key),
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("The note-generation service timed out.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Could not connect to the note-generation service."
            ) from exc

        if response.is_error:
            raise ProviderError(
                f"The note-generation service returned HTTP {response.status_code}."
            )
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content was not text")
            note_data = json.loads(_strip_code_fence(content))
            return SOAPNote.model_validate(note_data)
        except (ValueError, KeyError, IndexError, TypeError, ValidationError) as exc:
            raise ProviderError(
                "The note-generation service returned a response outside the required SOAP schema."
            ) from exc


def _api_url(base_url: str, endpoint: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ProviderConfigurationError("The model endpoint URL is not configured.")
    if base.lower().endswith("/v1"):
        return f"{base}/{endpoint.lstrip('/')}"
    return f"{base}/v1/{endpoint.lstrip('/')}"


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    return stripped.strip()


def _parse_segments(payload: dict[str, Any]) -> list[TranscriptSegment]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raw_segments = payload.get("words")
    if not isinstance(raw_segments, list):
        return []

    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("word") or "").strip()
        if not text:
            continue
        speaker_value = (
            item.get("speaker")
            if item.get("speaker") is not None
            else item.get("speaker_id", item.get("speaker_tag"))
        )
        start = _as_float(item.get("start", item.get("start_time")))
        end = _as_float(item.get("end", item.get("end_time")))
        segments.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                speaker=_normalize_speaker(speaker_value),
            )
        )
    return segments


def _normalize_speaker(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return f"Speaker {value + 1}"
    raw = str(value).strip()
    if not raw:
        return None
    match = re.fullmatch(r"(?:speaker[_\s-]*)?(\d+)", raw, re.IGNORECASE)
    if match:
        return f"Speaker {int(match.group(1)) + 1}"
    return raw


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
