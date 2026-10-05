"""Orchestration for audio-to-draft and transcript-to-draft workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from app.config import Settings
from app.schemas import ClinicalEntity, ProcessResponse, TranscriptSegment
from app.services.clinical_nlp import (
    extract_clinical_entities,
    has_speaker_labels,
)
from app.services.nvidia import ASRResult, NVIDIAASRClient, NVIDIAChatClient
from app.services.phi import redact_phi


class ScribePipeline:
    def __init__(
        self,
        settings: Settings,
        asr_client: NVIDIAASRClient,
        chat_client: NVIDIAChatClient,
    ) -> None:
        self.settings = settings
        self.asr_client = asr_client
        self.chat_client = chat_client

    async def process_audio(
        self,
        normalized_audio: bytes,
        *,
        language: str,
        specialty: str,
    ) -> ProcessResponse:
        asr_result = await self.asr_client.transcribe(normalized_audio, language)
        transcript = _transcript_from_asr(asr_result)
        return await self.process_transcript(
            transcript,
            specialty=specialty,
            language=language,
            source="audio",
            segments=asr_result.segments,
        )

    async def process_transcript(
        self,
        transcript: str,
        *,
        specialty: str,
        language: str | None = None,
        source: Literal["audio", "transcript"] = "transcript",
        segments: Sequence[TranscriptSegment] = (),
    ) -> ProcessResponse:
        # Run rule extraction over the text that will actually be sent to the LLM,
        # so a masked direct identifier is not accidentally reintroduced as a span.
        llm_transcript = (
            redact_phi(
                transcript,
                use_presidio=self.settings.use_presidio,
                language=language or self.settings.default_language,
            )
            if self.settings.phi_redaction_enabled
            else transcript
        )
        entities: list[ClinicalEntity] = extract_clinical_entities(llm_transcript)
        note = await self.chat_client.generate_note(
            llm_transcript,
            entities,
            specialty.strip() or "Primary care",
        )

        speaker_labels_available = bool(
            has_speaker_labels(transcript)
            or any(segment.speaker for segment in segments)
        )
        warnings = [
            "AI-generated draft only. A licensed clinician must verify, edit, and sign before use.",
            "Rule-based clinical spans are limited and are not a diagnosis, coding result, or safety check.",
            "Speaker roles are not verified. Confirm who said what before relying on attribution.",
        ]
        if self.settings.phi_redaction_enabled:
            warnings.append(
                "Text identifier masking is best-effort and does not anonymize audio or guarantee de-identification."
            )
        return ProcessResponse(
            source=source,
            transcript=transcript,
            segments=list(segments),
            note=note,
            entities=entities,
            warnings=warnings,
            speaker_labels_available=speaker_labels_available,
        )


def _transcript_from_asr(result: ASRResult) -> str:
    """Render diarized segments when every segment has a speaker label.

    Segment-aware ASR providers differ in their response shape. Only use this
    reconstruction when the response gives speaker tags for every segment;
    otherwise retain the provider's transcript instead of guessing alignment.
    """
    if not result.segments or not all(segment.speaker for segment in result.segments):
        return result.transcript

    turns: list[tuple[str, list[str]]] = []
    for segment in result.segments:
        speaker = segment.speaker or "Unknown"
        if turns and turns[-1][0] == speaker:
            turns[-1][1].append(segment.text)
        else:
            turns.append((speaker, [segment.text]))
    return "\n".join(
        f"{speaker}: {' '.join(words).strip()}" for speaker, words in turns
    )
