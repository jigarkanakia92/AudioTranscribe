"""Validated request and response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClinicalEntity(StrictModel):
    text: str
    category: Literal["symptom", "medication", "vital", "allergy"]
    negated: bool = False
    speaker: str | None = None


class TranscriptSegment(StrictModel):
    start: float | None = None
    end: float | None = None
    text: str
    speaker: str | None = None


class SOAPNote(StrictModel):
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    follow_up: list[str] = Field(default_factory=list)


class GenerateNoteRequest(StrictModel):
    transcript: str = Field(min_length=1, max_length=50_000)
    specialty: str = Field(default="Primary care", max_length=80)
    language: str = Field(default="en-US", max_length=20)
    privacy_confirmed: bool


class ProcessResponse(StrictModel):
    source: Literal["audio", "transcript"]
    transcript: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    note: SOAPNote
    entities: list[ClinicalEntity] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    speaker_labels_available: bool = False
