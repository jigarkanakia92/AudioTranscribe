from app.services.clinical_nlp import (
    extract_clinical_entities,
    has_role_labels,
    has_speaker_labels,
)
from app.services.phi import PHIRedactionError, redact_phi
import app.services.phi as phi_module
import pytest


def test_extracts_literal_entities_with_negation_and_speakers():
    transcript = (
        "Patient: No chest pain, but reports cough and takes metformin 500 mg.\n"
        "Doctor: Blood pressure 120/80 mm Hg. Allergy to penicillin."
    )

    entities = extract_clinical_entities(transcript)

    assert [
        (item.text.lower(), item.category, item.negated, item.speaker)
        for item in entities
    ] == [
        ("chest pain", "symptom", True, "Patient"),
        ("cough", "symptom", False, "Patient"),
        ("metformin 500 mg", "medication", False, "Patient"),
        ("blood pressure 120/80 mm hg", "vital", False, "Doctor"),
        ("penicillin", "allergy", False, "Doctor"),
    ]
    assert has_role_labels(transcript)
    assert has_speaker_labels(transcript)


def test_negation_scope_resets_after_contrast():
    entities = extract_clinical_entities(
        "Patient: Denies fever, cough, or shortness of breath but reports fatigue."
    )
    by_text = {entity.text.lower(): entity for entity in entities}

    assert by_text["fever"].negated is True
    assert by_text["cough"].negated is True
    assert by_text["shortness of breath"].negated is True
    assert by_text["fatigue"].negated is False


def test_extracts_percent_vitals_without_trailing_punctuation():
    entities = extract_clinical_entities("Doctor: SpO2 98%, pulse 72 bpm.")
    assert [(entity.text, entity.category) for entity in entities] == [
        ("SpO2 98%", "vital"),
        ("pulse 72 bpm", "vital"),
    ]


def test_unknown_speaker_is_not_inferred():
    entities = extract_clinical_entities("Chest pain was mentioned.")
    assert entities[0].speaker is None
    assert not has_role_labels("Speaker 1: I have a cough.")
    assert has_speaker_labels("Speaker 1: I have a cough.")


def test_masks_common_direct_identifiers():
    text = "Contact sam@example.com, call 212-555-0119, MRN: ZXCV1234; DOB: 01/02/1980."
    redacted = redact_phi(text)

    assert "sam@example.com" not in redacted
    assert "212-555-0119" not in redacted
    assert "ZXCV1234" not in redacted
    assert "01/02/1980" not in redacted
    assert redacted.count("[REDACTED]") == 4


def test_presidio_requested_rejects_non_english_transcripts():
    with pytest.raises(PHIRedactionError, match="English transcripts only"):
        redact_phi("Paciente: tos.", use_presidio=True, language="es-US")


def test_presidio_requested_fails_closed_if_unavailable(monkeypatch):
    def unavailable():
        raise ImportError("optional dependencies are not installed")

    monkeypatch.setattr(phi_module, "_presidio_engines", unavailable)
    with pytest.raises(PHIRedactionError):
        redact_phi("John Smith has a cough.", use_presidio=True)
