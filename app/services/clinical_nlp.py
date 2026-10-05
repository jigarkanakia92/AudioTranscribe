"""Conservative, transparent clinical-text heuristics.

This is deliberately not presented as medical-grade NER or coding. Every item
returned is a literal span from the supplied transcript; uncertain concepts are
left out instead of normalized into a diagnosis.
"""

from __future__ import annotations

import re

from app.schemas import ClinicalEntity


_SPEAKER_PREFIX = re.compile(
    r"^\s*(doctor|clinician|provider|physician|patient|nurse|speaker\s*\d+)\s*:\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

_SYMPTOMS = (
    "shortness of breath",
    "chest pain",
    "abdominal pain",
    "back pain",
    "joint pain",
    "sore throat",
    "headache",
    "dizziness",
    "nausea",
    "vomiting",
    "diarrhea",
    "constipation",
    "fatigue",
    "weakness",
    "fever",
    "cough",
    "wheezing",
    "palpitations",
    "rash",
    "swelling",
    "pain",
)

_MEDICATIONS = (
    "acetaminophen",
    "paracetamol",
    "ibuprofen",
    "aspirin",
    "naproxen",
    "metformin",
    "lisinopril",
    "amoxicillin",
    "atorvastatin",
    "levothyroxine",
    "albuterol",
    "losartan",
    "metoprolol",
    "omeprazole",
    "insulin",
    "sertraline",
    "amlodipine",
    "prednisone",
)

_SYMPTOM_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(term) for term in sorted(_SYMPTOMS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
_MEDICATION_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(term) for term in sorted(_MEDICATIONS, key=len, reverse=True))
    + r")\b(?:\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|units?))?",
    re.IGNORECASE,
)
_VITAL_RE = re.compile(
    r"\b(?:"
    r"(?:blood\s+pressure|BP)\s*[:=]?\s*\d{2,3}\s*/\s*\d{2,3}(?:\s*mm\s*Hg)?|"
    r"(?:temperature|temp)\s*[:=]?\s*\d{2,3}(?:\.\d+)?\s*(?:°\s*[FC]|degrees?\s*[FC])?|"
    r"(?:pulse|heart\s+rate|HR)\s*[:=]?\s*\d{2,3}\s*(?:bpm)?|"
    r"(?:oxygen\s+saturation|SpO2|O2\s+sat)\s*[:=]?\s*\d{1,3}\s*%?"
    r")(?=$|[^A-Za-z0-9_])",
    re.IGNORECASE,
)
_ALLERGY_RE = re.compile(
    r"\b(?:allergic\s+to|allergy\s+to|allergies\s*[:=])\s*"
    r"([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*)?)",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:no|denies?|denied|without|negative\s+for|not\s+(?:experiencing|taking|using|having)|"
    r"does\s+not\s+(?:have|take|use)|never|absence\s+of)\b",
    re.IGNORECASE,
)
_CONTRAST_RE = re.compile(r"\b(?:but|however|although|yet|except)\b", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?;\n]")


def extract_clinical_entities(transcript: str) -> list[ClinicalEntity]:
    """Extract a small set of literal symptoms, medications, vitals, and allergies."""
    found: list[tuple[int, ClinicalEntity]] = []
    offset = 0

    for line in transcript.splitlines() or [transcript]:
        match = _SPEAKER_PREFIX.match(line)
        if match:
            speaker = _canonical_speaker(match.group(1))
            content = match.group(2)
            content_offset = offset + match.start(2)
        else:
            speaker = None
            content = line
            content_offset = offset

        _collect_matches(
            content,
            content_offset,
            speaker,
            _SYMPTOM_RE,
            "symptom",
            found,
        )
        _collect_matches(
            content,
            content_offset,
            speaker,
            _MEDICATION_RE,
            "medication",
            found,
        )
        _collect_matches(
            content,
            content_offset,
            speaker,
            _VITAL_RE,
            "vital",
            found,
        )
        for allergy_match in _ALLERGY_RE.finditer(content):
            start, end = allergy_match.span(1)
            literal = allergy_match.group(1)
            found.append(
                (
                    content_offset + start,
                    ClinicalEntity(
                        text=literal,
                        category="allergy",
                        negated=_is_negated(content, start),
                        speaker=speaker,
                    ),
                )
            )
        offset += len(line) + 1

    found.sort(key=lambda item: item[0])
    unique: list[ClinicalEntity] = []
    seen: set[tuple[str, str, str | None]] = set()
    for _, entity in found:
        key = (entity.category, entity.text.casefold(), entity.speaker)
        if key not in seen:
            seen.add(key)
            unique.append(entity)
    return unique


def has_role_labels(transcript: str) -> bool:
    """True only when the source explicitly labels a patient and/or care team."""
    return bool(
        re.search(
            r"^\s*(?:doctor|clinician|provider|physician|patient|nurse)\s*:",
            transcript,
            re.IGNORECASE | re.MULTILINE,
        )
    )


def has_speaker_labels(transcript: str) -> bool:
    return bool(_SPEAKER_PREFIX.search(transcript))


def _collect_matches(
    text: str,
    offset: int,
    speaker: str | None,
    pattern: re.Pattern[str],
    category: str,
    output: list[tuple[int, ClinicalEntity]],
) -> None:
    for match in pattern.finditer(text):
        start, end = match.span()
        output.append(
            (
                offset + start,
                ClinicalEntity(
                    text=text[start:end].strip(),
                    category=category,  # type: ignore[arg-type]
                    negated=_is_negated(text, start),
                    speaker=speaker,
                ),
            )
        )


def _is_negated(text: str, start: int) -> bool:
    """A short-scope NegEx-like heuristic; it is not a clinical NLP model."""
    left = text[:start]
    sentence_boundaries = list(_SENTENCE_BOUNDARY_RE.finditer(left))
    if sentence_boundaries:
        left = left[sentence_boundaries[-1].end() :]
    if len(left) > 100:
        left = left[-100:]

    contrasts = list(_CONTRAST_RE.finditer(left))
    if contrasts:
        left = left[contrasts[-1].end() :]
    return bool(_NEGATION_RE.search(left))


def _canonical_speaker(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.strip())
    if normalized.lower().startswith("speaker"):
        number = re.search(r"\d+", normalized)
        return f"Speaker {number.group(0)}" if number else "Speaker"
    return normalized.title()
