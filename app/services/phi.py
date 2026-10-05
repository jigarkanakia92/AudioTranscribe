"""Best-effort masking of direct identifiers before text reaches the LLM.

This is a defense-in-depth helper, not a HIPAA de-identification guarantee.
Audio sent to an ASR service cannot be anonymized by this text-only function.
"""

from __future__ import annotations

import re
from functools import lru_cache


class PHIRedactionError(RuntimeError):
    """Requested Presidio processing is unavailable; fail closed before LLM use."""


# These patterns intentionally focus on common direct identifiers. They cannot
# reliably identify names, indirect identifiers, or every regional format.
_DIRECT_IDENTIFIER_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}\b"),
    re.compile(
        r"\b(?:MRN|medical\s+record(?:\s+number)?|patient\s+ID)"
        r"\s*[:#]?\s*[A-Z0-9-]{4,}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:DOB|date\s+of\s+birth)\s*[:#]?\s*"
        r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
        r"\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
)


def redact_phi(
    text: str, *, use_presidio: bool = False, language: str = "en-US"
) -> str:
    """Mask common direct identifiers and, optionally, run Presidio NER first.

    Presidio is used for English transcripts only. When explicitly enabled but
    unavailable, this function raises rather than silently sending unredacted
    text onward. Regex masking is always applied as a final pass.
    """
    result = text
    if use_presidio:
        if not language.lower().startswith("en"):
            raise PHIRedactionError(
                "Presidio is configured for English transcripts only. "
                "Use an English transcript or disable Presidio only after reviewing your data-handling requirements."
            )
        try:
            analyzer, anonymizer = _presidio_engines()
            findings = analyzer.analyze(text=result, language="en")
            if findings:
                result = anonymizer.anonymize(
                    text=result,
                    analyzer_results=findings,
                    operators={
                        "DEFAULT": {"type": "replace", "new_value": "[REDACTED]"}
                    },
                ).text
        except Exception as exc:  # Fail closed: do not send this transcript to the LLM.
            raise PHIRedactionError(
                "Presidio is enabled but could not redact this transcript. "
                "Install the optional Presidio dependencies and English spaCy model."
            ) from exc

    for pattern in _DIRECT_IDENTIFIER_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


@lru_cache(maxsize=1)
def _presidio_engines():
    """Create a small local English Presidio pipeline on first use."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )
    return analyzer, AnonymizerEngine()
