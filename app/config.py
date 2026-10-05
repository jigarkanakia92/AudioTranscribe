"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration; secrets are read only from the process environment."""

    app_env: str = "development"
    asr_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    asr_model: str = "nvidia/parakeet-tdt-0.6b-v2"
    llm_model: str = "meta/llama-3.3-70b-instruct"
    default_language: str = "en-US"
    request_timeout_seconds: float = 180.0
    max_audio_bytes: int = 25 * 1024 * 1024
    max_normalized_audio_bytes: int = 128 * 1024 * 1024
    max_audio_duration_seconds: int = 60 * 60
    max_transcript_chars: int = 50_000
    asr_api_key: str = field(default="", repr=False)
    llm_api_key: str = field(default="", repr=False)
    use_presidio: bool = False
    phi_redaction_enabled: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings, preferring service-specific keys over the shared NVIDIA key."""
        common_key = os.getenv("NVIDIA_API_KEY", "").strip()
        max_transcript_chars = _positive_int("MAX_TRANSCRIPT_CHARS", 50_000)
        if max_transcript_chars > 50_000:
            raise ValueError("MAX_TRANSCRIPT_CHARS cannot exceed the hard cap of 50000")
        return cls(
            app_env=os.getenv("APP_ENV", "development").strip() or "development",
            asr_base_url=os.getenv(
                "ASR_BASE_URL", "https://integrate.api.nvidia.com/v1"
            )
            .strip()
            .rstrip("/"),
            llm_base_url=os.getenv(
                "LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"
            )
            .strip()
            .rstrip("/"),
            asr_model=os.getenv("ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v2").strip(),
            llm_model=os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct").strip(),
            default_language=os.getenv("DEFAULT_LANGUAGE", "en-US").strip(),
            request_timeout_seconds=_positive_float("REQUEST_TIMEOUT_SECONDS", 180.0),
            max_audio_bytes=_positive_int("MAX_AUDIO_MB", 25) * 1024 * 1024,
            max_normalized_audio_bytes=(
                _positive_int("MAX_NORMALIZED_AUDIO_MB", 128) * 1024 * 1024
            ),
            max_audio_duration_seconds=_positive_int(
                "MAX_AUDIO_DURATION_SECONDS", 3600
            ),
            max_transcript_chars=max_transcript_chars,
            asr_api_key=os.getenv("ASR_API_KEY", "").strip() or common_key,
            llm_api_key=os.getenv("LLM_API_KEY", "").strip() or common_key,
            use_presidio=_bool_env("USE_PRESIDIO", default=False),
            phi_redaction_enabled=_bool_env("PHI_REDACTION_ENABLED", default=True),
        )

    def asr_is_configured(self) -> bool:
        return _provider_is_configured(self.asr_base_url, self.asr_api_key)

    def llm_is_configured(self) -> bool:
        return _provider_is_configured(self.llm_base_url, self.llm_api_key)


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _provider_is_configured(base_url: str, api_key: str) -> bool:
    """Local NIM deployments may be unauthenticated; the NVIDIA hosted API may not."""
    if not base_url:
        return False
    try:
        parsed_url = urlsplit(base_url)
    except ValueError:
        return False
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        return False
    hostname = parsed_url.hostname.lower()
    if hostname == "integrate.api.nvidia.com":
        return bool(api_key)
    return True
