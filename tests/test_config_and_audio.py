import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services import audio
from app.services.audio import (
    AudioError,
    AudioTooLong,
    normalize_audio,
    validate_audio_filename,
)


def test_settings_load_service_specific_credentials_and_limits(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "shared")
    monkeypatch.setenv("ASR_API_KEY", "asr-only")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("MAX_AUDIO_MB", "12")
    monkeypatch.setenv("USE_PRESIDIO", "false")

    settings = Settings.from_env()

    assert settings.asr_api_key == "asr-only"
    assert settings.llm_api_key == "shared"
    assert settings.max_audio_bytes == 12 * 1024 * 1024
    assert settings.use_presidio is False


def test_settings_reject_invalid_positive_limits(monkeypatch):
    monkeypatch.setenv("MAX_AUDIO_MB", "-1")
    with pytest.raises(ValueError, match="positive integer"):
        Settings.from_env()


def test_audio_filename_allowlist():
    assert validate_audio_filename("consultation.WAV") == ".wav"
    assert validate_audio_filename("consultation.webm") == ".webm"
    with pytest.raises(AudioError, match="Unsupported audio format"):
        validate_audio_filename("notes.txt")


def test_normalizer_requires_ffmpeg(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda _command: None)
    with pytest.raises(AudioError, match="ffmpeg and ffprobe are required"):
        normalize_audio(b"audio")


def test_normalizer_probes_and_converts_to_expected_format(monkeypatch):
    monkeypatch.setattr(
        audio.shutil,
        "which",
        lambda command: f"/usr/bin/{command}",
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "ffprobe" in command[0]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"format": {"duration": "2.0"}}).encode(),
                stderr=b"",
            )
        return SimpleNamespace(returncode=0, stdout=b"RIFFnormalized", stderr=b"")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    normalized = normalize_audio(
        b"source", max_output_bytes=200_000, max_duration_seconds=30
    )

    assert normalized == b"RIFFnormalized"
    assert len(commands) == 2
    assert "-protocol_whitelist" in commands[1]
    assert commands[1][commands[1].index("-ac") + 1] == "1"
    assert commands[1][commands[1].index("-ar") + 1] == "16000"


def test_normalizer_rejects_duration_before_conversion(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda command: f"/usr/bin/{command}")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "31"}}).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    with pytest.raises(AudioTooLong):
        normalize_audio(b"source", max_output_bytes=1024, max_duration_seconds=30)
    assert len(calls) == 1
