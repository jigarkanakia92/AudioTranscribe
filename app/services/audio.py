"""Audio normalization for the ASR HTTP API.

The original upload is held only for the duration of the request. FFmpeg reads
from stdin and writes a normalized 16 kHz mono WAV to stdout; no source audio
is saved into the application tree.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import PurePath


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
    ".aac",
}


class AudioError(ValueError):
    """The uploaded audio cannot be safely processed."""


class AudioTooLong(AudioError):
    """The recording exceeds the configured duration limit."""


def validate_audio_filename(filename: str | None) -> str:
    """Return the lower-case extension or reject files outside the allowlist."""
    extension = PurePath(filename or "").suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise AudioError(f"Unsupported audio format. Supported formats: {allowed}.")
    return extension


def normalize_audio(
    audio_bytes: bytes,
    *,
    max_output_bytes: int = 128 * 1024 * 1024,
    max_duration_seconds: int = 3600,
) -> bytes:
    """Convert audio to PCM S16LE, 16 kHz, mono WAV using bounded FFmpeg calls."""
    if not audio_bytes:
        raise AudioError("The uploaded audio file is empty.")
    if max_output_bytes <= 0 or max_duration_seconds <= 0:
        raise ValueError("Audio limits must be positive")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise AudioError(
            "Audio conversion is unavailable: ffmpeg and ffprobe are required."
        )

    probe_command = [
        ffprobe,
        "-v",
        "error",
        "-protocol_whitelist",
        "pipe",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        "-i",
        "pipe:0",
    ]
    try:
        probe = subprocess.run(
            probe_command,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Audio inspection timed out.") from exc
    except OSError as exc:
        raise AudioError("Audio inspection could not be started.") from exc

    if probe.returncode != 0:
        raise AudioError("The file is not a supported or readable audio recording.")
    try:
        probe_result = json.loads(probe.stdout)
        duration = float(probe_result["format"]["duration"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise AudioError("The recording duration could not be determined.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise AudioError("The uploaded audio contains no playable duration.")
    if duration > max_duration_seconds:
        raise AudioTooLong(
            f"Recordings must be {max_duration_seconds // 60} minutes or shorter."
        )
    estimated_output_bytes = int(duration * 16_000 * 2) + 64 * 1024
    if estimated_output_bytes > max_output_bytes:
        raise AudioTooLong(
            "The normalized recording exceeds the configured size limit."
        )

    convert_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-protocol_whitelist",
        "pipe",
        "-i",
        "pipe:0",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-threads",
        "1",
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        converted = subprocess.run(
            convert_command,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(30, min(180, int(duration * 2 + 15))),
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Audio conversion timed out.") from exc
    except OSError as exc:
        raise AudioError("Audio conversion could not be started.") from exc

    if converted.returncode != 0:
        raise AudioError("The audio could not be normalized for transcription.")
    if not converted.stdout:
        raise AudioError("The uploaded audio contains no decodable audio stream.")
    if len(converted.stdout) > max_output_bytes:
        raise AudioTooLong(
            "The normalized recording exceeds the configured size limit."
        )
    return converted.stdout
