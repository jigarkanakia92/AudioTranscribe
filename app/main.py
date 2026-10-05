"""FastAPI application for the AudioTranscribe clinical scribe prototype."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.schemas import GenerateNoteRequest, ProcessResponse
from app.services.audio import (
    AudioError,
    AudioTooLong,
    normalize_audio,
    validate_audio_filename,
)
from app.services.nvidia import (
    NVIDIAASRClient,
    NVIDIAChatClient,
    ProviderConfigurationError,
    ProviderError,
)
from app.services.phi import PHIRedactionError
from app.services.pipeline import ScribePipeline


STATIC_DIR = Path(__file__).parent / "static"
_LANGUAGE_PATTERN = re.compile(r"^(?:[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*|multi)$")


def create_app(
    settings: Settings | None = None,
    *,
    asr_client: Any | None = None,
    chat_client: Any | None = None,
    audio_normalizer=normalize_audio,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    runtime_asr = asr_client or NVIDIAASRClient(runtime_settings)
    runtime_chat = chat_client or NVIDIAChatClient(runtime_settings)
    pipeline = ScribePipeline(runtime_settings, runtime_asr, runtime_chat)

    application = FastAPI(
        title="AudioTranscribe",
        description="Privacy-conscious audio-to-clinical-note drafting prototype.",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.settings = runtime_settings
    application.state.pipeline = pipeline

    @application.middleware("http")
    async def privacy_and_security_headers(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_size = int(content_length)
            except ValueError:
                request_size = 0
            if request_size > 0:
                max_request_size = runtime_settings.max_audio_bytes + 512 * 1024
                if (
                    request.url.path == "/api/process"
                    and request_size > max_request_size
                ):
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": "The upload exceeds the configured audio size limit."
                        },
                    )
                if (
                    request.url.path == "/api/notes/generate"
                    and request_size
                    > runtime_settings.max_transcript_chars * 4 + 16_384
                ):
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": "The transcript exceeds the configured size limit."
                        },
                    )

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data: blob:; media-src blob:; connect-src 'self'; "
            "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
        )
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(RequestValidationError)
    async def sanitized_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Never echo submitted transcript/audio metadata in validation responses.
        safe_errors = [
            {
                "loc": error.get("loc", []),
                "msg": error.get("msg", "Invalid request."),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    @application.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "AudioTranscribe",
            "providers": {
                "asr_configured": runtime_settings.asr_is_configured(),
                "llm_configured": runtime_settings.llm_is_configured(),
            },
            "server_side_storage": False,
        }

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"}
        )

    @application.post("/api/process", response_model=ProcessResponse)
    async def process_audio(
        file: UploadFile = File(...),
        language: str = Form(default=runtime_settings.default_language),
        specialty: str = Form(default="Primary care"),
        privacy_confirmed: bool = Form(...),
    ) -> ProcessResponse:
        _require_privacy_confirmation(privacy_confirmed)
        _validate_language(language)
        if len(specialty) > 80:
            raise HTTPException(
                status_code=422, detail="Specialty must be 80 characters or fewer."
            )
        try:
            validate_audio_filename(file.filename)
        except AudioError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

        try:
            audio_bytes = await file.read(runtime_settings.max_audio_bytes + 1)
        finally:
            await file.close()
        if not audio_bytes:
            raise HTTPException(
                status_code=422, detail="The uploaded audio file is empty."
            )
        if len(audio_bytes) > runtime_settings.max_audio_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio files must be {runtime_settings.max_audio_bytes // (1024 * 1024)} MB or smaller."
                ),
            )

        try:
            normalized_audio = await asyncio.to_thread(
                audio_normalizer,
                audio_bytes,
                max_output_bytes=runtime_settings.max_normalized_audio_bytes,
                max_duration_seconds=runtime_settings.max_audio_duration_seconds,
            )
        except AudioTooLong as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except AudioError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            return await pipeline.process_audio(
                normalized_audio,
                language=language,
                specialty=specialty,
            )
        except ProviderConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except PHIRedactionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post("/api/notes/generate", response_model=ProcessResponse)
    async def generate_note(request: GenerateNoteRequest) -> ProcessResponse:
        _require_privacy_confirmation(request.privacy_confirmed)
        _validate_language(request.language)
        transcript = request.transcript.strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="Transcript cannot be blank.")
        if len(transcript) > runtime_settings.max_transcript_chars:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Transcripts must be {runtime_settings.max_transcript_chars} characters or shorter."
                ),
            )
        try:
            return await pipeline.process_transcript(
                transcript,
                specialty=request.specialty,
                language=request.language,
            )
        except ProviderConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except PHIRedactionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    application.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR, check_dir=True),
        name="static",
    )
    return application


def _require_privacy_confirmation(confirmed: bool) -> None:
    if not confirmed:
        raise HTTPException(
            status_code=400,
            detail=(
                "Confirm that you are authorized to process this data and that the configured "
                "ASR and LLM endpoints are approved for it."
            ),
        )


def _validate_language(language: str) -> None:
    if not _LANGUAGE_PATTERN.fullmatch(language.strip()):
        raise HTTPException(
            status_code=422, detail="Language must be a valid BCP-47-style code."
        )


app = create_app()
