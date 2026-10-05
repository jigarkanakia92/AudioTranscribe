# AudioTranscribe

A Docker-ready prototype for drafting a structured SOAP note from a visit recording or a pasted transcript. It combines NVIDIA Speech NIM-compatible transcription, conservative clinical-text heuristics, identifier masking, and an NVIDIA NIM/OpenAI-compatible chat model. A clinician must review and sign every draft.

> **Clinical privacy and safety:** this is a prototype, not a HIPAA-compliant medical product. The hosted NVIDIA API endpoints are external services; do not send identifiable patient audio or text unless your organization has explicitly approved the endpoint and its terms for that data. Text masking is best-effort and does not anonymize audio or guarantee de-identification. The software does not diagnose, prescribe, code, or write to an EHR.

## What is implemented

- Single-page review workspace for an audio upload or a pasted transcript.
- Audio validation and FFmpeg conversion to 16 kHz, mono PCM WAV. Default upload limit is 25 MB and one hour.
- Configurable ASR request to NVIDIA Speech NIM's `POST /v1/audio/transcriptions` interface.
- Configurable LLM request to an OpenAI-compatible `POST /v1/chat/completions` interface, with a constrained SOAP JSON response validated by Pydantic.
- Conservative, literal-span extraction for a small symptom, medication, vital, and allergy vocabulary, with a short-scope negation heuristic. It is a review aid, not medical-grade NER.
- Best-effort direct-identifier masking before transcript text is sent to the LLM. Optional English Presidio NER can be installed in the Docker image.
- Editable Subjective, Objective, Assessment, Plan, and Follow-up sections; copy and local text download.
- Privacy confirmation on each request, request-size limits, sanitized validation errors, security headers, no visit database, and no application-side file retention. Compose mounts `/tmp` as a temporary filesystem.
- API health endpoint: `GET /api/health`.

### Implementation flow

```text
Browser upload / pasted transcript
  → privacy acknowledgement + file/size/language validation
  → FFmpeg/ffprobe normalization (audio uploads only)
  → NVIDIA Speech NIM-compatible ASR (audio uploads only)
  → best-effort text masking + literal clinical-span extraction
  → NVIDIA NIM/OpenAI-compatible chat completion
  → Pydantic SOAP schema validation
  → editable browser draft, copy, or local text download
```

The API processes each encounter in memory for one request and does not create a visit record. Only text is masked before the LLM call; source audio is sent as-is to the configured ASR service.

### Repository layout

```text
app/main.py                 FastAPI endpoints, privacy checks, static UI serving
app/config.py               Environment-backed settings
app/services/audio.py       FFmpeg conversion and upload limits
app/services/nvidia.py      ASR and chat-completion HTTP clients
app/services/clinical_nlp.py  Literal symptom/medication/vital/allergy rules
app/services/phi.py         Regex masking and optional Presidio adapter
app/services/pipeline.py    ASR → spans → SOAP orchestration
app/static/                 Responsive browser workspace
Dockerfile, compose.yaml    Non-root container and hardened local deployment
.env.example                Documented runtime configuration template
tests/                      API, provider, audio, and NLP tests
```

### Scope and limitations

This repository started as an empty shell, so the current app is a functional MVP rather than the entire proposed production architecture. In particular:

- The default ASR HTTP response is text-only. This app does **not** load NeMo Sortformer or run diarization itself. It can preserve speaker-tagged `segments`/`words` returned by a compatible ASR endpoint; otherwise it marks speaker attribution as unverified. Paste `Doctor:` / `Patient:` labels when available and always verify attribution.
- Clinical extraction is a small deterministic ruleset, not scispaCy, medspaCy, negspaCy, or MedCAT. It does not map to UMLS, SNOMED, ICD-10, or CPT.
- There is no drug-reference RAG, medication recommendation, NeMo Guardrails runtime, FHIR/EHR integration, authentication, audit database, or electronic signature flow.
- The Docker Compose stack runs the web app only. NVIDIA model services are configured as external endpoints; model containers and their GPU requirements are not bundled.

## Quick start with Docker Compose

### 1. Prerequisites

- Docker Engine and the Docker Compose v2 plugin.
- An NVIDIA API key for hosted NVIDIA inference, **or** reachable self-hosted ASR and LLM endpoints.
- For audio uploads, the app image includes `ffmpeg` and `ffprobe`.

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set at least `NVIDIA_API_KEY` for the default hosted endpoints. Do not commit `.env`. If using self-hosted services, set `ASR_BASE_URL` and `LLM_BASE_URL` to those services and use their credentials if required. Configuration details are in the [environment variables](#environment-variables) table.

### 3. Build and start

```bash
docker compose up --build -d
docker compose ps
```

Open **http://localhost:8000**. The Compose port is bound to `127.0.0.1` by default; it is not exposed on every host network interface. Check service status and logs with:

```bash
curl http://localhost:8000/api/health
docker compose logs -f app
```

Stop the stack with:

```bash
docker compose down
```

After changing environment variables or Docker build options, recreate the container with `docker compose up --build -d`.

If no hosted key is configured, the UI still loads but reports that AI services need configuration. The health endpoint reports provider readiness without exposing credentials.

## NVIDIA and self-hosted endpoint configuration

The application accepts these interfaces:

- **ASR:** `POST {ASR_BASE_URL}/audio/transcriptions`, multipart `file`, `model`, `language`, and `response_format=json` fields. The default model setting is `nvidia/parakeet-tdt-0.6b-v2`, an English model. Choose a language supported by the model you configure; for multilingual use, select an appropriate Canary or multilingual Parakeet model where available.
- **LLM:** `POST {LLM_BASE_URL}/chat/completions`, using a system/user prompt, low temperature, and JSON response mode. The default model setting is `meta/llama-3.3-70b-instruct`.

Set base URLs with the `/v1` suffix (as in the examples). If a configured URL does not already end in `/v1`, the app appends that path. Model IDs and hosted API availability are controlled by NVIDIA and may change; verify the current NVIDIA API Catalog documentation and terms before use.

For example, to point to NIM services running on the Docker host:

```dotenv
ASR_BASE_URL=http://host.docker.internal:9000/v1
LLM_BASE_URL=http://host.docker.internal:8001/v1
ASR_MODEL=nvidia/parakeet-tdt-0.6b-v2
LLM_MODEL=your-served-chat-model
ASR_API_KEY=
LLM_API_KEY=
```

Compose adds `host.docker.internal` for the Docker host. If the model services run in other containers, connect them to the same Docker network and use their service DNS names instead. The Compose file here does not download or start NIM model containers.

## Using the app

1. Choose **Audio file** or **Paste transcript**.
2. For audio, choose or drop a supported file. The original is normalized in a temporary process and sent to your configured ASR endpoint.
3. Select language and visit context. If available, include `Doctor:` and `Patient:` speaker labels in pasted transcripts.
4. Confirm that you are authorized to process the content and that the configured endpoints are approved for it (or that the content is de-identified).
5. Select **Create note draft**. Review the source transcript, entity hints, speaker attribution, and every SOAP field. Edit the note in the browser, then copy it or download it to your device.

There is no automatic save or EHR write. The privacy checkbox is an explicit acknowledgement, not a consent workflow or compliance control.

## API examples

Transcribe audio and draft a SOAP note in one request:

```bash
curl -X POST http://localhost:8000/api/process \
  -F 'file=@consultation.wav' \
  -F 'language=en-US' \
  -F 'specialty=Family medicine' \
  -F 'privacy_confirmed=true'
```

Draft from an existing transcript:

```bash
curl -X POST http://localhost:8000/api/notes/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "transcript": "Patient: I have had a cough for three days. Doctor: Blood pressure 120/80.",
    "specialty": "Primary care",
    "language": "en-US",
    "privacy_confirmed": true
  }'
```

Check readiness:

```bash
curl http://localhost:8000/api/health
```

The API returns a structured note, transcript, literal entity hints, any speaker segments provided by ASR, and review warnings. Interactive API documentation is disabled in this prototype.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `APP_PORT` | `8000` | Host port published by Compose (loopback only). |
| `APP_ENV` | `production` in Compose | Environment label. |
| `NVIDIA_API_KEY` | empty | Shared NVIDIA API key; required for the default hosted NVIDIA endpoints. |
| `ASR_API_KEY` | falls back to `NVIDIA_API_KEY` | Optional ASR-specific bearer key. |
| `LLM_API_KEY` | falls back to `NVIDIA_API_KEY` | Optional LLM-specific bearer key. |
| `ASR_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA Speech NIM-compatible endpoint or self-hosted ASR URL. |
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA NIM chat endpoint or self-hosted OpenAI-compatible chat URL. |
| `ASR_MODEL` | `nvidia/parakeet-tdt-0.6b-v2` | Speech model ID accepted by the ASR endpoint. |
| `LLM_MODEL` | `meta/llama-3.3-70b-instruct` | Chat model ID accepted by the LLM endpoint. |
| `DEFAULT_LANGUAGE` | `en-US` | Initial language setting. |
| `REQUEST_TIMEOUT_SECONDS` | `180` | Upstream request timeout. |
| `MAX_AUDIO_MB` | `25` | Maximum original audio upload size. |
| `MAX_NORMALIZED_AUDIO_MB` | `128` | Maximum normalized WAV size. |
| `MAX_AUDIO_DURATION_SECONDS` | `3600` | Maximum audio duration. |
| `MAX_TRANSCRIPT_CHARS` | `50000` | Maximum manually submitted transcript length. |
| `PHI_REDACTION_ENABLED` | `true` | Apply regex masking of common direct identifiers to text before the LLM request. This cannot anonymize audio or guarantee de-identification. |
| `USE_PRESIDIO` | `false` | Also use local English Presidio recognizers before LLM use. Requires the optional image build below. |
| `INSTALL_PRESIDIO` | `false` | Docker build argument to install Presidio and `en_core_web_sm`. |

To build the optional English Presidio NLP layer, set both `INSTALL_PRESIDIO=true` and `USE_PRESIDIO=true` in `.env`, then rebuild:

```bash
docker compose up --build -d
```

If Presidio is enabled but unavailable, transcript drafting fails closed rather than sending the transcript unredacted. The regex fallback remains limited, and Presidio is not a substitute for an approved data-handling design.

## Local development and tests

Python 3.11+ and FFmpeg are required for local audio uploads. On Debian/Ubuntu, install FFmpeg with `sudo apt-get install ffmpeg`. If you want live inference, copy `.env.example` to `.env` and set provider keys, then:

```bash
cp .env.example .env
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env
```

Visit **http://localhost:8000**. Use `0.0.0.0` when the app must be reachable through a development preview or container proxy. For normal local use, do not expose this unauthenticated prototype to a public network.

Run the unit and API tests and lint checks with:

```bash
pytest -q
ruff check app tests
ruff format --check app tests
```

The test suite stubs the model endpoints; it does not require an API key, GPU, or live NVIDIA calls.

## Security and deployment notes

- The hosted free/developer endpoint must not be assumed to include a BAA or HIPAA safeguards. Confirm the provider terms, organization approval, and contractual coverage before sending any PHI. For sensitive deployments, use an approved private/self-hosted inference environment and review the whole infrastructure.
- The text redactor runs after ASR. Audio itself is sent to the ASR endpoint and is **not** redacted. Even for text, the default regex rules can miss names and indirect identifiers. Optional Presidio improves coverage but is not a compliance guarantee.
- No authentication, account management, audit trail, database encryption, consent management, or EHR signing is included. Compose publishes the app to loopback to reduce accidental exposure; add TLS, authentication/authorization, network controls, secure logging, retention policies, and an audit process before any shared deployment.
- The clinician remains responsible for checking transcription accuracy, negation, speaker attribution, medication details, and generated note content before signing or copying it into a record.
