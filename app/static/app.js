const MAX_AUDIO_BYTES = 25 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = new Set(["wav", "mp3", "m4a", "flac", "ogg", "opus", "webm", "aac"]);

const audioTab = document.getElementById("audio-tab");
const transcriptTab = document.getElementById("transcript-tab");
const audioPanel = document.getElementById("audio-panel");
const transcriptPanel = document.getElementById("transcript-panel");
const fileInput = document.getElementById("audio-file");
const dropzone = document.getElementById("dropzone");
const fileCard = document.getElementById("file-card");
const audioPreview = document.getElementById("audio-preview");
const generateButton = document.getElementById("generate-button");
const errorMessage = document.getElementById("error-message");
const toast = document.getElementById("toast");

let activeSource = "audio";
let selectedFile = null;
let audioObjectUrl = null;
let busy = false;
let toastTimeout = null;

function setSource(source) {
  activeSource = source;
  const audioSelected = source === "audio";
  audioTab.classList.toggle("active", audioSelected);
  transcriptTab.classList.toggle("active", !audioSelected);
  audioTab.setAttribute("aria-selected", String(audioSelected));
  transcriptTab.setAttribute("aria-selected", String(!audioSelected));
  audioPanel.hidden = !audioSelected;
  transcriptPanel.hidden = audioSelected;
  hideError();
}

audioTab.addEventListener("click", () => setSource("audio"));
transcriptTab.addEventListener("click", () => setSource("transcript"));

document.getElementById("transcript-input").addEventListener("input", (event) => {
  document.getElementById("char-count").textContent = `${event.target.value.length.toLocaleString()} / 50,000`;
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  const [file] = fileInput.files || [];
  if (file) setSelectedFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag-over");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag-over");
  });
});
dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer?.files || [];
  if (file) setSelectedFile(file);
});

document.getElementById("remove-file").addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  fileCard.hidden = true;
  dropzone.hidden = false;
  audioPreview.pause();
  audioPreview.removeAttribute("src");
  audioPreview.hidden = true;
  if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
  audioObjectUrl = null;
  hideError();
});

function setSelectedFile(file) {
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  if (!ACCEPTED_EXTENSIONS.has(extension)) {
    showError("Choose a supported audio file: WAV, MP3, M4A, FLAC, OGG, OPUS, WEBM, or AAC.");
    return;
  }
  if (file.size > MAX_AUDIO_BYTES) {
    showError("This file is larger than the 25 MB upload limit.");
    return;
  }
  if (file.size === 0) {
    showError("This audio file is empty.");
    return;
  }
  if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
  selectedFile = file;
  audioObjectUrl = URL.createObjectURL(file);
  document.getElementById("file-name").textContent = file.name;
  document.getElementById("file-size").textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  audioPreview.src = audioObjectUrl;
  audioPreview.hidden = false;
  fileCard.hidden = false;
  dropzone.hidden = true;
  hideError();
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}
function hideError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function setBusy(value) {
  busy = value;
  generateButton.disabled = value;
  generateButton.classList.toggle("is-loading", value);
  generateButton.setAttribute("aria-busy", String(value));
  const label = generateButton.querySelector(".button-label");
  label.textContent = value ? "Working…" : "Create note draft";
}

function getApiError(payload, status) {
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item?.msg).filter(Boolean).join(" ") || `Request failed (${status}).`;
  }
  return `The request could not be completed (${status}). Check the service configuration and try again.`;
}

async function submitRequest() {
  if (busy) return;
  hideError();
  if (!document.getElementById("privacy-confirmed").checked) {
    showError("Confirm authorization and endpoint approval before processing this content.");
    return;
  }

  const language = document.getElementById("language-select").value;
  const specialty = document.getElementById("specialty-select").value;
  let url;
  let options;
  if (activeSource === "audio") {
    if (!selectedFile) {
      showError("Add a visit recording before creating the note draft.");
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("language", language);
    formData.append("specialty", specialty);
    formData.append("privacy_confirmed", "true");
    url = "/api/process";
    options = { method: "POST", body: formData };
  } else {
    const transcript = document.getElementById("transcript-input").value.trim();
    if (!transcript) {
      showError("Paste a transcript before creating the note draft.");
      return;
    }
    url = "/api/notes/generate";
    options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, specialty, language, privacy_confirmed: true }),
    };
  }

  setBusy(true);
  const statusLabel = generateButton.querySelector(".button-label");
  const stages = activeSource === "audio"
    ? ["Preparing audio…", "Transcribing visit…", "Drafting SOAP note…"]
    : ["Reviewing transcript…", "Structuring clinical facts…", "Drafting SOAP note…"];
  let stageIndex = 0;
  const stageTimer = window.setInterval(() => {
    stageIndex = Math.min(stageIndex + 1, stages.length - 1);
    statusLabel.textContent = stages[stageIndex];
  }, 5000);

  try {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(getApiError(payload, response.status));
    renderResult(payload);
    showToast("Draft ready. Review all sections before use.");
  } catch (error) {
    showError(error instanceof Error ? error.message : "The request could not be completed.");
  } finally {
    window.clearInterval(stageTimer);
    setBusy(false);
  }
}

generateButton.addEventListener("click", submitRequest);

function renderResult(result) {
  document.getElementById("empty-state").hidden = true;
  document.getElementById("result-content").hidden = false;
  document.getElementById("result-actions").hidden = false;
  document.getElementById("source-kind").textContent = result.source === "audio" ? "Audio transcription" : "Pasted transcript";

  const note = result.note || {};
  document.getElementById("note-subjective").value = note.subjective || "";
  document.getElementById("note-objective").value = note.objective || "";
  document.getElementById("note-assessment").value = note.assessment || "";
  document.getElementById("note-plan").value = note.plan || "";
  document.getElementById("note-follow-up").value = Array.isArray(note.follow_up) ? note.follow_up.join("\n") : "";

  document.getElementById("source-transcript").textContent = result.transcript || "No transcript was returned.";
  const speakerTag = document.getElementById("speaker-tag");
  speakerTag.textContent = result.speaker_labels_available
    ? "Speaker labels present · verify roles"
    : "Speaker attribution unverified";

  renderEntities(result.entities || []);
  renderWarnings(result.warnings || []);
  document.getElementById("transcript-details").open = false;
}

function renderEntities(entities) {
  const panel = document.getElementById("insight-panel");
  const list = document.getElementById("entity-list");
  const empty = document.getElementById("entity-empty");
  panel.hidden = false;
  list.replaceChildren();
  empty.hidden = entities.length > 0;
  for (const entity of entities) {
    const chip = document.createElement("span");
    chip.className = `entity-chip${entity.negated ? " negated" : ""}`;
    const text = document.createElement("span");
    text.textContent = entity.text;
    const category = document.createElement("strong");
    category.textContent = entity.negated ? `negated ${entity.category}` : entity.category;
    chip.append(text, category);
    list.append(chip);
  }
}

function renderWarnings(warnings) {
  const container = document.getElementById("warnings-list");
  container.replaceChildren();
  for (const warning of warnings) {
    const item = document.createElement("div");
    item.className = "warning-item";
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 16 16");
    icon.setAttribute("fill", "none");
    icon.setAttribute("class", "warning-icon");
    icon.setAttribute("aria-hidden", "true");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", "M8 2.2 14 13H2L8 2.2Z");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "1.3");
    path.setAttribute("stroke-linejoin", "round");
    const mark = document.createElementNS("http://www.w3.org/2000/svg", "path");
    mark.setAttribute("d", "M8 5.5v3.2m0 1.9h.01");
    mark.setAttribute("stroke", "currentColor");
    mark.setAttribute("stroke-width", "1.4");
    mark.setAttribute("stroke-linecap", "round");
    icon.append(path, mark);
    const text = document.createElement("span");
    text.textContent = warning;
    item.append(icon, text);
    container.append(item);
  }
}

function composeNote() {
  const parts = [
    ["Subjective", document.getElementById("note-subjective").value.trim()],
    ["Objective", document.getElementById("note-objective").value.trim()],
    ["Assessment", document.getElementById("note-assessment").value.trim()],
    ["Plan", document.getElementById("note-plan").value.trim()],
    ["Follow-up / action items", document.getElementById("note-follow-up").value.trim()],
  ];
  return parts.map(([heading, value]) => `${heading}\n${value || "Not documented."}`).join("\n\n");
}

async function copyNote() {
  try {
    await navigator.clipboard.writeText(composeNote());
    showToast("SOAP draft copied to clipboard.");
  } catch {
    showToast("Clipboard access is unavailable in this browser.");
  }
}

document.getElementById("copy-button").addEventListener("click", copyNote);
document.getElementById("download-button").addEventListener("click", () => {
  const file = new Blob([composeNote()], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(file);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "clinical-note-draft.txt";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  showToast("Draft downloaded to this device.");
});

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(toastTimeout);
  toastTimeout = window.setTimeout(() => toast.classList.remove("show"), 2600);
}

async function loadHealth() {
  const status = document.getElementById("service-status");
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Health check failed");
    const health = await response.json();
    const ready = health.providers?.asr_configured && health.providers?.llm_configured;
    status.innerHTML = "";
    const dot = document.createElement("span");
    dot.className = `status-dot${ready ? "" : " warn"}`;
    const label = document.createElement("span");
    label.textContent = ready ? "AI services configured" : "AI services need configuration";
    status.append(dot, label);
    status.title = ready
      ? "ASR and note-generation endpoints are configured."
      : "Set provider keys or configure self-hosted endpoints in the environment.";
  } catch {
    status.innerHTML = "";
    const dot = document.createElement("span");
    dot.className = "status-dot warn";
    const label = document.createElement("span");
    label.textContent = "Service status unavailable";
    status.append(dot, label);
  }
}

loadHealth();
