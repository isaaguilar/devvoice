import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

const STATUS_LABELS = {
  idle: "Idle",
  loading_model: "Loading model",
  capturing_selection: "Capturing",
  rewriting_text: "Rewriting",
  synthesizing: "Synthesizing",
  speaking: "Speaking",
  ready: "Ready",
  error: "Error",
};

const elements = {
  statusPill: document.querySelector("#status-pill"),
  statusDetail: document.querySelector("#status-detail"),
  hotkeyLabel: document.querySelector("#hotkey-label"),
  configPath: document.querySelector("#config-path"),
  logPath: document.querySelector("#log-path"),
  httpApiLabel: document.querySelector("#http-api-label"),
  ttsBackendLabel: document.querySelector("#tts-backend-label"),
  apiKeyState: document.querySelector("#api-key-state"),
  settingsForm: document.querySelector("#settings-form"),
  saveSettings: document.querySelector("#save-settings"),
  geminiEnabled: document.querySelector("#gemini-enabled"),
  apiKey: document.querySelector("#api-key"),
  geminiModel: document.querySelector("#gemini-model"),
  geminiPrompt: document.querySelector("#gemini-prompt"),
  voiceModel: document.querySelector("#voice-model"),
  voicePreset: document.querySelector("#voice-preset"),
  ttsPrecision: document.querySelector("#tts-precision"),
  ttsPrecisionNote: document.querySelector("#tts-precision-note"),
  defaultChunkSize: document.querySelector("#default-chunk-size"),
  shortcut: document.querySelector("#shortcut"),
  manualText: document.querySelector("#manual-text"),
  speakSelection: document.querySelector("#speak-selection"),
  speakManual: document.querySelector("#speak-manual"),
  pausePlayback: document.querySelector("#pause-playback"),
  stopPlayback: document.querySelector("#stop-playback"),
  showWindow: document.querySelector("#show-window"),
  openLog: document.querySelector("#open-log"),
  lastSelection: document.querySelector("#last-selection"),
  lastOutput: document.querySelector("#last-output"),
  lastError: document.querySelector("#last-error"),
  modelInstructionsTitle: document.querySelector("#model-instructions-title"),
  modelInstructionsSummary: document.querySelector("#model-instructions-summary"),
  modelInstructionsAttributes: document.querySelector("#model-instructions-attributes"),
  modelInstructionsExample: document.querySelector("#model-instructions-example"),
};

let configPath = "-";
let logPath = "-";
function requireElement(element, selector) {
  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

function setText(element, value) {
  if (element) {
    element.textContent = value;
  }
}

function populateVoicePresets(voices, selectedPreset) {
  const select = elements.voicePreset;
  if (!select) return;
  while (select.options.length > 1) {
    select.remove(1);
  }
  for (const voice of voices) {
    const option = document.createElement("option");
    option.value = voice;
    option.textContent = voice;
    select.appendChild(option);
  }
  select.value = selectedPreset;
}

function clearVoicePresets() {
  const select = elements.voicePreset;
  if (!select) return;
  while (select.options.length > 1) {
    select.remove(1);
  }
  select.value = "";
}

function currentSettingsInput() {
  const apiKey = requireElement(elements.apiKey, "#api-key").value.trim();
  return {
    geminiEnabled: requireElement(elements.geminiEnabled, "#gemini-enabled").checked,
    geminiModel: requireElement(elements.geminiModel, "#gemini-model").value.trim(),
    geminiPrompt: requireElement(elements.geminiPrompt, "#gemini-prompt").value.trim(),
    voiceModel: requireElement(elements.voiceModel, "#voice-model").value,
    voicePreset: requireElement(elements.voicePreset, "#voice-preset").value,
    ttsPrecision: requireElement(elements.ttsPrecision, "#tts-precision").value,
    defaultChunkSize: Number.parseInt(
      requireElement(elements.defaultChunkSize, "#default-chunk-size").value,
      10,
    ),
    shortcut: requireElement(elements.shortcut, "#shortcut").value.trim(),
    apiKey: apiKey.length > 0 ? apiKey : null,
  };
}

function syncPrecisionControls(voiceModel, selectedPrecision) {
  const select = elements.ttsPrecision;
  if (!select) return;
  const isVibeVoice = voiceModel === "vibevoice";
  for (const option of select.options) {
    option.disabled = isVibeVoice && option.value !== "auto";
  }
  if (isVibeVoice && selectedPrecision !== "auto") {
    select.value = "auto";
  }
  if (elements.ttsPrecisionNote) {
    elements.ttsPrecisionNote.textContent = isVibeVoice
      ? "VibeVoice 1.5B now runs through the MLX backend and uses auto precision only. Other models can still use the Rust precision options."
      : "Auto preserves the current fast 0.5B behavior. Non-VibeVoice models can still use F16, BF16, or F32.";
  }
}

function renderSettings(snapshot) {
  const apiKeyInput = requireElement(elements.apiKey, "#api-key");
  requireElement(elements.geminiEnabled, "#gemini-enabled").checked = snapshot.config.geminiEnabled;
  requireElement(elements.geminiModel, "#gemini-model").value = snapshot.config.geminiModel;
  requireElement(elements.geminiPrompt, "#gemini-prompt").value = snapshot.config.geminiPrompt;
  requireElement(elements.voiceModel, "#voice-model").value = snapshot.config.voiceModel;
  populateVoicePresets(snapshot.availableVoices || [], snapshot.config.voicePreset || "");
  requireElement(elements.ttsPrecision, "#tts-precision").value = snapshot.config.ttsPrecision || "auto";
  syncPrecisionControls(snapshot.config.voiceModel, snapshot.config.ttsPrecision || "auto");
  requireElement(elements.defaultChunkSize, "#default-chunk-size").value = String(
    snapshot.config.defaultChunkSize,
  );
  requireElement(elements.shortcut, "#shortcut").value = snapshot.config.shortcut;
  apiKeyInput.placeholder = snapshot.apiKeyPresent
    ? "Saved in Keychain. Leave blank to keep it."
    : "Leave blank to keep the saved key";

  const apiState = snapshot.config.geminiEnabled
    ? snapshot.apiKeyPresent
      ? "Gemini rewrite is enabled. The API key is stored in Keychain, and saving with this field blank keeps the current key."
      : "Gemini rewrite is enabled, but no API key is stored in Keychain yet."
    : "Gemini rewrite is disabled.";
  setText(elements.apiKeyState, apiState);
}

function renderModelInstructions(snapshot) {
  const instructions = snapshot.modelInstructions;
  if (!instructions) {
    return;
  }

  setText(elements.modelInstructionsTitle, instructions.modelLabel || "-");
  setText(elements.modelInstructionsSummary, instructions.summary || "-");
  setText(elements.modelInstructionsExample, instructions.curlExample || "-");

  const container = elements.modelInstructionsAttributes;
  if (!container) {
    return;
  }

  container.innerHTML = "";
  const attributes = instructions.attributes || [];
  if (attributes.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No per-request query params are exposed for this model yet.";
    container.appendChild(empty);
    return;
  }

  for (const attribute of attributes) {
    const card = document.createElement("article");
    card.className = "instruction-item";

    const label = document.createElement("strong");
    label.textContent = `${attribute.queryParam}`;

    const title = document.createElement("p");
    title.className = "instruction-item__title";
    title.textContent = attribute.label;

    const description = document.createElement("p");
    description.className = "muted";
    description.textContent = attribute.description;

    card.appendChild(label);
    card.appendChild(title);
    card.appendChild(description);
    container.appendChild(card);
  }
}

function renderSnapshot(snapshot) {
  const pill = requireElement(elements.statusPill, "#status-pill");
  pill.textContent = STATUS_LABELS[snapshot.status] || snapshot.status;
  pill.className = `status-pill status-pill--${snapshot.status}`;

  setText(elements.statusDetail, snapshot.statusDetail);
  setText(elements.hotkeyLabel, `Hotkey: ${snapshot.config.shortcut}`);
  setText(elements.configPath, `Config: ${configPath}`);
  setText(elements.logPath, `Log: ${logPath}`);

  const port = snapshot.config.httpPort;
  if (port > 0) {
    setText(elements.httpApiLabel, `HTTP API: http://127.0.0.1:${port}`);
  } else {
    setText(elements.httpApiLabel, "HTTP API: disabled");
  }
  setText(
    elements.ttsBackendLabel,
    `TTS backend: ${snapshot.ttsBackendStatus || snapshot.ttsRuntimeLabel || "-"}`,
  );
  setText(elements.lastSelection, snapshot.lastSelection || "-");
  setText(elements.lastOutput, snapshot.lastPreparedText || "-");
  setText(elements.lastError, snapshot.lastError || "-");
  renderModelInstructions(snapshot);
  populateVoicePresets(snapshot.availableVoices || [], snapshot.config.voicePreset || "");
  requireElement(elements.pausePlayback, "#pause-playback").textContent = snapshot.playbackPaused
    ? "Resume"
    : "Pause";
}

function setBusy(disabled) {
  for (const key of [
    "saveSettings",
    "speakSelection",
    "speakManual",
    "pausePlayback",
    "stopPlayback",
    "showWindow",
    "openLog",
  ]) {
    requireElement(elements[key], `#${key}`).disabled = disabled;
  }
}

async function refreshSnapshot() {
  const snapshot = await invoke("get_snapshot");
  renderSettings(snapshot);
  renderSnapshot(snapshot);
}

async function loadInitialState() {
  [configPath, logPath] = await Promise.all([
    invoke("get_config_path"),
    invoke("get_log_path"),
  ]);
  await refreshSnapshot();
}

document.addEventListener("DOMContentLoaded", async () => {
  const form = requireElement(elements.settingsForm, "#settings-form");
  const manualText = requireElement(elements.manualText, "#manual-text");

  await loadInitialState();
  await listen("state-changed", (event) => {
    renderSnapshot(event.payload);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) {
      return;
    }
    setBusy(true);
    try {
      const snapshot = await invoke("save_settings", { input: currentSettingsInput() });
      renderSettings(snapshot);
      renderSnapshot(snapshot);
    } finally {
      setBusy(false);
    }
  });

  requireElement(elements.voiceModel, "#voice-model").addEventListener("change", async () => {
    syncPrecisionControls(
      requireElement(elements.voiceModel, "#voice-model").value,
      requireElement(elements.ttsPrecision, "#tts-precision").value,
    );
    clearVoicePresets();
    if (!form.reportValidity()) {
      return;
    }
    setBusy(true);
    try {
      const snapshot = await invoke("save_settings", { input: currentSettingsInput() });
      renderSettings(snapshot);
      renderSnapshot(snapshot);
    } finally {
      setBusy(false);
    }
  });

  requireElement(elements.speakSelection, "#speak-selection").addEventListener("click", async () => {
    setBusy(true);
    try {
      const outcome = await invoke("speak_selection");
      if (outcome) {
        setText(elements.lastSelection, outcome.rawSelection || "-");
        setText(elements.lastOutput, outcome.preparedText || "-");
      }
      await refreshSnapshot();
    } finally {
      setBusy(false);
    }
  });

  requireElement(elements.speakManual, "#speak-manual").addEventListener("click", async () => {
    const text = manualText.value.trim();
    if (!text) {
      setText(elements.lastError, "Enter some manual text first.");
      return;
    }

    setBusy(true);
    try {
      const outcome = await invoke("speak_manual_text", { text });
      if (outcome) {
        setText(elements.lastSelection, outcome.rawSelection || "-");
        setText(elements.lastOutput, outcome.preparedText || "-");
      }
      await refreshSnapshot();
    } finally {
      setBusy(false);
    }
  });

  requireElement(elements.pausePlayback, "#pause-playback").addEventListener("click", async () => {
    await invoke("toggle_pause_playback");
    await refreshSnapshot();
  });

  requireElement(elements.stopPlayback, "#stop-playback").addEventListener("click", async () => {
    await invoke("stop_playback");
    await refreshSnapshot();
  });

  requireElement(elements.showWindow, "#show-window").addEventListener("click", async () => {
    await invoke("show_main_window");
  });

  requireElement(elements.openLog, "#open-log").addEventListener("click", async () => {
    await invoke("open_log_file");
  });
});
