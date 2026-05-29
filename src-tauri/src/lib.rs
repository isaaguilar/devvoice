mod config;
mod gemini;
mod http_api;
mod state;
mod tts;

use crate::config::{normalize_shortcut, AppConfig, ConfigStore};
use crate::gemini::GeminiProvider;
use crate::state::{
    AppSnapshot, AppStatus, ModelInstructionAttribute, ModelInstructions, SettingsInput,
    SpeakOutcome, SpeechOverrides,
};
use crate::tts::{
    initialize_tracing, log_file_path, split_into_speech_chunks, PlaybackController, TtsService,
    VibeVoicePresetInfo, WarmupInfo,
};
use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{image::Image, AppHandle, Emitter, Manager, State};
use tokio::sync::mpsc;
use tracing::info;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

const TRAY_ID: &str = "devvoice-tray";
const HOTKEY_DOUBLE_PRESS_WINDOW: Duration = Duration::from_millis(450);

fn is_active_status(status: AppStatus) -> bool {
    matches!(
        status,
        AppStatus::CapturingSelection
            | AppStatus::RewritingText
            | AppStatus::Synthesizing
            | AppStatus::Speaking
    )
}

enum SpeechMode {
    Direct(u64),
    Queued { queue_gen: u64, item_token: u64 },
}

pub(crate) enum QueueSignal {
    NewItem,
    Stop,
}

#[derive(Clone)]
struct QueuedSpeechRequest {
    text: String,
    overrides: SpeechOverrides,
}

pub struct SpeechQueue {
    items: Mutex<VecDeque<QueuedSpeechRequest>>,
    tx: mpsc::UnboundedSender<QueueSignal>,
    generation: AtomicU64,
    current_item_generation: AtomicU64,
}

impl SpeechQueue {
    fn new(tx: mpsc::UnboundedSender<QueueSignal>) -> Self {
        Self {
            items: Mutex::new(VecDeque::new()),
            tx,
            generation: AtomicU64::new(0),
            current_item_generation: AtomicU64::new(0),
        }
    }

    fn enqueue(&self, request: QueuedSpeechRequest) -> usize {
        let mut items = self.items.lock().unwrap();
        items.push_back(request);
        let len = items.len();
        let _ = self.tx.send(QueueSignal::NewItem);
        len
    }

    fn pop(&self) -> Option<QueuedSpeechRequest> {
        self.items.lock().unwrap().pop_front()
    }

    fn clear_and_invalidate(&self) {
        self.items.lock().unwrap().clear();
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.current_item_generation.fetch_add(1, Ordering::SeqCst);
        let _ = self.tx.send(QueueSignal::Stop);
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    fn current_item_token(&self) -> u64 {
        self.current_item_generation.load(Ordering::SeqCst)
    }

    fn skip_current_item(&self) {
        self.current_item_generation.fetch_add(1, Ordering::SeqCst);
    }

    pub fn len(&self) -> usize {
        self.items.lock().unwrap().len()
    }
}

pub struct AppRuntime {
    store: ConfigStore,
    config: RwLock<AppConfig>,
    runtime: Mutex<RuntimeState>,
    client: Client,
    tts: TtsService,
    playback: PlaybackController,
    pub speech_queue: SpeechQueue,
}

struct RuntimeState {
    status: AppStatus,
    status_detail: String,
    last_selection: Option<String>,
    last_prepared_text: Option<String>,
    last_error: Option<String>,
    model_ready: bool,
    playback_paused: bool,
    available_voices: Vec<String>,
    last_hotkey_press: Option<PendingHotkeyPress>,
    next_hotkey_press_id: u64,
}

#[derive(Clone, Copy)]
struct PendingHotkeyPress {
    id: u64,
    pressed_at: Instant,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            status: AppStatus::Idle,
            status_detail: "Ready to queue selected text.".to_owned(),
            last_selection: None,
            last_prepared_text: None,
            last_error: None,
            model_ready: false,
            playback_paused: false,
            available_voices: Vec::new(),
            last_hotkey_press: None,
            next_hotkey_press_id: 0,
        }
    }
}

impl AppRuntime {
    fn load() -> Result<(Self, mpsc::UnboundedReceiver<QueueSignal>)> {
        let (store, config) = ConfigStore::load()?;
        let client = Client::builder()
            .user_agent("DevVoice/0.1.0")
            .build()
            .context("create DevVoice HTTP client")?;
        let tts = TtsService::new(store.data_dir().join("tts-assets"));
        let playback = PlaybackController::new()?;
        let (queue_tx, queue_rx) = mpsc::unbounded_channel();
        let speech_queue = SpeechQueue::new(queue_tx);

        Ok((
            Self {
                store,
                config: RwLock::new(config),
                runtime: Mutex::new(RuntimeState::default()),
                client,
                tts,
                playback,
                speech_queue,
            },
            queue_rx,
        ))
    }

    pub fn snapshot(&self) -> AppSnapshot {
        let config = self.config.read().unwrap().clone();
        let runtime = self.runtime.lock().unwrap();
        let queue_length = self.speech_queue.len();
        let can_skip = !self.playback.is_empty() || is_active_status(runtime.status);
        let model_instructions = describe_model_instructions(config.voice_model);
        let tts_runtime_label = self
            .tts
            .loaded_runtime_label(config.voice_model, config.tts_precision);
        let tts_backend_status = self
            .tts
            .backend_status(config.voice_model, config.tts_precision);

        AppSnapshot {
            status: runtime.status,
            status_detail: runtime.status_detail.clone(),
            config,
            api_key_present: self
                .store
                .read_api_key()
                .map(|key| key.is_some())
                .unwrap_or(false),
            model_ready: runtime.model_ready,
            playback_paused: runtime.playback_paused,
            queue_length,
            can_skip,
            last_selection: runtime.last_selection.clone(),
            last_prepared_text: runtime.last_prepared_text.clone(),
            last_error: runtime.last_error.clone(),
            available_voices: runtime.available_voices.clone(),
            tts_runtime_label,
            tts_backend_status,
            model_instructions,
        }
    }

    pub fn current_runtime_label(&self) -> Option<String> {
        let config = self.current_config();
        self.tts
            .loaded_runtime_label(config.voice_model, config.tts_precision)
    }

    pub fn current_backend_status(&self) -> String {
        let config = self.current_config();
        self.tts
            .backend_status(config.voice_model, config.tts_precision)
    }

    fn emit_snapshot(&self, app: &AppHandle) {
        if let Err(error) = app.emit("state-changed", self.snapshot()) {
            eprintln!("failed to emit state update: {error}");
        }
    }

    fn mark_status(&self, app: &AppHandle, status: AppStatus, detail: impl Into<String>) {
        let detail = detail.into();
        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.status = status;
            runtime.status_detail = detail;
            if !matches!(status, AppStatus::Error) {
                runtime.last_error = None;
            }
        }
        update_tray_icon(app, status);
        self.emit_snapshot(app);
    }

    fn mark_error(&self, app: &AppHandle, detail: impl Into<String>) {
        let detail = detail.into();
        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.status = AppStatus::Error;
            runtime.status_detail = detail.clone();
            runtime.last_error = Some(detail);
        }
        update_tray_icon(app, AppStatus::Error);
        self.emit_snapshot(app);
    }

    fn current_config(&self) -> AppConfig {
        self.config.read().unwrap().clone()
    }

    pub fn save_settings(&self, app: &AppHandle, input: SettingsInput) -> Result<AppSnapshot> {
        let mut config = self.current_config();
        let old_shortcut = config.shortcut.clone();
        let old_voice_model = config.voice_model;
        let old_precision = config.tts_precision;

        config.gemini_enabled = input.gemini_enabled;
        if !input.gemini_model.trim().is_empty() {
            config.gemini_model = input.gemini_model.trim().to_owned();
        }
        if !input.gemini_prompt.trim().is_empty() {
            config.gemini_prompt = input.gemini_prompt.trim().to_owned();
        }
        config.voice_model = input.voice_model;
        config.voice_preset = input.voice_preset;
        config.default_chunk_size = input.default_chunk_size;
        config.tts_precision = input.tts_precision;
        if config.voice_model == crate::config::VoiceModel::VibeVoice {
            config.tts_precision = crate::config::TtsPrecision::Auto;
        }
        if !input.shortcut.trim().is_empty() {
            config.shortcut = normalize_shortcut(&input.shortcut);
        }

        let api_key_status = if let Some(api_key) = input
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.store.set_api_key(Some(api_key))?;
            "Gemini API key updated."
        } else if self.store.read_api_key()?.is_some() {
            "Existing Gemini API key kept."
        } else {
            "No Gemini API key is currently stored."
        };
        self.store.save_config(&config)?;
        {
            let mut guard = self.config.write().unwrap();
            *guard = config.clone();
        }
        if old_voice_model != config.voice_model || old_precision != config.tts_precision {
            self.tts.reset_auto_precision_override(config.voice_model);
        }
        self.tts
            .prune_cache(config.voice_model, config.tts_precision);

        if old_shortcut != config.shortcut {
            self.rebind_shortcut(app)?;
        }

        info!("Settings saved. {api_key_status}");
        self.mark_status(
            app,
            AppStatus::Ready,
            format!("Settings saved. {api_key_status}"),
        );
        Ok(self.snapshot())
    }

    pub async fn warmup_model(&self, app: &AppHandle) -> Result<()> {
        let config = self.current_config();
        self.tts
            .prune_cache(config.voice_model, config.tts_precision);
        self.mark_status(
            app,
            AppStatus::LoadingModel,
            "Downloading and warming the TTS model...",
        );

        match self
            .tts
            .warmup(config.voice_model, config.tts_precision)
            .await
        {
            Ok(warmup) => {
                {
                    let mut runtime = self.runtime.lock().unwrap();
                    runtime.model_ready = true;
                    runtime.available_voices = warmup.voices.clone();
                }
                self.mark_status(
                    app,
                    AppStatus::Ready,
                    if warmup.warmed_inference {
                        format!(
                            "Model ready with {} voice presets on {} after inference warmup ({} ms).",
                            warmup.voices.len(),
                            warmup.runtime_label,
                            warmup.warmup_duration_ms
                        )
                    } else {
                        format!(
                            "Model ready with {} voice presets on {}.",
                            warmup.voices.len(),
                            warmup.runtime_label
                        )
                    },
                );
                Ok(())
            }
            Err(error) => {
                self.mark_error(app, error.to_string());
                Err(error)
            }
        }
    }

    pub fn is_active(&self) -> bool {
        let runtime = self.runtime.lock().unwrap();
        is_active_status(runtime.status)
    }

    pub fn has_skippable_work(&self) -> bool {
        if !self.playback.is_empty() {
            return true;
        }

        let runtime = self.runtime.lock().unwrap();
        is_active_status(runtime.status)
    }

    fn clear_pending_hotkey_press(&self) {
        let mut runtime = self.runtime.lock().unwrap();
        runtime.last_hotkey_press = None;
    }

    fn consume_pending_hotkey_press(&self, press_id: u64) -> bool {
        let mut runtime = self.runtime.lock().unwrap();
        match runtime.last_hotkey_press {
            Some(pending) if pending.id == press_id => {
                runtime.last_hotkey_press = None;
                true
            }
            _ => false,
        }
    }

    fn schedule_single_tap_hotkey(&self, app: &AppHandle, press_id: u64) {
        let app_handle = app.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(HOTKEY_DOUBLE_PRESS_WINDOW).await;

            let runtime = app_handle.state::<AppRuntime>();
            if !runtime.consume_pending_hotkey_press(press_id) {
                return;
            }

            if let Err(error) = runtime.enqueue_selection(&app_handle).await {
                runtime.mark_error(&app_handle, error.to_string());
            }
        });
    }

    pub async fn speak_selection(&self, app: &AppHandle) -> Result<SpeakOutcome> {
        self.clear_pending_hotkey_press();
        self.speech_queue.clear_and_invalidate();
        self.mark_status(
            app,
            AppStatus::CapturingSelection,
            "Capturing selected text from the frontmost app...",
        );
        let job = self.playback.begin_job()?;
        let selection = match self.tts.capture_selection() {
            Ok(selection) => selection,
            Err(error) => {
                self.mark_error(app, error.to_string());
                return Err(error);
            }
        };

        self.process_text(
            app,
            selection.raw,
            SpeechMode::Direct(job),
            SpeechOverrides::default(),
        )
        .await
    }

    pub async fn speak_manual_text(&self, app: &AppHandle, text: String) -> Result<SpeakOutcome> {
        self.clear_pending_hotkey_press();
        self.speech_queue.clear_and_invalidate();
        let job = self.playback.begin_job()?;
        self.process_text(
            app,
            text,
            SpeechMode::Direct(job),
            SpeechOverrides::default(),
        )
        .await
    }

    pub async fn enqueue_selection(&self, app: &AppHandle) -> Result<usize> {
        self.clear_pending_hotkey_press();
        self.mark_status(
            app,
            AppStatus::CapturingSelection,
            "Capturing selected text from the frontmost app...",
        );
        let selection = match self.tts.capture_selection() {
            Ok(selection) => selection,
            Err(error) => {
                self.mark_error(app, error.to_string());
                return Err(error);
            }
        };

        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.last_selection = Some(selection.raw.clone());
            runtime.playback_paused = false;
        }
        self.emit_snapshot(app);
        self.enqueue_text(app, selection.raw, SpeechOverrides::default())
    }

    pub fn enqueue_manual_text(&self, app: &AppHandle, text: String) -> Result<usize> {
        self.clear_pending_hotkey_press();
        let trimmed = text.trim().to_owned();
        if trimmed.is_empty() {
            let error = anyhow!("Enter text to queue.");
            self.mark_error(app, error.to_string());
            return Err(error);
        }

        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.last_selection = Some(trimmed.clone());
            runtime.playback_paused = false;
        }
        self.emit_snapshot(app);

        self.enqueue_text(app, trimmed, SpeechOverrides::default())
    }

    pub fn enqueue_text(
        &self,
        app: &AppHandle,
        text: String,
        overrides: SpeechOverrides,
    ) -> Result<usize> {
        let text = text.trim().to_owned();
        if text.is_empty() {
            let error = anyhow!("No text was provided.");
            self.mark_error(app, error.to_string());
            return Err(error);
        }
        let overrides = overrides.normalized();
        self.validate_overrides(&overrides)?;

        let position = self
            .speech_queue
            .enqueue(QueuedSpeechRequest { text, overrides });
        let should_mark_ready = {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.playback_paused = false;
            !matches!(
                runtime.status,
                AppStatus::Speaking
                    | AppStatus::Synthesizing
                    | AppStatus::RewritingText
                    | AppStatus::LoadingModel
            )
        };
        if self.playback.is_empty() && should_mark_ready {
            let detail = if position > 1 {
                format!("Queued {position} items for playback.")
            } else {
                "Queued 1 item for playback.".to_owned()
            };
            self.mark_status(app, AppStatus::Ready, detail);
        } else {
            self.emit_snapshot(app);
        }
        Ok(position)
    }

    fn validate_overrides(&self, overrides: &SpeechOverrides) -> Result<()> {
        self.validate_overrides_for_model(self.current_config().voice_model, overrides)
    }

    fn validate_overrides_for_model(
        &self,
        model: crate::config::VoiceModel,
        overrides: &SpeechOverrides,
    ) -> Result<()> {
        match model {
            crate::config::VoiceModel::VibeVoiceRealtime => {
                if overrides.description.is_some()
                    || overrides.reference_preset_id.is_some()
                    || overrides.reference_preset_name.is_some()
                    || overrides.reference_audio_path.is_some()
                    || overrides.cfg_scale.is_some()
                    || overrides.temperature.is_some()
                    || overrides.max_tokens.is_some()
                {
                    return Err(anyhow!(
                        "VibeVoice Realtime supports voice_preset, style, chunk_size, save_audio, and output_dir query params."
                    ));
                }
            }
            crate::config::VoiceModel::VibeVoice => {
                if overrides.voice_preset.is_some()
                    || overrides.style.is_some()
                    || overrides.description.is_some()
                {
                    return Err(anyhow!(
                        "VibeVoice 1.5B does not use voice_preset, style, or description query params. Use reference_audio_path plus speaker-formatted text instead."
                    ));
                }
                let reference_selector_count = [
                    overrides.reference_audio_path.is_some(),
                    overrides.reference_preset_id.is_some(),
                    overrides.reference_preset_name.is_some(),
                ]
                .into_iter()
                .filter(|present| *present)
                .count();
                if reference_selector_count > 1 {
                    return Err(anyhow!(
                        "Provide only one of reference_audio_path, reference_preset_id, or reference_preset_name for VibeVoice 1.5B."
                    ));
                }
                if let Some(path) = overrides.reference_audio_path.as_deref() {
                    if !Path::new(path).exists() {
                        return Err(anyhow!(
                            "Reference audio file does not exist: {path}. reference_audio_path is optional. If you use it, replace the example path with a real local WAV or MP3 file such as /Users/you/Desktop/reference-voice.wav."
                        ));
                    }
                }
            }
            crate::config::VoiceModel::ParlerTts => {
                if overrides.reference_preset_id.is_some()
                    || overrides.reference_preset_name.is_some()
                    || overrides.reference_audio_path.is_some()
                    || overrides.cfg_scale.is_some()
                    || overrides.temperature.is_some()
                    || overrides.max_tokens.is_some()
                {
                    return Err(anyhow!(
                        "Parler-TTS supports voice_preset, description, style, chunk_size, save_audio, and output_dir query params."
                    ));
                }
            }
            crate::config::VoiceModel::Kokoro => {
                if overrides.style.is_some()
                    || overrides.description.is_some()
                    || overrides.reference_preset_id.is_some()
                    || overrides.reference_preset_name.is_some()
                    || overrides.reference_audio_path.is_some()
                    || overrides.cfg_scale.is_some()
                    || overrides.temperature.is_some()
                    || overrides.max_tokens.is_some()
                {
                    return Err(anyhow!(
                        "Kokoro currently supports voice_preset, chunk_size, save_audio, and output_dir query params."
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn create_vibevoice_preset(
        &self,
        reference_audio_path: &str,
        name: Option<&str>,
    ) -> Result<VibeVoicePresetInfo> {
        self.tts.create_vibevoice_preset(reference_audio_path, name)
    }

    pub fn list_vibevoice_presets(&self) -> Result<Vec<VibeVoicePresetInfo>> {
        self.tts.list_vibevoice_presets()
    }

    pub async fn warmup_vibevoice(
        &self,
        app: &AppHandle,
        overrides: SpeechOverrides,
    ) -> Result<WarmupInfo> {
        let overrides = overrides.normalized();
        self.validate_overrides_for_model(crate::config::VoiceModel::VibeVoice, &overrides)?;
        self.mark_status(
            app,
            AppStatus::LoadingModel,
            "Running VibeVoice 1.5B inference warmup...",
        );
        let precision = self.current_config().tts_precision;
        match self.tts.warmup_vibevoice(precision, &overrides).await {
            Ok(warmup) => {
                if self.current_config().voice_model == crate::config::VoiceModel::VibeVoice {
                    let mut runtime = self.runtime.lock().unwrap();
                    runtime.model_ready = true;
                    runtime.available_voices = warmup.voices.clone();
                }
                self.mark_status(
                    app,
                    AppStatus::Ready,
                    format!(
                        "VibeVoice 1.5B warmup finished on {} in {} ms.",
                        warmup.runtime_label, warmup.warmup_duration_ms
                    ),
                );
                Ok(warmup)
            }
            Err(error) => {
                self.mark_error(app, error.to_string());
                Err(error)
            }
        }
    }

    pub fn skip_current_item(&self, app: &AppHandle) -> Result<AppSnapshot> {
        self.clear_pending_hotkey_press();
        if !self.has_skippable_work() {
            self.mark_status(app, AppStatus::Ready, "Nothing is queued right now.");
            return Ok(self.snapshot());
        }

        self.speech_queue.skip_current_item();
        self.playback.stop()?;
        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.playback_paused = false;
        }
        let detail = if self.speech_queue.len() > 0 {
            "Skipping the current message and continuing with the queue."
        } else {
            "Skipped the current message. Queue is empty."
        };
        self.mark_status(app, AppStatus::Ready, detail);
        Ok(self.snapshot())
    }

    pub async fn handle_shortcut(&self, app: &AppHandle) -> Result<()> {
        let now = Instant::now();
        let pending_press_id = {
            let mut runtime = self.runtime.lock().unwrap();
            if let Some(last) = runtime.last_hotkey_press {
                if now.duration_since(last.pressed_at) <= HOTKEY_DOUBLE_PRESS_WINDOW {
                    runtime.last_hotkey_press = None;
                    None
                } else {
                    runtime.next_hotkey_press_id += 1;
                    let press_id = runtime.next_hotkey_press_id;
                    runtime.last_hotkey_press = Some(PendingHotkeyPress {
                        id: press_id,
                        pressed_at: now,
                    });
                    Some(press_id)
                }
            } else {
                runtime.next_hotkey_press_id += 1;
                let press_id = runtime.next_hotkey_press_id;
                runtime.last_hotkey_press = Some(PendingHotkeyPress {
                    id: press_id,
                    pressed_at: now,
                });
                Some(press_id)
            }
        };

        if let Some(press_id) = pending_press_id {
            self.schedule_single_tap_hotkey(app, press_id);
        } else if self.has_skippable_work() {
            self.skip_current_item(app)?;
        }

        Ok(())
    }

    async fn process_text(
        &self,
        app: &AppHandle,
        raw_selection: String,
        mode: SpeechMode,
        overrides: SpeechOverrides,
    ) -> Result<SpeakOutcome> {
        let raw_selection = raw_selection.trim().to_owned();
        if raw_selection.is_empty() {
            let error = anyhow!("No text was provided.");
            self.mark_error(app, error.to_string());
            return Err(error);
        }

        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.last_selection = Some(raw_selection.clone());
            runtime.playback_paused = false;
        }
        self.emit_snapshot(app);

        let config = self.current_config();
        let (prepared_text, used_gemini) =
            match self.prepare_text(app, &raw_selection, &config).await {
                Ok(result) => result,
                Err(error) => {
                    self.mark_error(app, error.to_string());
                    return Err(error);
                }
            };

        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.last_prepared_text = Some(prepared_text.clone());
        }
        self.emit_snapshot(app);

        let chunk_size = overrides.chunk_size.or(Some(config.default_chunk_size));
        let chunks = split_into_speech_chunks(&prepared_text, chunk_size);
        let total = chunks.len();
        let mut first_voice: Option<String> = None;
        let mut total_audio_secs: f64 = 0.0;
        let mut synthesized_chunks = Vec::with_capacity(total);
        let should_save_audio =
            overrides.save_audio.unwrap_or(false) || overrides.output_dir.is_some();
        let mut completed = true;

        for (i, chunk) in chunks.into_iter().enumerate() {
            let cancelled = match &mode {
                SpeechMode::Direct(job) => !self.playback.is_current_job(*job),
                SpeechMode::Queued {
                    queue_gen,
                    item_token,
                } => {
                    self.speech_queue.generation() != *queue_gen
                        || self.speech_queue.current_item_token() != *item_token
                }
            };
            if cancelled {
                completed = false;
                self.mark_status(app, AppStatus::Ready, "Playback cancelled.");
                break;
            }

            let status_detail = if total > 1 {
                format!("Synthesizing chunk {} of {total}...", i + 1)
            } else {
                "Synthesizing speech with the local neural voice...".to_owned()
            };
            self.mark_status(app, AppStatus::Synthesizing, status_detail);

            let (audio, voice_name) = match self
                .tts
                .synthesize(
                    &chunk,
                    config.voice_model,
                    config.tts_precision,
                    &config.voice_preset,
                    &overrides,
                )
                .await
            {
                Ok(result) => result,
                Err(error) => {
                    self.mark_error(app, format!("{error:#}"));
                    return Err(error);
                }
            };

            if i == 0 {
                first_voice = voice_name;
                {
                    let mut runtime = self.runtime.lock().unwrap();
                    runtime.model_ready = true;
                }
            }

            let sample_count = audio.samples.len() as f64;
            let divisor = audio.sample_rate as f64 * audio.channels as f64;
            if divisor > 0.0 {
                total_audio_secs += sample_count / divisor;
            }
            synthesized_chunks.push(audio.clone());

            let appended = match &mode {
                SpeechMode::Direct(job) => self.playback.append_audio(*job, audio)?,
                SpeechMode::Queued { .. } => {
                    self.playback.append_audio_unchecked(audio)?;
                    true
                }
            };
            if !appended {
                completed = false;
                self.mark_status(app, AppStatus::Ready, "Playback cancelled.");
                break;
            }

            if i == 0 {
                let detail = match &first_voice {
                    Some(name) => format!("Speaking with {name}."),
                    None => "Speaking with the default voice.".to_owned(),
                };
                self.mark_status(app, AppStatus::Speaking, detail);
            }
        }

        if completed && should_save_audio && !synthesized_chunks.is_empty() {
            let output_path = save_synthesized_audio(
                config.voice_model,
                overrides.output_dir.as_deref(),
                &synthesized_chunks,
            )?;
            info!("Saved synthesized audio to {}", output_path.display());
        }

        Ok(SpeakOutcome {
            raw_selection,
            prepared_text,
            used_gemini,
            audio_duration_secs: total_audio_secs,
        })
    }

    async fn prepare_text(
        &self,
        app: &AppHandle,
        raw_selection: &str,
        config: &AppConfig,
    ) -> Result<(String, bool)> {
        if !config.gemini_enabled {
            return Ok((raw_selection.to_owned(), false));
        }

        let api_key = self
            .store
            .read_api_key()?
            .ok_or_else(|| anyhow!("Gemini rewrite is enabled, but no API key is stored."))?;

        self.mark_status(
            app,
            AppStatus::RewritingText,
            format!("Rewriting with {}...", config.gemini_model),
        );

        let provider = GeminiProvider::new(
            self.client.clone(),
            api_key,
            config.gemini_model.clone(),
            config.gemini_prompt.clone(),
        );
        let text = provider.rewrite(raw_selection).await?;
        Ok((text, true))
    }

    pub fn stop_playback(&self, app: &AppHandle) -> Result<AppSnapshot> {
        self.clear_pending_hotkey_press();
        self.speech_queue.clear_and_invalidate();
        self.playback.stop()?;
        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.playback_paused = false;
        }
        self.mark_status(app, AppStatus::Ready, "Playback stopped.");
        Ok(self.snapshot())
    }

    pub fn toggle_pause_playback(&self, app: &AppHandle) -> Result<AppSnapshot> {
        let paused = self.playback.toggle_pause()?;
        {
            let mut runtime = self.runtime.lock().unwrap();
            runtime.playback_paused = paused;
            runtime.status = if paused {
                AppStatus::Speaking
            } else {
                AppStatus::Speaking
            };
            runtime.status_detail = if paused {
                "Playback paused.".to_owned()
            } else {
                "Playback resumed.".to_owned()
            };
        }
        self.emit_snapshot(app);
        Ok(self.snapshot())
    }

    fn rebind_shortcut(&self, app: &AppHandle) -> Result<()> {
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            app.global_shortcut()
                .unregister_all()
                .map_err(|error| anyhow!(error.to_string()))?;
            register_shortcut(app, &self.current_config().shortcut)?;
        }

        Ok(())
    }
}

fn save_synthesized_audio(
    model: crate::config::VoiceModel,
    output_dir: Option<&str>,
    chunks: &[any_tts::AudioSamples],
) -> Result<std::path::PathBuf> {
    let combined = combine_audio_chunks(chunks)?;
    let output_dir = output_dir
        .map(std::path::PathBuf::from)
        .unwrap_or_else(default_audio_output_dir);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("create output directory {}", output_dir.display()))?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let file_name = format!("devvoice-{}-{timestamp}.wav", model_slug(model));
    let output_path = output_dir.join(file_name);
    combined
        .save_wav(&output_path)
        .with_context(|| format!("save synthesized audio to {}", output_path.display()))?;
    Ok(output_path)
}

fn combine_audio_chunks(chunks: &[any_tts::AudioSamples]) -> Result<any_tts::AudioSamples> {
    let first = chunks
        .first()
        .ok_or_else(|| anyhow!("no synthesized audio was available to save"))?;
    let mut samples = Vec::new();
    for chunk in chunks {
        if chunk.sample_rate != first.sample_rate || chunk.channels != first.channels {
            return Err(anyhow!(
                "cannot combine synthesized chunks with mismatched audio formats"
            ));
        }
        samples.extend_from_slice(&chunk.samples);
    }
    Ok(any_tts::AudioSamples::new(samples, first.sample_rate))
}

fn default_audio_output_dir() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("Downloads")
}

fn model_slug(model: crate::config::VoiceModel) -> &'static str {
    match model {
        crate::config::VoiceModel::VibeVoiceRealtime => "vibevoice-realtime",
        crate::config::VoiceModel::VibeVoice => "vibevoice-1-5b",
        crate::config::VoiceModel::ParlerTts => "parler-tts",
        crate::config::VoiceModel::Kokoro => "kokoro",
    }
}

fn describe_model_instructions(model: crate::config::VoiceModel) -> ModelInstructions {
    let mut attributes = match model {
        crate::config::VoiceModel::VibeVoiceRealtime => vec![
            ModelInstructionAttribute {
                query_param: "voice_preset".to_owned(),
                label: "Voice preset override".to_owned(),
                description: "Overrides the saved voice preset for this single /speak request.".to_owned(),
            },
            ModelInstructionAttribute {
                query_param: "style".to_owned(),
                label: "Instruction prompt".to_owned(),
                description: "Overrides the realtime model instruction prompt used to shape delivery for this request.".to_owned(),
            },
        ],
        crate::config::VoiceModel::VibeVoice => vec![
            ModelInstructionAttribute {
                query_param: "reference_audio_path".to_owned(),
                label: "Reference voice file".to_owned(),
                description: "Optional. Path to a local WAV or MP3 file used to seed speaker identity for this request. A clean 3 to 10 second clip of one person speaking works best. You can record your own voice, clip existing speech you have rights to use, or reuse another generated sample.".to_owned(),
            },
            ModelInstructionAttribute {
                query_param: "reference_preset_id".to_owned(),
                label: "Saved reference preset".to_owned(),
                description: "Optional. Reuses a VibeVoice preset created earlier from a reference clip, so /speak can skip rebuilding that voice seed from raw audio on every request.".to_owned(),
            },
            ModelInstructionAttribute {
                query_param: "reference_preset_name".to_owned(),
                label: "Saved reference preset by name".to_owned(),
                description: "Optional. Looks up a VibeVoice preset by its unique creation name instead of by preset ID.".to_owned(),
            },
            ModelInstructionAttribute {
                query_param: "cfg_scale".to_owned(),
                label: "Guidance scale".to_owned(),
                description: "Overrides the VibeVoice classifier-free guidance scale for this request. Default is 1.3.".to_owned(),
            },
            ModelInstructionAttribute {
                query_param: "temperature".to_owned(),
                label: "Sampling temperature".to_owned(),
                description: "Overrides the VibeVoice sampling temperature for this request. Default is 0.0.".to_owned(),
            },
            ModelInstructionAttribute {
                query_param: "max_tokens".to_owned(),
                label: "Generation budget".to_owned(),
                description: "Caps how many generation tokens VibeVoice can spend on this request. Cost scales steeply, so 32 is a better starting point for short utterances and you should only increase it when outputs are being cut off.".to_owned(),
            },
        ],
        crate::config::VoiceModel::ParlerTts => vec![
            ModelInstructionAttribute {
                query_param: "voice_preset".to_owned(),
                label: "Voice preset override".to_owned(),
                description: "Selects one of the curated Parler preset descriptions for this request.".to_owned(),
            },
            ModelInstructionAttribute {
                query_param: "description".to_owned(),
                label: "Full voice description".to_owned(),
                description: "Replaces the preset description with your own full descriptive prompt for Parler-TTS.".to_owned(),
            },
            ModelInstructionAttribute {
                query_param: "style".to_owned(),
                label: "Style suffix".to_owned(),
                description: "Appends extra style guidance to the preset or custom description for this request.".to_owned(),
            },
        ],
        crate::config::VoiceModel::Kokoro => vec![ModelInstructionAttribute {
            query_param: "voice_preset".to_owned(),
            label: "Voice preset override".to_owned(),
            description: "Overrides the saved voice preset for this single /speak request.".to_owned(),
        }],
    };
    attributes.extend([
        ModelInstructionAttribute {
            query_param: "chunk_size".to_owned(),
            label: "Chunk size in characters".to_owned(),
            description: "Optional. Sets the target chunk size in characters for this request. If omitted, DevVoice uses your saved default chunk size. Use 0 to disable chunking entirely.".to_owned(),
        },
        ModelInstructionAttribute {
            query_param: "save_audio".to_owned(),
            label: "Save synthesized audio".to_owned(),
            description: "Optional. Set to true to save the synthesized WAV for this request.".to_owned(),
        },
        ModelInstructionAttribute {
            query_param: "output_dir".to_owned(),
            label: "Saved audio directory".to_owned(),
            description: "Optional. Directory where saved WAV files should be written. If omitted, DevVoice uses $HOME/Downloads. Supplying output_dir also implies saving audio.".to_owned(),
        },
    ]);

    let summary = match model {
        crate::config::VoiceModel::VibeVoiceRealtime => {
            "Fastest local model. Use voice_preset and style for delivery changes. You can also control chunking and optionally save the synthesized WAV per request."
        }
        crate::config::VoiceModel::VibeVoice => {
            "Long-form expressive model. DevVoice now runs VibeVoice 1.5B through the MLX backend on macOS, using auto precision only. The app warms the model automatically on startup, and you can also call /vibevoice/warmup to preload a saved preset before a real request. Use Speaker 0:, Speaker 1:, and similar labels directly in the request body for dialogue. You can seed voice identity with a raw reference_audio_path, a reusable reference_preset_id, or a unique reference_preset_name created ahead of time. You can also control chunking and optionally save the synthesized WAV per request."
        }
        crate::config::VoiceModel::ParlerTts => {
            "Most flexible model for descriptive voice control. You can override the preset, send a full description, or append a style suffix per request."
        }
        crate::config::VoiceModel::Kokoro => {
            "Lightweight model with limited control surface. Use voice preset overrides per request."
        }
    };

    let curl_example = match model {
        crate::config::VoiceModel::VibeVoiceRealtime => {
            r#"curl -X POST "http://127.0.0.1:9876/speak?voice_preset=en-Emma_woman&style=Read%20this%20like%20an%20excited%20demo%20for%20engineers.&chunk_size=0&save_audio=true" --data "Ship the build after the tests pass.""#
        }
        crate::config::VoiceModel::VibeVoice => {
            "Create a reusable preset once, using a unique name:\n\
curl -X POST \"http://127.0.0.1:9876/vibevoice/presets\" -H \"Content-Type: application/json\" -d '{\"name\":\"my-demo-voice\",\"referenceAudioPath\":\"/Users/you/Desktop/reference-voice.wav\"}'\n\
\n\
Warm the 1.5B model and that preset before your first real request:\n\
curl -X POST \"http://127.0.0.1:9876/vibevoice/warmup\" -H \"Content-Type: application/json\" -d '{\"referencePresetName\":\"my-demo-voice\"}'\n\
\n\
Reuse a saved preset by unique name for synthesis:\n\
curl -X POST \"http://127.0.0.1:9876/speak?reference_preset_name=my-demo-voice&cfg_scale=1.3&temperature=0.0&max_tokens=32\" --data-binary $'Speaker 0: Explain the rollout calmly.\\nSpeaker 1: Reply with more excitement.'\n\
\n\
Or reuse a saved preset by internal id:\n\
curl -X POST \"http://127.0.0.1:9876/speak?reference_preset_id=my-demo-voice-1716420000&cfg_scale=1.3&temperature=0.0&max_tokens=32\" --data-binary $'Speaker 0: Explain the rollout calmly.\\nSpeaker 1: Reply with more excitement.'\n\
\n\
No seed voice, use the model default:\n\
curl -X POST \"http://127.0.0.1:9876/speak?cfg_scale=1.3&temperature=0.0&max_tokens=32&chunk_size=220\" --data-binary $'Speaker 0: Explain the rollout calmly.\\nSpeaker 1: Reply with more excitement.'\n\
\n\
Optional seed voice, replace the path with a real local file:\n\
curl -X POST \"http://127.0.0.1:9876/speak?reference_audio_path=/Users/you/Desktop/reference-voice.wav&cfg_scale=1.3&temperature=0.0&max_tokens=32&save_audio=true\" --data-binary $'Speaker 0: Explain the rollout calmly.\\nSpeaker 1: Reply with more excitement.'"
        }
        crate::config::VoiceModel::ParlerTts => {
            r#"curl -X POST "http://127.0.0.1:9876/speak?voice_preset=senior_developer&style=Sound%20more%20enthusiastic%20without%20rushing." --data "Explain the rollout plan.""#
        }
        crate::config::VoiceModel::Kokoro => {
            r#"curl -X POST "http://127.0.0.1:9876/speak?voice_preset=af_heart" --data "Use the lightweight local model for this one.""#
        }
    }
    .to_owned();

    ModelInstructions {
        model_label: model.display_name().to_owned(),
        summary: summary.to_owned(),
        curl_example,
        attributes,
    }
}

fn register_shortcut(app: &AppHandle, shortcut: &str) -> Result<()> {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        let shortcut = normalize_shortcut(shortcut);
        app.global_shortcut()
            .on_shortcut(shortcut.as_str(), |app, _shortcut, event| {
                if event.state != ShortcutState::Pressed {
                    return;
                }

                let app = app.clone();
                tauri::async_runtime::spawn(async move {
                    let runtime = app.state::<AppRuntime>();
                    if let Err(error) = runtime.handle_shortcut(&app).await {
                        eprintln!("shortcut handler error: {error}");
                    }
                });
            })
            .map_err(|error| anyhow!(error.to_string()))?;
    }

    Ok(())
}

fn show_main_window_inner(app: &AppHandle) -> Result<()> {
    if let Some(window) = app.get_webview_window("main") {
        window.show().map_err(|error| anyhow!(error.to_string()))?;
        window
            .set_focus()
            .map_err(|error| anyhow!(error.to_string()))?;
    }
    Ok(())
}

fn open_log_file_inner() -> Result<()> {
    Command::new("open")
        .arg(log_file_path())
        .status()
        .map_err(|error| anyhow!("open DevVoice log file: {error}"))?;
    Ok(())
}

fn build_tray(app: &AppHandle) -> Result<()> {
    let show = MenuItem::with_id(app, "show", "Show DevVoice", true, None::<&str>)?;
    let queue_selection = MenuItem::with_id(
        app,
        "queue-selection",
        "Queue Selection",
        true,
        None::<&str>,
    )?;
    let skip_next = MenuItem::with_id(app, "skip-next", "Skip to Next Item", true, None::<&str>)?;
    let pause = MenuItem::with_id(app, "pause", "Pause/Resume", true, None::<&str>)?;
    let stop = MenuItem::with_id(app, "stop", "Stop", true, None::<&str>)?;
    let open_log = MenuItem::with_id(app, "open-log", "Open Log File", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;

    let menu = Menu::with_items(
        app,
        &[
            &show,
            &queue_selection,
            &skip_next,
            &pause,
            &stop,
            &open_log,
            &separator,
            &quit,
        ],
    )?;
    let (icon, is_template) = tray_icon_image(AppStatus::Idle);

    TrayIconBuilder::with_id(TRAY_ID)
        .icon(icon)
        .menu(&menu)
        .icon_as_template(is_template)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "show" => {
                let _ = show_main_window_inner(app);
            }
            "queue-selection" => {
                let app = app.clone();
                tauri::async_runtime::spawn(async move {
                    let runtime = app.state::<AppRuntime>();
                    let _ = runtime.enqueue_selection(&app).await;
                });
            }
            "skip-next" => {
                let runtime = app.state::<AppRuntime>();
                let _ = runtime.skip_current_item(app);
            }
            "pause" => {
                let runtime = app.state::<AppRuntime>();
                let _ = runtime.toggle_pause_playback(app);
            }
            "stop" => {
                let runtime = app.state::<AppRuntime>();
                let _ = runtime.stop_playback(app);
            }
            "open-log" => {
                let _ = open_log_file_inner();
            }
            "quit" => {
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let _ = show_main_window_inner(tray.app_handle());
            }
        })
        .build(app)?;

    Ok(())
}

fn update_tray_icon(app: &AppHandle, status: AppStatus) {
    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        let (icon, is_template) = tray_icon_image(status);
        let _ = tray.set_icon(Some(icon));
        let _ = tray.set_icon_as_template(is_template);
    }
}

fn tray_icon_image(status: AppStatus) -> (Image<'static>, bool) {
    let (rgba, is_template) = match status {
        AppStatus::Idle | AppStatus::Ready => (rgba(0, 0, 0, 255), true),
        AppStatus::Speaking => (rgba(34, 197, 94, 255), false),
        AppStatus::Error => (rgba(239, 68, 68, 255), false),
        AppStatus::LoadingModel
        | AppStatus::CapturingSelection
        | AppStatus::RewritingText
        | AppStatus::Synthesizing => (rgba(250, 204, 21, 255), false),
    };
    (render_lips_icon(rgba), is_template)
}

const fn rgba(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
    [r, g, b, a]
}

fn render_lips_icon(color: [u8; 4]) -> Image<'static> {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const SAMPLES: usize = 4;
    let mut pixels = vec![0_u8; (WIDTH * HEIGHT * 4) as usize];

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let mut coverage = 0.0_f32;

            for sy in 0..SAMPLES {
                for sx in 0..SAMPLES {
                    let fx = (x as f32 + (sx as f32 + 0.5) / SAMPLES as f32) / WIDTH as f32;
                    let fy = (y as f32 + (sy as f32 + 0.5) / SAMPLES as f32) / HEIGHT as f32;
                    let px = fx * 2.0 - 1.0;
                    let py = fy * 2.0 - 1.0;

                    if lips_mask(px, py) {
                        coverage += 1.0;
                    }
                }
            }

            let alpha = (coverage / (SAMPLES * SAMPLES) as f32 * color[3] as f32).round() as u8;
            let idx = ((y * WIDTH + x) * 4) as usize;
            pixels[idx] = color[0];
            pixels[idx + 1] = color[1];
            pixels[idx + 2] = color[2];
            pixels[idx + 3] = alpha;
        }
    }

    Image::new_owned(pixels, WIDTH, HEIGHT)
}

fn lips_mask(x: f32, y: f32) -> bool {
    let upper_left = ellipse(x + 0.38, y + 0.07, 0.39, 0.25);
    let upper_right = ellipse(x - 0.38, y + 0.07, 0.39, 0.25);
    let upper_center = ellipse(x, y + 0.02, 0.29, 0.13);
    let lower = ellipse(x, y - 0.22, 0.78, 0.38);
    let outer = (upper_left || upper_right || upper_center || lower) && y > -0.88 && y < 0.88;

    let inner_mouth = ellipse(x, y - 0.03, 0.46, 0.08) || ellipse(x, y - 0.13, 0.16, 0.05);
    outer && !inner_mouth
}

fn ellipse(x: f32, y: f32, rx: f32, ry: f32) -> bool {
    (x * x) / (rx * rx) + (y * y) / (ry * ry) <= 1.0
}

#[tauri::command]
fn get_snapshot(state: State<'_, AppRuntime>) -> AppSnapshot {
    state.snapshot()
}

#[tauri::command]
fn get_config_path(state: State<'_, AppRuntime>) -> String {
    state.store.config_path().to_string_lossy().to_string()
}

#[tauri::command]
fn get_log_path() -> String {
    log_file_path().to_string_lossy().to_string()
}

#[tauri::command]
async fn save_settings(
    app: AppHandle,
    state: State<'_, AppRuntime>,
    input: SettingsInput,
) -> Result<AppSnapshot, String> {
    let old_model = state.current_config().voice_model;
    let old_precision = state.current_config().tts_precision;
    let snapshot = state
        .save_settings(&app, input)
        .map_err(|error| error.to_string())?;

    if snapshot.config.voice_model != old_model || snapshot.config.tts_precision != old_precision {
        let app_clone = app.clone();
        tauri::async_runtime::spawn(async move {
            let runtime = app_clone.state::<AppRuntime>();
            if let Err(error) = runtime.warmup_model(&app_clone).await {
                eprintln!("warmup after model change failed: {error}");
            }
        });
    }

    Ok(snapshot)
}

#[tauri::command]
async fn speak_selection(
    app: AppHandle,
    state: State<'_, AppRuntime>,
) -> Result<AppSnapshot, String> {
    state
        .enqueue_selection(&app)
        .await
        .map(|_| state.snapshot())
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn speak_manual_text(
    app: AppHandle,
    state: State<'_, AppRuntime>,
    text: String,
) -> Result<AppSnapshot, String> {
    state
        .enqueue_manual_text(&app, text)
        .map(|_| state.snapshot())
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn skip_current_item(app: AppHandle, state: State<'_, AppRuntime>) -> Result<AppSnapshot, String> {
    state
        .skip_current_item(&app)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn stop_playback(app: AppHandle, state: State<'_, AppRuntime>) -> Result<AppSnapshot, String> {
    state.stop_playback(&app).map_err(|error| error.to_string())
}

#[tauri::command]
fn toggle_pause_playback(
    app: AppHandle,
    state: State<'_, AppRuntime>,
) -> Result<AppSnapshot, String> {
    state
        .toggle_pause_playback(&app)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn show_main_window(app: AppHandle) -> Result<(), String> {
    show_main_window_inner(&app).map_err(|error| error.to_string())
}

#[tauri::command]
fn open_log_file() -> Result<(), String> {
    open_log_file_inner().map_err(|error| error.to_string())
}

/// Polls the player and resets the app status to Ready once audio finishes.
/// Without this, the status stays "Speaking" after playback completes and
/// the hotkey / queue worker can get confused.
async fn playback_watcher(app: AppHandle) {
    let runtime = app.state::<AppRuntime>();
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let status = {
            let rt = runtime.runtime.lock().unwrap();
            rt.status
        };
        if status == AppStatus::Speaking
            && runtime.playback.is_empty()
            && runtime.speech_queue.len() == 0
        {
            runtime.mark_status(&app, AppStatus::Ready, "Playback finished.");
        }
    }
}

async fn queue_worker(app: AppHandle, mut rx: mpsc::UnboundedReceiver<QueueSignal>) {
    let runtime = app.state::<AppRuntime>();

    while let Some(signal) = rx.recv().await {
        if matches!(signal, QueueSignal::Stop) {
            tracing::debug!("queue_worker: received Stop signal, skipping");
            continue;
        }

        let queue_gen = runtime.speech_queue.generation();
        tracing::info!("queue_worker: received NewItem signal, gen={queue_gen}");

        // Wait for the audio player to drain before the queue starts.
        // This avoids interrupting a hotkey speak that is still playing.
        loop {
            if runtime.speech_queue.generation() != queue_gen {
                tracing::info!("queue_worker: generation changed during idle wait, aborting");
                break;
            }
            if runtime.playback.is_empty() {
                tracing::info!("queue_worker: player is empty, proceeding");
                break;
            }
            tracing::debug!("queue_worker: waiting for player to drain...");
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        if runtime.speech_queue.generation() != queue_gen {
            continue;
        }

        let mut items_played = 0;

        loop {
            if runtime.speech_queue.generation() != queue_gen {
                tracing::info!("queue_worker: generation changed, breaking inner loop");
                break;
            }

            let request = match runtime.speech_queue.pop() {
                Some(request) => request,
                None => {
                    tracing::info!(
                        "queue_worker: queue empty, inner loop done (played {items_played} items)"
                    );
                    break;
                }
            };

            if runtime.speech_queue.generation() != queue_gen {
                break;
            }

            let item_token = runtime.speech_queue.current_item_token();
            tracing::info!(
                "queue_worker: processing item {} ({} chars)",
                items_played + 1,
                request.text.len()
            );
            match runtime
                .process_text(
                    &app,
                    request.text,
                    SpeechMode::Queued {
                        queue_gen,
                        item_token,
                    },
                    request.overrides,
                )
                .await
            {
                Ok(outcome) => {
                    tracing::info!(
                        "queue_worker: item {} done, audio={:.1}s",
                        items_played + 1,
                        outcome.audio_duration_secs
                    );
                }
                Err(e) => {
                    tracing::error!("queue worker: speech failed: {e:#}");
                }
            }

            items_played += 1;

            loop {
                if runtime.speech_queue.generation() != queue_gen {
                    break;
                }
                if runtime.playback.is_empty() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }

            // Drain any signals that arrived during processing.
            loop {
                match rx.try_recv() {
                    Ok(QueueSignal::Stop) => {}
                    Ok(QueueSignal::NewItem) => {}
                    Err(_) => break,
                }
            }
        }

        if runtime.speech_queue.generation() == queue_gen
            && items_played > 0
            && runtime.playback.is_empty()
        {
            tracing::info!("queue_worker: batch complete ({items_played} items)");
            runtime.mark_status(&app, AppStatus::Ready, "Queue complete.");
        }
    }
}

pub fn run() {
    if let Err(error) = initialize_tracing() {
        eprintln!("DevVoice failed to initialize logging: {error:#}");
    }

    let (runtime, queue_rx) = match AppRuntime::load() {
        Ok(result) => result,
        Err(error) => {
            eprintln!("DevVoice failed to initialize: {error:#}");
            std::process::exit(1);
        }
    };

    let builder = tauri::Builder::default()
        .manage(runtime)
        .setup(move |app| {
            build_tray(&app.handle())?;

            {
                let runtime = app.state::<AppRuntime>();
                runtime.rebind_shortcut(&app.handle())?;
            }

            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let runtime = app_handle.state::<AppRuntime>();
                if let Err(error) = runtime.warmup_model(&app_handle).await {
                    eprintln!("warmup failed: {error}");
                }
            });

            let http_port = {
                let runtime = app.state::<AppRuntime>();
                runtime.current_config().http_port
            };
            if http_port > 0 {
                let app_handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    http_api::start(app_handle, http_port).await;
                });
            }

            {
                let app_handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    queue_worker(app_handle, queue_rx).await;
                });
            }

            {
                let app_handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    playback_watcher(app_handle).await;
                });
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_snapshot,
            get_config_path,
            get_log_path,
            save_settings,
            speak_selection,
            speak_manual_text,
            skip_current_item,
            stop_playback,
            toggle_pause_playback,
            show_main_window,
            open_log_file,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        });

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    let builder = builder.plugin(tauri_plugin_global_shortcut::Builder::new().build());

    if let Err(error) = builder.run(tauri::generate_context!()) {
        eprintln!("DevVoice runtime error: {error}");
        std::process::exit(1);
    }
}
