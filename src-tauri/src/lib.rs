mod config;
mod gemini;
mod http_api;
mod state;
mod tts;

use crate::config::{AppConfig, ConfigStore, normalize_shortcut};
use crate::gemini::GeminiProvider;
use crate::state::{AppSnapshot, AppStatus, SettingsInput, SpeakOutcome};
use crate::tts::{PlaybackController, TtsService, initialize_tracing, log_file_path, split_into_speech_chunks};
use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use std::collections::VecDeque;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::Duration;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State, image::Image};
use tokio::sync::mpsc;
use tracing::info;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

const TRAY_ID: &str = "devvoice-tray";

enum SpeechMode {
    Direct(u64),
    Queued(u64),
}

pub(crate) enum QueueSignal {
    NewItem,
    Stop,
}

pub struct SpeechQueue {
    items: Mutex<VecDeque<String>>,
    tx: mpsc::UnboundedSender<QueueSignal>,
    generation: AtomicU64,
}

impl SpeechQueue {
    fn new(tx: mpsc::UnboundedSender<QueueSignal>) -> Self {
        Self {
            items: Mutex::new(VecDeque::new()),
            tx,
            generation: AtomicU64::new(0),
        }
    }

    pub fn enqueue(&self, text: String) -> usize {
        let mut items = self.items.lock().unwrap();
        items.push_back(text);
        let len = items.len();
        let _ = self.tx.send(QueueSignal::NewItem);
        len
    }

    fn pop(&self) -> Option<String> {
        self.items.lock().unwrap().pop_front()
    }

    fn clear_and_invalidate(&self) {
        self.items.lock().unwrap().clear();
        self.generation.fetch_add(1, Ordering::SeqCst);
        let _ = self.tx.send(QueueSignal::Stop);
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
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
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            status: AppStatus::Idle,
            status_detail: "Ready to speak selected text.".to_owned(),
            last_selection: None,
            last_prepared_text: None,
            last_error: None,
            model_ready: false,
            playback_paused: false,
            available_voices: Vec::new(),
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

        Ok((Self {
            store,
            config: RwLock::new(config),
            runtime: Mutex::new(RuntimeState::default()),
            client,
            tts,
            playback,
            speech_queue,
        }, queue_rx))
    }

    pub fn snapshot(&self) -> AppSnapshot {
        let config = self.config.read().unwrap().clone();
        let runtime = self.runtime.lock().unwrap();

        AppSnapshot {
            status: runtime.status,
            status_detail: runtime.status_detail.clone(),
            config,
            api_key_present: self.store.read_api_key().map(|key| key.is_some()).unwrap_or(false),
            model_ready: runtime.model_ready,
            playback_paused: runtime.playback_paused,
            last_selection: runtime.last_selection.clone(),
            last_prepared_text: runtime.last_prepared_text.clone(),
            last_error: runtime.last_error.clone(),
            available_voices: runtime.available_voices.clone(),
        }
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

        config.gemini_enabled = input.gemini_enabled;
        if !input.gemini_model.trim().is_empty() {
            config.gemini_model = input.gemini_model.trim().to_owned();
        }
        if !input.gemini_prompt.trim().is_empty() {
            config.gemini_prompt = input.gemini_prompt.trim().to_owned();
        }
        config.voice_model = input.voice_model;
        config.voice_preset = input.voice_preset;
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

        if old_shortcut != config.shortcut {
            self.rebind_shortcut(app)?;
        }

        info!("Settings saved. {api_key_status}");
        self.mark_status(app, AppStatus::Ready, format!("Settings saved. {api_key_status}"));
        Ok(self.snapshot())
    }

    pub async fn warmup_model(&self, app: &AppHandle) -> Result<()> {
        let voice_model = self.current_config().voice_model;
        self.mark_status(
            app,
            AppStatus::LoadingModel,
            "Downloading and warming the TTS model on Metal...",
        );

        match self.tts.warmup(voice_model).await {
            Ok(voices) => {
                {
                    let mut runtime = self.runtime.lock().unwrap();
                    runtime.model_ready = true;
                    runtime.available_voices = voices.clone();
                }
                self.mark_status(
                    app,
                    AppStatus::Ready,
                    format!("Model ready with {} voice presets.", voices.len()),
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
        matches!(
            runtime.status,
            AppStatus::Speaking | AppStatus::Synthesizing | AppStatus::RewritingText | AppStatus::CapturingSelection
        )
    }

    pub async fn speak_selection(&self, app: &AppHandle) -> Result<SpeakOutcome> {
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

        self.process_text(app, selection.raw, SpeechMode::Direct(job)).await
    }

    pub async fn speak_manual_text(&self, app: &AppHandle, text: String) -> Result<SpeakOutcome> {
        self.speech_queue.clear_and_invalidate();
        let job = self.playback.begin_job()?;
        self.process_text(app, text, SpeechMode::Direct(job)).await
    }

    async fn process_text(
        &self,
        app: &AppHandle,
        raw_selection: String,
        mode: SpeechMode,
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
        let (prepared_text, used_gemini) = match self.prepare_text(app, &raw_selection, &config).await
        {
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

        let chunks = split_into_speech_chunks(&prepared_text);
        let total = chunks.len();
        let mut first_voice: Option<String> = None;
        let mut total_audio_secs: f64 = 0.0;

        for (i, chunk) in chunks.into_iter().enumerate() {
            let cancelled = match &mode {
                SpeechMode::Direct(job) => !self.playback.is_current_job(*job),
                SpeechMode::Queued(queue_gen) => self.speech_queue.generation() != *queue_gen,
            };
            if cancelled {
                self.mark_status(app, AppStatus::Ready, "Playback cancelled.");
                break;
            }

            let status_detail = if total > 1 {
                format!("Synthesizing chunk {} of {total}...", i + 1)
            } else {
                "Synthesizing speech with the local neural voice...".to_owned()
            };
            self.mark_status(app, AppStatus::Synthesizing, status_detail);

            let (audio, voice_name) =
                match self.tts.synthesize(&chunk, config.voice_model, &config.voice_preset).await {
                    Ok(result) => result,
                    Err(error) => {
                        self.mark_error(app, error.to_string());
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

            let appended = match &mode {
                SpeechMode::Direct(job) => self.playback.append_audio(*job, audio)?,
                SpeechMode::Queued(_) => {
                    self.playback.append_audio_unchecked(audio)?;
                    true
                }
            };
            if !appended {
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
                    // Check the actual player state, not the app status enum,
                    // because the status can be stale (stays "Speaking" after
                    // audio finishes).
                    if !runtime.playback.is_empty() {
                        let _ = runtime.stop_playback(&app);
                    } else if let Err(error) = runtime.speak_selection(&app).await {
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
    let speak = MenuItem::with_id(app, "speak-selection", "Speak Selection", true, None::<&str>)?;
    let pause = MenuItem::with_id(app, "pause", "Pause/Resume", true, None::<&str>)?;
    let stop = MenuItem::with_id(app, "stop", "Stop", true, None::<&str>)?;
    let open_log = MenuItem::with_id(app, "open-log", "Open Log File", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;

    let menu = Menu::with_items(
        app,
        &[&show, &speak, &pause, &stop, &open_log, &separator, &quit],
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
            "speak-selection" => {
                let app = app.clone();
                tauri::async_runtime::spawn(async move {
                    let runtime = app.state::<AppRuntime>();
                    let _ = runtime.speak_selection(&app).await;
                });
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
    let snapshot = state.save_settings(&app, input).map_err(|error| error.to_string())?;

    if snapshot.config.voice_model != old_model {
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
) -> Result<SpeakOutcome, String> {
    state
        .speak_selection(&app)
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn speak_manual_text(
    app: AppHandle,
    state: State<'_, AppRuntime>,
    text: String,
) -> Result<SpeakOutcome, String> {
    state
        .speak_manual_text(&app, text)
        .await
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
        let synthesis_start = std::time::Instant::now();
        let mut cumulative_audio_secs: f64 = 0.0;

        loop {
            if runtime.speech_queue.generation() != queue_gen {
                tracing::info!("queue_worker: generation changed, breaking inner loop");
                break;
            }

            let text = match runtime.speech_queue.pop() {
                Some(text) => text,
                None => {
                    tracing::info!("queue_worker: queue empty, inner loop done (played {items_played} items)");
                    break;
                }
            };

            // Between items, wait for previous audio to finish, then announce.
            if items_played > 0 {
                let elapsed = synthesis_start.elapsed();
                let target = Duration::from_secs_f64(cumulative_audio_secs);
                if let Some(remaining) = target.checked_sub(elapsed) {
                    tracing::info!("queue_worker: waiting {remaining:?} for previous audio to finish");
                    tokio::time::sleep(remaining).await;
                }

                if runtime.speech_queue.generation() != queue_gen {
                    break;
                }

                tracing::info!("queue_worker: announcing next response");
                let announce = "Now reading the next response.".to_owned();
                match runtime
                    .process_text(&app, announce, SpeechMode::Queued(queue_gen))
                    .await
                {
                    Ok(outcome) => {
                        cumulative_audio_secs += outcome.audio_duration_secs;
                    }
                    Err(e) => {
                        tracing::error!("queue worker: announcement failed: {e}");
                    }
                }
            }

            if runtime.speech_queue.generation() != queue_gen {
                break;
            }

            tracing::info!(
                "queue_worker: processing item {} ({} chars)",
                items_played + 1,
                text.len()
            );
            match runtime
                .process_text(&app, text, SpeechMode::Queued(queue_gen))
                .await
            {
                Ok(outcome) => {
                    tracing::info!(
                        "queue_worker: item {} done, audio={:.1}s",
                        items_played + 1,
                        outcome.audio_duration_secs
                    );
                    cumulative_audio_secs += outcome.audio_duration_secs;
                }
                Err(e) => {
                    tracing::error!("queue worker: speech failed: {e}");
                }
            }

            items_played += 1;

            // Drain any signals that arrived during processing.
            loop {
                match rx.try_recv() {
                    Ok(QueueSignal::Stop) => {}
                    Ok(QueueSignal::NewItem) => {}
                    Err(_) => break,
                }
            }
        }

        // Reset status after finishing a queue batch so subsequent
        // queue signals are not blocked by a stale Speaking status.
        if runtime.speech_queue.generation() == queue_gen && items_played > 0 {
            // Wait for the last item's audio to actually finish playing.
            let elapsed = synthesis_start.elapsed();
            let target = Duration::from_secs_f64(cumulative_audio_secs);
            if let Some(remaining) = target.checked_sub(elapsed) {
                tracing::info!("queue_worker: waiting {remaining:?} for final audio to finish");
                tokio::time::sleep(remaining).await;
            }
            tracing::info!("queue_worker: batch complete ({items_played} items, {cumulative_audio_secs:.1}s audio)");
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
