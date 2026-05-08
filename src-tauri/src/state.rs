use crate::config::AppConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AppStatus {
    Idle,
    LoadingModel,
    CapturingSelection,
    RewritingText,
    Synthesizing,
    Speaking,
    Ready,
    Error,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSnapshot {
    pub status: AppStatus,
    pub status_detail: String,
    pub config: AppConfig,
    pub api_key_present: bool,
    pub model_ready: bool,
    pub playback_paused: bool,
    pub last_selection: Option<String>,
    pub last_prepared_text: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsInput {
    pub gemini_enabled: bool,
    pub gemini_model: String,
    pub gemini_prompt: String,
    pub voice_gender: crate::config::VoiceGender,
    pub shortcut: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakOutcome {
    pub raw_selection: String,
    pub prepared_text: String,
    pub used_gemini: bool,
    #[serde(skip)]
    pub audio_duration_secs: f64,
}
