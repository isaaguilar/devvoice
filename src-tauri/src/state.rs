use crate::config::AppConfig;
use serde::{Deserialize, Deserializer, Serialize};

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
    pub available_voices: Vec<String>,
    pub model_instructions: ModelInstructions,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsInput {
    pub gemini_enabled: bool,
    pub gemini_model: String,
    pub gemini_prompt: String,
    pub voice_model: crate::config::VoiceModel,
    pub voice_preset: String,
    pub tts_precision: crate::config::TtsPrecision,
    pub shortcut: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechOverrides {
    #[serde(default, alias = "voicePreset", alias = "voice_preset")]
    pub voice_preset: Option<String>,
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, alias = "referenceAudioPath", alias = "reference_audio_path")]
    pub reference_audio_path: Option<String>,
    #[serde(
        default,
        alias = "cfgScale",
        alias = "cfg_scale",
        deserialize_with = "deserialize_optional_f64"
    )]
    pub cfg_scale: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub temperature: Option<f64>,
    #[serde(
        default,
        alias = "maxTokens",
        alias = "max_tokens",
        deserialize_with = "deserialize_optional_usize"
    )]
    pub max_tokens: Option<usize>,
    #[serde(
        default,
        alias = "chunkSize",
        alias = "chunk_size",
        deserialize_with = "deserialize_optional_usize"
    )]
    pub chunk_size: Option<usize>,
    #[serde(
        default,
        alias = "saveAudio",
        alias = "save_audio",
        deserialize_with = "deserialize_optional_bool"
    )]
    pub save_audio: Option<bool>,
    #[serde(default, alias = "outputDir", alias = "output_dir")]
    pub output_dir: Option<String>,
}

impl SpeechOverrides {
    pub fn normalized(self) -> Self {
        Self {
            voice_preset: trim_option(self.voice_preset),
            style: trim_option(self.style),
            description: trim_option(self.description),
            reference_audio_path: trim_option(self.reference_audio_path),
            cfg_scale: self.cfg_scale,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            chunk_size: self.chunk_size,
            save_audio: self.save_audio,
            output_dir: trim_option(self.output_dir),
        }
    }
}

fn trim_option(value: Option<String>) -> Option<String> {
    value.map(|value| value.trim().to_owned()).filter(|value| !value.is_empty())
}

fn deserialize_optional_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<StringOrNumber>::deserialize(deserializer)?;
    value
        .map(|value| match value {
            StringOrNumber::String(text) => text.parse::<f64>().map_err(serde::de::Error::custom),
            StringOrNumber::F64(number) => Ok(number),
            StringOrNumber::U64(number) => Ok(number as f64),
        })
        .transpose()
}

fn deserialize_optional_usize<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<StringOrNumber>::deserialize(deserializer)?;
    value
        .map(|value| match value {
            StringOrNumber::String(text) => text.parse::<usize>().map_err(serde::de::Error::custom),
            StringOrNumber::F64(number) => {
                if number.fract() == 0.0 && number >= 0.0 {
                    Ok(number as usize)
                } else {
                    Err(serde::de::Error::custom("expected a whole number"))
                }
            }
            StringOrNumber::U64(number) => usize::try_from(number)
                .map_err(|_| serde::de::Error::custom("value is too large")),
        })
        .transpose()
}

fn deserialize_optional_bool<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<BoolLike>::deserialize(deserializer)?;
    value
        .map(|value| match value {
            BoolLike::Bool(flag) => Ok(flag),
            BoolLike::String(text) => match text.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                _ => Err(serde::de::Error::custom("expected a boolean value")),
            },
        })
        .transpose()
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrNumber {
    String(String),
    F64(f64),
    U64(u64),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum BoolLike {
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInstructions {
    pub model_label: String,
    pub summary: String,
    pub curl_example: String,
    pub attributes: Vec<ModelInstructionAttribute>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInstructionAttribute {
    pub query_param: String,
    pub label: String,
    pub description: String,
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
