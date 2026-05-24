use crate::config::{TtsPrecision, VoiceModel, DEFAULT_CHUNK_SIZE};
use crate::state::SpeechOverrides;
use any_tts::mel::resample_linear;
use any_tts::models::vibevoice::config::VibeVoicePreprocessorConfig;
use any_tts::{
    load_model, AudioSamples, DType, DeviceSelection, ModelType, ReferenceAudio, SynthesisRequest,
    TtsConfig, TtsModel, VoiceEmbedding,
};
use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType as CandleDType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::parler_tts::{Config as ParlerConfig, Model as ParlerModel};
use get_selected_text::get_selected_text;
use hf_hub::HFClientSync;
use macos_accessibility_client::accessibility::{
    application_is_trusted, application_is_trusted_with_prompt,
};
use reqwest::blocking::Client as BlockingClient;
use rodio::buffer::SamplesBuffer;
use rodio::{DeviceSinkBuilder, MixerDeviceSink, Player};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::num::{NonZeroU16, NonZeroU32};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// VibeVoice Realtime 0.5B
const VR_OWNER: &str = "microsoft";
const VR_NAME: &str = "VibeVoice-Realtime-0.5B";
const VR_ID: &str = "microsoft/VibeVoice-Realtime-0.5B";
const VR_DIR: &str = "vibevoice-realtime-0.5b";

// VibeVoice 1.5B
const VV_OWNER: &str = "microsoft";
const VV_NAME: &str = "VibeVoice-1.5B";
const VV_ID: &str = "microsoft/VibeVoice-1.5B";
const VV_DIR: &str = "vibevoice-1.5b";

// Kokoro 82M
const KO_OWNER: &str = "hexgrad";
const KO_NAME: &str = "Kokoro-82M";
const KO_ID: &str = "hexgrad/Kokoro-82M";
const KO_DIR: &str = "kokoro-82m";

// Parler-TTS Mini v1
const PA_OWNER: &str = "parler-tts";
const PA_NAME: &str = "parler-tts-mini-v1";
const PA_ID: &str = "parler-tts/parler-tts-mini-v1";
const PA_DIR: &str = "parler-tts-mini-v1";
const PARLER_MAX_STEPS: usize = 768;
const PARLER_DEFAULT_PRESET: &str = "senior_developer";
const PARLER_VOICE_PRESETS: &[(&str, &str)] = &[
    (
        "senior_developer",
        "A calm speaker delivers technical prose with precise enunciation, measured pacing, and a close-mic studio sound. The voice is clear, direct, and confident, like an experienced software engineer explaining architecture to another engineer.",
    ),
    (
        "clear_female",
        "A female speaker delivers clear, natural speech at a moderate pace with a clean studio recording. The voice sounds articulate, balanced, and easy to understand.",
    ),
    (
        "clear_male",
        "A male speaker delivers clear, natural speech at a moderate pace with a clean studio recording. The voice sounds articulate, balanced, and easy to understand.",
    ),
    (
        "warm_narrator",
        "A warm narrator delivers polished speech with steady pacing and a high-quality close recording. The voice is natural, smooth, and expressive without sounding theatrical.",
    ),
];

const TOKENIZER_FALLBACK_OWNER: &str = "Qwen";
const TOKENIZER_FALLBACK_MODEL: &str = "Qwen2.5-0.5B";
const VOICE_PRESET_BASE_URL: &str =
    "https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/voices/streaming_model";
const VOICE_PRESET_FILES: &[&str] = &[
    "en-Emma_woman.pt",
    "en-Grace_woman.pt",
    "en-Davis_man.pt",
    "en-Frank_man.pt",
    "en-Mike_man.pt",
    "en-Carter_man.pt",
    "in-Samuel_man.pt",
];
const STYLE_PROMPT: &str = "Read like an experienced senior software engineer presenting technical prose to another engineer. Keep terminology crisp, punctuation deliberate, and code-adjacent words precise.";
const VIBEVOICE_WARMUP_TEXT: &str = "Speaker 0: Warm up.";
const VIBEVOICE_MLX_MODEL_ID: &str = "gafiatulin/vibevoice-1.5b-mlx";
const VIBEVOICE_MLX_RUNTIME_LABEL: &str = "python:mlx-int8:no-semantic";
const VIBEVOICE_MLX_INSTALL_SPEC: &str =
    "git+https://github.com/gafiatulin/vibevoice-mlx.git@f513aa7877e77fefa1aebe87432855c407da3b87";
const VIBEVOICE_REFERENCE_EMBEDDING_MODEL_TYPE: &str = "vibevoice-reference-audio-v1";
const VIBEVOICE_MLX_WORKER_SCRIPT: &str = include_str!("vibevoice_mlx_worker.py");

pub struct CaptureResult {
    pub raw: String,
}

pub struct WarmupInfo {
    pub voices: Vec<String>,
    pub runtime_label: String,
    pub warmed_inference: bool,
    pub warmup_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VibeVoicePresetInfo {
    pub id: String,
    pub name: String,
    pub source_audio_path: String,
    pub clip_duration_secs: f32,
    pub sample_rate: u32,
    pub created_at_unix_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredVibeVoicePreset {
    #[serde(flatten)]
    info: VibeVoicePresetInfo,
    embedding: VoiceEmbedding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ModelCacheKey {
    voice_model: VoiceModel,
    precision: TtsPrecision,
}

enum LoadedModelKind {
    AnyTts(Arc<dyn TtsModel>),
    Parler(Arc<Mutex<ParlerRuntime>>),
}

struct LoadedModel {
    model: LoadedModelKind,
    voices: Arc<[String]>,
    runtime_label: String,
}

#[derive(Debug)]
enum SynthesisFailure {
    Prepare(anyhow::Error),
    Runtime(anyhow::Error),
}

impl SynthesisFailure {
    fn is_runtime(&self) -> bool {
        matches!(self, Self::Runtime(_))
    }

    fn into_error(self) -> anyhow::Error {
        match self {
            Self::Prepare(error) | Self::Runtime(error) => error,
        }
    }
}

#[derive(Clone)]
struct CachedReferenceAudio {
    modified_at: Option<SystemTime>,
    audio: ReferenceAudio,
}

#[derive(Serialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum VibeVoiceMlxRequest {
    Warmup {
        text: String,
        cfg_scale: f64,
        max_speech_tokens: usize,
    },
    Synthesize {
        text: String,
        reference_audio_path: Option<String>,
        output_path: String,
        cfg_scale: f64,
        max_speech_tokens: usize,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VibeVoiceMlxMetrics {
    #[serde(default)]
    load_ms: Option<u64>,
    #[serde(default)]
    encode_ms: Option<u64>,
    #[serde(default)]
    gen_ms: Option<u64>,
    #[serde(default)]
    audio_seconds: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VibeVoiceMlxResponse {
    ok: bool,
    #[serde(default)]
    runtime_label: Option<String>,
    #[serde(default)]
    metrics: Option<VibeVoiceMlxMetrics>,
    #[serde(default)]
    output_path: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

struct VibeVoiceMlxWorker {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl VibeVoiceMlxWorker {
    fn spawn(python_path: &Path, script_path: &Path) -> Result<Self> {
        let mut child = Command::new(python_path)
            .arg("-u")
            .arg(script_path)
            .arg("--model")
            .arg(VIBEVOICE_MLX_MODEL_ID)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "start VibeVoice MLX worker with {} {}",
                    python_path.display(),
                    script_path.display()
                )
            })?;
        let stdin = child
            .stdin
            .take()
            .context("capture stdin for VibeVoice MLX worker")?;
        let stdout = child
            .stdout
            .take()
            .context("capture stdout for VibeVoice MLX worker")?;
        Ok(Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
        })
    }

    fn send(&mut self, request: &VibeVoiceMlxRequest) -> Result<VibeVoiceMlxResponse> {
        let payload = serde_json::to_string(request).context("serialize VibeVoice MLX request")?;
        writeln!(self.stdin, "{payload}").context("write request to VibeVoice MLX worker")?;
        self.stdin
            .flush()
            .context("flush VibeVoice MLX worker stdin")?;
        let mut line = String::new();
        let bytes_read = self
            .stdout
            .read_line(&mut line)
            .context("read response from VibeVoice MLX worker")?;
        if bytes_read == 0 {
            let status = self
                .child
                .try_wait()
                .context("check VibeVoice MLX worker exit status")?;
            bail!("VibeVoice MLX worker exited before replying: {status:?}");
        }
        serde_json::from_str(line.trim_end()).context("parse VibeVoice MLX worker response")
    }
}

struct ParlerRuntime {
    model: ParlerModel,
    tokenizer: Tokenizer,
    sample_rate: u32,
    device: candle_core::Device,
}

pub struct TtsService {
    asset_root: PathBuf,
    cache: Mutex<HashMap<ModelCacheKey, Arc<OnceCell<Arc<LoadedModel>>>>>,
    auto_precision_overrides: Mutex<HashMap<VoiceModel, TtsPrecision>>,
    reference_audio_cache: Arc<Mutex<HashMap<PathBuf, CachedReferenceAudio>>>,
    vibevoice_preset_cache: Arc<Mutex<HashMap<String, VoiceEmbedding>>>,
    vibevoice_mlx_worker: Mutex<Option<VibeVoiceMlxWorker>>,
    vibevoice_runtime_label: Mutex<Option<String>>,
}

impl TtsService {
    pub fn new(asset_root: PathBuf) -> Self {
        Self {
            asset_root,
            cache: Mutex::new(HashMap::new()),
            auto_precision_overrides: Mutex::new(HashMap::new()),
            reference_audio_cache: Arc::new(Mutex::new(HashMap::new())),
            vibevoice_preset_cache: Arc::new(Mutex::new(HashMap::new())),
            vibevoice_mlx_worker: Mutex::new(None),
            vibevoice_runtime_label: Mutex::new(None),
        }
    }

    fn data_dir(&self) -> Result<PathBuf> {
        self.asset_root
            .parent()
            .map(Path::to_path_buf)
            .context("determine DevVoice data directory")
    }

    fn vibevoice_mlx_requested(&self, voice_model: VoiceModel, precision: TtsPrecision) -> bool {
        voice_model == VoiceModel::VibeVoice
            && cfg!(target_os = "macos")
            && precision == TtsPrecision::Auto
    }

    fn vibevoice_mlx_python_path(&self) -> Option<PathBuf> {
        if !cfg!(target_os = "macos") {
            return None;
        }
        self.asset_root
            .parent()
            .map(|data_dir| data_dir.join("vibevoice-mlx-venv/bin/python"))
            .filter(|path| path.is_file())
    }

    pub fn backend_status(&self, voice_model: VoiceModel, precision: TtsPrecision) -> String {
        match voice_model {
            VoiceModel::VibeVoice if !cfg!(target_os = "macos") => {
                "VibeVoice 1.5B requires macOS because DevVoice only supports the MLX backend for this model."
                    .to_owned()
            }
            VoiceModel::VibeVoice if precision != TtsPrecision::Auto => {
                "VibeVoice 1.5B requires auto precision because it now runs only through the MLX backend."
                    .to_owned()
            }
            VoiceModel::VibeVoice => {
                if let Some(runtime_label) = self.vibevoice_runtime_label.lock().unwrap().clone() {
                    format!("MLX ready on {runtime_label}")
                } else if self.vibevoice_mlx_python_path().is_some() {
                    "MLX installed, waiting for warmup.".to_owned()
                } else {
                    "MLX runtime missing. DevVoice will provision it automatically on first 1.5B use."
                        .to_owned()
                }
            }
            _ => self
                .loaded_runtime_label(voice_model, precision)
                .map(|runtime_label| format!("Built-in backend ready on {runtime_label}"))
                .unwrap_or_else(|| format!("{} uses the built-in Rust backend.", voice_model.display_name())),
        }
    }

    async fn ensure_vibevoice_mlx_runtime(&self, precision: TtsPrecision) -> Result<()> {
        if !cfg!(target_os = "macos") {
            bail!(
                "VibeVoice 1.5B now requires macOS because DevVoice only supports the MLX backend for this model."
            );
        }
        if precision != TtsPrecision::Auto {
            bail!(
                "VibeVoice 1.5B now requires auto precision because it runs only through the MLX backend."
            );
        }
        if self.vibevoice_mlx_python_path().is_some() {
            return Ok(());
        }
        let data_dir = self.data_dir()?;
        tokio::task::spawn_blocking(move || provision_vibevoice_mlx_runtime(&data_dir))
            .await
            .context("join MLX runtime provisioning task")??;
        if self.vibevoice_mlx_python_path().is_none() {
            bail!(
                "VibeVoice MLX provisioning completed, but the Python runtime was still not found."
            );
        }
        Ok(())
    }

    pub async fn warmup(
        &self,
        voice_model: VoiceModel,
        precision: TtsPrecision,
    ) -> Result<WarmupInfo> {
        if voice_model == VoiceModel::VibeVoice {
            return self
                .warmup_vibevoice(precision, &SpeechOverrides::default())
                .await;
        }
        let warmup_started = Instant::now();
        let loaded = self.ensure_loaded(voice_model, precision).await?;
        Ok(WarmupInfo {
            voices: loaded.voices.iter().cloned().collect(),
            runtime_label: loaded.runtime_label.clone(),
            warmed_inference: false,
            warmup_duration_ms: warmup_started.elapsed().as_millis() as u64,
        })
    }

    pub async fn warmup_vibevoice(
        &self,
        precision: TtsPrecision,
        overrides: &SpeechOverrides,
    ) -> Result<WarmupInfo> {
        let _ = overrides;
        self.ensure_vibevoice_mlx_runtime(precision).await?;
        self.warmup_vibevoice_mlx().await
    }

    async fn warmup_vibevoice_mlx(&self) -> Result<WarmupInfo> {
        let warmup_started = Instant::now();
        let response = self.send_vibevoice_mlx_request(VibeVoiceMlxRequest::Warmup {
            text: VIBEVOICE_WARMUP_TEXT.to_owned(),
            cfg_scale: 1.3,
            max_speech_tokens: 32,
        })?;
        let runtime_label = response
            .runtime_label
            .clone()
            .unwrap_or_else(|| VIBEVOICE_MLX_RUNTIME_LABEL.to_owned());
        *self.vibevoice_runtime_label.lock().unwrap() = Some(runtime_label.clone());
        let warmup_duration_ms = warmup_started.elapsed().as_millis() as u64;
        if let Some(metrics) = response.metrics.as_ref() {
            info!(
                "Completed VibeVoice MLX warmup in {} ms (load={:?} encode={:?} gen={:?}).",
                warmup_duration_ms, metrics.load_ms, metrics.encode_ms, metrics.gen_ms
            );
        } else {
            info!(
                "Completed VibeVoice MLX warmup in {} ms.",
                warmup_duration_ms
            );
        }
        Ok(WarmupInfo {
            voices: vec!["Speaker 0".to_owned()],
            runtime_label,
            warmed_inference: true,
            warmup_duration_ms,
        })
    }

    async fn synthesize_with_vibevoice_mlx(
        &self,
        spoken_text: &str,
        overrides: &SpeechOverrides,
    ) -> Result<(AudioSamples, Option<String>)> {
        let request_started = Instant::now();
        let reference_audio_path = self
            .resolve_vibevoice_reference_audio_source(overrides)?
            .map(|path| path.to_string_lossy().to_string());
        let output_dir = self.asset_root.join("mlx-output");
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("create MLX output directory at {}", output_dir.display()))?;
        let output_path = output_dir.join(format!(
            "vibevoice-mlx-{}.wav",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        let max_speech_tokens = vibevoice_effective_max_tokens(spoken_text, overrides.max_tokens);
        let response = self.send_vibevoice_mlx_request(VibeVoiceMlxRequest::Synthesize {
            text: spoken_text.to_owned(),
            reference_audio_path,
            output_path: output_path.to_string_lossy().to_string(),
            cfg_scale: overrides.cfg_scale.unwrap_or(1.3),
            max_speech_tokens,
        })?;
        let runtime_label = response
            .runtime_label
            .clone()
            .unwrap_or_else(|| VIBEVOICE_MLX_RUNTIME_LABEL.to_owned());
        *self.vibevoice_runtime_label.lock().unwrap() = Some(runtime_label.clone());
        if let Some(metrics) = response.metrics.as_ref() {
            info!(
                "VibeVoice MLX synthesized in {} ms (load={:?} encode={:?} gen={:?} audio_seconds={:?}) on {}.",
                request_started.elapsed().as_millis(),
                metrics.load_ms,
                metrics.encode_ms,
                metrics.gen_ms,
                metrics.audio_seconds,
                runtime_label
            );
        }
        let returned_output_path = response
            .output_path
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or(output_path.clone());
        let audio = AudioSamples::from_wav_file(&returned_output_path).with_context(|| {
            format!(
                "load MLX-generated VibeVoice audio from {}",
                returned_output_path.display()
            )
        })?;
        if let Err(error) = fs::remove_file(&returned_output_path) {
            warn!(
                "Failed to remove temporary MLX audio {}: {error}",
                returned_output_path.display()
            );
        }
        Ok((audio, None))
    }

    fn resolve_vibevoice_reference_audio_source(
        &self,
        overrides: &SpeechOverrides,
    ) -> Result<Option<PathBuf>> {
        if let Some(path) = overrides
            .reference_audio_path
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Ok(Some(
                fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path)),
            ));
        }
        if let Some(preset_id) = overrides
            .reference_preset_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return self
                .load_vibevoice_preset_info_by_id(preset_id)
                .map(|preset| Some(PathBuf::from(preset.source_audio_path)));
        }
        if let Some(preset_name) = overrides
            .reference_preset_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return self
                .load_vibevoice_preset_info_by_name(preset_name)
                .map(|preset| Some(PathBuf::from(preset.source_audio_path)));
        }
        Ok(None)
    }

    fn send_vibevoice_mlx_request(
        &self,
        request: VibeVoiceMlxRequest,
    ) -> Result<VibeVoiceMlxResponse> {
        let python_path = self.vibevoice_mlx_python_path().ok_or_else(|| {
            anyhow!("VibeVoice MLX is unavailable because the Python runtime was not found.")
        })?;
        let script_path = self.ensure_vibevoice_mlx_worker_script()?;
        let mut worker_guard = self.vibevoice_mlx_worker.lock().unwrap();
        if worker_guard.is_none() {
            *worker_guard = Some(VibeVoiceMlxWorker::spawn(&python_path, &script_path)?);
        }
        let response = match worker_guard
            .as_mut()
            .context("VibeVoice MLX worker failed to initialize")?
            .send(&request)
        {
            Ok(response) => response,
            Err(first_error) => {
                warn!("Restarting VibeVoice MLX worker after error: {first_error:#}");
                *worker_guard = Some(VibeVoiceMlxWorker::spawn(&python_path, &script_path)?);
                worker_guard
                    .as_mut()
                    .context("VibeVoice MLX worker failed to restart")?
                    .send(&request)?
            }
        };
        if !response.ok {
            bail!(
                "VibeVoice MLX worker error: {}",
                response.error.as_deref().unwrap_or("unknown worker error")
            );
        }
        Ok(response)
    }

    fn ensure_vibevoice_mlx_worker_script(&self) -> Result<PathBuf> {
        let data_dir = self
            .asset_root
            .parent()
            .context("determine DevVoice data directory for MLX worker")?;
        fs::create_dir_all(data_dir)
            .with_context(|| format!("create data directory at {}", data_dir.display()))?;
        let script_path = data_dir.join("vibevoice-mlx-worker.py");
        let needs_write = match fs::read_to_string(&script_path) {
            Ok(existing) => existing != VIBEVOICE_MLX_WORKER_SCRIPT,
            Err(_) => true,
        };
        if needs_write {
            fs::write(&script_path, VIBEVOICE_MLX_WORKER_SCRIPT)
                .with_context(|| format!("write MLX worker script to {}", script_path.display()))?;
        }
        Ok(script_path)
    }

    pub fn capture_selection(&self) -> Result<CaptureResult> {
        ensure_accessibility()?;
        let selected_text = get_selected_text()
            .map_err(|error| anyhow!("capture selected text from the active app: {error}"))?;
        let selected_text = selected_text.trim().to_string();
        if selected_text.is_empty() {
            bail!("No highlighted text was detected in the active application.");
        }

        info!(
            "Captured selection ({} chars): {}",
            selected_text.chars().count(),
            selected_text
        );

        Ok(CaptureResult { raw: selected_text })
    }

    pub async fn synthesize(
        &self,
        text: &str,
        voice_model: VoiceModel,
        precision: TtsPrecision,
        voice_preset: &str,
        overrides: &SpeechOverrides,
    ) -> Result<(AudioSamples, Option<String>)> {
        let spoken_text = normalize_technical_text(text);
        let overrides = overrides.clone().normalized();
        if voice_model == VoiceModel::VibeVoice {
            self.ensure_vibevoice_mlx_runtime(precision).await?;
            return self
                .synthesize_with_vibevoice_mlx(&spoken_text, &overrides)
                .await;
        }
        let loaded = self.ensure_loaded(voice_model, precision).await?;
        let initial_runtime = loaded.runtime_label.clone();

        match self
            .synthesize_with_loaded(&spoken_text, voice_model, voice_preset, &overrides, loaded)
            .await
        {
            Ok(result) => Ok(result),
            Err(error)
                if voice_model == VoiceModel::VibeVoice
                    && precision == TtsPrecision::Auto
                    && initial_runtime.starts_with("metal")
                    && error.is_runtime() =>
            {
                warn!(
                    "VibeVoice 1.5B runtime failed on {initial_runtime}: {}. Falling back to CPU f32 for auto mode.",
                    error.into_error()
                );
                self.force_vibevoice_auto_cpu_fallback();
                let loaded = self.ensure_loaded(voice_model, precision).await?;
                self.synthesize_with_loaded(
                    &spoken_text,
                    voice_model,
                    voice_preset,
                    &overrides,
                    loaded,
                )
                .await
                .map_err(SynthesisFailure::into_error)
            }
            Err(error) => Err(error.into_error()),
        }
    }

    pub fn prune_cache(&self, voice_model: VoiceModel, precision: TtsPrecision) {
        let keep = ModelCacheKey {
            voice_model,
            precision,
        };
        self.cache.lock().unwrap().retain(|key, _| *key == keep);
    }

    pub fn loaded_runtime_label(
        &self,
        voice_model: VoiceModel,
        precision: TtsPrecision,
    ) -> Option<String> {
        if self.vibevoice_mlx_requested(voice_model, precision) {
            if let Some(runtime_label) = self.vibevoice_runtime_label.lock().unwrap().clone() {
                return Some(runtime_label);
            }
        }
        let key = ModelCacheKey {
            voice_model,
            precision,
        };
        let cell = self.cache.lock().unwrap().get(&key).cloned()?;
        cell.get().map(|loaded| loaded.runtime_label.clone())
    }

    pub fn reset_auto_precision_override(&self, voice_model: VoiceModel) {
        self.auto_precision_overrides
            .lock()
            .unwrap()
            .remove(&voice_model);
    }

    pub fn create_vibevoice_preset(
        &self,
        reference_audio_path: &str,
        name: Option<&str>,
    ) -> Result<VibeVoicePresetInfo> {
        let canonical_path = fs::canonicalize(reference_audio_path)
            .unwrap_or_else(|_| PathBuf::from(reference_audio_path));
        let source_audio = load_reference_audio_cached(
            &self.reference_audio_cache,
            canonical_path.to_string_lossy().as_ref(),
        )?;
        let preprocessor_config = load_vibevoice_preprocessor_config(&self.asset_root)?;
        let normalized = normalize_vibevoice_reference_audio(&source_audio, &preprocessor_config);
        let embedding = VoiceEmbedding::new(
            normalized.clone(),
            vec![normalized.len()],
            VIBEVOICE_REFERENCE_EMBEDDING_MODEL_TYPE,
        );
        let preset_name = name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| {
                canonical_path
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "VibeVoice preset".to_owned());
        let preset_dir = self.ensure_vibevoice_preset_dir()?;
        ensure_unique_vibevoice_preset_name(&preset_dir, &preset_name)?;
        let preset_slug = slugify_preset_name(&preset_name);
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let created_at_unix_secs = created_at.as_secs();
        let preset_id = format!("{preset_slug}-{}", created_at.as_millis());
        let managed_audio_path = copy_vibevoice_preset_audio(
            &canonical_path,
            &self.ensure_vibevoice_preset_audio_dir()?,
            &preset_id,
        )?;
        let preset = StoredVibeVoicePreset {
            info: VibeVoicePresetInfo {
                id: preset_id.clone(),
                name: preset_name,
                source_audio_path: managed_audio_path.to_string_lossy().to_string(),
                clip_duration_secs: source_audio.duration_secs(),
                sample_rate: preprocessor_config.audio_processor.sampling_rate,
                created_at_unix_secs,
            },
            embedding,
        };
        let preset_path = vibevoice_preset_path(&preset_dir, &preset_id);
        fs::write(
            &preset_path,
            serde_json::to_vec_pretty(&preset).context("serialize VibeVoice preset")?,
        )
        .with_context(|| format!("write VibeVoice preset {}", preset_path.display()))?;
        self.vibevoice_preset_cache
            .lock()
            .unwrap()
            .insert(preset_id, preset.embedding.clone());
        Ok(preset.info)
    }

    pub fn list_vibevoice_presets(&self) -> Result<Vec<VibeVoicePresetInfo>> {
        let preset_dir = self.asset_root.join("vibevoice-presets");
        if !preset_dir.exists() {
            return Ok(Vec::new());
        }
        let mut presets = Vec::new();
        for entry in fs::read_dir(&preset_dir)
            .with_context(|| format!("read VibeVoice preset directory {}", preset_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let stored = load_stored_vibevoice_preset(&path)?;
            presets.push(stored.info);
        }
        presets.sort_by(|left, right| right.created_at_unix_secs.cmp(&left.created_at_unix_secs));
        Ok(presets)
    }

    fn load_vibevoice_preset_info_by_id(&self, preset_id: &str) -> Result<VibeVoicePresetInfo> {
        let preset_id = validate_vibevoice_preset_id(preset_id)?;
        let preset_path = vibevoice_preset_path(&self.ensure_vibevoice_preset_dir()?, &preset_id);
        let stored = load_stored_vibevoice_preset(&preset_path)?;
        Ok(stored.info)
    }

    fn load_vibevoice_preset_info_by_name(&self, preset_name: &str) -> Result<VibeVoicePresetInfo> {
        let preset_dir = self.ensure_vibevoice_preset_dir()?;
        let stored = find_vibevoice_preset_by_name(&preset_dir, preset_name)?;
        Ok(stored.info)
    }

    fn load_vibevoice_preset_embedding(&self, preset_id: &str) -> Result<VoiceEmbedding> {
        let preset_id = validate_vibevoice_preset_id(preset_id)?;
        if let Some(embedding) = self.vibevoice_preset_cache.lock().unwrap().get(&preset_id) {
            return Ok(embedding.clone());
        }
        let preset_path = vibevoice_preset_path(&self.ensure_vibevoice_preset_dir()?, &preset_id);
        let stored = load_stored_vibevoice_preset(&preset_path)?;
        let embedding = stored.embedding.clone();
        self.vibevoice_preset_cache
            .lock()
            .unwrap()
            .insert(preset_id, embedding.clone());
        Ok(embedding)
    }

    fn load_vibevoice_preset_embedding_by_name(&self, preset_name: &str) -> Result<VoiceEmbedding> {
        let preset_dir = self.ensure_vibevoice_preset_dir()?;
        let stored = find_vibevoice_preset_by_name(&preset_dir, preset_name)?;
        let preset_id = stored.info.id.clone();
        let embedding = stored.embedding.clone();
        self.vibevoice_preset_cache
            .lock()
            .unwrap()
            .insert(preset_id, embedding.clone());
        Ok(embedding)
    }

    fn ensure_vibevoice_preset_dir(&self) -> Result<PathBuf> {
        let dir = self.asset_root.join("vibevoice-presets");
        fs::create_dir_all(&dir)
            .with_context(|| format!("create VibeVoice preset directory {}", dir.display()))?;
        Ok(dir)
    }

    fn ensure_vibevoice_preset_audio_dir(&self) -> Result<PathBuf> {
        let dir = self.asset_root.join("vibevoice-preset-audio");
        fs::create_dir_all(&dir).with_context(|| {
            format!("create VibeVoice preset audio directory {}", dir.display())
        })?;
        Ok(dir)
    }

    async fn ensure_loaded(
        &self,
        voice_model: VoiceModel,
        precision: TtsPrecision,
    ) -> Result<Arc<LoadedModel>> {
        if voice_model == VoiceModel::VibeVoice {
            bail!(
                "VibeVoice 1.5B now runs only through the MLX backend. Use auto precision on macOS."
            );
        }
        self.ensure_loaded_with_key(
            ModelCacheKey {
                voice_model,
                precision,
            },
            move |asset_root| load_selected_model(asset_root, voice_model, precision),
        )
        .await
    }

    async fn ensure_loaded_with_key<F>(
        &self,
        key: ModelCacheKey,
        loader: F,
    ) -> Result<Arc<LoadedModel>>
    where
        F: FnOnce(PathBuf) -> Result<LoadedModel> + Send + 'static,
    {
        let key = ModelCacheKey {
            voice_model: key.voice_model,
            precision: key.precision,
        };
        let cell = {
            let mut cache = self.cache.lock().unwrap();
            cache
                .entry(key)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        cell.get_or_try_init(|| {
            let asset_root = self.asset_root.clone();
            async move {
                tokio::task::spawn_blocking(move || loader(asset_root))
                    .await
                    .context("wait for model load task")?
                    .map(Arc::new)
            }
        })
        .await
        .map(Arc::clone)
    }

    async fn synthesize_with_loaded(
        &self,
        spoken_text: &str,
        voice_model: VoiceModel,
        voice_preset: &str,
        overrides: &SpeechOverrides,
        loaded: Arc<LoadedModel>,
    ) -> std::result::Result<(AudioSamples, Option<String>), SynthesisFailure> {
        let selected_voice = overrides.voice_preset.as_deref().unwrap_or(voice_preset);
        let voice = resolve_voice(loaded.voices.as_ref(), selected_voice);
        let runtime_label = loaded.runtime_label.clone();
        let spoken_text = spoken_text.to_owned();
        let overrides = overrides.clone();
        let reference_audio_cache = Arc::clone(&self.reference_audio_cache);
        let preset_embedding = if let Some(preset_id) = overrides.reference_preset_id.as_deref() {
            Some(
                self.load_vibevoice_preset_embedding(preset_id)
                    .map_err(SynthesisFailure::Prepare)?,
            )
        } else if let Some(preset_name) = overrides.reference_preset_name.as_deref() {
            Some(
                self.load_vibevoice_preset_embedding_by_name(preset_name)
                    .map_err(SynthesisFailure::Prepare)?,
            )
        } else {
            None
        };

        tokio::task::spawn_blocking(move || {
            let total_started = Instant::now();
            if let Some(ref voice_name) = voice {
                info!("Using voice preset: {voice_name}");
            } else {
                info!("Using model default voice preset");
            }
            info!("Running {} on {runtime_label}", voice_model.display_name());

            match &loaded.model {
                LoadedModelKind::AnyTts(model) => {
                    let request_started = Instant::now();
                    let request = build_request(
                        &spoken_text,
                        voice_model,
                        voice.as_deref(),
                        &overrides,
                        &reference_audio_cache,
                        preset_embedding.clone(),
                    )
                    .map_err(SynthesisFailure::Prepare)?;
                    if voice_model == VoiceModel::VibeVoice {
                        info!(
                            "VibeVoice 1.5B built request in {:?} on {}.",
                            request_started.elapsed(),
                            runtime_label
                        );
                    }
                    let label = voice_model.display_name();
                    let synth_started = Instant::now();
                    let audio = model
                        .synthesize(&request)
                        .context(format!("synthesize speech with {label}"))
                        .map_err(SynthesisFailure::Runtime)?;
                    if voice_model == VoiceModel::VibeVoice {
                        info!(
                            "VibeVoice 1.5B synthesized {} chars in {:?} on {}. Total {:?}.",
                            spoken_text.chars().count(),
                            synth_started.elapsed(),
                            runtime_label,
                            total_started.elapsed()
                        );
                    }
                    Ok((audio, voice))
                }
                LoadedModelKind::Parler(runtime) => {
                    let audio =
                        synthesize_with_parler(runtime, &spoken_text, voice.as_deref(), &overrides)
                            .context("synthesize speech with Parler-TTS")
                            .map_err(SynthesisFailure::Runtime)?;
                    Ok((audio, voice))
                }
            }
        })
        .await
        .map_err(|error| SynthesisFailure::Runtime(anyhow!(error)))?
    }

    fn force_vibevoice_auto_cpu_fallback(&self) {
        self.auto_precision_overrides
            .lock()
            .unwrap()
            .insert(VoiceModel::VibeVoice, TtsPrecision::F32);
        self.cache.lock().unwrap().remove(&ModelCacheKey {
            voice_model: VoiceModel::VibeVoice,
            precision: TtsPrecision::Auto,
        });
    }
}

fn load_selected_model(
    asset_root: PathBuf,
    voice_model: VoiceModel,
    precision: TtsPrecision,
) -> Result<LoadedModel> {
    match voice_model {
        VoiceModel::VibeVoiceRealtime => load_vibevoice_realtime(asset_root, precision),
        VoiceModel::VibeVoice => load_vibevoice(asset_root, precision),
        VoiceModel::ParlerTts => load_parler(asset_root, precision),
        VoiceModel::Kokoro => load_kokoro(asset_root, precision),
    }
}

fn load_vibevoice_realtime(asset_root: PathBuf, precision: TtsPrecision) -> Result<LoadedModel> {
    let cache_dir = prepare_vibevoice_realtime_snapshot(&asset_root)?;
    load_on_metal(
        "VibeVoice Realtime 0.5B",
        ModelType::VibeVoiceRealtime,
        VR_ID,
        cache_dir,
        requested_dtype(VoiceModel::VibeVoiceRealtime, precision),
        precision == TtsPrecision::Auto,
    )
}

fn load_vibevoice(asset_root: PathBuf, precision: TtsPrecision) -> Result<LoadedModel> {
    let prepare_started = Instant::now();
    let cache_dir = prepare_vibevoice_snapshot(&asset_root)?;
    let prepare_elapsed = prepare_started.elapsed();
    let dtype = requested_dtype(VoiceModel::VibeVoice, precision);
    let load_started = Instant::now();
    let loaded = if dtype == DType::F32 {
        // Upstream any-tts 0.1.1 completes VibeVoice 1.5B requests on CPU F32,
        // while the same upstream Metal F32 path hangs for small requests.
        load_on_device(
            "VibeVoice 1.5B",
            ModelType::VibeVoice,
            VV_ID,
            cache_dir,
            DeviceSelection::Cpu,
            DType::F32,
        )?
    } else {
        load_on_metal(
            "VibeVoice 1.5B",
            ModelType::VibeVoice,
            VV_ID,
            cache_dir,
            dtype,
            false,
        )?
    };
    info!(
        "VibeVoice 1.5B prepared assets in {:?} and initialized runtime in {:?} using {}.",
        prepare_elapsed,
        load_started.elapsed(),
        loaded.runtime_label
    );
    Ok(loaded)
}

fn load_kokoro(asset_root: PathBuf, precision: TtsPrecision) -> Result<LoadedModel> {
    let cache_dir = prepare_kokoro_snapshot(&asset_root)?;
    load_on_metal(
        "Kokoro-82M",
        ModelType::Kokoro,
        KO_ID,
        cache_dir,
        requested_dtype(VoiceModel::Kokoro, precision),
        precision == TtsPrecision::Auto,
    )
}

fn load_parler(asset_root: PathBuf, precision: TtsPrecision) -> Result<LoadedModel> {
    info!("Preparing Parler-TTS model assets from {}", PA_ID);
    let model_dir = prepare_parler_snapshot(&asset_root)?;
    info!("Initializing Parler-TTS Metal device");
    let device = DeviceSelection::Metal(0)
        .resolve()
        .context("initialize Metal device for Parler-TTS")?;
    let dtype = to_candle_dtype(requested_dtype(VoiceModel::ParlerTts, precision));
    let model_file = model_dir.join("model.safetensors");
    let tokenizer_file = model_dir.join("tokenizer.json");
    let config_file = model_dir.join("config.json");

    info!("Loading Parler-TTS weights with {}", dtype_label(dtype));
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device) }
        .context("load Parler-TTS weights")?;
    let config: ParlerConfig =
        serde_json::from_slice(&std::fs::read(&config_file).context("read Parler config.json")?)
            .context("parse Parler config.json")?;
    info!("Loading Parler-TTS tokenizer");
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(|error| {
        anyhow!(
            "load Parler tokenizer from {}: {error}",
            tokenizer_file.display()
        )
    })?;
    info!("Building Parler-TTS model graph on Metal");
    let model = ParlerModel::new(&config, vb).context("build Parler-TTS model")?;
    let runtime = ParlerRuntime {
        model,
        tokenizer,
        sample_rate: config.audio_encoder.sampling_rate,
        device,
    };
    info!("Parler-TTS model ready");

    Ok(LoadedModel {
        model: LoadedModelKind::Parler(Arc::new(Mutex::new(runtime))),
        voices: parler_voice_names(),
        runtime_label: format!("metal:0 ({})", dtype_label(dtype)),
    })
}

fn requested_dtype(voice_model: VoiceModel, precision: TtsPrecision) -> DType {
    match precision {
        TtsPrecision::Auto => match voice_model {
            VoiceModel::Kokoro => DType::F16,
            VoiceModel::VibeVoice | VoiceModel::VibeVoiceRealtime | VoiceModel::ParlerTts => {
                DType::F32
            }
        },
        TtsPrecision::F32 => DType::F32,
        TtsPrecision::F16 => DType::F16,
        TtsPrecision::BF16 => DType::BF16,
    }
}

fn load_on_metal(
    label: &str,
    model_type: ModelType,
    hf_model_id: &str,
    cache_dir: PathBuf,
    requested_dtype: DType,
    allow_f32_fallback: bool,
) -> Result<LoadedModel> {
    let try_load = |dtype: DType| -> Result<LoadedModel> {
        load_on_device(
            label,
            model_type,
            hf_model_id,
            cache_dir.clone(),
            DeviceSelection::Metal(0),
            dtype,
        )
    };

    match try_load(requested_dtype) {
        Ok(loaded) => Ok(loaded),
        Err(error) if allow_f32_fallback && requested_dtype != DType::F32 => {
            warn!(
                "Loading {label} with {} failed on Metal: {error}. Falling back to f32.",
                requested_dtype.label()
            );
            try_load(DType::F32)
        }
        Err(error) => Err(error),
    }
}

fn load_on_device(
    label: &str,
    model_type: ModelType,
    hf_model_id: &str,
    cache_dir: PathBuf,
    device_selection: DeviceSelection,
    dtype: DType,
) -> Result<LoadedModel> {
    let runtime_label = format!("{} ({})", device_selection.label(), dtype.label());
    let config = TtsConfig::new(model_type)
        .with_hf_model_id(hf_model_id)
        .with_model_path(cache_dir.to_string_lossy().into_owned())
        .with_device(device_selection)
        .with_dtype(dtype);

    let model =
        load_model(config).context(format!("load {label} on {}", runtime_label.as_str()))?;
    let model: Arc<dyn TtsModel> = Arc::from(model);
    let voices: Arc<[String]> = model.supported_voices().into();
    info!("Loaded {label} using {runtime_label}");
    Ok(LoadedModel {
        model: LoadedModelKind::AnyTts(model),
        voices,
        runtime_label,
    })
}

pub struct PlaybackController {
    _stream: MixerDeviceSink,
    player: Mutex<Player>,
    generation: AtomicU64,
}

impl PlaybackController {
    pub fn new() -> Result<Self> {
        let stream = DeviceSinkBuilder::open_default_sink().context("open default audio device")?;
        let player = Player::connect_new(&stream.mixer());
        Ok(Self {
            _stream: stream,
            player: Mutex::new(player),
            generation: AtomicU64::new(0),
        })
    }

    pub fn begin_job(&self) -> Result<u64> {
        let job = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let player = self
            .player
            .lock()
            .map_err(|_| anyhow!("playback mutex poisoned"))?;
        player.stop();
        player.play();
        Ok(job)
    }

    /// Append audio to the playback queue without stopping current playback.
    /// Used for streaming chunks after the first one.
    pub fn append_audio(&self, job: u64, audio: AudioSamples) -> Result<bool> {
        if self.generation.load(Ordering::SeqCst) != job {
            return Ok(false);
        }

        let source = SamplesBuffer::new(
            NonZeroU16::new(audio.channels).ok_or_else(|| anyhow!("audio had zero channels"))?,
            NonZeroU32::new(audio.sample_rate)
                .ok_or_else(|| anyhow!("audio had zero sample rate"))?,
            audio.samples,
        );
        let player = self
            .player
            .lock()
            .map_err(|_| anyhow!("playback mutex poisoned"))?;
        player.append(source);
        Ok(true)
    }

    pub fn is_current_job(&self, job: u64) -> bool {
        self.generation.load(Ordering::SeqCst) == job
    }

    /// Append audio without checking the job generation.
    /// Used by the speech queue, which manages its own cancellation.
    pub fn append_audio_unchecked(&self, audio: AudioSamples) -> Result<()> {
        let source = SamplesBuffer::new(
            NonZeroU16::new(audio.channels).ok_or_else(|| anyhow!("audio had zero channels"))?,
            NonZeroU32::new(audio.sample_rate)
                .ok_or_else(|| anyhow!("audio had zero sample rate"))?,
            audio.samples,
        );
        let player = self
            .player
            .lock()
            .map_err(|_| anyhow!("playback mutex poisoned"))?;
        player.append(source);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        let player = self.player.lock().unwrap();
        player.empty()
    }

    pub fn stop(&self) -> Result<()> {
        self.generation.fetch_add(1, Ordering::SeqCst);
        let player = self
            .player
            .lock()
            .map_err(|_| anyhow!("playback mutex poisoned"))?;
        player.stop();
        Ok(())
    }

    pub fn toggle_pause(&self) -> Result<bool> {
        let player = self
            .player
            .lock()
            .map_err(|_| anyhow!("playback mutex poisoned"))?;
        if player.is_paused() {
            player.play();
            Ok(false)
        } else {
            player.pause();
            Ok(true)
        }
    }
}

pub fn log_file_path() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Library/Logs/DevVoice/devvoice.log")
}

pub fn initialize_tracing() -> Result<()> {
    let log_path = log_file_path();
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create log directory {}", parent.display()))?;
    }

    let file_path = log_path.clone();
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("devvoice=info")))
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_target(false)
                .with_writer(move || {
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&file_path)
                        .expect("open DevVoice log file")
                }),
        )
        .try_init()
        .map_err(|error| anyhow!("initialize tracing: {error}"))?;

    info!("DevVoice log file: {}", log_path.display());
    Ok(())
}

fn prepare_vibevoice_realtime_snapshot(asset_root: &Path) -> Result<PathBuf> {
    let cache_root = asset_root.join("hf-cache");
    std::fs::create_dir_all(&cache_root).with_context(|| {
        format!(
            "create Hugging Face cache directory {}",
            cache_root.display()
        )
    })?;

    let local_dir = asset_root.join(VR_DIR);
    std::fs::create_dir_all(&local_dir)
        .with_context(|| format!("create model directory {}", local_dir.display()))?;

    let client = hf_hub::HFClient::builder()
        .cache_dir(&cache_root)
        .build()
        .context("create hf-hub client")?;
    let client = HFClientSync::from_inner(client).context("create blocking hf-hub client")?;

    let repo = client.model(VR_OWNER, VR_NAME);
    repo.download_file()
        .filename("config.json")
        .local_dir(local_dir.clone())
        .send()
        .context("download VibeVoice Realtime config.json")?;
    repo.download_file()
        .filename("model.safetensors")
        .local_dir(local_dir.clone())
        .send()
        .context("download VibeVoice Realtime model.safetensors")?;
    let _ = repo
        .download_file()
        .filename("preprocessor_config.json")
        .local_dir(local_dir.clone())
        .send();
    let _ = repo
        .snapshot_download()
        .allow_patterns(vec!["voices/*.pt".to_string()])
        .local_dir(local_dir.clone())
        .send();

    let tokenizer_path = local_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        info!(
            "tokenizer.json not published in {}, fetching fallback tokenizer from {}/{}",
            VR_ID, TOKENIZER_FALLBACK_OWNER, TOKENIZER_FALLBACK_MODEL
        );
        client
            .model(TOKENIZER_FALLBACK_OWNER, TOKENIZER_FALLBACK_MODEL)
            .download_file()
            .filename("tokenizer.json")
            .local_dir(local_dir.clone())
            .send()
            .context("download fallback tokenizer.json")?;
    }

    download_voice_presets(&local_dir.join("voices"))?;
    ensure_required_assets(
        &local_dir,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    Ok(local_dir)
}

fn prepare_vibevoice_snapshot(asset_root: &Path) -> Result<PathBuf> {
    let cache_root = asset_root.join("hf-cache");
    std::fs::create_dir_all(&cache_root).with_context(|| {
        format!(
            "create Hugging Face cache directory {}",
            cache_root.display()
        )
    })?;

    let local_dir = asset_root.join(VV_DIR);
    std::fs::create_dir_all(&local_dir)
        .with_context(|| format!("create model directory {}", local_dir.display()))?;

    let client = hf_hub::HFClient::builder()
        .cache_dir(&cache_root)
        .build()
        .context("create hf-hub client")?;
    let client = HFClientSync::from_inner(client).context("create blocking hf-hub client")?;

    let repo = client.model(VV_OWNER, VV_NAME);
    repo.download_file()
        .filename("config.json")
        .local_dir(local_dir.clone())
        .send()
        .context("download VibeVoice 1.5B config.json")?;

    // VibeVoice 1.5B may use sharded weights; try single file first, then sharded.
    let single = repo
        .download_file()
        .filename("model.safetensors")
        .local_dir(local_dir.clone())
        .send();
    if single.is_err() {
        info!("Single model.safetensors not found for VibeVoice 1.5B, trying sharded weights");
        let _ = repo
            .snapshot_download()
            .allow_patterns(vec!["model-*.safetensors".to_string()])
            .local_dir(local_dir.clone())
            .send();
    }

    let _ = repo
        .download_file()
        .filename("preprocessor_config.json")
        .local_dir(local_dir.clone())
        .send();

    let tokenizer_path = local_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        info!(
            "tokenizer.json not published in {}, fetching fallback tokenizer from {}/{}",
            VV_ID, TOKENIZER_FALLBACK_OWNER, TOKENIZER_FALLBACK_MODEL
        );
        let tokenizer_result = repo
            .download_file()
            .filename("tokenizer.json")
            .local_dir(local_dir.clone())
            .send();
        if tokenizer_result.is_err() {
            client
                .model(TOKENIZER_FALLBACK_OWNER, TOKENIZER_FALLBACK_MODEL)
                .download_file()
                .filename("tokenizer.json")
                .local_dir(local_dir.clone())
                .send()
                .context("download fallback tokenizer.json for VibeVoice 1.5B")?;
        }
    }

    ensure_required_assets(&local_dir, &["config.json", "tokenizer.json"])?;
    Ok(local_dir)
}

fn prepare_kokoro_snapshot(asset_root: &Path) -> Result<PathBuf> {
    let cache_root = asset_root.join("hf-cache");
    std::fs::create_dir_all(&cache_root).with_context(|| {
        format!(
            "create Hugging Face cache directory {}",
            cache_root.display()
        )
    })?;

    let local_dir = asset_root.join(KO_DIR);
    std::fs::create_dir_all(&local_dir)
        .with_context(|| format!("create model directory {}", local_dir.display()))?;

    let client = hf_hub::HFClient::builder()
        .cache_dir(&cache_root)
        .build()
        .context("create hf-hub client")?;
    let client = HFClientSync::from_inner(client).context("create blocking hf-hub client")?;

    let repo = client.model(KO_OWNER, KO_NAME);
    repo.download_file()
        .filename("config.json")
        .local_dir(local_dir.clone())
        .send()
        .context("download Kokoro config.json")?;

    // Kokoro uses a .pth weight file; try safetensors first, then .pth.
    let safetensors = repo
        .download_file()
        .filename("model.safetensors")
        .local_dir(local_dir.clone())
        .send();
    if safetensors.is_err() {
        info!("model.safetensors not found for Kokoro, trying .pth weights");
        let _ = repo
            .snapshot_download()
            .allow_patterns(vec!["*.pth".to_string()])
            .local_dir(local_dir.clone())
            .send();
    }

    let _ = repo
        .snapshot_download()
        .allow_patterns(vec!["voices/*.pt".to_string()])
        .local_dir(local_dir.clone())
        .send();

    ensure_required_assets(&local_dir, &["config.json"])?;
    Ok(local_dir)
}

fn prepare_parler_snapshot(asset_root: &Path) -> Result<PathBuf> {
    let cache_root = asset_root.join("hf-cache");
    std::fs::create_dir_all(&cache_root).with_context(|| {
        format!(
            "create Hugging Face cache directory {}",
            cache_root.display()
        )
    })?;

    let local_dir = asset_root.join(PA_DIR);
    std::fs::create_dir_all(&local_dir)
        .with_context(|| format!("create model directory {}", local_dir.display()))?;

    let client = hf_hub::HFClient::builder()
        .cache_dir(&cache_root)
        .build()
        .context("create hf-hub client")?;
    let client = HFClientSync::from_inner(client).context("create blocking hf-hub client")?;

    let repo = client.model(PA_OWNER, PA_NAME);
    for file in ["config.json", "tokenizer.json", "model.safetensors"] {
        if local_dir.join(file).exists() {
            continue;
        }
        repo.download_file()
            .filename(file)
            .local_dir(local_dir.clone())
            .send()
            .with_context(|| format!("download Parler-TTS asset {file}"))?;
    }

    ensure_required_assets(
        &local_dir,
        &["config.json", "tokenizer.json", "model.safetensors"],
    )?;
    Ok(local_dir)
}

fn ensure_required_assets(model_dir: &Path, required: &[&str]) -> Result<()> {
    for file in required {
        let path = model_dir.join(file);
        if !path.exists() {
            bail!("Missing required model asset: {}", path.display());
        }
    }

    Ok(())
}

fn download_voice_presets(voices_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(voices_dir)
        .with_context(|| format!("create voice preset directory {}", voices_dir.display()))?;

    let client = BlockingClient::builder()
        .user_agent("DevVoice/0.1.0")
        .build()
        .context("create HTTP client for voice preset downloads")?;

    for voice_file in VOICE_PRESET_FILES {
        let destination = voices_dir.join(voice_file);
        if destination.exists() {
            continue;
        }

        let url = format!("{VOICE_PRESET_BASE_URL}/{voice_file}");
        info!("Downloading voice preset {} from {}", voice_file, url);
        let bytes = client
            .get(&url)
            .send()
            .and_then(|response| response.error_for_status())
            .context("download voice preset")?
            .bytes()
            .context("read voice preset bytes")?;
        std::fs::write(&destination, &bytes)
            .with_context(|| format!("write voice preset {}", destination.display()))?;
    }

    Ok(())
}

fn ensure_accessibility() -> Result<()> {
    if application_is_trusted() {
        info!("Accessibility permission already granted.");
        return Ok(());
    }

    warn!("Accessibility permission missing, requesting macOS prompt.");
    if application_is_trusted_with_prompt() {
        info!("Accessibility permission granted after prompt.");
        return Ok(());
    }

    bail!(
        "DevVoice needs Accessibility access. Approve it in System Settings > Privacy & Security > Accessibility, then retry."
    );
}

fn resolve_voice(voices: &[String], preset: &str) -> Option<String> {
    if preset.is_empty() {
        return voices.first().cloned();
    }
    let lower = preset.to_ascii_lowercase();
    voices
        .iter()
        .find(|v| v.to_ascii_lowercase() == lower)
        .or_else(|| {
            voices
                .iter()
                .find(|v| v.to_ascii_lowercase().contains(&lower))
        })
        .cloned()
        .or_else(|| voices.first().cloned())
}

fn parler_voice_names() -> Arc<[String]> {
    PARLER_VOICE_PRESETS
        .iter()
        .map(|(name, _)| (*name).to_owned())
        .collect::<Vec<_>>()
        .into()
}

fn parler_voice_description(preset: Option<&str>) -> &'static str {
    let requested = preset.unwrap_or(PARLER_DEFAULT_PRESET);
    PARLER_VOICE_PRESETS
        .iter()
        .find(|(name, _)| *name == requested)
        .map(|(_, description)| *description)
        .unwrap_or(PARLER_VOICE_PRESETS[0].1)
}

fn synthesize_with_parler(
    runtime: &Arc<Mutex<ParlerRuntime>>,
    text: &str,
    preset: Option<&str>,
    overrides: &SpeechOverrides,
) -> Result<AudioSamples> {
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("Parler-TTS model mutex poisoned"))?;
    let description = parler_request_description(preset, overrides);
    let prompt_tokens = runtime
        .tokenizer
        .encode(text, true)
        .map_err(|error| anyhow!("tokenize Parler prompt: {error}"))?
        .get_ids()
        .to_vec();
    let prompt_tokens = Tensor::new(prompt_tokens, &runtime.device)?.unsqueeze(0)?;
    let description_tokens = runtime
        .tokenizer
        .encode(description, true)
        .map_err(|error| anyhow!("tokenize Parler description: {error}"))?
        .get_ids()
        .to_vec();
    let description_tokens = Tensor::new(description_tokens, &runtime.device)?.unsqueeze(0)?;

    let codes = runtime
        .model
        .generate(
            &prompt_tokens,
            &description_tokens,
            LogitsProcessor::new(0, Some(0.0), None),
            PARLER_MAX_STEPS,
        )
        .context("generate Parler audio codes")?;
    let codes = codes.to_dtype(CandleDType::I64)?.unsqueeze(0)?;
    let pcm = runtime
        .model
        .audio_encoder
        .decode_codes(&codes.to_device(&runtime.device)?)?
        .i((0, 0))?
        .to_vec1::<f32>()
        .context("decode Parler audio to PCM")?;

    Ok(AudioSamples::new(normalize_pcm(pcm), runtime.sample_rate))
}

fn normalize_pcm(samples: Vec<f32>) -> Vec<f32> {
    let peak = samples
        .iter()
        .fold(0.0_f32, |max, sample| max.max(sample.abs()));
    let gain = if peak > 0.0 {
        (0.95 / peak).min(4.0)
    } else {
        1.0
    };
    samples
        .into_iter()
        .map(|sample| (sample * gain).clamp(-1.0, 1.0))
        .collect()
}

fn to_candle_dtype(dtype: DType) -> CandleDType {
    match dtype {
        DType::F32 => CandleDType::F32,
        DType::F16 => CandleDType::F16,
        DType::BF16 => CandleDType::BF16,
    }
}

fn dtype_label(dtype: CandleDType) -> &'static str {
    match dtype {
        CandleDType::F32 => "f32",
        CandleDType::F16 => "f16",
        CandleDType::BF16 => "bf16",
        _ => "unknown",
    }
}

fn build_request(
    text: &str,
    model: VoiceModel,
    voice: Option<&str>,
    overrides: &SpeechOverrides,
    reference_audio_cache: &Arc<Mutex<HashMap<PathBuf, CachedReferenceAudio>>>,
    preset_embedding: Option<VoiceEmbedding>,
) -> Result<SynthesisRequest> {
    let mut request = SynthesisRequest::new(text).with_language("en");

    match model {
        VoiceModel::VibeVoiceRealtime => {
            request = request
                .with_instruct(overrides.style.as_deref().unwrap_or(STYLE_PROMPT))
                .with_temperature(0.15);
        }
        VoiceModel::VibeVoice => {
            let effective_max_tokens = vibevoice_effective_max_tokens(text, overrides.max_tokens);
            if let Some(requested_max_tokens) = overrides
                .max_tokens
                .filter(|requested| *requested != effective_max_tokens)
            {
                info!(
                    "Capping VibeVoice 1.5B max_tokens from {} to {} for a short prompt.",
                    requested_max_tokens, effective_max_tokens
                );
            }
            if let Some(embedding) = preset_embedding {
                request = request.with_voice_embedding(embedding);
            } else if let Some(path) = overrides.reference_audio_path.as_deref() {
                request = request.with_reference_audio(load_reference_audio_cached(
                    reference_audio_cache,
                    path,
                )?);
            }
            request = request
                .with_cfg_scale(overrides.cfg_scale.unwrap_or(1.3))
                .with_temperature(overrides.temperature.unwrap_or(0.0))
                .with_max_tokens(effective_max_tokens);
        }
        VoiceModel::ParlerTts => {}
        VoiceModel::Kokoro => {
            // Kokoro uses its built-in phonemizer; minimal config needed.
        }
    }

    if let Some(name) = voice {
        request = request.with_voice(name.to_owned());
    }

    Ok(request)
}

fn parler_request_description(preset: Option<&str>, overrides: &SpeechOverrides) -> String {
    if let Some(description) = overrides.description.as_deref() {
        if let Some(style) = overrides.style.as_deref() {
            return format!("{description} {style}");
        }
        return description.to_owned();
    }

    let mut description = parler_voice_description(preset).to_owned();
    if let Some(style) = overrides.style.as_deref() {
        description.push(' ');
        description.push_str(style);
    }
    description
}

fn vibevoice_max_tokens(text: &str) -> usize {
    let estimated = text.split_whitespace().count().saturating_mul(4) + 16;
    estimated.clamp(32, 128)
}

fn vibevoice_effective_max_tokens(text: &str, requested: Option<usize>) -> usize {
    let recommended = vibevoice_max_tokens(text);
    match requested {
        Some(value)
            if text.split_whitespace().count() <= 24
                && text.chars().count() <= 160
                && value > recommended =>
        {
            recommended
        }
        Some(value) => value,
        None => recommended,
    }
}

fn load_reference_audio(path: &str) -> Result<ReferenceAudio> {
    let audio = AudioSamples::from_audio_file(path)
        .with_context(|| format!("load reference audio from {}", path))?;
    Ok(ReferenceAudio::new(audio.samples, audio.sample_rate))
}

fn load_reference_audio_cached(
    cache: &Arc<Mutex<HashMap<PathBuf, CachedReferenceAudio>>>,
    path: &str,
) -> Result<ReferenceAudio> {
    let path = fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));
    let metadata =
        fs::metadata(&path).with_context(|| format!("read metadata for {}", path.display()))?;
    let modified_at = metadata.modified().ok();

    if let Some(cached) = cache.lock().unwrap().get(&path) {
        if cached.modified_at == modified_at {
            info!(
                "Reusing cached VibeVoice 1.5B reference audio from {} ({:.1}s clip).",
                path.display(),
                cached.audio.duration_secs()
            );
            return Ok(cached.audio.clone());
        }
    }

    let load_started = Instant::now();
    let audio = load_reference_audio(&path.to_string_lossy())?;
    if audio.duration_secs() > 10.0 {
        warn!(
            "Reference audio {} is {:.1}s long. VibeVoice 1.5B works best with a clean 3 to 10 second clip.",
            path.display(),
            audio.duration_secs()
        );
    }
    info!(
        "Loaded VibeVoice 1.5B reference audio from {} in {:?} ({:.1}s clip).",
        path.display(),
        load_started.elapsed(),
        audio.duration_secs()
    );
    cache.lock().unwrap().insert(
        path,
        CachedReferenceAudio {
            modified_at,
            audio: audio.clone(),
        },
    );
    Ok(audio)
}

fn load_vibevoice_preprocessor_config(asset_root: &Path) -> Result<VibeVoicePreprocessorConfig> {
    let config_path = asset_root.join(VV_DIR).join("preprocessor_config.json");
    if config_path.exists() {
        return VibeVoicePreprocessorConfig::from_file(&config_path)
            .map_err(|error| anyhow!("load VibeVoice preprocessor config: {error}"));
    }
    Ok(VibeVoicePreprocessorConfig::default())
}

fn normalize_vibevoice_reference_audio(
    audio: &ReferenceAudio,
    config: &VibeVoicePreprocessorConfig,
) -> Vec<f32> {
    let resampled = if audio.sample_rate != config.audio_processor.sampling_rate {
        resample_linear(
            &audio.samples,
            audio.sample_rate,
            config.audio_processor.sampling_rate,
        )
    } else {
        audio.samples.clone()
    };
    if !config.db_normalize {
        return resampled;
    }
    normalize_dbfs(
        &resampled,
        config.audio_processor.target_d_b_fs,
        config.audio_processor.eps,
    )
}

fn normalize_dbfs(samples: &[f32], target_db_fs: f32, eps: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let rms =
        (samples.iter().map(|value| value * value).sum::<f32>() / samples.len() as f32).sqrt();
    let scalar = 10f32.powf(target_db_fs / 20.0) / (rms + eps);
    samples
        .iter()
        .map(|value| (value * scalar).clamp(-1.0, 1.0))
        .collect()
}

fn slugify_preset_name(name: &str) -> String {
    let mut slug = String::new();
    let mut last_was_dash = false;
    for ch in name.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            slug.push('-');
            last_was_dash = true;
        }
    }
    let slug = slug.trim_matches('-').chars().take(48).collect::<String>();
    if slug.is_empty() {
        "preset".to_owned()
    } else {
        slug
    }
}

fn normalize_vibevoice_preset_name(name: &str) -> String {
    name.trim().to_lowercase()
}

fn ensure_unique_vibevoice_preset_name(preset_dir: &Path, preset_name: &str) -> Result<()> {
    let normalized_name = normalize_vibevoice_preset_name(preset_name);
    if normalized_name.is_empty() {
        bail!("Preset name cannot be empty.");
    }
    let existing = load_all_vibevoice_presets(preset_dir)?;
    if let Some(duplicate) = existing
        .into_iter()
        .find(|preset| normalize_vibevoice_preset_name(&preset.info.name) == normalized_name)
    {
        bail!(
            "A VibeVoice preset named '{}' already exists with id '{}'. Use a unique name.",
            duplicate.info.name,
            duplicate.info.id
        );
    }
    Ok(())
}

fn validate_vibevoice_preset_id(preset_id: &str) -> Result<String> {
    let preset_id = preset_id.trim();
    if preset_id.is_empty() {
        bail!("Preset id cannot be empty.");
    }
    if !preset_id
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        bail!("Preset id must use only letters, numbers, hyphens, or underscores.");
    }
    Ok(preset_id.to_owned())
}

fn vibevoice_preset_path(preset_dir: &Path, preset_id: &str) -> PathBuf {
    preset_dir.join(format!("{preset_id}.json"))
}

fn copy_vibevoice_preset_audio(
    source_path: &Path,
    audio_dir: &Path,
    preset_id: &str,
) -> Result<PathBuf> {
    let extension = source_path
        .extension()
        .and_then(|value| value.to_str())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("wav");
    let destination = audio_dir.join(format!("{preset_id}.{extension}"));
    fs::copy(source_path, &destination).with_context(|| {
        format!(
            "copy VibeVoice reference audio from {} to {}",
            source_path.display(),
            destination.display()
        )
    })?;
    Ok(destination)
}

fn load_stored_vibevoice_preset(path: &Path) -> Result<StoredVibeVoicePreset> {
    let bytes =
        fs::read(path).with_context(|| format!("read VibeVoice preset {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parse VibeVoice preset")
}

fn load_all_vibevoice_presets(preset_dir: &Path) -> Result<Vec<StoredVibeVoicePreset>> {
    if !preset_dir.exists() {
        return Ok(Vec::new());
    }
    let mut presets = Vec::new();
    for entry in fs::read_dir(preset_dir)
        .with_context(|| format!("read VibeVoice preset directory {}", preset_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        presets.push(load_stored_vibevoice_preset(&path)?);
    }
    Ok(presets)
}

fn find_vibevoice_preset_by_name(
    preset_dir: &Path,
    preset_name: &str,
) -> Result<StoredVibeVoicePreset> {
    let normalized_name = normalize_vibevoice_preset_name(preset_name);
    if normalized_name.is_empty() {
        bail!("Preset name cannot be empty.");
    }
    let matches = load_all_vibevoice_presets(preset_dir)?
        .into_iter()
        .filter(|preset| normalize_vibevoice_preset_name(&preset.info.name) == normalized_name)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => bail!(
            "No VibeVoice preset named '{}' was found.",
            preset_name.trim()
        ),
        [preset] => Ok(preset.clone()),
        _ => bail!(
            "Multiple VibeVoice presets named '{}' were found. Use reference_preset_id instead.",
            preset_name.trim()
        ),
    }
}

fn provision_vibevoice_mlx_runtime(data_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(data_dir)
        .with_context(|| format!("create DevVoice data directory {}", data_dir.display()))?;
    let venv_dir = data_dir.join("vibevoice-mlx-venv");
    let python_path = venv_dir.join("bin/python");
    if python_path.is_file() && validate_vibevoice_mlx_runtime(&python_path).is_ok() {
        return Ok(python_path);
    }

    info!(
        "Provisioning VibeVoice MLX runtime in {}.",
        venv_dir.display()
    );
    let mut create_venv = Command::new("python3");
    create_venv.arg("-m").arg("venv").arg(&venv_dir);
    run_command(
        &mut create_venv,
        "create the VibeVoice MLX virtual environment",
    )?;

    let mut upgrade_pip = Command::new(&python_path);
    upgrade_pip
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--upgrade")
        .arg("pip");
    run_command(
        &mut upgrade_pip,
        "upgrade pip for the VibeVoice MLX runtime",
    )?;

    let mut install_runtime = Command::new(&python_path);
    install_runtime
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg(VIBEVOICE_MLX_INSTALL_SPEC)
        .arg("scipy");
    run_command(
        &mut install_runtime,
        "install the VibeVoice MLX runtime dependencies",
    )?;

    validate_vibevoice_mlx_runtime(&python_path)?;
    info!(
        "Provisioned VibeVoice MLX runtime at {}.",
        python_path.display()
    );
    Ok(python_path)
}

fn validate_vibevoice_mlx_runtime(python_path: &Path) -> Result<()> {
    let mut validate = Command::new(python_path);
    validate.arg("-c").arg("import vibevoice_mlx, scipy");
    run_command(&mut validate, "validate the VibeVoice MLX runtime")
}

fn run_command(command: &mut Command, description: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("start command to {description}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("process exited with status {}", output.status)
    };
    bail!("Failed to {description}: {detail}");
}

fn normalize_technical_text(input: &str) -> String {
    let mut text = input.replace("O(log n)", "big O of log n");
    text = text.replace("O(n)", "big O of n");
    text = text.replace("O(1)", "big O of constant time");
    text = text.replace("O(", "big O of ");
    text = text.replace("impl Trait", "impl trait");
    text = text.replace("K8s", "Kubernetes");
    text = text.replace("ElastiCache", "Elastic Cache");
    text = text.replace("EKS", "E K S");
    text = text.replace("AKS", "A K S");
    text = text.replace("S3", "S 3");
    text = text.replace("CRDs", "C R Ds");
    text = text.replace("CRD", "C R D");
    text = text.replace("createdb", "create database");
    text = text.replace("GPU", "G P U");
    text = text.replace("CPU", "C P U");
    text = text.replace("CLI", "C L I");
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split text into chunks at sentence boundaries so the first chunk can be
/// synthesized and played quickly while later chunks are still being generated.
///
/// `chunk_size` is measured in characters. `Some(0)` disables chunking.
pub fn split_into_speech_chunks(text: &str, chunk_size: Option<usize>) -> Vec<String> {
    let text = text.trim();
    let target_chunk_chars = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    if target_chunk_chars == 0 || text.chars().count() <= target_chunk_chars {
        return vec![text.to_owned()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut char_count: usize = 0;

    for ch in text.chars() {
        current.push(ch);
        char_count += 1;

        let reached_sentence_boundary =
            char_count >= target_chunk_chars && (ch == '.' || ch == '!' || ch == '?');
        let reached_fallback_boundary =
            char_count >= target_chunk_chars.saturating_mul(2) && ch.is_whitespace();
        if reached_sentence_boundary || reached_fallback_boundary {
            let trimmed = current.trim().to_owned();
            if !trimmed.is_empty() {
                chunks.push(trimmed);
            }
            current.clear();
            char_count = 0;
        }
    }

    let remainder = current.trim().to_owned();
    if !remainder.is_empty() {
        if let Some(last) = chunks
            .last_mut()
            .filter(|_| remainder.chars().count() < target_chunk_chars / 2)
        {
            last.push(' ');
            last.push_str(&remainder);
        } else {
            chunks.push(remainder);
        }
    }

    if chunks.is_empty() {
        chunks.push(text.to_owned());
    }

    chunks
}
