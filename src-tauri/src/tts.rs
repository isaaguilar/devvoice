use crate::config::{DEFAULT_CHUNK_SIZE, TtsPrecision, VoiceModel};
use crate::state::SpeechOverrides;
use any_tts::{
    AudioSamples, DType, DeviceSelection, ModelType, ReferenceAudio, SynthesisRequest, TtsConfig,
    TtsModel,
    load_model,
};
use anyhow::{Context, Result, anyhow, bail};
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
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

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

pub struct CaptureResult {
    pub raw: String,
}

pub struct WarmupInfo {
    pub voices: Vec<String>,
    pub runtime_label: String,
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

struct ParlerRuntime {
    model: ParlerModel,
    tokenizer: Tokenizer,
    sample_rate: u32,
    device: candle_core::Device,
}

pub struct TtsService {
    asset_root: PathBuf,
    cache: Mutex<HashMap<ModelCacheKey, Arc<OnceCell<Arc<LoadedModel>>>>>,
}

impl TtsService {
    pub fn new(asset_root: PathBuf) -> Self {
        Self {
            asset_root,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub async fn warmup(
        &self,
        voice_model: VoiceModel,
        precision: TtsPrecision,
    ) -> Result<WarmupInfo> {
        let loaded = self.ensure_loaded(voice_model, precision).await?;
        Ok(WarmupInfo {
            voices: loaded.voices.iter().cloned().collect(),
            runtime_label: loaded.runtime_label.clone(),
        })
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
        let loaded = self.ensure_loaded(voice_model, precision).await?;
        let selected_voice = overrides.voice_preset.as_deref().unwrap_or(voice_preset);
        let voice = resolve_voice(loaded.voices.as_ref(), selected_voice);
        let runtime_label = loaded.runtime_label.clone();
        let overrides = overrides.clone().normalized();

        tokio::task::spawn_blocking(move || {
            if let Some(ref voice_name) = voice {
                info!("Using voice preset: {voice_name}");
            } else {
                info!("Using model default voice preset");
            }
            info!("Running {} on {runtime_label}", voice_model.display_name());

            match &loaded.model {
                LoadedModelKind::AnyTts(model) => {
                    let request =
                        build_request(&spoken_text, voice_model, voice.as_deref(), &overrides)?;
                    let label = voice_model.display_name();
                    let audio = model
                        .synthesize(&request)
                        .context(format!("synthesize speech with {label}"))?;
                    Ok((audio, voice))
                }
                LoadedModelKind::Parler(runtime) => {
                    let audio =
                        synthesize_with_parler(runtime, &spoken_text, voice.as_deref(), &overrides)
                        .context("synthesize speech with Parler-TTS")?;
                    Ok((audio, voice))
                }
            }
        })
        .await
        .context("wait for synthesis task")?
    }

    pub fn prune_cache(&self, voice_model: VoiceModel, precision: TtsPrecision) {
        let keep = ModelCacheKey {
            voice_model,
            precision,
        };
        self.cache.lock().unwrap().retain(|key, _| *key == keep);
    }

    async fn ensure_loaded(
        &self,
        voice_model: VoiceModel,
        precision: TtsPrecision,
    ) -> Result<Arc<LoadedModel>> {
        let key = ModelCacheKey {
            voice_model,
            precision,
        };
        let cell = {
            let mut cache = self.cache.lock().unwrap();
            cache.entry(key)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        cell.get_or_try_init(|| {
            let asset_root = self.asset_root.clone();
            async move {
                tokio::task::spawn_blocking(move || {
                    load_selected_model(asset_root, voice_model, precision)
                })
                .await
                .context("wait for model load task")?
                .map(Arc::new)
            }
        })
        .await
        .map(Arc::clone)
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
    let cache_dir = prepare_vibevoice_snapshot(&asset_root)?;
    let dtype = requested_dtype(VoiceModel::VibeVoice, precision);
    if dtype == DType::F32 {
        // Upstream any-tts 0.1.1 completes VibeVoice 1.5B requests on CPU F32,
        // while the same upstream Metal F32 path hangs for small requests.
        return load_on_device(
            "VibeVoice 1.5B",
            ModelType::VibeVoice,
            VV_ID,
            cache_dir,
            DeviceSelection::Cpu,
            DType::F32,
        );
    }
    load_on_metal(
        "VibeVoice 1.5B",
        ModelType::VibeVoice,
        VV_ID,
        cache_dir,
        dtype,
        precision == TtsPrecision::Auto,
    )
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
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device) }
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
        .with(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("devvoice=info")),
        )
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
    std::fs::create_dir_all(&cache_root)
        .with_context(|| format!("create Hugging Face cache directory {}", cache_root.display()))?;

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
    ensure_required_assets(&local_dir, &["config.json", "tokenizer.json", "model.safetensors"])?;
    Ok(local_dir)
}

fn prepare_vibevoice_snapshot(asset_root: &Path) -> Result<PathBuf> {
    let cache_root = asset_root.join("hf-cache");
    std::fs::create_dir_all(&cache_root)
        .with_context(|| format!("create Hugging Face cache directory {}", cache_root.display()))?;

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
    std::fs::create_dir_all(&cache_root)
        .with_context(|| format!("create Hugging Face cache directory {}", cache_root.display()))?;

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
    std::fs::create_dir_all(&cache_root)
        .with_context(|| format!("create Hugging Face cache directory {}", cache_root.display()))?;

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

    ensure_required_assets(&local_dir, &["config.json", "tokenizer.json", "model.safetensors"])?;
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
    let gain = if peak > 0.0 { (0.95 / peak).min(4.0) } else { 1.0 };
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
) -> Result<SynthesisRequest> {
    let mut request = SynthesisRequest::new(text).with_language("en");

    match model {
        VoiceModel::VibeVoiceRealtime => {
            request = request
                .with_instruct(overrides.style.as_deref().unwrap_or(STYLE_PROMPT))
                .with_temperature(0.15);
        }
        VoiceModel::VibeVoice => {
            if let Some(path) = overrides.reference_audio_path.as_deref() {
                request = request.with_reference_audio(load_reference_audio(path)?);
            }
            request = request
                .with_cfg_scale(overrides.cfg_scale.unwrap_or(1.3))
                .with_temperature(overrides.temperature.unwrap_or(0.0))
                .with_max_tokens(overrides.max_tokens.unwrap_or_else(|| vibevoice_max_tokens(text)));
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

fn load_reference_audio(path: &str) -> Result<ReferenceAudio> {
    let audio = AudioSamples::from_audio_file(path)
        .with_context(|| format!("load reference audio from {}", path))?;
    Ok(ReferenceAudio::new(audio.samples, audio.sample_rate))
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
