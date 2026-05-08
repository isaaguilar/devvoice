use crate::config::VoiceGender;
use any_tts::{
    AudioSamples, DType, DeviceSelection, ModelType, SynthesisRequest, TtsConfig, TtsModel,
    load_model,
};
use anyhow::{Context, Result, anyhow, bail};
use get_selected_text::get_selected_text;
use hf_hub::HFClientSync;
use macos_accessibility_client::accessibility::{
    application_is_trusted, application_is_trusted_with_prompt,
};
use reqwest::blocking::Client as BlockingClient;
use rodio::buffer::SamplesBuffer;
use rodio::{DeviceSinkBuilder, MixerDeviceSink, Player};
use std::fs::OpenOptions;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

const MODEL_OWNER: &str = "microsoft";
const MODEL_NAME: &str = "VibeVoice-Realtime-0.5B";
const MODEL_ID: &str = "microsoft/VibeVoice-Realtime-0.5B";
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

struct LoadedModel {
    model: Arc<dyn TtsModel>,
    voices: Vec<String>,
}

pub struct TtsService {
    asset_root: PathBuf,
    model: OnceCell<Arc<LoadedModel>>,
}

impl TtsService {
    pub fn new(asset_root: PathBuf) -> Self {
        Self {
            asset_root,
            model: OnceCell::new(),
        }
    }

    pub async fn warmup(&self) -> Result<Vec<String>> {
        Ok(self.ensure_loaded().await?.voices.clone())
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
        voice_gender: VoiceGender,
    ) -> Result<(AudioSamples, Option<String>)> {
        let spoken_text = normalize_technical_text(text);
        let loaded = self.ensure_loaded().await?;
        let voices = loaded.voices.clone();
        let model = Arc::clone(&loaded.model);

        tokio::task::spawn_blocking(move || {
            let voice = select_voice(&voices, voice_gender);
            if let Some(ref voice_name) = voice {
                info!("Using voice preset: {voice_name}");
            } else {
                info!("Using model default voice preset");
            }

            let mut request = SynthesisRequest::new(&spoken_text)
                .with_language("en")
                .with_instruct(STYLE_PROMPT)
                .with_temperature(0.15);

            if let Some(ref voice_name) = voice {
                request = request.with_voice(voice_name.clone());
            }

            let audio = model
                .synthesize(&request)
                .context("synthesize speech with VibeVoice Realtime")?;
            Ok((audio, voice))
        })
        .await
        .context("wait for synthesis task")?
    }

    async fn ensure_loaded(&self) -> Result<Arc<LoadedModel>> {
        self.model
            .get_or_try_init(|| async {
                let asset_root = self.asset_root.clone();
                tokio::task::spawn_blocking(move || load_model_from_disk(asset_root))
                    .await
                    .context("wait for model load task")?
                    .map(Arc::new)
            })
            .await
            .map(Arc::clone)
    }
}

fn load_model_from_disk(asset_root: PathBuf) -> Result<LoadedModel> {
    let cache_dir = prepare_model_snapshot(&asset_root)?;
    let config = TtsConfig::new(ModelType::VibeVoiceRealtime)
        .with_hf_model_id(MODEL_ID)
        .with_model_path(cache_dir.to_string_lossy().into_owned())
        .with_device(DeviceSelection::Metal(0))
        .with_dtype(DType::F32);

    let model = load_model(config).context("load VibeVoice Realtime on Metal")?;
    let model: Arc<dyn TtsModel> = Arc::from(model);
    let voices = model.supported_voices();
    Ok(LoadedModel { model, voices })
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

fn prepare_model_snapshot(asset_root: &Path) -> Result<PathBuf> {
    let cache_root = asset_root.join("hf-cache");
    std::fs::create_dir_all(&cache_root)
        .with_context(|| format!("create Hugging Face cache directory {}", cache_root.display()))?;

    let local_dir = asset_root.join("vibevoice-realtime-0.5b");
    std::fs::create_dir_all(&local_dir)
        .with_context(|| format!("create DevVoice model directory {}", local_dir.display()))?;

    let client = hf_hub::HFClient::builder()
        .cache_dir(&cache_root)
        .build()
        .context("create hf-hub client")?;
    let client = HFClientSync::from_inner(client).context("create blocking hf-hub client")?;

    let repo = client.model(MODEL_OWNER, MODEL_NAME);
    repo.download_file()
        .filename("config.json")
        .local_dir(local_dir.clone())
        .send()
        .context("download VibeVoice config.json")?;
    repo.download_file()
        .filename("model.safetensors")
        .local_dir(local_dir.clone())
        .send()
        .context("download VibeVoice model.safetensors")?;
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
            MODEL_ID, TOKENIZER_FALLBACK_OWNER, TOKENIZER_FALLBACK_MODEL
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
    ensure_required_assets(&local_dir)?;
    Ok(local_dir)
}

fn ensure_required_assets(model_dir: &Path) -> Result<()> {
    for file in ["config.json", "tokenizer.json", "model.safetensors"] {
        let path = model_dir.join(file);
        if !path.exists() {
            bail!("Missing required model asset: {}", path.display());
        }
    }

    let voices_dir = model_dir.join("voices");
    if !voices_dir.exists() {
        warn!("voice preset directory is missing, the default voice will be used");
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

fn select_voice(voices: &[String], voice_gender: VoiceGender) -> Option<String> {
    let preferred = match voice_gender {
        VoiceGender::Woman => &["en-emma_woman", "en-grace_woman", "emma", "grace"][..],
        VoiceGender::Man => &[
            "en-davis_man",
            "en-frank_man",
            "en-mike_man",
            "en-carter_man",
            "in-samuel_man",
            "davis",
            "frank",
            "mike",
            "carter",
            "samuel",
        ][..],
    };

    for needle in preferred {
        if let Some(voice) = voices
            .iter()
            .find(|voice| voice.to_ascii_lowercase().contains(needle))
        {
            return Some(voice.clone());
        }
    }

    voices.first().cloned()
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
pub fn split_into_speech_chunks(text: &str) -> Vec<String> {
    const MIN_CHUNK_CHARS: usize = 100;

    let text = text.trim();
    if text.chars().count() <= MIN_CHUNK_CHARS * 2 {
        return vec![text.to_owned()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut char_count: usize = 0;

    for ch in text.chars() {
        current.push(ch);
        char_count += 1;

        if char_count >= MIN_CHUNK_CHARS && (ch == '.' || ch == '!' || ch == '?') {
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
        if let Some(last) = chunks.last_mut().filter(|_| remainder.len() < 50) {
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
