use super::{
    resolve_voice, run_command, send_mlx_request, write_if_changed, VibeVoiceMlxResponse,
    VibeVoiceMlxWorker, WarmupInfo, VOICE_PRESET_FILES,
};
use crate::config::{TtsPrecision, VoiceModel};
use crate::state::SpeechOverrides;
use any_tts::AudioSamples;
use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::OnceCell;
use tracing::{info, warn};

const MODEL_ID: &str = "mlx-community/VibeVoice-Realtime-0.5B-8bit";
pub(super) const RUNTIME_LABEL: &str = "python:mlx-audio-int8:realtime";
const WORKER_SCRIPT: &str = include_str!("../vibevoice_realtime_mlx_worker.py");
const REQUIREMENTS: &str = include_str!("../vibevoice_realtime_mlx_requirements.txt");
const VENV_DIR: &str = "vibevoice-realtime-mlx-venv";

#[derive(Serialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum Request {
    Warmup {
        voice: String,
    },
    Synthesize {
        text: String,
        voice: String,
        output_path: String,
    },
}

pub(super) fn requested(model: VoiceModel, precision: TtsPrecision) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && model == VoiceModel::VibeVoiceRealtime
        && precision == TtsPrecision::Auto
}

pub(super) struct RealtimeMlx {
    asset_root: PathBuf,
    runtime_ready: OnceCell<()>,
    worker: Mutex<Option<VibeVoiceMlxWorker>>,
    runtime_label: Mutex<Option<String>>,
}

impl RealtimeMlx {
    pub(super) fn new(asset_root: PathBuf) -> Self {
        Self {
            asset_root,
            runtime_ready: OnceCell::new(),
            worker: Mutex::new(None),
            runtime_label: Mutex::new(None),
        }
    }

    fn data_dir(&self) -> Result<&Path> {
        self.asset_root
            .parent()
            .context("determine DevVoice data directory for Realtime MLX")
    }

    pub(super) fn runtime_label(&self) -> Option<String> {
        self.runtime_label.lock().unwrap().clone()
    }

    pub(super) fn backend_status(&self) -> String {
        if let Some(label) = self.runtime_label() {
            format!("MLX ready on {label}")
        } else if self
            .data_dir()
            .is_ok_and(|dir| dir.join(VENV_DIR).join("bin/python").is_file())
        {
            "Realtime MLX installed, waiting for warmup.".to_owned()
        } else {
            "Realtime MLX runtime missing. DevVoice will provision it automatically on first use."
                .to_owned()
        }
    }

    async fn ensure_runtime(self: &Arc<Self>) -> Result<()> {
        self.runtime_ready
            .get_or_try_init(|| async {
                let service = Arc::clone(self);
                tokio::task::spawn_blocking(move || {
                    let data_dir = service.data_dir()?;
                    provision_runtime(data_dir)?;
                    write_if_changed(
                        &data_dir.join("vibevoice-realtime-mlx-worker.py"),
                        WORKER_SCRIPT,
                    )
                })
                .await
                .context("join Realtime MLX provisioning task")??;
                Ok::<(), anyhow::Error>(())
            })
            .await?;
        Ok(())
    }

    fn send(&self, request: &Request) -> Result<VibeVoiceMlxResponse> {
        let data_dir = self.data_dir()?;
        let script_path = data_dir.join("vibevoice-realtime-mlx-worker.py");
        let response = send_mlx_request(
            &self.worker,
            &data_dir.join(VENV_DIR).join("bin/python"),
            &script_path,
            MODEL_ID,
            request,
        )?;
        let label = response
            .runtime_label
            .as_ref()
            .context("Realtime MLX worker did not report its runtime label")?;
        if label != RUNTIME_LABEL {
            bail!("Unexpected Realtime MLX runtime label: {label}");
        }
        *self.runtime_label.lock().unwrap() = Some(label.clone());
        if let Some(metrics) = response.metrics.as_ref() {
            info!(
                "Realtime MLX request completed (load={:?} ms, voice={:?} ms, gen={:?} ms, audio_seconds={:?}) on {}.",
                metrics.load_ms, metrics.encode_ms, metrics.gen_ms, metrics.audio_seconds, label
            );
        }
        Ok(response)
    }

    pub(super) async fn warmup(self: &Arc<Self>) -> Result<WarmupInfo> {
        let started = Instant::now();
        self.ensure_runtime().await?;
        let service = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            let voices = voice_names();
            service.send(&Request::Warmup {
                voice: voices[0].clone(),
            })?;
            Ok(WarmupInfo {
                voices,
                runtime_label: service
                    .runtime_label()
                    .context("Realtime MLX warmup did not initialize the runtime")?,
                warmed_inference: true,
                warmup_duration_ms: started.elapsed().as_millis() as u64,
            })
        })
        .await
        .context("join Realtime MLX warmup task")?
    }

    pub(super) async fn synthesize(
        self: &Arc<Self>,
        text: &str,
        voice_preset: &str,
        overrides: &SpeechOverrides,
    ) -> Result<(AudioSamples, Option<String>)> {
        if overrides.style.is_some() {
            bail!(
                "Realtime MLX (auto precision) does not support style instructions. Select F16, BF16, or F32 to use the Rust backend with style."
            );
        }
        self.ensure_runtime().await?;
        let voice = resolve_voice(
            &voice_names(),
            overrides.voice_preset.as_deref().unwrap_or(voice_preset),
        )
        .context("Realtime MLX has no voice presets")?;
        let text = text.to_owned();
        let service = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            let output_dir = service.asset_root.join("mlx-output");
            fs::create_dir_all(&output_dir).context("create Realtime MLX output directory")?;
            let output_path = output_dir.join(format!(
                "realtime-mlx-{}.wav",
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
            ));
            let result = service
                .send(&Request::Synthesize {
                    text,
                    voice: voice.clone(),
                    output_path: output_path.to_string_lossy().into_owned(),
                })
                .and_then(|_| {
                    AudioSamples::from_wav_file(&output_path)
                        .context("read Realtime MLX synthesized audio")
                });
            if output_path.exists() {
                if let Err(error) = fs::remove_file(&output_path) {
                    warn!(
                        "Failed to remove Realtime MLX audio {}: {error}",
                        output_path.display()
                    );
                }
            }
            result.map(|audio| (audio, Some(voice)))
        })
        .await
        .context("join Realtime MLX synthesis task")?
    }
}

fn voice_names() -> Vec<String> {
    VOICE_PRESET_FILES
        .iter()
        .map(|file| file.trim_end_matches(".pt").to_owned())
        .collect()
}

fn validate_runtime(python_path: &Path) -> Result<()> {
    let mut command = Command::new(python_path);
    command.arg("-c").arg(
        "from importlib.metadata import version; \
         assert version('mlx-audio') == '0.6.6'; \
         import mlx.core, numpy, soundfile; \
         from mlx_audio.tts.models.vibevoice.vibevoice import Model",
    );
    run_command(&mut command, "validate the Realtime MLX runtime")
}

fn provision_runtime(data_dir: &Path) -> Result<()> {
    fs::create_dir_all(data_dir).context("create Realtime MLX data directory")?;
    let venv_dir = data_dir.join(VENV_DIR);
    let python_path = venv_dir.join("bin/python");
    if python_path.is_file() {
        match validate_runtime(&python_path) {
            Ok(()) => return Ok(()),
            Err(error) => warn!("Repairing Realtime MLX runtime: {error:#}"),
        }
    } else {
        let candidates = [
            data_dir.join("vibevoice-mlx-venv/bin/python"),
            PathBuf::from("/opt/homebrew/bin/python3"),
            PathBuf::from("/usr/local/bin/python3"),
            PathBuf::from("python3"),
        ];
        let python = candidates
            .iter()
            .find(|python| {
                Command::new(python)
                    .args([
                        "-c",
                        "import platform, sys; \
                         sys.exit(0 if sys.version_info >= (3, 11) and platform.machine() == 'arm64' else 1)",
                    ])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
            })
            .ok_or_else(|| {
                anyhow!("Realtime MLX requires an Apple Silicon Python 3.11 or newer. Install Python and retry model warmup.")
            })?;
        info!("Creating Realtime MLX runtime in {}.", venv_dir.display());
        let mut create = Command::new(python);
        create.args(["-m", "venv"]).arg(&venv_dir);
        run_command(&mut create, "create the Realtime MLX virtual environment")?;
    }
    let requirements_path = data_dir.join("vibevoice-realtime-mlx-requirements.txt");
    write_if_changed(&requirements_path, REQUIREMENTS)?;
    let mut install = Command::new(&python_path);
    install
        .args(["-m", "pip", "install", "--requirement"])
        .arg(&requirements_path);
    info!("Installing the pinned Realtime MLX runtime dependencies.");
    run_command(
        &mut install,
        "install the Realtime MLX runtime dependencies",
    )?;
    validate_runtime(&python_path)
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::tts::TtsService;

    #[test]
    fn realtime_auto_rejects_style_before_provisioning() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let service = TtsService::new(PathBuf::from("unused-test-data/tts-assets"));
        let overrides = SpeechOverrides {
            style: Some("Read calmly.".to_owned()),
            ..Default::default()
        };
        let error = runtime
            .block_on(service.synthesize(
                "Hello.",
                VoiceModel::VibeVoiceRealtime,
                TtsPrecision::Auto,
                "",
                &overrides,
            ))
            .unwrap_err();
        assert!(error.to_string().contains("does not support style"));
        assert!(service
            .loaded_runtime_label(VoiceModel::VibeVoiceRealtime, TtsPrecision::Auto)
            .is_none());
    }

    #[test]
    #[ignore = "downloads the MLX runtime and model; set DEVVOICE_TEST_DATA_DIR explicitly"]
    fn realtime_mlx_smoke() -> Result<()> {
        let data_dir = PathBuf::from(
            std::env::var_os("DEVVOICE_TEST_DATA_DIR")
                .context("set DEVVOICE_TEST_DATA_DIR to an isolated test directory")?,
        );
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(async {
            let service = TtsService::new(data_dir.join("tts-assets"));
            let warmup = service
                .warmup(VoiceModel::VibeVoiceRealtime, TtsPrecision::Auto)
                .await?;
            assert!(warmup.warmed_inference);
            assert_eq!(warmup.runtime_label, RUNTIME_LABEL);
            assert_eq!(warmup.voices.len(), VOICE_PRESET_FILES.len());
            println!("Realtime MLX warmup: {} ms", warmup.warmup_duration_ms);
            let text = "The build is ready. Ship the update after the tests pass.";
            for preset in ["", "en-Emma_woman", "Grace"] {
                let overrides = SpeechOverrides {
                    voice_preset: (!preset.is_empty()).then(|| preset.to_owned()),
                    ..Default::default()
                };
                let started = Instant::now();
                let (audio, voice) = service
                    .synthesize(
                        text,
                        VoiceModel::VibeVoiceRealtime,
                        TtsPrecision::Auto,
                        "",
                        &overrides,
                    )
                    .await?;
                assert_eq!(
                    voice.as_deref(),
                    Some(if preset == "Grace" {
                        "en-Grace_woman"
                    } else {
                        "en-Emma_woman"
                    })
                );
                assert_eq!(audio.sample_rate, 24000);
                assert!(audio.duration_secs() > 1.0);
                assert!(audio.samples.iter().all(|sample| sample.is_finite()));
                assert!(audio.samples.iter().any(|sample| sample.abs() > 0.001));
                println!(
                    "Realtime MLX {:?}: {} ms for {:.2} s of audio",
                    voice,
                    started.elapsed().as_millis(),
                    audio.duration_secs()
                );
            }
            assert_eq!(
                service
                    .loaded_runtime_label(VoiceModel::VibeVoiceRealtime, TtsPrecision::Auto)
                    .as_deref(),
                Some(RUNTIME_LABEL)
            );
            assert!(service
                .loaded_runtime_label(VoiceModel::VibeVoiceRealtime, TtsPrecision::F32)
                .is_none());
            assert_eq!(
                fs::read_dir(data_dir.join("tts-assets/mlx-output"))?.count(),
                0
            );
            drop(service);

            if std::env::var_os("DEVVOICE_TEST_RUST_BASELINE").is_some() {
                let service = TtsService::new(data_dir.join("tts-assets"));
                service
                    .warmup(VoiceModel::VibeVoiceRealtime, TtsPrecision::F32)
                    .await?;
                for _ in 0..2 {
                    let started = Instant::now();
                    let (audio, _) = service
                        .synthesize(
                            text,
                            VoiceModel::VibeVoiceRealtime,
                            TtsPrecision::F32,
                            "en-Emma_woman",
                            &SpeechOverrides::default(),
                        )
                        .await?;
                    assert!(audio.duration_secs() > 1.0);
                    println!(
                        "Realtime Rust F32: {} ms for {:.2} s of audio",
                        started.elapsed().as_millis(),
                        audio.duration_secs()
                    );
                }
            }
            Ok(())
        })
    }
}
