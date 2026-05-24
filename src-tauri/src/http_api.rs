use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tauri::{AppHandle, Manager};
use tokio::net::TcpListener;
use tracing::{error, info};

use crate::state::SpeechOverrides;
use crate::tts::VibeVoicePresetInfo;
use crate::AppRuntime;

#[derive(Clone)]
struct HttpState {
    app: AppHandle,
}

#[derive(Serialize)]
struct SpeakResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    prepared_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct EnqueueResponse {
    ok: bool,
    queued: bool,
    position: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    status_detail: String,
    model_ready: bool,
    playback_paused: bool,
    queue_length: usize,
    tts_backend_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tts_runtime_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
}

#[derive(Serialize)]
struct VibeVoicePresetResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    preset: Option<VibeVoicePresetInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct VibeVoicePresetListResponse {
    ok: bool,
    presets: Vec<VibeVoicePresetInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct VibeVoiceWarmupResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warmup_duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warmed_inference: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct SpeakQuery {
    #[serde(flatten)]
    overrides: SpeechOverrides,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateVibeVoicePresetRequest {
    reference_audio_path: String,
    name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WarmupVibeVoiceRequest {
    #[serde(flatten)]
    overrides: SpeechOverrides,
}

async fn speak_handler(
    State(state): State<HttpState>,
    Query(query): Query<SpeakQuery>,
    body: String,
) -> (StatusCode, Json<EnqueueResponse>) {
    let text = body.trim().to_owned();
    if text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(EnqueueResponse {
                ok: false,
                queued: false,
                position: 0,
                error: Some("No text provided.".to_owned()),
            }),
        );
    }

    let runtime = state.app.state::<AppRuntime>();
    match runtime.enqueue_text(&state.app, text, query.overrides) {
        Ok(position) => (
            StatusCode::OK,
            Json(EnqueueResponse {
                ok: true,
                queued: true,
                position,
                error: None,
            }),
        ),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(EnqueueResponse {
                ok: false,
                queued: false,
                position: 0,
                error: Some(error.to_string()),
            }),
        ),
    }
}

async fn status_handler(State(state): State<HttpState>) -> Json<StatusResponse> {
    let runtime = state.app.state::<AppRuntime>();
    let snapshot = runtime.snapshot();
    Json(StatusResponse {
        status: format!("{:?}", snapshot.status),
        status_detail: snapshot.status_detail,
        model_ready: snapshot.model_ready,
        playback_paused: snapshot.playback_paused,
        queue_length: runtime.speech_queue.len(),
        tts_backend_status: runtime.current_backend_status(),
        tts_runtime_label: runtime.current_runtime_label(),
        last_error: snapshot.last_error,
    })
}

async fn stop_handler(State(state): State<HttpState>) -> (StatusCode, Json<SpeakResponse>) {
    let runtime = state.app.state::<AppRuntime>();
    match runtime.stop_playback(&state.app) {
        Ok(_) => (
            StatusCode::OK,
            Json(SpeakResponse {
                ok: true,
                prepared_text: None,
                error: None,
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(SpeakResponse {
                ok: false,
                prepared_text: None,
                error: Some(e.to_string()),
            }),
        ),
    }
}

async fn create_vibevoice_preset_handler(
    State(state): State<HttpState>,
    Json(request): Json<CreateVibeVoicePresetRequest>,
) -> (StatusCode, Json<VibeVoicePresetResponse>) {
    let runtime = state.app.state::<AppRuntime>();
    match runtime.create_vibevoice_preset(&request.reference_audio_path, request.name.as_deref()) {
        Ok(preset) => (
            StatusCode::OK,
            Json(VibeVoicePresetResponse {
                ok: true,
                preset: Some(preset),
                error: None,
            }),
        ),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(VibeVoicePresetResponse {
                ok: false,
                preset: None,
                error: Some(error.to_string()),
            }),
        ),
    }
}

async fn list_vibevoice_presets_handler(
    State(state): State<HttpState>,
) -> (StatusCode, Json<VibeVoicePresetListResponse>) {
    let runtime = state.app.state::<AppRuntime>();
    match runtime.list_vibevoice_presets() {
        Ok(presets) => (
            StatusCode::OK,
            Json(VibeVoicePresetListResponse {
                ok: true,
                presets,
                error: None,
            }),
        ),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(VibeVoicePresetListResponse {
                ok: false,
                presets: Vec::new(),
                error: Some(error.to_string()),
            }),
        ),
    }
}

async fn warmup_vibevoice_handler(
    State(state): State<HttpState>,
    Json(request): Json<WarmupVibeVoiceRequest>,
) -> (StatusCode, Json<VibeVoiceWarmupResponse>) {
    let runtime = state.app.state::<AppRuntime>();
    match runtime
        .warmup_vibevoice(&state.app, request.overrides)
        .await
    {
        Ok(warmup) => (
            StatusCode::OK,
            Json(VibeVoiceWarmupResponse {
                ok: true,
                runtime_label: Some(warmup.runtime_label),
                warmup_duration_ms: Some(warmup.warmup_duration_ms),
                warmed_inference: Some(warmup.warmed_inference),
                error: None,
            }),
        ),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(VibeVoiceWarmupResponse {
                ok: false,
                runtime_label: None,
                warmup_duration_ms: None,
                warmed_inference: None,
                error: Some(error.to_string()),
            }),
        ),
    }
}

pub async fn start(app: AppHandle, port: u16) {
    let state = HttpState { app };
    let router = Router::new()
        .route("/speak", post(speak_handler))
        .route(
            "/vibevoice/presets",
            post(create_vibevoice_preset_handler).get(list_vibevoice_presets_handler),
        )
        .route("/vibevoice/warmup", post(warmup_vibevoice_handler))
        .route("/status", get(status_handler))
        .route("/stop", post(stop_handler))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    info!("DevVoice HTTP API listening on http://{addr}");

    match TcpListener::bind(addr).await {
        Ok(listener) => {
            if let Err(e) = axum::serve(listener, router).await {
                error!("HTTP API server error: {e}");
            }
        }
        Err(e) => {
            error!("Failed to bind HTTP API on {addr}: {e}");
        }
    }
}
