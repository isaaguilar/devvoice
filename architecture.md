# DevVoice VibeVoice 1.5B architecture draft

## Goal

Keep VibeVoice 1.5B available as the highest-quality local model, but route it through a persistent MLX worker on Apple Silicon so repeated requests are practical instead of taking tens of minutes.

## High-level shape

```text
UI / HTTP client
    |
    v
AppRuntime
    |
    v
TtsService
    |
    +--> Non-VibeVoice or non-MLX case
    |        |
    |        v
    |    Rust any-tts / Candle backend
    |
    +--> VibeVoice 1.5B + macOS + auto precision + MLX venv present
             |
             v
       Persistent Python MLX worker
             |
             +--> cached 1.5B model
             +--> cached voice embeddings by reference audio path
             |
             v
       temporary WAV output
             |
             v
       AudioSamples -> playback queue -> optional saved WAV
```

## Runtime selection

The MLX path is only selected when all of the following are true:

- the selected model is `VibeVoice` (the 1.5B model)
- the selected precision is `auto`
- the app is running on macOS
- `~/Library/Application Support/com.isa.devvoice/vibevoice-mlx-venv/bin/python` exists

If the model is `VibeVoice 1.5B`, DevVoice now treats MLX as the supported backend. The old Rust/Candle 1.5B path is no longer treated as the supported fallback for normal use.

## Startup sequence

When the app opens, `AppRuntime::warmup_model()` runs automatically in the background.

For VibeVoice 1.5B with MLX available, that startup warmup:

- starts the Python worker if it is not already running
- loads the quantized MLX 1.5B model into memory
- runs a tiny warmup inference
- stores the runtime label as `python:mlx-int8:no-semantic`
- marks the model as ready in app state

That means you do not have to call a new API just to make 1.5B usable. The app already starts warming it when it launches.

If the MLX runtime is missing, DevVoice can provision the Python virtual environment and dependencies into the app data directory before continuing.

## Request flow

### 1. Create a preset, optional but recommended

`POST /vibevoice/presets` stores:

- preset metadata
- the existing any-tts voice embedding
- a DevVoice-managed copy of the reference audio for new presets

Today, the MLX backend does not use the stored any-tts embedding directly. Instead, DevVoice resolves the preset back to the managed `sourceAudioPath` and lets the MLX worker build or reuse its own cached voice embedding from that audio file.

### 2. Warm a specific preset, optional

`POST /vibevoice/warmup` is now best understood as an optimization call, not a required setup step.

It is useful when you want the first real request for a specific voice to be faster because it can:

- ensure the MLX model is already loaded
- optionally pre-encode the reference voice for a preset name, preset id, or raw audio path

### 3. Synthesize

`POST /speak` still remains the main synthesis API.

For MLX-backed 1.5B requests:

1. `AppRuntime` validates the request and enqueues it.
2. `TtsService` decides whether the MLX path is active.
3. If MLX is active, Rust sends a JSON command to the long-lived Python worker over stdin.
4. The worker reuses the loaded model.
5. If a reference voice is provided, the worker reuses a cached embedding keyed by the reference audio path when possible.
6. The worker writes a temporary WAV file.
7. Rust reads that WAV back into `AudioSamples`.
8. DevVoice queues playback and optionally saves the final combined WAV to the requested output directory or `~/Downloads`.

## API surface

The core `/speak` call is unchanged.

The useful VibeVoice-related endpoints are:

- `POST /vibevoice/presets` to create a reusable named or id-based preset
- `GET /vibevoice/presets` to inspect saved presets
- `POST /vibevoice/warmup` to preload the model and optionally pre-encode a specific voice
- `GET /status` to confirm `model_ready` and inspect `tts_runtime_label`

The most relevant query parameters for `/speak` are:

- `reference_preset_name`
- `reference_preset_id`
- `reference_audio_path`
- `cfg_scale`
- `temperature`
- `max_tokens`
- `chunk_size`
- `save_audio`
- `output_dir`

Exactly one of `reference_preset_name`, `reference_preset_id`, or `reference_audio_path` should be supplied for a reference-voice request.

## What a user should do in practice

After opening the app:

1. Set the model to `VibeVoice`.
2. Leave precision at `auto` if you want the MLX path.
3. Wait until the app status says the model is ready, or confirm with `GET /status`.
4. If you already have a preset, call `/speak` with `reference_preset_name` or `reference_preset_id`.
5. If you want the very first request for that voice to be faster, call `/vibevoice/warmup` first.
6. Reuse the same preset in the same app session for the best repeat latency.

## Known limitations

- The MLX acceleration currently only applies to VibeVoice 1.5B on macOS in `auto` precision mode.
- Older presets created before the durable-audio change may still depend on their original external source path.
- `/speak` is queue-based. The HTTP response confirms the request was accepted, not that playback or WAV saving has finished yet.
- In headless or dark-wake situations, playback can still fail even when synthesis itself succeeds.

## Future improvements

- Preserve a durable copy of reference audio when creating presets so MLX does not depend on a temporary source path.
- Optionally auto-provision the MLX virtual environment instead of requiring manual setup.
- Expose MLX availability more clearly in the UI.
- Consider MLX-native preset artifacts instead of rebuilding from source audio on demand.
