# DevVoice

DevVoice is a desktop app and local HTTP API for speaking selected text with several local TTS models, including VibeVoice Realtime 0.5B and the higher-quality VibeVoice 1.5B model.

## VibeVoice Realtime 0.5B with MLX

Select `VibeVoice Realtime 0.5B` and leave precision on `auto` to use the persistent 8-bit MLX Audio backend on Apple Silicon macOS. DevVoice provisions a separate Python environment at `~/Library/Application Support/com.isa.devvoice/vibevoice-realtime-mlx-venv`, downloads the quantized model and voice presets, and runs an inference warmup automatically. Setup requires an Apple Silicon Python 3.11 or newer; an existing 1.5B MLX Python environment can provide the interpreter, but its packages are not changed.

The backend uses `mlx-audio==0.6.6` and the pinned `ae6a233e1b2d6d5e2315e7030581323cc8191bb3` revision of `mlx-community/VibeVoice-Realtime-0.5B-8bit`. The model and current voice conditioning stay loaded across requests. All seven existing English and Indian-English preset names are preserved, including `en-Emma_woman`, `en-Grace_woman`, and `in-Samuel_man`. Status reports `python:mlx-audio-int8:realtime` after warmup.

MLX Audio does not support Realtime `style` instructions. Requests containing `style` in Auto mode fail with an explicit error rather than ignoring the instruction. Select `f16`, `bf16`, or `f32` to keep using the built-in Rust backend with its existing controls. The 1.5B MLX backend and environment are unchanged.

```bash
curl -X POST "http://127.0.0.1:9876/speak?voice_preset=en-Emma_woman" \
  --data "Ship the build after the tests pass."
```

The first setup includes package and model downloads; subsequent startup and speech requests reuse the local runtime and cached model files. MLX accelerates generation, but DevVoice still hands audio to playback at its existing text-chunk boundaries.

## VibeVoice 1.5B usage now

The new 1.5B flow keeps the model practical by using a persistent MLX worker on macOS when:

- the selected model is `VibeVoice`
- precision is `auto`
- the MLX Python environment exists under `~/Library/Application Support/com.isa.devvoice/vibevoice-mlx-venv`

If the MLX environment is missing, DevVoice now provisions it automatically on first 1.5B use. The old non-accelerated Rust 1.5B fallback is no longer treated as a supported path.

## Do I need to warm up the model manually?

No, not for normal use.

When the app opens, DevVoice already starts a background warmup for the currently selected model. For VibeVoice 1.5B with MLX available, that startup warmup loads the MLX model and runs a tiny inference pass automatically.

The manual warmup API is still useful, but now it is optional and mainly for reducing the first request time for a specific voice.

## Recommended user flow after opening the app

1. Open DevVoice.
2. Select `VibeVoice` as the voice model.
3. Leave precision set to `auto`.
4. Wait until the UI says the model is ready, or check `GET /status` and wait for `"model_ready": true`.
5. If you already have a preset, call `/speak` with `reference_preset_name` or `reference_preset_id`.
6. If you want the first request for that voice to be faster, call `/vibevoice/warmup` once with that preset before the first real `/speak`.
7. Reuse the same preset in the same app session for the best repeat performance.

## Status check

You can confirm the active backend with:

```bash
curl http://127.0.0.1:9876/status
```

When the MLX path is active for 1.5B, `tts_runtime_label` should show:

```json
{
  "model_ready": true,
  "tts_runtime_label": "python:mlx-int8:no-semantic"
}
```

## MLX setup for the accelerated 1.5B path

DevVoice now auto-provisions the MLX runtime on first 1.5B use. If you want to install it yourself ahead of time, you can still create the virtual environment manually:

```bash
DATA_DIR="$HOME/Library/Application Support/com.isa.devvoice"
python3 -m venv "$DATA_DIR/vibevoice-mlx-venv"
source "$DATA_DIR/vibevoice-mlx-venv/bin/activate"
pip install --upgrade pip
pip install 'git+https://github.com/gafiatulin/vibevoice-mlx.git@f513aa7877e77fefa1aebe87432855c407da3b87' scipy
```

After that, relaunch DevVoice and keep precision on `auto` for the MLX worker to be used.

## API calls

### Create a preset once

```bash
curl -X POST "http://127.0.0.1:9876/vibevoice/presets" \
  -H "Content-Type: application/json" \
  -d '{"name":"my-demo-voice","referenceAudioPath":"/Users/you/Desktop/reference-voice.wav"}'
```

Names are unique, so `reference_preset_name` is safe to use later. New presets now copy the reference clip into DevVoice-managed storage so they do not depend on the original temporary source path.

### List presets

```bash
curl http://127.0.0.1:9876/vibevoice/presets
```

### Optional preset warmup

```bash
curl -X POST "http://127.0.0.1:9876/vibevoice/warmup" \
  -H "Content-Type: application/json" \
  -d '{"referencePresetName":"my-demo-voice"}'
```

Use this when you want the first real synthesis for that voice to avoid paying the model-load and voice-encode cost right before playback.

### Speak with a preset by name

```bash
curl -X POST "http://127.0.0.1:9876/speak?reference_preset_name=my-demo-voice&cfg_scale=1.3&temperature=0.0&max_tokens=96&save_audio=true" \
  -d "I'm your worst nightmare."
```

### Speak with a preset by id

```bash
curl -X POST "http://127.0.0.1:9876/speak?reference_preset_id=my-demo-voice-1716420000&cfg_scale=1.3&temperature=0.0&max_tokens=96&save_audio=true" \
  -d "I'm your worst nightmare."
```

### Speak directly from a raw reference audio path

```bash
curl -X POST "http://127.0.0.1:9876/speak?reference_audio_path=/Users/you/Desktop/reference-voice.wav&cfg_scale=1.3&temperature=0.0&max_tokens=96&save_audio=true" \
  -d "I'm your worst nightmare."
```

## Preset durability note

New presets now keep a DevVoice-managed copy of the reference clip, so they no longer depend on the original source path remaining available. Older presets created before this change may still point at external files.

## Practical latency expectations

With the persistent MLX path working, the rough behavior on the tested machine was:

- startup warmup in about 20 seconds
- first preset-backed request in about 30 seconds
- repeated request with the same preset in the same app session in about 6 seconds

That is still slower than VibeVoice Realtime 0.5B, but it is dramatically faster than the previous tens-of-minutes Rust path for VibeVoice 1.5B.

## More detail

See `architecture.md` for the current 1.5B request lifecycle, component layout, fallback behavior, and known limitations.
