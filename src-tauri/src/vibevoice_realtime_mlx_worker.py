import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from mlx_audio.tts.utils import load_model


MODEL_REVISION = "ae6a233e1b2d6d5e2315e7030581323cc8191bb3"
RUNTIME_LABEL = "python:mlx-audio-int8:realtime"


class Worker:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.voices = set()
        self.voice = None

    def ensure_loaded(self):
        if self.model is not None:
            return 0
        started = time.perf_counter()
        model_path = Path(
            snapshot_download(
                self.model_id,
                revision=MODEL_REVISION,
                allow_patterns=[
                    "*.json",
                    "model*.safetensors",
                    "voices/en-*.safetensors",
                    "voices/in-Samuel_man.safetensors",
                ],
            )
        )
        model = load_model(model_path)
        voices = {path.stem for path in (model_path / "voices").glob("*.safetensors")}
        if not voices:
            raise RuntimeError("The Realtime MLX model has no voice presets")
        self.voices = voices
        self.model = model
        return int((time.perf_counter() - started) * 1000)

    def run(self, command: dict):
        operation = command["command"]
        if operation not in {"warmup", "synthesize"}:
            raise ValueError(f"Unknown Realtime MLX command: {operation}")
        if command.get("style"):
            raise ValueError("Realtime MLX does not support style instructions")
        text = "Warm up." if operation == "warmup" else command["text"].strip()
        if not text:
            raise ValueError("No text was provided")
        output_path = command.get("output_path")
        if operation == "synthesize" and not output_path:
            raise ValueError("Synthesis requires an output_path")

        load_ms = self.ensure_loaded()
        voice = command["voice"]
        if voice not in self.voices:
            raise ValueError(f"Unknown Realtime MLX voice preset: {voice}")
        started = time.perf_counter()
        if voice != self.voice:
            self.model.load_voice(voice)
            self.voice = voice
        encode_ms = int((time.perf_counter() - started) * 1000)
        text_tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
        max_tokens = 32 if operation == "warmup" else max(512, len(text_tokens) * 6)
        started = time.perf_counter()
        chunks = []
        for result in self.model.generate(
            text=text,
            max_tokens=max_tokens,
            cfg_scale=1.5,
            stream=False,
            verbose=False,
        ):
            chunks.append(np.asarray(result.audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Realtime MLX did not produce audio")
        audio = np.concatenate(chunks)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("Realtime MLX produced empty or non-finite audio")
        gen_ms = int((time.perf_counter() - started) * 1000)
        if output_path:
            sf.write(output_path, audio, self.model.sample_rate, subtype="PCM_16")
        return {
            "ok": True,
            "runtimeLabel": RUNTIME_LABEL,
            "outputPath": output_path,
            "metrics": {
                "loadMs": load_ms,
                "encodeMs": encode_ms,
                "genMs": gen_ms,
                "audioSeconds": audio.size / self.model.sample_rate,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    worker = Worker(args.model)
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        try:
            # Keep model download and inference diagnostics off the JSON protocol.
            with contextlib.redirect_stdout(sys.stderr):
                response = worker.run(json.loads(raw_line))
        except Exception as error:
            response = {
                "ok": False,
                "error": f"{error}\n{traceback.format_exc()}",
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
