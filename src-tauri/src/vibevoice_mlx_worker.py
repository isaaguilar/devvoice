import argparse
import json
import sys
import time
import traceback

import mlx.core as mx
import soundfile as sf

from vibevoice_mlx.e2e_pipeline import (
    _detect_tokenizer,
    encode_voice_reference,
    tokenize_text,
)
from vibevoice_mlx.generate import GenerationOptions, generate
from vibevoice_mlx.load_weights import load_model


class Worker:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.config = None
        self.tokenizer_name = None
        self.voice_cache = {}

    def ensure_loaded(self):
        if self.model is not None:
            return 0
        started = time.perf_counter()
        self.model, self.config = load_model(self.model_id, quantize_bits=8)
        self.tokenizer_name = _detect_tokenizer(self.model_id, self.config)
        return int((time.perf_counter() - started) * 1000)

    def build_request(self, text: str, reference_audio_path: str | None):
        encode_ms = 0
        if reference_audio_path:
            cached_embeds = self.voice_cache.get(reference_audio_path)
            if cached_embeds is None:
                result = tokenize_text(
                    text,
                    self.tokenizer_name,
                    self.config,
                    ref_audio=[reference_audio_path],
                )
                started = time.perf_counter()
                for speaker in result.speakers:
                    speaker.cached_embeds = encode_voice_reference(
                        speaker.ref_audio_np,
                        speaker.num_vae_tokens,
                        self.model,
                        self.config,
                        self.model_id,
                    )
                    cached_embeds = speaker.cached_embeds
                    break
                encode_ms = int((time.perf_counter() - started) * 1000)
                if cached_embeds is None:
                    raise RuntimeError("reference audio did not produce speaker embeddings")
                self.voice_cache[reference_audio_path] = cached_embeds
            speaker_embeds = [(len(cached_embeds), cached_embeds)]
            result = tokenize_text(
                text,
                self.tokenizer_name,
                self.config,
                speaker_embeds=speaker_embeds,
            )
            input_ids = result.input_ids
            voice_embeds = {}
            embeds_mx = mx.array(cached_embeds).astype(mx.float16)
            for speaker in result.speakers:
                for i, pos in enumerate(speaker.speech_embed_positions):
                    if i < embeds_mx.shape[0]:
                        voice_embeds[pos] = embeds_mx[i : i + 1]
            return input_ids, voice_embeds, encode_ms
        input_ids = tokenize_text(text, self.tokenizer_name, self.config)
        return input_ids, None, encode_ms

    def run(self, command: dict):
        load_ms = self.ensure_loaded()
        text = command.get("text") or "Speaker 0: Warm up."
        reference_audio_path = command.get("reference_audio_path")
        input_ids, voice_embeds, encode_ms = self.build_request(text, reference_audio_path)
        opts = GenerationOptions(
            solver="dpm",
            diffusion_steps=10,
            cfg_scale=float(command.get("cfg_scale", 1.3)),
            max_speech_tokens=int(command.get("max_speech_tokens", 96)),
            seed=42,
        )
        started = time.perf_counter()
        audio, metrics = generate(
            model=self.model,
            input_ids=input_ids,
            opts=opts,
            semantic_encoder_fn=None,
            semantic_reset_fn=None,
            voice_embeds=voice_embeds,
        )
        gen_ms = int((time.perf_counter() - started) * 1000)
        output_path = command.get("output_path")
        if output_path:
            sf.write(output_path, audio, 24000)
        return {
            "ok": True,
            "runtimeLabel": "python:mlx-int8:no-semantic",
            "outputPath": output_path,
            "metrics": {
                "loadMs": load_ms,
                "encodeMs": encode_ms,
                "genMs": gen_ms,
                "audioSeconds": metrics.audio_samples / 24000.0,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    worker = Worker(args.model)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
            response = worker.run(command)
        except Exception as error:
            response = {
                "ok": False,
                "error": f"{error}\n{traceback.format_exc()}",
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
