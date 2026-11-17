import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

import vibevoice_realtime_mlx_worker as worker_module


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.worker = worker_module.Worker("test-model")
        self.model = Mock()
        self.model.sample_rate = 24000
        self.model.tokenizer.encode.side_effect = lambda text, **kwargs: text.split()
        self.model.generate.return_value = [
            SimpleNamespace(audio=np.full(240, 0.25, dtype=np.float32))
        ]
        self.worker.model = self.model
        self.worker.voices = {"en-Emma_woman", "en-Grace_woman"}

    def test_synthesis_writes_audio_and_reuses_the_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "speech.wav")
            command = {
                "command": "synthesize",
                "text": "The build is ready.",
                "voice": "en-Emma_woman",
                "output_path": path,
            }
            for _ in range(2):
                result = self.worker.run(command)
                self.assertTrue(result["ok"])
                self.assertEqual(result["runtimeLabel"], worker_module.RUNTIME_LABEL)
                self.assertEqual(result["metrics"]["loadMs"], 0)
                self.assertEqual(result["outputPath"], path)
            audio, sample_rate = sf.read(path)
            self.assertEqual(sample_rate, 24000)
            np.testing.assert_allclose(audio, 0.25)
            self.model.load_voice.assert_called_once_with("en-Emma_woman")

    def test_warmup_and_voice_switch_do_not_reuse_the_previous_voice(self):
        for voice in ["en-Emma_woman", "en-Grace_woman"]:
            self.worker.run({"command": "warmup", "voice": voice})
        self.assertEqual(self.model.load_voice.call_count, 2)
        self.model.load_voice.assert_called_with("en-Grace_woman")
        self.assertEqual(self.model.generate.call_args.kwargs["max_tokens"], 32)

    def test_long_text_has_a_growing_token_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            self.worker.run(
                {
                    "command": "synthesize",
                    "text": "word " * 300,
                    "voice": "en-Emma_woman",
                    "output_path": str(Path(directory) / "long.wav"),
                }
            )
        self.assertEqual(self.model.generate.call_args.kwargs["max_tokens"], 1800)

    def test_invalid_requests_are_not_silently_accepted(self):
        requests = [
            {"command": "unknown"},
            {"command": "warmup", "voice": "../en-Emma_woman"},
            {"command": "warmup", "voice": "en-Emma_woman", "style": "calm"},
            {"command": "synthesize", "text": " "},
            {"command": "synthesize", "text": "Hello."},
        ]
        for command in requests:
            with self.subTest(command=command):
                with self.assertRaises(ValueError):
                    self.worker.run(command)
        self.model.load_voice.assert_not_called()
        self.model.generate.assert_not_called()

    def test_empty_and_non_finite_audio_are_errors(self):
        for audio in [[], [float("nan")], [float("inf")]]:
            with self.subTest(audio=audio):
                self.model.generate.return_value = [
                    SimpleNamespace(audio=np.asarray(audio, dtype=np.float32))
                ]
                with self.assertRaisesRegex(RuntimeError, "empty or non-finite"):
                    self.worker.run({"command": "warmup", "voice": "en-Emma_woman"})

    def test_model_load_is_pinned_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            voices_dir = Path(directory) / "voices"
            voices_dir.mkdir()
            (voices_dir / "en-Emma_woman.safetensors").touch()
            self.worker.model = None
            with (
                patch.object(
                    worker_module, "snapshot_download", return_value=directory
                ) as download,
                patch.object(worker_module, "load_model", return_value=self.model) as load,
            ):
                for _ in range(2):
                    self.worker.run({"command": "warmup", "voice": "en-Emma_woman"})
                download.assert_called_once()
                self.assertEqual(
                    download.call_args.kwargs["revision"], worker_module.MODEL_REVISION
                )
                load.assert_called_once()

    def test_protocol_keeps_diagnostics_off_stdout_and_reports_errors(self):
        def run(command):
            print("model diagnostic")
            if command["command"] == "fail":
                raise ValueError("synthesis failed")
            return {"ok": True}

        stdin = io.StringIO('\n{"command":"fail"}\n{"command":"warmup"}\n')
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(worker_module.sys, "argv", ["worker", "--model", "test-model"]),
            patch.object(worker_module.sys, "stdin", stdin),
            patch.object(worker_module.Worker, "run", side_effect=run),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            worker_module.main()
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertFalse(responses[0]["ok"])
        self.assertIn("synthesis failed", responses[0]["error"])
        self.assertTrue(responses[1]["ok"])
        self.assertIn("model diagnostic", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
