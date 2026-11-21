"""Regression checks for quick generation and its initial form state."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.voice_playground import build_app
from voxforge.playground.service import PlaygroundService


class QuickVoiceTest(unittest.TestCase):
    def test_reference_gate_and_generation_handler(self):
        with tempfile.TemporaryDirectory() as temp:
            service = PlaygroundService(Path(temp))
            app = build_app(service)
            events = {event.fn.__name__: event for event in app.fns.values() if event.fn}
            gate = events["generation_button"]
            engine, _profile = gate.inputs
            self.assertEqual(engine.value, "voxcpm2")
            self.assertFalse(gate.outputs[0].interactive)
            for name in ("xtts", "voxcpm2", "chatterbox-v3"):
                self.assertFalse(gate.fn(name, None)["interactive"])
                self.assertTrue(gate.fn(name, "voice")["interactive"])
            self.assertTrue(gate.fn("mock", None)["interactive"])
            event = events["on_generate"]
            args = [component.value for component in event.inputs]
            with patch.object(service, "generate_many", return_value=([], [])) as generate:
                self.assertIsNone(event.fn(*args)[-1])
                generate.assert_not_called()
                args[6] = "voice"
                event.fn(*args)
                self.assertEqual(generate.call_args.args[1:3], (["voxcpm2"], "voice"))
