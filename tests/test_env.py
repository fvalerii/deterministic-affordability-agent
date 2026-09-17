"""Tests for `.env` loading and Anthropic key placeholders."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from tools.env import anthropic_api_key  # noqa: E402


class TestAnthropicKeyPlaceholder(unittest.TestCase):
    def setUp(self):
        self._previous = os.environ.get("ANTHROPIC_API_KEY")

    def tearDown(self):
        if self._previous is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = self._previous

    def test_placeholder_is_treated_as_missing(self):
        os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key_here"
        self.assertIsNone(anthropic_api_key())

    def test_empty_value_is_missing(self):
        os.environ["ANTHROPIC_API_KEY"] = ""
        self.assertIsNone(anthropic_api_key())

    def test_real_looking_key_is_returned(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-not-a-real-secret"
        self.assertEqual(anthropic_api_key(), "sk-ant-test-not-a-real-secret")


class TestLoadEnvironment(unittest.TestCase):
    def test_main_loads_dotenv_without_overriding_os_env(self):
        import main as pipeline

        os.environ["BUYORWAIT_DOTENV_PROBE"] = "from-os"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".env").write_text(
                "BUYORWAIT_DOTENV_PROBE=from-file\nBUYORWAIT_DOTENV_ONLY=from-file\n",
                encoding="utf-8",
            )
            loaded = pipeline.load_environment(root)
            self.assertIsNotNone(loaded)
            self.assertEqual(os.environ["BUYORWAIT_DOTENV_PROBE"], "from-os")
            self.assertEqual(os.environ["BUYORWAIT_DOTENV_ONLY"], "from-file")
        os.environ.pop("BUYORWAIT_DOTENV_ONLY", None)
        os.environ.pop("BUYORWAIT_DOTENV_PROBE", None)


if __name__ == "__main__":
    unittest.main()
