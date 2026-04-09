"""
AI client wrappers for all four debate models.
Each client streams text chunks via a generator.
Missing API keys produce a graceful error message instead of crashing.
"""

import os
import logging
from typing import Generator

logger = logging.getLogger(__name__)

# ── Model config ─────────────────────────────────────────────────────────────

MODELS = {
    "claude": {
        "model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
        "display": "Claude",
        "color": "#f97316",
    },
    "gpt": {
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "display": "GPT",
        "color": "#10b981",
    },
    "grok": {
        "model": "grok-2-latest",
        "env_key": "GROK_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "display": "Grok",
        "color": "#8b5cf6",
    },
    "gemini": {
        "model": "gemini-2.0-flash",
        "env_key": "GEMINI_API_KEY",
        "display": "Gemini",
        "color": "#3b82f6",
    },
}

MAX_TOKENS = 1500


# ── Main client ───────────────────────────────────────────────────────────────

class AIClient:
    """Lazy-initialised wrappers for all four AI providers."""

    def __init__(self):
        self._anthropic = None
        self._openai = None
        self._grok = None
        self._genai = None

    # ── Lazy client accessors ────────────────────────────────────────────────

    def _claude_client(self):
        if self._anthropic is None:
            try:
                from anthropic import Anthropic
                key = os.getenv("ANTHROPIC_API_KEY")
                if key:
                    self._anthropic = Anthropic(api_key=key)
            except ImportError:
                logger.warning("anthropic package not installed")
        return self._anthropic

    def _openai_client(self):
        if self._openai is None:
            try:
                from openai import OpenAI
                key = os.getenv("OPENAI_API_KEY")
                if key:
                    self._openai = OpenAI(api_key=key)
            except ImportError:
                logger.warning("openai package not installed")
        return self._openai

    def _grok_client(self):
        if self._grok is None:
            try:
                from openai import OpenAI
                key = os.getenv("GROK_API_KEY")
                if key:
                    self._grok = OpenAI(
                        api_key=key,
                        base_url="https://api.x.ai/v1",
                    )
            except ImportError:
                logger.warning("openai package not installed (needed for Grok)")
        return self._grok

    def _genai_module(self):
        if self._genai is None:
            try:
                import google.generativeai as genai
                key = os.getenv("GEMINI_API_KEY")
                if key:
                    genai.configure(api_key=key)
                    self._genai = genai
            except ImportError:
                logger.warning("google-generativeai package not installed")
        return self._genai

    # ── Public API ───────────────────────────────────────────────────────────

    def stream(
        self, model_key: str, prompt: str, system_prompt: str = ""
    ) -> Generator[str, None, None]:
        """Yield text chunks from the given model."""
        if model_key == "claude":
            yield from self._stream_claude(prompt, system_prompt)
        elif model_key == "gpt":
            yield from self._stream_openai("gpt", prompt, system_prompt)
        elif model_key == "grok":
            yield from self._stream_openai("grok", prompt, system_prompt)
        elif model_key == "gemini":
            yield from self._stream_gemini(prompt, system_prompt)
        else:
            yield f"[Unknown model: {model_key}]"

    def get_facilitator_model(self) -> str:
        """Return the best available model to act as facilitator."""
        for key in ("claude", "gpt", "gemini", "grok"):
            if os.getenv(MODELS[key]["env_key"]):
                return key
        return "claude"  # will surface a config error gracefully

    def available_models(self) -> list[str]:
        return [k for k in MODELS if os.getenv(MODELS[k]["env_key"])]

    # ── Provider implementations ─────────────────────────────────────────────

    def _stream_claude(self, prompt: str, system_prompt: str) -> Generator[str, None, None]:
        client = self._claude_client()
        if not client:
            yield "[Claude: set ANTHROPIC_API_KEY to enable]"
            return
        try:
            with client.messages.stream(
                model=MODELS["claude"]["model"],
                max_tokens=MAX_TOKENS,
                system=system_prompt or "You are a debate participant.",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as exc:
            logger.exception("Claude stream error")
            yield f"\n[Claude error: {exc}]"

    def _stream_openai(
        self, client_key: str, prompt: str, system_prompt: str
    ) -> Generator[str, None, None]:
        client = self._openai_client() if client_key == "gpt" else self._grok_client()
        cfg = MODELS[client_key]
        if not client:
            yield f"[{cfg['display']}: set {cfg['env_key']} to enable]"
            return
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                stream=True,
                max_tokens=MAX_TOKENS,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            logger.exception("%s stream error", client_key)
            yield f"\n[{cfg['display']} error: {exc}]"

    def _stream_gemini(self, prompt: str, system_prompt: str) -> Generator[str, None, None]:
        genai = self._genai_module()
        if not genai:
            yield "[Gemini: set GEMINI_API_KEY to enable]"
            return
        try:
            kwargs = {}
            if system_prompt:
                kwargs["system_instruction"] = system_prompt
            model = genai.GenerativeModel(MODELS["gemini"]["model"], **kwargs)
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.exception("Gemini stream error")
            yield f"\n[Gemini error: {exc}]"
