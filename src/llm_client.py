# src/llm_client.py

"""Ollama HTTP client with streaming support."""

from __future__ import annotations

import json
import sys
from typing import List, Optional

import requests

from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


class OllamaClient:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout_s: Optional[int] = None,
    ):
        cfg = get_settings().llm
        self.model = model or cfg.model
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.url = f"{self.base_url}/api/generate"
        self.request_timeout_s = request_timeout_s or cfg.request_timeout_s
        logger.debug("OllamaClient configured model=%s base_url=%s", self.model, self.base_url)

    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str:
        """
        Generate an answer using Ollama with optional streaming output.

        Returns the complete generated response as a string.
        """
        if context:
            full_prompt = "\n\n".join(context) + f"\n\nQuestion: {prompt}\nAnswer:"
        else:
            full_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "stream": True,
        }

        try:
            response = requests.post(self.url, json=payload, stream=True, timeout=self.request_timeout_s)
            response.raise_for_status()

            final_text = ""

            if stream:
                logger.debug("Streaming LLM output from %s", self.url)
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    token = data.get("response", "")
                    if token:
                        final_text += token
                        sys.stdout.write(token)
                        sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                for chunk in response.text.strip().split("\n"):
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if "response" in data:
                        final_text += data["response"]

            return final_text.strip() or "No response generated."

        except Exception as exc:
            logger.error("Ollama request failed: %s", exc)
            return f"[Error communicating with Ollama: {exc}]"

    def stream_generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Yield individual response tokens as they arrive from Ollama.

        Generator interface for use with HTTP streaming (SSE). The caller
        controls how to surface tokens — e.g. the FastAPI ``/ask`` route
        emits them as ``token`` SSE events.

        On error, raises rather than returning a string (callers wrap
        and emit an ``error`` event).
        """
        if context:
            full_prompt = "\n\n".join(context) + f"\n\nQuestion: {prompt}\nAnswer:"
        else:
            full_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "stream": True,
        }

        response = requests.post(
            self.url, json=payload, stream=True, timeout=self.request_timeout_s
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            token = data.get("response", "")
            if token:
                yield token


if __name__ == "__main__":
    client = OllamaClient()
    ans = client.generate("Explain reinforcement learning in simple terms.")
    print("\n\nFinal Answer:\n", ans)
