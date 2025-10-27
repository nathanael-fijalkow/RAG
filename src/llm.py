from __future__ import annotations
import os
from typing import Dict, List, Optional

import google.generativeai as genai
from huggingface_hub import InferenceClient

from .config import get_settings

class LLM:
    def __init__(self):
        self.settings = get_settings()

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        if self.settings.llm_provider == "gemini":
            return self._gemini_generate(prompt, system)
        elif self.settings.llm_provider == "huggingface":
            return self._hf_generate(prompt, system)
        else:
            raise ValueError("Unsupported LLM_PROVIDER")

    def _gemini_generate(self, prompt: str, system: Optional[str]) -> str:
        api_key = self.settings.gemini_api_key
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            system_instruction=system
        )
        response = model.generate_content(prompt)
        return response.text.strip()

    def _hf_generate(self, prompt: str, system: Optional[str]) -> str:
        if not self.settings.hf_api_key:
            raise RuntimeError("HF_API_KEY not set")
        client = InferenceClient(
            model=self.settings.hf_llm_model,
        )
        response_text = client.text_generation(
            prompt=prompt
        )
        return response_text.strip()