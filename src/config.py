import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    llm_provider: str = "gemini"  # or 'huggingface'
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    hf_api_key: str | None = None
    hf_llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        hf_api_key=os.getenv("HUGGINGFACE_API_KEY"),
        hf_llm_model=os.getenv("HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    )
