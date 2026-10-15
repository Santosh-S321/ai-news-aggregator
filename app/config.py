from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Defaults per provider; any of these can be overridden with env vars.
PROVIDER_DEFAULTS = {
    "openai": {
        "summary_model": "gpt-4o-mini",
        "rerank_model": "gpt-4.1-mini",
        "embedding_model": "text-embedding-3-small",
        "workers": 4,
        "rpm": 0,  # 0 = no client-side cap
    },
    "gemini": {
        "summary_model": "gemini-3.5-flash-lite",
        "rerank_model": "gemini-3.5-flash",
        "embedding_model": "gemini-embedding-001",
        "workers": 2,  # free tier has low requests-per-minute limits
        "rpm": 12,  # free tier allows 15/min per model; stay under it
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_news"

    # LLM provider: "openai" or "gemini" (Gemini has a free tier)
    llm_provider: str = "openai"
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Leave blank to use the provider's default model
    summary_model: str = ""
    rerank_model: str = ""
    embedding_model: str = ""

    # Pipeline behaviour
    lookback_hours: int = 48
    top_n: int = 10
    candidate_n: int = 30
    dup_similarity: float = 0.88
    max_content_chars: int = 30_000
    workers: int = 0  # 0 = provider default
    requests_per_minute: int = -1  # -1 = provider default, 0 = unlimited
    enable_youtube_transcripts: bool = False

    # Email (any SMTP provider)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: str = ""

    sources_file: str = str(ROOT / "config" / "sources.yaml")
    profile_file: str = str(ROOT / "config" / "profile.yaml")

    @property
    def provider(self) -> str:
        name = self.llm_provider.strip().lower()
        if name not in PROVIDER_DEFAULTS:
            raise ValueError(f"LLM_PROVIDER must be 'openai' or 'gemini', got {name!r}")
        return name

    @property
    def api_key(self) -> str:
        return self.gemini_api_key if self.provider == "gemini" else self.openai_api_key

    @property
    def base_url(self) -> str | None:
        return GEMINI_BASE_URL if self.provider == "gemini" else None

    @property
    def summary_model_name(self) -> str:
        return self.summary_model or PROVIDER_DEFAULTS[self.provider]["summary_model"]

    @property
    def rerank_model_name(self) -> str:
        return self.rerank_model or PROVIDER_DEFAULTS[self.provider]["rerank_model"]

    @property
    def embedding_model_name(self) -> str:
        return self.embedding_model or PROVIDER_DEFAULTS[self.provider]["embedding_model"]

    @property
    def rpm_limit(self) -> int:
        if self.requests_per_minute >= 0:
            return self.requests_per_minute
        return PROVIDER_DEFAULTS[self.provider]["rpm"]

    @property
    def worker_count(self) -> int:
        return self.workers or PROVIDER_DEFAULTS[self.provider]["workers"]

    @property
    def sqlalchemy_url(self) -> str:
        """Render/Heroku give postgres:// or postgresql://; we use psycopg v3."""
        url = self.database_url
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()


class SourceConfig(BaseModel):
    name: str
    type: str = "rss"  # rss | youtube
    url: str | None = None  # rss feed URL
    channel_id: str | None = None  # youtube channel id
    fetch_full_text: bool = False  # download + extract the article page
    enabled: bool = True


class Profile(BaseModel):
    role: str = ""
    expertise_level: str = ""
    interests: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)

    def as_text(self) -> str:
        parts = [f"Role: {self.role}", f"Expertise: {self.expertise_level}"]
        parts.append("Interests: " + "; ".join(self.interests))
        if self.avoid:
            parts.append("Not interested in: " + "; ".join(self.avoid))
        return "\n".join(parts)


def load_sources(path: str | None = None) -> list[SourceConfig]:
    data = yaml.safe_load(Path(path or get_settings().sources_file).read_text())
    return [s for s in (SourceConfig(**item) for item in data["sources"]) if s.enabled]


def load_profile(path: str | None = None) -> Profile:
    return Profile(**yaml.safe_load(Path(path or get_settings().profile_file).read_text()))