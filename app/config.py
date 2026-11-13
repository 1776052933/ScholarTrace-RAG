"""Validated, server-owned model provider configuration for M2."""
from __future__ import annotations

import os
from ai_components.provider import DEFAULT_BASE_URL, DEFAULT_MODEL, resolve_provider
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: SecretStr = SecretStr("")
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    base_url: str = Field(default=DEFAULT_BASE_URL, min_length=1)
    timeout_seconds: float = Field(default=60.0, gt=0)

    @classmethod
    def from_env(cls) -> "ModelSettings":
        def legacy_key() -> str:
            try:
                from API_key.key_manager import key_get
                return str(key_get("deepseek") or "").strip()
            except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                return ""
        provider = resolve_provider(key_fallback=legacy_key)
        return cls(
            api_key=SecretStr(provider.api_key),
            model=provider.model,
            base_url=provider.base_url,
            timeout_seconds=os.getenv("DEEPSEEK_TIMEOUT") or 60.0,
        )

    def client_kwargs(self) -> dict[str, str]:
        return {
            "api_key": self.api_key.get_secret_value(),
            "model": self.model,
            "base_url": self.base_url.rstrip("/"),
        }
