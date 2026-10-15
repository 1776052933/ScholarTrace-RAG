"""Validated, server-owned model provider configuration for M2."""
from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: SecretStr = SecretStr("")
    model: str = Field(default="deepseek-chat", min_length=1)
    base_url: str = Field(default="https://api.deepseek.com", min_length=1)
    timeout_seconds: float = Field(default=60.0, gt=0)

    @classmethod
    def from_env(cls) -> "ModelSettings":
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            try:
                from API_key.key_manager import key_get
                key = str(key_get("deepseek") or "").strip()
            except (ImportError, OSError, TypeError, ValueError):
                pass
        return cls(
            api_key=SecretStr(key),
            model=os.getenv("DEEPSEEK_MODEL") or "deepseek-chat",
            base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            timeout_seconds=os.getenv("DEEPSEEK_TIMEOUT") or 60.0,
        )

    def client_kwargs(self) -> dict[str, str]:
        return {
            "api_key": self.api_key.get_secret_value(),
            "model": self.model,
            "base_url": self.base_url.rstrip("/"),
        }
