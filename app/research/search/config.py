"""搜索 Agent 配置。"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class SearchAgentConfig:
    max_rounds: int = 3
    results_per_task: int = 10
    target_papers: int = 15
    min_relevant_papers: int = 5
    relevance_threshold: float = 0.25
    quality_threshold: float = 0.72
    request_timeout_seconds: float = 20.0
    use_llm: bool = False
    serpapi_api_key: str = ""
    openalex_api_key: str = ""
    openalex_mailto: str = ""

    @classmethod
    def from_env(cls) -> "SearchAgentConfig":
        env = os.environ
        return cls(
            max_rounds=int(env.get("SEARCH_AGENT_MAX_ROUNDS", cls.max_rounds)),
            results_per_task=int(env.get("SEARCH_AGENT_RESULTS_PER_TASK", cls.results_per_task)),
            target_papers=int(env.get("SEARCH_AGENT_TARGET_PAPERS", cls.target_papers)),
            min_relevant_papers=int(env.get("SEARCH_AGENT_MIN_RELEVANT", cls.min_relevant_papers)),
            relevance_threshold=float(env.get("SEARCH_AGENT_RELEVANCE_THRESHOLD", cls.relevance_threshold)),
            quality_threshold=float(env.get("SEARCH_AGENT_QUALITY_THRESHOLD", cls.quality_threshold)),
            request_timeout_seconds=float(env.get("SEARCH_AGENT_TIMEOUT", cls.request_timeout_seconds)),
            use_llm=env.get("SEARCH_AGENT_USE_LLM", "false").lower() in {"1", "true", "yes"},
            serpapi_api_key=_serpapi_key(env.get("SERPAPI_API_KEY", "")),
            openalex_api_key=_optional_key("openalex", env.get("OPENALEX_API_KEY", "")),
            openalex_mailto=env.get("OPENALEX_MAILTO", ""),
        )


def _serpapi_key(env_value: str) -> str:
    """优先读取环境变量，否则使用项目 API_key 密钥管理器。"""
    if env_value.strip():
        return env_value.strip()
    try:
        from API_key.key_manager import key_get

        return str(key_get("serpapi") or "").strip()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return ""


def _optional_key(name: str, env_value: str) -> str:
    if env_value.strip():
        return env_value.strip()
    try:
        from API_key.key_manager import key_get

        return str(key_get(name) or "").strip()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return ""
