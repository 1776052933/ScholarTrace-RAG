"""查询理解流水线的配置模块。"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class QueryUnderstandingConfig:
    """运行时设置，环境变量为可选项。"""

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    timeout_seconds: float = 30.0
    max_subqueries: int = 6
    max_keywords: int = 20
    use_llm: bool = True
    fallback_to_rules: bool = True
    language: str = "auto"

    @classmethod
    def from_env(cls, prefix: str = "QUERY_UNDERSTANDING_") -> "QueryUnderstandingConfig":
        """从环境变量构建配置，不依赖项目的 key_manager。"""
        env = os.environ
        return cls(
            model=env.get(prefix + "MODEL", cls.model),
            base_url=env.get(prefix + "BASE_URL", cls.base_url),
            api_key=env.get(prefix + "API_KEY", ""),
            timeout_seconds=float(env.get(prefix + "TIMEOUT_SECONDS", cls.timeout_seconds)),
            max_subqueries=int(env.get(prefix + "MAX_SUBQUERIES", cls.max_subqueries)),
            max_keywords=int(env.get(prefix + "MAX_KEYWORDS", cls.max_keywords)),
            use_llm=env.get(prefix + "USE_LLM", "true").lower() not in {"0", "false", "no"},
            fallback_to_rules=env.get(prefix + "FALLBACK_TO_RULES", "true").lower()
            not in {"0", "false", "no"},
            language=env.get(prefix + "LANGUAGE", cls.language),
        )
