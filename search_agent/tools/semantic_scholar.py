"""带同源限速和 429 退避的 Semantic Scholar Graph API 搜索。"""

from email.utils import parsedate_to_datetime
import json
import logging
import os
from threading import Lock
import time
from typing import Any, Callable, List, Optional
from urllib import error, parse, request

from ..schema import PaperMetadata


logger = logging.getLogger(__name__)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class SemanticScholarTool:
    """线程安全的 Semantic Scholar 搜索客户端。

    Search Agent 会并发执行不同数据源，但该工具会串行化自身请求，以免多个
    子查询同时击中 Semantic Scholar 的单源速率限制。
    """

    source = "semantic_scholar"

    def __init__(
        self,
        api_key: str = "",
        timeout_seconds: float = 20.0,
        *,
        min_interval_seconds: float = 1.0,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
        cooldown_seconds: float = 60.0,
        _sleep: Callable[[float], None] = time.sleep,
        _clock: Callable[[], float] = time.monotonic,
        _urlopen: Callable[..., Any] = request.urlopen,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must not be negative")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must not be negative")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")
        self.api_key = api_key.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.min_interval_seconds = float(min_interval_seconds)
        self.max_retries = max_retries
        self.backoff_seconds = float(backoff_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self._sleep = _sleep
        self._clock = _clock
        self._urlopen = _urlopen
        self._request_lock = Lock()
        self._last_request_started: Optional[float] = None
        self._cooldown_until = 0.0
        self._cooldown_logged = False

    def search(self, query: str, limit: int = 10) -> List[PaperMetadata]:
        """搜索论文；重试耗尽或其他 API 异常时返回空列表。"""
        if not query.strip() or limit < 1:
            return []
        req = self._request(query, limit)
        payload = self._request_json(req, query)
        if payload is None:
            return []
        return _papers(payload)

    def _request(self, query: str, limit: int) -> request.Request:
        params = parse.urlencode({
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,year,authors,url,citationCount,venue,externalIds",
        })
        headers = {"User-Agent": "Auto-paper-search-agent/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return request.Request(
            f"https://api.semanticscholar.org/graph/v1/paper/search?{params}",
            headers=headers,
        )

    def _request_json(
        self,
        req: request.Request,
        query: str,
    ) -> Optional[dict[str, Any]]:
        with self._request_lock:
            if self._clock() < self._cooldown_until:
                if not self._cooldown_logged:
                    logger.info(
                        "Semantic Scholar 处于限流冷却期，暂时跳过本轮排队请求"
                    )
                    self._cooldown_logged = True
                return None
            for attempt in range(self.max_retries + 1):
                self._wait_for_rate_slot()
                self._last_request_started = self._clock()
                try:
                    with self._urlopen(req, timeout=self.timeout_seconds) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    self._cooldown_until = 0.0
                    self._cooldown_logged = False
                    return payload
                except error.HTTPError as exc:
                    if exc.code in _RETRYABLE_STATUS and attempt < self.max_retries:
                        delay = _retry_delay(
                            exc,
                            attempt=attempt,
                            backoff_seconds=self.backoff_seconds,
                        )
                        logger.info(
                            "Semantic Scholar HTTP %d，%.1f 秒后重试（%d/%d）：%s",
                            exc.code,
                            delay,
                            attempt + 1,
                            self.max_retries,
                            query[:80],
                        )
                        self._sleep(delay)
                        continue
                    if exc.code == 429:
                        self._cooldown_until = (
                            self._clock() + self.cooldown_seconds
                        )
                        self._cooldown_logged = False
                    logger.warning(
                        "Semantic Scholar API 搜索失败：HTTP %d，query=%s",
                        exc.code,
                        query[:80],
                    )
                    return None
                except (error.URLError, TimeoutError, OSError) as exc:
                    if attempt < self.max_retries:
                        delay = self.backoff_seconds * (2 ** attempt)
                        logger.info(
                            "Semantic Scholar 网络异常，%.1f 秒后重试（%d/%d）：%s",
                            delay,
                            attempt + 1,
                            self.max_retries,
                            type(exc).__name__,
                        )
                        self._sleep(delay)
                        continue
                    logger.warning("Semantic Scholar API 搜索失败：%s", exc)
                    return None
                except json.JSONDecodeError as exc:
                    logger.warning("Semantic Scholar API 返回无效 JSON：%s", exc)
                    return None
        return None

    def _wait_for_rate_slot(self) -> None:
        if self._last_request_started is None:
            return
        remaining = (
            self.min_interval_seconds
            - (self._clock() - self._last_request_started)
        )
        if remaining > 0:
            self._sleep(remaining)


def search(query: str, limit: int = 10) -> List[PaperMetadata]:
    """模块级便捷接口。"""
    return SemanticScholarTool(
        api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", "") or _project_key(),
    ).search(query, limit)


def _papers(payload: dict[str, Any]) -> List[PaperMetadata]:
    papers = []
    for item in payload.get("data", []):
        try:
            if not item.get("title"):
                continue
            papers.append(PaperMetadata(
                title=item["title"],
                authors=[
                    author.get("name", "")
                    for author in item.get("authors", [])
                    if author.get("name")
                ],
                abstract=item.get("abstract") or "",
                year=item.get("year"),
                venue=item.get("venue") or "",
                citation_count=item.get("citationCount"),
                url=item.get("url") or "",
                source=SemanticScholarTool.source,
                doi=(item.get("externalIds") or {}).get("DOI"),
                sources=[SemanticScholarTool.source],
            ))
        except (TypeError, ValueError):
            logger.debug(
                "跳过格式异常的 Semantic Scholar 论文记录",
                exc_info=True,
            )
    return papers


def _retry_delay(
    exception: error.HTTPError,
    *,
    attempt: int,
    backoff_seconds: float,
) -> float:
    header = exception.headers.get("Retry-After") if exception.headers else None
    retry_after = _parse_retry_after(header)
    if retry_after is not None:
        return min(60.0, max(0.0, retry_after))
    return min(60.0, backoff_seconds * (2 ** attempt))


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            now = time.time()
            return max(0.0, target.timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return None


def _project_key() -> str:
    try:
        from API_key.key_manager import key_get

        return str(key_get("semantic_scholar") or "").strip()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return ""
