"""
config_manager.py — 配置管理器

单例模式的配置中心，支持多源加载、优先级链、热更新和点号路径访问。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml


# ── 默认配置 ──────────────────────────────────────────────────────

_DEFAULT_CONFIG: Dict[str, Any] = {
    "engines": {
        "priority": ["arxiv", "crossref", "google_scholar"],
        "fallback_enabled": True,
    },
    "download": {
        "concurrent": 3,
        "timeout": 30,
        "retry": 3,
        "chunk_size": 8192,
        "resume": True,
    },
    "cache": {
        "enabled": True,
        "ttl": 86400,  # 24 小时
        "max_memory_items": 1000,
        "db_path": "cache/paper_cache.db",
        "cleanup_interval": 3600,  # 1 小时
    },
    "output": {
        "default_dir": "papers",
        "rename_by_metadata": True,
        "filename_template": "{first_author}_{year}_{title}",
    },
    "search": {
        "max_results": 10,
        "min_year": None,
        "sort_by": "relevance",
        "timeout": 30,
    },
    "proxy": {
        "enabled": False,
        "http": "",
        "https": "",
        "username": "",
        "password": "",
    },
    "logging": {
        "level": "INFO",
        "file": "logs/paper_downloader.log",
    },
}

# 环境变量 → 配置键映射
_ENV_MAPPING: Dict[str, str] = {
    "PAPER_DOWNLOAD_DIR":       "output.default_dir",
    "PAPER_CONCURRENT":         "download.concurrent",
    "PAPER_TIMEOUT":            "download.timeout",
    "PAPER_RETRY":              "download.retry",
    "PAPER_CACHE_ENABLED":      "cache.enabled",
    "PAPER_CACHE_TTL":          "cache.ttl",
    "PAPER_PROXY_HTTP":         "proxy.http",
    "PAPER_PROXY_HTTPS":        "proxy.https",
    "PAPER_LOG_LEVEL":          "logging.level",
    "PAPER_ENGINES":            "engines.priority",
}


class ConfigManager:
    """配置管理器（单例）。

    支持从 YAML / JSON / 环境变量加载配置，优先级为:
        CLI 参数 > 环境变量 > 配置文件 > 默认值

    支持热更新和点号路径读写（如 ``"download.concurrent"``）。

    Usage::

        cfg = ConfigManager()
        cfg.load_yaml("config.yaml")

        timeout = cfg.get("download.timeout", 30)
        cfg.set("download.concurrent", 5)

        # 环境变量方式
        export PAPER_CONCURRENT=10
    """

    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "ConfigManager":
        """单例模式 — 全局唯一配置实例。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器。

        仅在首次调用时执行实际初始化。

        Args:
            config_path: YAML/JSON 配置文件路径。
        """
        if self._initialized:  # type: ignore[has-type]
            if config_path:
                self.load_file(config_path)
            return

        self._initialized = True
        self._config = deepcopy(_DEFAULT_CONFIG)
        self._watch_callbacks: List[Callable[[str, Any, Any], None]] = []
        self.logger = logging.getLogger(self.__class__.__name__)

        # 1) 加载默认
        self._config = deepcopy(_DEFAULT_CONFIG)

        # 2) 加载环境变量
        self._load_env()

        # 3) 加载配置文件
        if config_path:
            self.load_file(config_path)

    # ── 文件加载 ──────────────────────────────────────────────────

    def load_file(self, path: str) -> None:
        """从 YAML 或 JSON 文件加载配置（自动识别格式）。

        加载的配置会合并覆盖已有值。

        Args:
            path: 配置文件路径。

        Raises:
            FileNotFoundError: 文件不存在。
        """
        p = Path(path)
        if not p.exists():
            self.logger.warning("配置文件不存在: %s", path)
            return

        with open(p, "r", encoding="utf-8") as f:
            if p.suffix.lower() in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            elif p.suffix.lower() == ".json":
                data = json.load(f)
            else:
                self.logger.warning("不支持的配置文件格式: %s", p.suffix)
                return

        if isinstance(data, dict):
            self._deep_merge(self._config, data)
            self.logger.info("配置已加载: %s", path)

    def load_yaml(self, path: str) -> None:
        """加载 YAML 配置文件。"""
        self.load_file(path)

    def load_json(self, path: str) -> None:
        """加载 JSON 配置文件。"""
        self.load_file(path)

    def load_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载配置（如 CLI 参数）。

        最高优先级，直接合并覆盖。

        Args:
            data: 配置字典。
        """
        self._deep_merge(self._config, data)
        self.logger.debug("配置已从字典更新: %s", list(data.keys()))

    def _load_env(self) -> None:
        """从环境变量加载配置。"""
        for env_key, config_path in _ENV_MAPPING.items():
            value = os.environ.get(env_key)
            if value is not None:
                # 类型转换
                converted = self._convert_env_value(value)
                if converted is not None:
                    self._set_by_path(config_path, converted)
                    self.logger.debug("环境变量 %s=%s → %s", env_key, value, config_path)

    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """将环境变量字符串转为合适的类型。"""
        # 布尔
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        # 整数
        try:
            return int(value)
        except ValueError:
            pass
        # 列表（逗号分隔）
        if "," in value:
            return [v.strip() for v in value.split(",")]
        return value

    # ── 读取 ──────────────────────────────────────────────────────

    def get(self, path: str, default: Any = None) -> Any:
        """通过点号路径读取配置值。

        Args:
            path:    配置路径，如 "download.concurrent"。
            default: 路径不存在时返回的默认值。

        Returns:
            配置值。

        Example::

            cfg.get("cache.enabled")  # True
            cfg.get("nonexistent", 42)  # 42
        """
        return self._get_by_path(path, default)

    def _get_by_path(self, path: str, default: Any = None) -> Any:
        """沿点号路径遍历嵌套字典获取值。"""
        keys = path.split(".")
        node: Any = self._config
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def all(self) -> Dict[str, Any]:
        """返回全部配置的深拷贝。"""
        return deepcopy(self._config)

    # ── 写入 ──────────────────────────────────────────────────────

    def set(self, path: str, value: Any) -> None:
        """设置配置值（支持热更新）。

        通知所有已注册的变更监听器。

        Args:
            path:  配置路径。
            value: 新值。
        """
        old = self._get_by_path(path)
        self._set_by_path(path, value)
        new = self._get_by_path(path)

        if old != new:
            for cb in self._watch_callbacks:
                try:
                    cb(path, old, new)
                except Exception as exc:
                    self.logger.warning("配置变更回调异常: %s", exc)

    def _set_by_path(self, path: str, value: Any) -> None:
        """沿路径设置嵌套字典的值，不存在的中间键自动创建。"""
        keys = path.split(".")
        node: Any = self._config
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """批量更新配置。

        Args:
            updates: 配置键值对字典。
        """
        self._deep_merge(self._config, updates)

    # ── 热更新监听 ────────────────────────────────────────────────

    def on_change(self, callback: Callable[[str, Any, Any], None]) -> None:
        """注册配置变更监听器。

        Args:
            callback: 签名为 (path: str, old_value: Any, new_value: Any) -> None。

        Example::

            def on_timeout_change(path, old, new):
                if path == "download.timeout":
                    print(f"超时从 {old}s 变为 {new}s")

            cfg.on_change(on_timeout_change)
        """
        self._watch_callbacks.append(callback)

    def remove_on_change(self, callback: Callable) -> bool:
        """移除变更监听器。"""
        try:
            self._watch_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # ── 持久化 ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """将当前配置保存到文件。

        Args:
            path: YAML 或 JSON 文件路径。
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            if p.suffix.lower() in (".yaml", ".yml"):
                yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
            else:
                json.dump(self._config, f, ensure_ascii=False, indent=2)

        self.logger.info("配置已保存: %s", path)

    def reset(self) -> None:
        """重置为默认配置。"""
        self._config = deepcopy(_DEFAULT_CONFIG)
        self._load_env()
        self.logger.info("配置已重置为默认值")

    # ── 工具 ──────────────────────────────────────────────────────

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """递归合并 override 到 base 中。"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManager._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value) if isinstance(value, dict) else value

    def __contains__(self, path: str) -> bool:
        return self._get_by_path(path, _SENTINEL) is not _SENTINEL

    def __repr__(self) -> str:
        return f"ConfigManager(keys={list(self._config.keys())})"


_SENTINEL = object()
