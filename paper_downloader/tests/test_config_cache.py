"""
test_config_cache.py — 配置/缓存/验证器模块测试.

覆盖 ConfigManager、CacheManager、cached 装饰器、validators、
以及 SearchFactory 缓存集成。
"""

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import yaml

from paper_downloader.src.config.config_manager import ConfigManager
from paper_downloader.src.cache.cache_manager import CacheManager
from paper_downloader.src.cache.cache_decorator import cached, invalidate_cache, get_cache, set_cache
from paper_downloader.src.utils.validators import (
    validate_title,
    validate_doi,
    validate_url,
    sanitize_filename,
    validate_title_or_raise,
    validate_doi_or_raise,
    validate_url_or_raise,
    is_probable_title,
)
from paper_downloader.src.search_engines.search_factory import SearchFactory


# ═══════════════════════════════════════════════════════════════════
# ConfigManager 测试
# ═══════════════════════════════════════════════════════════════════

class TestConfigManager:
    """配置管理器测试."""

    def teardown_method(self):
        ConfigManager._instance = None

    def test_singleton(self):
        """单例模式."""
        a = ConfigManager()
        b = ConfigManager()
        assert a is b

    def test_default_values(self):
        """默认值."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        assert cfg.get("download.concurrent") == 3
        assert cfg.get("cache.enabled") is True
        assert cfg.get("cache.ttl") == 86400
        assert cfg.get("output.default_dir") == "papers"

    def test_get_with_default(self):
        """路径不存在返回默认值."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        assert cfg.get("nonexistent.path", 42) == 42
        assert cfg.get("a.b.c.d") is None

    def test_set_and_get(self):
        """设置和读取."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        cfg.set("download.timeout", 60)
        assert cfg.get("download.timeout") == 60

    def test_set_nested_path(self):
        """自动创建中间键."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        cfg.set("new.section.key", "value")
        assert cfg.get("new.section.key") == "value"

    def test_update(self):
        """批量更新."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        cfg.update({"download": {"timeout": 99, "retry": 5}})
        assert cfg.get("download.timeout") == 99
        assert cfg.get("download.retry") == 5
        # 未更新的键保持原值
        assert cfg.get("download.concurrent") == 3

    def test_load_yaml(self, tmp_path):
        """从 YAML 文件加载."""
        ConfigManager._instance = None
        f = tmp_path / "cfg.yaml"
        f.write_text("download:\n  timeout: 45\n  retry: 7\n")

        cfg = ConfigManager()
        cfg.load_yaml(str(f))
        assert cfg.get("download.timeout") == 45
        assert cfg.get("download.retry") == 7

    def test_load_json(self, tmp_path):
        """从 JSON 文件加载."""
        ConfigManager._instance = None
        f = tmp_path / "cfg.json"
        json.dump({"download": {"concurrent": 8}}, f.open("w"))

        cfg = ConfigManager()
        cfg.load_json(str(f))
        assert cfg.get("download.concurrent") == 8

    def test_load_dict(self):
        """从字典加载（最高优先级）."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        cfg.load_dict({"download": {"timeout": 120}})
        assert cfg.get("download.timeout") == 120

    def test_env_variable(self, monkeypatch):
        """环境变量加载."""
        ConfigManager._instance = None
        monkeypatch.setenv("PAPER_TIMEOUT", "42")
        monkeypatch.setenv("PAPER_CACHE_ENABLED", "false")

        cfg = ConfigManager()
        assert cfg.get("download.timeout") == 42
        assert cfg.get("cache.enabled") is False

    def test_env_list_value(self, monkeypatch):
        """环境变量逗号分隔列表."""
        ConfigManager._instance = None
        monkeypatch.setenv("PAPER_ENGINES", "arxiv,crossref")

        cfg = ConfigManager()
        assert cfg.get("engines.priority") == ["arxiv", "crossref"]

    def test_on_change_callback(self):
        """配置变更监听."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        changes: List[tuple] = []

        cfg.on_change(lambda path, old, new: changes.append((path, old, new)))
        cfg.set("download.timeout", 99)

        assert len(changes) == 1
        assert changes[0][0] == "download.timeout"
        assert changes[0][2] == 99

    def test_remove_callback(self):
        """移除监听器."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        called = []

        def cb(p, o, n): called.append(p)

        cfg.on_change(cb)
        cfg.set("download.timeout", 1)
        assert len(called) == 1

        cfg.remove_on_change(cb)
        cfg.set("download.timeout", 2)
        assert len(called) == 1  # 未再增加

    def test_save_to_yaml(self, tmp_path):
        """保存配置."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        cfg.set("download.timeout", 88)
        p = str(tmp_path / "saved.yaml")
        cfg.save(p)
        assert Path(p).exists()

    def test_reset(self):
        """重置为默认."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        cfg.set("download.timeout", 999)
        cfg.reset()
        assert cfg.get("download.timeout") == 30  # 默认值

    def test_all_returns_copy(self):
        """all() 返回深拷贝."""
        ConfigManager._instance = None
        cfg = ConfigManager()
        data = cfg.all()
        data["download"]["timeout"] = 999
        assert cfg.get("download.timeout") == 30  # 原值未变


# ═══════════════════════════════════════════════════════════════════
# CacheManager 测试
# ═══════════════════════════════════════════════════════════════════

class TestCacheManager:
    """缓存管理器测试."""

    def setup_method(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.cache = CacheManager(db_path=self.db_path, max_memory=50, default_ttl=3600)

    def teardown_method(self):
        self.cache.close()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_set_and_get(self):
        """基本读写."""
        self.cache.set("test:key1", {"data": [1, 2, 3]})
        result = self.cache.get("test:key1")
        assert result == {"data": [1, 2, 3]}

    def test_get_missing(self):
        """未命中返回 None."""
        assert self.cache.get("nonexistent:key") is None

    def test_get_expired(self):
        """过期返回 None."""
        self.cache.set("exp:key", "value", ttl=-1)  # 负 TTL = 已过期
        result = self.cache.get("exp:key")
        assert result is None

    def test_delete(self):
        """删除条目."""
        self.cache.set("del:key", "value")
        assert self.cache.get("del:key") == "value"
        self.cache.delete("del:key")
        assert self.cache.get("del:key") is None

    def test_exists(self):
        """存在性检查."""
        self.cache.set("exists:key", "hello")
        assert self.cache.exists("exists:key") is True
        assert self.cache.exists("no:key") is False

    def test_clear_by_namespace(self):
        """按命名空间清空."""
        self.cache.set("ns:a:1", 1)
        self.cache.set("ns:a:2", 2)
        self.cache.set("ns:b:3", 3)

        count = self.cache.clear(namespace="ns:a:")
        assert count == 2
        assert self.cache.get("ns:a:1") is None
        assert self.cache.get("ns:b:3") == 3

    def test_clear_all(self):
        """清空全部."""
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        self.cache.clear()
        assert self.cache.get("a") is None
        assert self.cache.get("b") is None

    def test_cleanup_expired(self):
        """清理过期条目."""
        self.cache.set("fresh", "ok", ttl=3600)
        self.cache.set("stale", "old", ttl=-1)  # 负 TTL = 已过期
        removed = self.cache.cleanup()
        assert removed >= 1  # stale 被清除
        assert self.cache.get("fresh") == "ok"

    def test_memory_lru_eviction(self):
        """LRU 淘汰."""
        small_cache = CacheManager(db_path=tempfile.mktemp(suffix=".db"),
                                   max_memory=3, default_ttl=3600)
        for i in range(10):
            small_cache.set(f"key:{i}", i)

        # 只有最近 3 个在内存中
        stats = small_cache.stats()
        assert stats["memory_entries"] == 3
        small_cache.close()

    def test_disk_persistence(self):
        """磁盘持久化 — 新实例可读取."""
        self.cache.set("disk:key", "persistent")
        self.cache.close()

        # 新实例，同一数据库
        cache2 = CacheManager(db_path=self.db_path)
        val = cache2.get("disk:key")
        assert val == "persistent"
        cache2.close()

    def test_stats(self):
        """统计信息."""
        self.cache.set("s1", "a")
        self.cache.set("s2", "b")
        stats = self.cache.stats()
        assert stats["memory_entries"] == 2
        assert stats["disk_entries"] >= 0

    def test_context_manager(self):
        """上下文管理器."""
        with CacheManager(db_path=tempfile.mktemp(suffix=".db")) as cm:
            cm.set("ctx:key", "ctx_val")
            assert cm.get("ctx:key") == "ctx_val"

    def test_complex_value(self):
        """复杂值序列化."""
        value = [
            {"title": "P1", "authors": ["Alice"], "year": "2024"},
            {"title": "P2", "authors": ["Bob"], "year": "2023", "doi": "10.xxx"},
        ]
        self.cache.set("complex", value)
        assert self.cache.get("complex") == value


# ═══════════════════════════════════════════════════════════════════
# cached 装饰器测试
# ═══════════════════════════════════════════════════════════════════

class TestCachedDecorator:
    """缓存装饰器测试."""

    def setup_method(self):
        self.cache = CacheManager(db_path=tempfile.mktemp(suffix=".db"), max_memory=100)
        set_cache(self.cache)

    def teardown_method(self):
        self.cache.close()

    def test_sync_cached(self):
        """同步函数缓存."""
        call_count = [0]

        @cached(ttl=3600)
        def expensive(x: int) -> int:
            call_count[0] += 1
            return x * 2

        assert expensive(5) == 10
        assert call_count[0] == 1
        assert expensive(5) == 10  # 缓存命中
        assert call_count[0] == 1

    def test_sync_different_args(self):
        """不同参数不同缓存."""
        call_count = [0]

        @cached(ttl=3600)
        def compute(x: int) -> int:
            call_count[0] += 1
            return x * 3

        compute(1)
        compute(2)
        assert call_count[0] == 2

    def test_ignore_args(self):
        """忽略参数."""
        call_count = [0]

        @cached(ttl=3600, ignore_args=True)
        def const_func(x: int) -> str:
            call_count[0] += 1
            return "fixed"

        const_func(1)
        const_func(999)  # 不同参数但缓存命中
        assert call_count[0] == 1

    def test_fixed_key(self):
        """固定缓存键."""
        call_count = [0]

        @cached(ttl=3600, key="my_fixed_key")
        def my_func() -> str:
            call_count[0] += 1
            return "result"

        assert my_func() == "result"
        assert my_func() == "result"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_async_cached(self):
        """异步函数缓存."""
        call_count = [0]

        @cached(ttl=3600)
        async def async_expensive(x: int) -> int:
            call_count[0] += 1
            return x + 10

        result = await async_expensive(5)
        assert result == 15
        assert call_count[0] == 1

        result = await async_expensive(5)
        assert result == 15
        assert call_count[0] == 1  # 缓存命中

    def test_invalidate_cache(self):
        """清除装饰器缓存."""
        call_count = [0]

        @cached(ttl=3600)
        def to_invalidate(x: int) -> int:
            call_count[0] += 1
            return x

        to_invalidate(1)
        assert call_count[0] == 1

        invalidate_cache()
        to_invalidate(1)
        assert call_count[0] == 2  # 缓存已清除


# ═══════════════════════════════════════════════════════════════════
# validators 测试
# ═══════════════════════════════════════════════════════════════════

class TestValidators:
    """输入验证器测试."""

    def test_validate_title_valid(self):
        """合法标题."""
        ok, err = validate_title("Attention Is All You Need")
        assert ok is True
        assert err is None

    def test_validate_title_empty(self):
        """空标题."""
        ok, err = validate_title("")
        assert ok is False
        assert "不能为空" in err  # type: ignore[union-attr]

    def test_validate_title_too_short(self):
        """太短."""
        ok, err = validate_title("A")
        assert ok is False

    def test_validate_title_too_long(self):
        """太长."""
        ok, err = validate_title("X" * 2000)
        assert ok is False

    def test_validate_title_only_numbers(self):
        """纯数字."""
        ok, err = validate_title("12345 67890")
        assert ok is False

    def test_validate_title_or_raise(self):
        """抛异常版."""
        assert validate_title_or_raise("  Hello World  ") == "Hello World"
        with pytest.raises(ValueError):
            validate_title_or_raise("")

    def test_validate_doi_valid(self):
        """合法 DOI."""
        ok, _ = validate_doi("10.1038/nature14539")
        assert ok is True

    def test_validate_doi_with_prefix(self):
        """带 URL 前缀."""
        ok, _ = validate_doi("https://doi.org/10.1234/abc.def")
        assert ok is True

    def test_validate_doi_invalid(self):
        """无效 DOI."""
        ok, err = validate_doi("not-a-doi")
        assert ok is False

    def test_validate_doi_or_raise_cleans(self):
        """清理 DOI 前缀."""
        result = validate_doi_or_raise("https://doi.org/10.1234/test")
        assert result == "10.1234/test"

    def test_validate_url_valid(self):
        """合法 URL."""
        ok, _ = validate_url("https://arxiv.org/pdf/2401.00001.pdf")
        assert ok is True

    def test_validate_url_no_scheme(self):
        """缺少协议."""
        ok, err = validate_url("arxiv.org/pdf/test.pdf")
        assert ok is False

    def test_validate_url_ftp(self):
        """FTP 协议."""
        ok, _ = validate_url("ftp://files.example.org/paper.pdf")
        assert ok is True

    def test_sanitize_filename(self):
        """文件名清理."""
        result = sanitize_filename('bad:name<test>.pdf')
        assert ":" not in result
        assert "<" not in result

    def test_sanitize_filename_empty(self):
        """空输入."""
        assert sanitize_filename("") == "untitled"

    def test_sanitize_filename_truncate(self):
        """截断."""
        long_name = "x" * 300 + ".pdf"
        result = sanitize_filename(long_name, max_len=100)
        assert len(result) <= 100
        assert result.endswith(".pdf")

    def test_is_probable_title(self):
        """标题启发式判断."""
        assert is_probable_title("Attention Is All You Need") is True
        assert is_probable_title("Deep Learning: A Comprehensive Survey") is True
        assert is_probable_title("ml") is False
        assert is_probable_title("") is False


# ═══════════════════════════════════════════════════════════════════
# SearchFactory 缓存集成测试
# ═══════════════════════════════════════════════════════════════════

class TestSearchFactoryCache:
    """SearchFactory 缓存集成."""

    def test_cache_store_and_hit(self):
        """搜索结果缓存."""
        cache = CacheManager(db_path=tempfile.mktemp(suffix=".db"), max_memory=100)

        factory = SearchFactory({
            "search": {"engines": ["arxiv"], "max_results": 5},
            "cache": {"enabled": True, "ttl": 3600},
            "concurrency": {"request_delay": 0},
        })
        factory.set_cache(cache)

        # Mock 搜索结果
        mock_result = [{"title": "Cached Paper", "authors": ["A"], "year": "2024", "source": "arxiv"}]

        with patch.object(factory, "_parallel_search", return_value=mock_result) as mock_search:
            r1 = factory.search_all("test query", max_results=5)
            assert len(r1) == 1
            assert mock_search.call_count == 1

            # 第二次应命中缓存
            r2 = factory.search_all("test query", max_results=5)
            assert len(r2) == 1
            assert mock_search.call_count == 1  # 未再调用搜索引擎

        cache.close()

    def test_disable_cache(self):
        """禁用缓存."""
        cache = CacheManager(db_path=tempfile.mktemp(suffix=".db"), max_memory=100)

        factory = SearchFactory({
            "search": {"engines": ["arxiv"], "max_results": 5},
            "cache": {"enabled": True},
            "concurrency": {"request_delay": 0},
        })
        factory.set_cache(cache)
        factory.disable_cache()

        mock_result = [{"title": "Fresh Paper", "authors": ["B"], "year": "2024", "source": "arxiv"}]
        with patch.object(factory, "_parallel_search", return_value=mock_result) as mock_search:
            factory.search_all("fresh query", max_results=5)
            factory.search_all("fresh query", max_results=5)
            assert mock_search.call_count == 2  # 每次都搜索

        cache.close()

    def test_no_cache_by_default(self):
        """默认不启用缓存."""
        factory = SearchFactory({
            "search": {"engines": ["arxiv"], "max_results": 5},
            "concurrency": {"request_delay": 0},
        })
        # 无 cache manager 时不应尝试查缓存
        mock_result = [{"title": "No Cache", "authors": ["C"], "year": "2024", "source": "arxiv"}]
        with patch.object(factory, "_parallel_search", return_value=mock_result) as mock_search:
            factory.search_all("no cache", max_results=5)
            factory.search_all("no cache", max_results=5)
            assert mock_search.call_count == 2


# ═══════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
