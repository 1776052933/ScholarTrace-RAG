"""
logger.py — 日志系统

封装 Python logging，提供统一的日志器获取和配置接口。
支持控制台+文件双输出、日志轮转、彩色标记。
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Optional


# ── 全局状态 ──────────────────────────────────────────────────────

_loggers: Dict[str, logging.Logger] = {}
_initialized = False
_log_level = logging.INFO
_log_format = "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
_log_datefmt = "%Y-%m-%d %H:%M:%S"
_log_file: Optional[str] = None
_log_max_bytes = 10 * 1024 * 1024  # 10 MB
_log_backup_count = 5

# 控制台是否使用彩色
_color_enabled = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False


# ── 配置 ──────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    color: bool = True,
) -> None:
    """全局日志配置。

    Args:
        level:        日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）。
        log_file:     日志文件路径，None 表示仅输出到控制台。
        log_format:   自定义格式字符串。
        max_bytes:    日志文件最大字节数（触发轮转）。
        backup_count: 保留的轮转文件数。
        color:        是否启用控制台彩色输出。
    """
    global _initialized, _log_level, _log_file, _log_max_bytes, _log_backup_count, _color_enabled

    _log_level = getattr(logging, level.upper(), logging.INFO)
    _log_file = log_file
    _log_max_bytes = max_bytes
    _log_backup_count = backup_count
    _color_enabled = color and sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False

    if log_format:
        global _log_format
        _log_format = log_format

    # 确保日志目录存在
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 刷新已创建的 logger
    for name, logger in _loggers.items():
        _configure_logger(logger, name)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器。

    自动配置控制台 + 文件双输出，首次调用时初始化。

    Args:
        name: 日志器名称（通常使用 __name__ 或模块路径）。

    Returns:
        配置好的 logging.Logger 实例。

    Example::

        logger = get_logger(__name__)
        logger.info("搜索完成，共命中 %d 篇", 10)
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(_log_level)

    # 避免传播到 root logger（防止重复输出）
    logger.propagate = False

    _configure_logger(logger, name)
    _loggers[name] = logger

    return logger


def _configure_logger(logger: logging.Logger, name: str) -> None:
    """为 logger 添加 handler（控制台 + 文件）。"""
    # 清除已有 handlers
    logger.handlers.clear()

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_log_level)
    if _color_enabled:
        console_handler.setFormatter(_ColoredFormatter(_log_format, _log_datefmt))
    else:
        console_handler.setFormatter(logging.Formatter(_log_format, _log_datefmt))
    logger.addHandler(console_handler)

    # 文件 handler（轮转）
    if _log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            _log_file,
            maxBytes=_log_max_bytes,
            backupCount=_log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(_log_level)
        file_handler.setFormatter(logging.Formatter(_log_format, _log_datefmt))
        logger.addHandler(file_handler)


# ── 便捷函数 ──────────────────────────────────────────────────────

def debug(msg: str, *args, **kwargs) -> None:
    """模块级 DEBUG 日志。"""
    logging.getLogger("paper_downloader").debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """模块级 INFO 日志。"""
    logging.getLogger("paper_downloader").info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """模块级 WARNING 日志。"""
    logging.getLogger("paper_downloader").warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """模块级 ERROR 日志。"""
    logging.getLogger("paper_downloader").error(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs) -> None:
    """模块级 EXCEPTION 日志（附 traceback）。"""
    logging.getLogger("paper_downloader").exception(msg, *args, **kwargs)


# ── 彩色格式化器 ──────────────────────────────────────────────────

class _ColoredFormatter(logging.Formatter):
    """ANSI 彩色日志格式化器。"""

    COLORS = {
        "DEBUG":    "\033[36m",  # 青色
        "INFO":     "\033[32m",  # 绿色
        "WARNING":  "\033[33m",  # 黄色
        "ERROR":    "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # 给级别名上色
        color = self.COLORS.get(record.levelname, "")
        if color:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)
