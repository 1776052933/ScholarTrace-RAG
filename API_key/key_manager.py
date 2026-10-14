"""
key_manager.py — 统一 API Key 管理

使用方式:
    from API_key.key_manager import key_get
    serpapi_key = key_get("serpapi")
    deepseek_key = key_get("deepseek")
"""

import json
import os

_JSON_PATH = os.path.join(os.path.dirname(__file__), "API_key.json")
_cache = None

# 项目运行所必需的 API 服务
REQUIRED_KEYS = ["serpapi", "deepseek"]


def _load() -> dict:
    """加载 JSON 文件（仅首次读取，后续命中缓存）。"""
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _cache = {}
    return _cache


def key_get(name: str) -> str:
    """
    获取指定服务的 API Key。

    查找顺序: JSON 文件 → 环境变量（{NAME}_API_KEY）→ 空字符串

    Args:
        name: 服务名，如 "serpapi"、"deepseek"

    Returns:
        API Key 字符串，未找到则返回 ""
    """
    keys = _load()
    if name in keys and keys[name]:
        return keys[name]

    env_var = f"{name.upper()}_API_KEY"
    return os.environ.get(env_var, "")


def verify_keys() -> bool:
    """
    检验 API_key.json 是否存在并完整。

    - 不存在则新建模板文件，并清除缓存
    - 存在但缺失必需 Key 或值为空，则打印缺失项警告

    Returns:
        True 表示所有必需 Key 已就绪，False 表示有缺失项
    """
    global _cache

    if not os.path.exists(_JSON_PATH):
        template = {k: "" for k in REQUIRED_KEYS}
        os.makedirs(os.path.dirname(_JSON_PATH), exist_ok=True)
        with open(_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        _cache = None
        print(f"[密钥管理器] 已创建 API_key.json 模板，请填入所需 Key:")
        print(f"              {_JSON_PATH}")
        for k in REQUIRED_KEYS:
            print(f"              - {k}: \"\"")
        return False

    keys = _load()
    missing = [k for k in REQUIRED_KEYS if k not in keys or not keys[k]]

    if missing:
        print(f"[密钥管理器] ⚠ API_key.json 缺少以下 Key: {', '.join(missing)}")
        return False

    return True
