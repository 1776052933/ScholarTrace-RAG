# 测试目录

从项目根目录执行 `python -m pytest -q tests -m "not slow and not network"`。`pytest.ini` 已将 `tests/` 设为默认收集目录，因此也可直接执行 `python -m pytest -q -m "not slow and not network"`。

| 位置 | 覆盖范围 |
| --- | --- |
| `tests/test_*.py` | API、工作流、RAG、评测契约等跨模块回归 |
| `tests/ui/` | Streamlit 界面与展示辅助函数 |
| `tests/evaluation/` | 评测工具与引文追踪 |
| `tests/downloader/` | 旧下载器的单元与集成测试；共享 fixtures 在本目录的 `conftest.py` |
| `tests/papers/`、`tests/research/` | 论文资源与研究模块的功能测试 |

真实服务、网络和模型质量评测依照 `docs/` 与 `evaluation/` 中的专用命令执行；常规 pytest 回归不发送真实模型请求。测试中的临时 SQLite、Chroma 及模拟模型用于验证代码路径，当前生产配置由 `.env` 与 Compose 指定。
