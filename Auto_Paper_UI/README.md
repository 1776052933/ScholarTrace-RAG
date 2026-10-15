# AutoPaper Studio（M2）

Streamlit 界面通过 HTTP 调用 FastAPI。研究任务由 API 执行，任务进度、结果和论文问答记录写入数据库；刷新界面或重启服务后，可从侧边栏恢复已完成的任务。UI 不再启动 `main.py` 子进程。

## 主要功能

- 输入研究问题、论文数量和检索选项，启动任务并查看分阶段进度、日志及结果
- 展示候选论文、下载/入库状态、摘要、知识抽取和引用证据
- 针对已入库论文继续提问，查看 PDF 页码与原文片段
- 取消任务、恢复历史任务、导出 Markdown 报告和 CSV

## 启动

推荐按[项目根目录的 M2 运行与验收说明](../docs/M2_RUN_AND_ACCEPTANCE.md)用 Docker Compose 启动 MySQL、API 和 UI。无 Docker 时，先启动本地 API 并执行 Alembic 迁移，再启动 UI：

```powershell
$env:AUTO_PAPER_API_URL='http://127.0.0.1:8000'
python -m streamlit run Auto_Paper_UI/ui_app.py --server.address 127.0.0.1
```

`run_ui.bat` 只启动界面；API 必须先运行。界面会显示 API 连接和服务端模型配置状态。模型、搜索 API Key 及数据库连接配置在 API 进程的环境变量中，界面不会收集或传送密钥。

## 数据与后续阶段

MySQL 存储论文元数据、任务与问答；PDF 和 Chroma 索引放在 `data/m2` 持久化目录。无 Docker 的本地验收可用独立 SQLite 文件，便于运行同一 API/UI 链路。M3 再升级检索索引与召回方法。
