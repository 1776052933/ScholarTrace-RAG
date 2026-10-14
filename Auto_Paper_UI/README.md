# AutoPaper Studio

这是 AutoPaper 项目的用户界面升级版。研究任务通过子进程调用上一级目录中的 `main.py`，并沿用项目的检索、排序、下载、RAG、摘要和知识抽取逻辑。

## 主要功能

- 输入研究主题和论文数量
- 控制下载、RAG、摘要和知识抽取
- 设置搜索轮次、下载来源和超时
- 缓存经过校验的论文评分，重复研究时减少 DeepSeek 调用
- 并发生成 Top 论文摘要和结构化知识
- 优先使用 PDF URL、arXiv ID 和 DOI，失败后再执行多源标题搜索
- Top-K 网络下载并发执行，并对损坏文件、HTML 页面和残缺文件做校验
- 每次研究自动保存脱敏日志，并保留最近 30 次运行
- 展示 Top 论文及原文链接
- 研究完成后按论文连续提问，并查看回答和证据片段
- 导出 Top 论文 Markdown 报告、完整精排论文 CSV 和运行日志

## 首次安装

在 PyCharm Terminal 中执行：

```powershell
conda activate pywork
cd D:\files\Auto_Paper-model
python -m pip install -r Auto_Paper_UI\requirements-ui.txt
```

## 启动界面

```powershell
cd D:\files\Auto_Paper-model
python -m streamlit run Auto_Paper_UI\ui_app.py
```

浏览器通常会自动打开 `http://localhost:8501`。也可以在已激活 `pywork` 环境后双击 `run_ui.bat`。

## API 配置

界面会继承启动终端中的 `DEEPSEEK_API_KEY` 和 `SERPAPI_API_KEY`，也会继续使用原项目 `API_key` 目录中的配置。界面中的“临时 API Key”只对当前一次运行有效，不会写入磁盘。

## 数据位置

- PDF：`Auto_Paper_UI/runtime/papers`
- RAG 向量库：`Auto_Paper_UI/runtime/chroma`
- 文献数据库：`Auto_Paper_UI/runtime/literature.db`
- 完整结果：`Auto_Paper_UI/outputs`
- 运行日志：与结果 JSON 同名，保存在 `Auto_Paper_UI/outputs`
- 评分缓存：`Auto_Paper_UI/runtime/ranking_evaluations.sqlite3`

这些目录与原命令行项目的数据隔离，便于比赛演示和清理。

## 下载策略

排序结果中的 PDF URL、arXiv ID 和 DOI 会一直传递到下载阶段。系统优先直接下载明确的开放全文；没有精确标识或直链失败时，才回退到 arXiv、OpenAlex、CrossRef、Semantic Scholar、Unpaywall 和 Europe PMC。Top-K 论文并发下载，数据库状态在主线程按排名顺序写入。

界面默认勾选“重试历史失败任务”，用于恢复临时网络错误。命令行重试时请添加 `--retry-failed-downloads`。
