# AutoPaper：学术论文检索与证据问答

输入研究问题后，项目依次完成查询理解与拆解、多来源论文检索、引文扩展、论文排序、开放全文获取，以及基于论文内容的摘要、知识抽取和问答。

项目采用模块化工作流。`SearchAgent` 在单个搜索流程内进行规划、检索、评价和多轮调整；其他以 Agent 命名的模块承担排序等职责。当前实现没有多个自主 Agent 之间的任务委派与协作机制。

## 主要模块

| 目录 | 职责 |
| --- | --- |
| `query_understanding/`、`query_decomposition/` | 理解和拆解研究问题 |
| `search_agent/` | 搜索 Google Scholar、arXiv、OpenAlex，并扩展引文关系 |
| `ranking_agent/` | 用 BGE-M3、元数据特征和大模型评分排序论文 |
| `paper_downloader/`、`literature_database/` | 获取开放全文，管理下载状态及论文记录 |
| `rag_reader/` | 解析全文、切分、向量检索、摘要、知识抽取与证据问答 |
| `evaluation/` | 论文检索评测；当前评测不覆盖全文问答质量 |
| `Auto_Paper_UI/` | Streamlit 界面和结果导出 |

## 本地运行

建议使用 Python 3.10。安装依赖：

```bash
python -m pip install -r Auto_Paper_UI/requirements-ui.txt
```

需要大模型功能时，在本地配置 `DEEPSEEK_API_KEY`。Google Scholar 搜索需要 `SERPAPI_API_KEY`；OpenAlex 引文扩展需要 `OPENALEX_API_KEY`。密钥只通过环境变量或本地配置提供，勿提交到仓库。

启动界面：

```bash
python -m streamlit run Auto_Paper_UI/ui_app.py --server.address 127.0.0.1
```

运行离线测试：

```bash
python -m pytest test_rag_parallel.py test_metadata_flow.py ranking_agent/test_parallel_ranking.py ranking_agent/test_evaluation_cache.py evaluation/test_evaluation.py -q
```

完整参数、评测口径和环境配置见[项目交付与快速启动指南](项目交付与快速启动指南.md)。数据集、PDF、缓存、运行结果和本地密钥均被 `.gitignore` 排除。