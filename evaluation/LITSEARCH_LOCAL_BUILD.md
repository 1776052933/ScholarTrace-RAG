# LitSearch 全量本地索引：2026-10-21

## 配置

- 原始数据：`data/benchmarks/litsearch/corpus_clean`，六个 Parquet 分片。
- 目标：64,183 篇，标题 + 摘要，每篇一个文档/向量；不下载 PDF。
- Embedding：已有 `data/m3/bge_m3_model` BGE-M3 权重，CPU、batch size 8。
- 当前镜像 `scholartrace:local`：torch 2.5.1+cpu，不支持 CUDA。
- Milvus：现有 `scholartrace_repro` 服务的独立集合
  `eval_current_64501a1afc8cf37e4bc22a34`，不使用或清空业务集合。
- 运行目录：`evaluation/runs/litsearch_full_v2_20261021`。
- 容器名：`scholartrace-litsearch-index-v2-20261021`。
- 离线加载权重，不注入模型 API 密钥，不调用生成模型；仅连接本机 Milvus。
- 固定全部 597 个问题用于后续检索，当前只执行 index，没有执行 retrieve/answer。
- 配置 dense、top-k 20，后续不同检索设置需按当前入口的 manifest 规则使用独立运行。

## 全库数据审计

已完整扫描全部分片：64,183 行、64,183 个唯一论文 ID，无重复；
597 个问题对应的 574 个不同标准相关论文 ID 全部存在。
**6,197 篇标题和摘要都为空**，其中包含 5 篇标准相关论文，影响 5 个问题。
保留这些 ID，使用官方 `Title: ...\nAbstract: ...` 格式，不编造正文，
不使用参考答案补齐，不从评分分母删除相关问题。缺失文本会限制可达到的检索效果。
我们按 corpus ID 保留每篇论文；官方示例 build_index 使用文本作字典键，可能合并相同文本。
因此这是全 ID 语料上的项目检索协议，不声称逐项复现官方示例索引或榜单。

官方模板来源：
https://github.com/princeton-nlp/LitSearch/blob/main/utils/utils.py

审计结果及完整空文本 ID 清单：
`evaluation/runs/litsearch_corpus_audit_20261021.json`。
其中同时记录已有 BGE-M3 权重文件 SHA256；本地模型配置没有远程 revision，
因此后续复现应核对权重哈希。

最初 `litsearch_full_20261021` / `scholartrace-litsearch-index-20261021` 是发现空文本问题前的
未完成运行，已停止，不能作为完整索引使用。修复后新建 v2 运行，未篡改旧 manifest 或删除数据。

## 检查是否完成

`index_progress.json` 是每批成功写入并读回后的进度。
`index_complete.json` 出现且 papers=64183 才表示完整建库结束。
`progress.sqlite3` 保存每篇的内容摘要与 chunk ID，不能删除或混用。
当前文档是操作记录，不是“已经完成”的证明；以运行产物和容器退出码为准。

## 2026-10-21 中断与恢复

第一次 v2 建库在 23,296 / 64,183 篇停止。报错为 `text exceeds Milvus VARCHAR byte limit`。
完整数据检查显示仅 2 篇超过原始 16,384 字节限制，最长 18,473 字节；
没有论文超过 32,768 字节。评测专属集合的 `text` 上限已就地改为 32,768 字节，
向量仓库改为读取集合实际字段限制。之前已入库论文和向量均保留，代码哈希迁移
记录在 `evaluation/runs/litsearch_full_v2_20261021/text_limit_migration.json`。
迁移脚本是 `scripts/migrate_litsearch_text_limit.py`，具备运行目录、集合、
checkpoint 数量、数据哈希、问题 ID、语料最大字节数的前置检查。
已重新启动原容器继续建库。迁移后运行了评分、当前评测与 Milvus 检索回归：24 passed。

```powershell
docker logs --tail 40 scholartrace-litsearch-index-v2-20261021
docker inspect scholartrace-litsearch-index-v2-20261021 --format '{{json .State}}'
Get-Content evaluation/runs/litsearch_full_v2_20261021/index_progress.json
Get-Content evaluation/runs/litsearch_full_v2_20261021/index_complete.json
```

## 中断后的继续方式

先确保 Docker 和原来的 Milvus 三个服务已启动；确认没有同名运行任务。
原容器保留命令和环境变量，结束或机器重启后可用：

```powershell
docker start scholartrace_repro-milvus_etcd-1 scholartrace_repro-milvus_minio-1 scholartrace_repro-milvus-1
docker start scholartrace-litsearch-index-v2-20261021
```

入口会核对代码、数据、配置、已有向量后续建；任一不一致会停止，不能强改 manifest 绕过。
本任务运行期间不要改动 evaluation/rag_reader/ai_components 的 Python 代码。
容器未设自动重启策略，关机或 Docker 停止会中断，需要手动继续。

## 本次评分代码验证

`tests/test_current_scoring.py tests/test_current_evaluation.py tests/test_langchain_architecture.py`：
31 passed，包含本地下载的 QASPER 官方脚本对照、LitSearch 空字段处理；全部断网执行。
这证明评分实现与入口行为通过测试，不是模型在数据集上的质量成绩。
