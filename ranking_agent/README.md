# Ranking Agent

`ranking_agent` 是 Auto_paper 的论文精排模块。它接收 Search Agent 返回的候选论文，根据用户研究目标对论文进行去重、多维评分和重新排序，最终输出相关性更高的论文列表。

## 1. Ranking Agent 功能

Ranking Agent 提供以下能力：

- 根据论文 ID 和规范化标题删除重复论文；
- 使用 `BAAI/bge-m3` 生成中英文查询及论文向量；
- 使用余弦相似度计算查询与论文的语义相关性；
- 识别论文方法与用户方法约束的匹配程度；
- 识别研究任务、实验内容及应用领域的匹配程度；
- 根据年份、引用次数、摘要完整度和 venue 评价论文质量；
- 先使用 BGE-M3、约束规则、质量和时效完成全量粗排；
- 仅使用 DeepSeek 评价粗排前 30～50 篇高潜力候选；
- 计算综合分数并划分论文相关等级；
- 按本次全部去重论文的得分分布，将前 30% 自适应标为高度相关；
- 根据综合分数、引用次数和发表年份完成精排。

处理流程：

```text
查询上下文 + Search Agent 候选论文
                ↓
             论文去重
                ↓
          查询与论文向量编码
                ↓
     语义 / 方法 / 任务 / 质量评分
                ↓
       选取前 30～50 篇高潜力候选
                ↓
           DeepSeek 辅助评价
                ↓
             综合分数计算
                ↓
       相关等级划分与论文重排序
                ↓
            RankingResult
```

## 2. 输入来源：Search Agent

Ranking Agent 的论文候选集合来自 `search_agent`。它可以直接接收：

- Search Agent 返回的 `SearchAgentResult`；
- `SearchAgentResult.papers`；
- `Paper` 对象列表；
- 论文字典列表；
- 包含 `papers` 字段的 JSON 对象。

典型模块关系：

```text
Query Understanding
        ↓
Query Decomposition
        ↓
Search Agent
        ↓
Ranking Agent
```

查询上下文需要包含以下主要字段：

```python
query = {
    "original_query": "近五年Transformer医学图像分割方法",
    "research_intent": "研究基于Transformer的医学图像分割方法",
    "entities": [
        {"name": "Transformer", "type": "method"},
        {"name": "Medical Image Segmentation", "type": "task"},
    ],
    "method_constraints": ["Transformer"],
    "domain_constraints": ["医学图像"],
    "task_type": "segmentation",
    "keywords": ["Transformer", "medical image segmentation"],
}
```

## 3. 输出：排序后的论文

`RankingAgent.run()` 返回 `RankingResult`：

```json
{
  "papers": [],
  "total": 0,
  "ranking_time": 0.0,
  "scores": {},
  "graph_points": [],
  "llm_evaluated_count": 0,
  "highly_relevant_count": 0,
  "adaptive_threshold": 0.0
}
```

字段说明：

- `papers`：去重并按综合分数降序排列的论文列表；
- `total`：最终论文数量；
- `ranking_time`：本次精排耗时，单位为秒。

每篇论文的详细评分保存在 Agent 的 `last_scores` 中：

```python
for paper in result.papers:
    score = agent.last_scores[paper.paper_id]
    print(paper.title, score.final_score, score.relevance_level)
```

Pipeline 会根据本次完整候选集自适应划分相关等级：

- `highly_relevant`：本次综合得分排名前 30% 的论文；
- `partially_relevant`：其余论文。

`RankingScorer` 中的固定阈值仅作为单篇评分的默认值，完整 Pipeline 会在所有论文评分完成后执行自适应分级。

## 4. 评分公式

每篇论文的最终分数均在 `0-1` 范围内：

```text
Final Score =
    semantic_score × 0.35
  + method_score   × 0.20
  + task_score     × 0.15
  + quality_score  × 0.15
  + time_score     × 0.05
  + llm_score      × 0.10
```

各维度含义：

| 评分 | 权重 | 说明 |
| --- | ---: | --- |
| `semantic_score` | 0.35 | 查询向量与论文标题、摘要向量的余弦相似度 |
| `method_score` | 0.20 | 论文方法与用户方法约束的匹配程度 |
| `task_score` | 0.15 | 论文任务、实验和应用领域的匹配程度 |
| `quality_score` | 0.15 | 引用质量、venue 和摘要完整度的综合评价 |
| `time_score` | 0.05 | 论文年份与当前年份的接近程度 |
| `llm_score` | 0.10 | DeepSeek 对论文整体相关性的评价 |

权重和等级阈值定义在 `config.py`，可以通过 `RankingConfig` 自定义。

## 5. 调用示例

接收 Search Agent 的完整输出：

```python
from ranking_agent.pipeline import RankingAgent

agent = RankingAgent()

result = agent.run(
    query,
    search_result,
)

for paper in result.papers:
    score = agent.last_scores[paper.paper_id]
    print(
        paper.title,
        score.final_score,
        score.relevance_level,
    )
```

如果已经取得 Search Agent 的论文列表，也可以直接调用：

```python
from ranking_agent.pipeline import RankingAgent

agent = RankingAgent()

result = agent.run(
    query,
    papers,
)
```

DeepSeek 配置：

```powershell
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

API Key 也可以由项目的 `API_key/key_manager.py` 提供。请勿将真实密钥写入源码。

安装向量模型依赖：

```bash
pip install sentence-transformers openai
```

运行模块示例：

```powershell
python -m ranking_agent.examples
```
