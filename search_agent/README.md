# Search Agent

`search_agent` 是 Auto_paper 的自主论文搜索模块。它接收 `query_decomposition` 产生的科研子查询，自动生成搜索计划，并调用多个论文数据库进行检索。

模块会评估论文的相关性、质量和研究问题覆盖度。当结果覆盖不足时，Agent 会分析缺失信息、生成新的搜索方向并进入下一轮搜索。默认最多执行 3 轮，最终返回跨轮合并、去重后的论文集合及完整运行状态。

## 1. 模块作用

Search Agent 提供以下能力：

- 根据拆解后的科研问题生成论文搜索计划；
- 为不同方法、任务、数据集和评价方向生成检索 Query；
- 并发调用 Google Scholar、arXiv 和 OpenAlex；
- 根据 DOI 或规范化标题合并重复论文；
- 评价论文相关性、论文质量和问题覆盖度；
- 根据缺失信息反思并生成下一轮 `SearchPlan`；
- 保存已搜索 Query、搜索轮次、论文、失败关键词和新增关键词；
- 达到停止条件或最多运行 3 轮后返回最终结果。

内部流程如下：

```text
SearchPlanner
    ↓
PaperSearcher
    ↓
SearchEvaluator
    ↓
SearchReflector（需要继续搜索时生成新计划）
    ↓
SearchMemory（保存并合并多轮状态）
```

## 2. 与其他模块的关系

Auto_paper 的科研查询处理流程为：

```text
Query Understanding
        ↓
Query Decomposition
        ↓
Search Agent
```

- `query_understanding`：理解用户自然语言科研查询，识别研究意图、实体、约束和关键词。
- `query_decomposition`：判断查询复杂度，并生成能够独立检索的子查询。
- `search_agent`：根据子查询制定检索计划、搜索论文、评价结果并执行多轮优化。

`search_agent` 不负责重新理解用户原始问题，而是直接消费查询拆解结果。

## 3. DeepSeek 配置

Planner、Evaluator 和 Reflector 可以使用 DeepSeek。模块通过 OpenAI 兼容接口调用 DeepSeek，不会在代码中保存 API Key。

PowerShell 配置示例：

```powershell
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
$env:SEARCH_AGENT_USE_LLM="true"
```

配置项说明：

| 环境变量 | 是否必须 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | 启用 LLM 时必须 | 无 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | 否 | `https://api.deepseek.com` | OpenAI 兼容接口地址 |
| `DEEPSEEK_MODEL` | 否 | `deepseek-chat` | 使用的模型 |
| `SEARCH_AGENT_USE_LLM` | 否 | `false` | 是否启用 LLM 规划、评价和反思 |

没有启用 LLM 时，Search Agent 使用内置规则完成规划、评价和反思，论文数据库检索仍可正常运行。

## 4. 论文 API 配置

Search Agent 默认搜索以下数据源：

| 数据源 | 配置 | 说明 |
| --- | --- | --- |
| Google Scholar | `API_key/API_key.json` 中的 `serpapi`，或 `SERPAPI_API_KEY` | 默认通过 SerpAPI Google Scholar 引擎调用 |
| arXiv | 无需 Key | 直接调用公开 Atom API |
| OpenAlex | `OPENALEX_MAILTO` | 可选；建议填写联系邮箱 |

PowerShell 配置示例：

```powershell
$env:SERPAPI_API_KEY="your-serpapi-key"
$env:OPENALEX_MAILTO="your-email@example.com"
```

还可以调整搜索行为：

```powershell
$env:SEARCH_AGENT_MAX_ROUNDS="3"
$env:SEARCH_AGENT_RESULTS_PER_TASK="10"
$env:SEARCH_AGENT_TARGET_PAPERS="15"
$env:SEARCH_AGENT_MIN_RELEVANT="5"
$env:SEARCH_AGENT_RELEVANCE_THRESHOLD="0.25"
$env:SEARCH_AGENT_QUALITY_THRESHOLD="0.72"
$env:SEARCH_AGENT_TIMEOUT="20"
```

任一论文 API 请求失败时，该数据源返回空结果，不会中断其他数据源或整个 Agent。

## 5. 调用示例

直接传入 `query_decomposition` 的输出：

```python
from search_agent.pipeline import SearchAgent

agent = SearchAgent(max_round=3)
search_result = agent.run(result)

print(search_result.papers)
print(search_result.final_evaluation)
print(search_result.stopped_reason)
```

其中 `result` 是查询拆解模块的返回对象，例如：

```python
result = {
    "original_query": "比较CNN和Transformer在目标检测中的性能",
    "research_intent": "比较CNN和Transformer目标检测性能",
    "need_decompose": True,
    "reason": "包含多个方法和比较关系",
    "sub_queries": [
        {
            "query": "CNN目标检测",
            "purpose": "检索CNN目标检测方法",
            "query_type": "method_search",
            "priority": "high",
            "english_query": "CNN object detection methods performance",
            "keywords": ["CNN", "object detection", "performance"],
        },
        {
            "query": "Transformer目标检测",
            "purpose": "检索Transformer目标检测方法",
            "query_type": "method_search",
            "priority": "high",
            "english_query": "Transformer object detection methods performance",
            "keywords": ["Transformer", "DETR", "object detection"],
        },
        {
            "query": "CNN和Transformer目标检测性能比较",
            "purpose": "比较两类方法的性能",
            "query_type": "comparison",
            "priority": "high",
            "english_query": "CNN versus Transformer object detection benchmark",
            "keywords": ["CNN", "Transformer", "benchmark"],
        },
    ],
}

agent = SearchAgent()
search_result = agent.run(result)
```

完整示例：

```powershell
python -m search_agent.examples
```

`run()` 返回 `SearchAgentResult`，主要字段包括：

- `papers`：最终去重论文集合；
- `rounds_completed`：实际搜索轮数；
- `stopped_reason`：停止搜索的原因；
- `iterations`：每轮搜索计划、论文增量、评价和反思记录；
- `final_evaluation`：最终相关性、质量、覆盖度和缺失信息评价。
