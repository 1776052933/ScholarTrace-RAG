# query_decomposition

`query_decomposition` 是科研论文自动检索流程中的查询拆解模块。它接收已经完成理解和结构化的科研查询，判断查询是否复杂，并生成可以独立提交给论文数据库的子查询。

## 1. 模块作用

模块提供以下能力：

- 判断科研查询是否需要拆解；
- 识别多方法、多任务、多研究对象和比较关系；
- 识别综述、评价、应用及发展趋势等研究维度；
- 将复杂科研问题拆解为多个互补的检索子问题；
- 为每个子查询生成用途说明、查询类型和执行优先级；
- 优化子查询的英文检索表达；
- 生成适用于 Semantic Scholar、arXiv、OpenAlex 等论文数据源的关键词；
- 使用 Pydantic 校验输入和输出，并支持 JSON 序列化。

模块执行流程：

```text
查询理解 JSON
      ↓
QueryAnalyzer
复杂度判断
      ↓
QueryDecomposer
生成独立子查询
      ↓
SubQueryOptimizer
优化英文检索式和关键词
      ↓
DecompositionResult
```

## 2. 输入格式

Pipeline 接收普通 Python 字典或 `QueryUnderstandingResult` 对象。

标准输入 JSON：

```json
{
  "original_query": "比较CNN和Transformer在目标检测中的性能",
  "research_intent": "比较CNN与Transformer目标检测方法的性能",
  "entities": [
    {
      "name": "CNN",
      "type": "model",
      "importance": "high"
    },
    {
      "name": "Transformer",
      "type": "model",
      "importance": "high"
    },
    {
      "name": "目标检测",
      "type": "task",
      "importance": "high"
    }
  ],
  "method_constraints": [
    "CNN",
    "Transformer"
  ],
  "domain_constraints": [
    "计算机视觉"
  ],
  "dataset_constraints": [],
  "task_type": "object_detection",
  "keywords": [
    "CNN",
    "Transformer",
    "Object Detection"
  ],
  "expanded_queries": [
    "CNN versus Transformer object detection performance"
  ]
}
```

输入字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `original_query` | `str` | 用户原始科研查询。 |
| `research_intent` | `str` | 查询理解阶段识别出的研究目标。 |
| `entities` | `list` | 方法、模型、任务、领域、数据集或应用实体。 |
| `method_constraints` | `list[str]` | 查询明确限定的方法或模型。 |
| `domain_constraints` | `list[str]` | 学科、行业或应用领域限制。 |
| `dataset_constraints` | `list[str]` | 明确指定的数据集或 benchmark。 |
| `task_type` | `str` | 标准化科研任务类型。 |
| `keywords` | `list[str]` | 查询理解阶段生成的检索关键词。 |
| `expanded_queries` | `list[str]` | 查询理解阶段生成的扩展检索式。 |

输入模型允许携带 `time_constraints` 等额外查询理解字段，方便直接传递完整 JSON。

## 3. 输出格式

`run()` 返回经过 Pydantic 校验的 `DecompositionResult`：

```json
{
  "original_query": "比较CNN和Transformer在目标检测中的性能",
  "need_decompose": true,
  "reason": "查询包含复杂研究动作：比较",
  "sub_queries": [
    {
      "query": "CNN 目标检测 models methods performance",
      "purpose": "分析 CNN 模型、方法及其性能表现",
      "query_type": "method_search",
      "priority": "high",
      "english_query": "CNN object detection models methods performance survey",
      "keywords": [
        "CNN",
        "object detection",
        "methods",
        "performance"
      ]
    },
    {
      "query": "Transformer 目标检测 models methods performance",
      "purpose": "分析 Transformer 模型、方法及其性能表现",
      "query_type": "method_search",
      "priority": "high",
      "english_query": "Transformer based object detection models methods performance survey",
      "keywords": [
        "Transformer",
        "object detection",
        "methods",
        "performance"
      ]
    },
    {
      "query": "CNN vs Transformer 目标检测 benchmark performance comparison",
      "purpose": "比较不同方法的性能与适用条件",
      "query_type": "comparison",
      "priority": "high",
      "english_query": "CNN vs Transformer object detection benchmark performance comparison",
      "keywords": [
        "CNN",
        "Transformer",
        "object detection",
        "benchmark",
        "performance comparison"
      ]
    }
  ]
}
```

`query_type` 可选值：

| 值 | 含义 |
| --- | --- |
| `method_search` | 方法或模型检索。 |
| `comparison` | 方法比较。 |
| `application` | 应用分析。 |
| `dataset_search` | 数据集或 benchmark 分析。 |
| `evaluation` | 指标和性能评价。 |
| `trend_analysis` | 发展过程和趋势分析。 |

`priority` 可选值：

- `high`：核心问题或必要的前置检索；
- `medium`：重要的补充分析；
- `low`：非关键扩展检索。

输出支持以下序列化方法：

```python
result.to_dict()
result.to_json(indent=2)
result.model_dump()
result.model_dump_json(indent=2)
```

## 4. 调用方法

### DeepSeek 模式

运行前配置：

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

调用 Pipeline：

```python
from query_decomposition.pipeline import QueryDecompositionPipeline


pipeline = QueryDecompositionPipeline()

result = pipeline.run(query_analysis_json)

print(result.to_json(indent=2))
```

### 离线规则模式

```python
from query_decomposition.pipeline import QueryDecompositionPipeline


pipeline = QueryDecompositionPipeline(use_llm=False)
result = pipeline.run(query_analysis_json)

print(result.to_json(indent=2))
```

### 访问子查询

```python
for sub_query in result.sub_queries:
    print(sub_query.query)
    print(sub_query.english_query)
    print(sub_query.keywords)
    print(sub_query.purpose)
    print(sub_query.query_type)
    print(sub_query.priority)
```

## 5. 与 query_understanding 的关系

三个模块在论文检索流程中的职责不同：

```text
用户自然语言科研问题
          ↓
query_understanding
理解用户意图，抽取实体、约束、关键词和扩展查询
          ↓
query_decomposition
判断复杂度，生成并优化可独立检索的子查询
          ↓
search_agent
根据英文检索式和关键词执行论文检索
```

### query_understanding

负责“理解”：

- 解析用户自然语言问题；
- 识别研究意图；
- 抽取方法、任务、领域、数据集和应用实体；
- 识别查询约束；
- 生成初始关键词和扩展查询。

### query_decomposition

负责“拆解”：

- 接收查询理解 JSON；
- 判断查询是否需要拆解；
- 生成方法、应用、比较、评价、数据集和趋势子查询；
- 优化英文检索式和搜索关键词；
- 输出结构化 `DecompositionResult`。

### search_agent

负责“检索”：

- 读取 `sub_queries`；
- 根据 `priority` 安排检索顺序；
- 使用 `english_query` 和 `keywords` 查询论文数据源；
- 返回与各子查询对应的论文结果。

模块间通过 JSON 数据传递，不需要共享内部实现。
