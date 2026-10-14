# query_understanding

`query_understanding` 是 Auto_paper 的科研查询理解模块。它接收用户输入的自然语言科研问题，将其解析、拆解并改写为可以用于论文检索的结构化查询结果。

## 1. 模块功能

模块按照以下顺序处理科研查询：

```text
用户自然语言查询
        ↓
科研意图与约束解析
        ↓
复杂问题判断与拆解
        ↓
论文检索式改写和关键词扩展
        ↓
QueryAnalysis
```

主要能力包括：

- 识别用户的研究意图；
- 抽取方法、模型、任务、领域、数据集和应用实体；
- 识别方法、领域、数据集及时间限制；
- 判断查询是否包含多个方法、任务或研究对象；
- 将复杂问题拆解为可以独立检索的子查询；
- 生成中文关键词、标准英文术语、缩写和同义词；
- 生成适用于 Semantic Scholar、arXiv 和 OpenAlex 的扩展检索式；
- 使用 Pydantic 校验结果，并支持 JSON 序列化。

## 2. 文件说明

```text
query_understanding/
├── __init__.py
├── config.py
├── schema.py
├── llm_client.py
├── prompts.py
├── query_parser.py
├── query_decomposer.py
├── query_rewriter.py
├── pipeline.py
├── examples.py
└── README.md
```

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 导出模块的公共类和数据模型。 |
| `config.py` | 保存查询理解相关的默认配置。 |
| `schema.py` | 定义 `Entity`、`TimeConstraint` 和 `QueryAnalysis` 等 Pydantic 数据结构。 |
| `llm_client.py` | 使用 OpenAI Python SDK 调用 DeepSeek OpenAI-compatible API。 |
| `prompts.py` | 定义查询分析、复杂问题拆解和检索式改写 Prompt。 |
| `query_parser.py` | 调用 DeepSeek 解析研究意图、实体和查询约束。 |
| `query_decomposer.py` | 判断查询复杂度，并更新 `sub_queries`。 |
| `query_rewriter.py` | 生成中英文关键词、同义词和 `expanded_queries`。 |
| `pipeline.py` | 按顺序编排 Parser、Decomposer 和 Rewriter，是推荐调用入口。 |
| `examples.py` | 使用真实 DeepSeek API 运行三个完整测试案例并输出 JSON。 |
| `README.md` | 模块配置与使用说明。 |

## 3. DeepSeek 配置方法

### 安装依赖

```bash
pip install openai pydantic
```

### 配置环境变量

PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

请勿将真实 API Key 写入源码或提交到版本控制。

其中：

- `DEEPSEEK_API_KEY`：DeepSeek API Key，必须配置；
- `DEEPSEEK_BASE_URL`：可选，默认值为 `https://api.deepseek.com`；
- `DEEPSEEK_MODEL`：可选，默认值为 `deepseek-chat`。

`DeepSeekClient` 也会读取项目现有密钥管理器中的 `deepseek` 配置。显式传入构造参数时，构造参数优先级最高。

## 4. 调用方式

推荐通过 `QueryUnderstandingPipeline` 使用完整查询理解流程：

```python
from query_understanding.pipeline import QueryUnderstandingPipeline


pipeline = QueryUnderstandingPipeline()

result = pipeline.run(
    "搜索近五年LLM用于蛋白质预测的论文"
)

print(result.to_json(indent=2))
```

`run()` 返回 `QueryAnalysis` 对象。可以直接访问结构化字段：

```python
print(result.research_intent)
print(result.entities)
print(result.sub_queries)
print(result.expanded_queries)
print(result.keywords)
```

可以通过以下方法序列化：

```python
# JSON 兼容字典
result_dict = result.to_dict()

# JSON 字符串
result_json = result.to_json(indent=2)
```

不使用 DeepSeek 的离线规则模式：

```python
pipeline = QueryUnderstandingPipeline(use_llm=False)
result = pipeline.run("近五年Transformer医学图像分割方法")
```

## 5. 输出 JSON 格式

完整结果使用以下结构：

```json
{
  "original_query": "用户原始科研查询",
  "research_intent": "识别出的研究目标",
  "entities": [
    {
      "name": "实体名称",
      "type": "method",
      "importance": "high"
    }
  ],
  "method_constraints": [],
  "domain_constraints": [],
  "dataset_constraints": [],
  "task_type": null,
  "time_constraints": {
    "start_year": null,
    "end_year": null,
    "description": null
  },
  "sub_queries": [],
  "expanded_queries": [],
  "keywords": []
}
```

实体 `type` 的可选值：

- `method`
- `task`
- `domain`
- `dataset`
- `model`
- `application`

实体 `importance` 的可选值：

- `high`
- `medium`
- `low`

## 最终调用效果

其他调用方只需要：

```python
from query_understanding.pipeline import QueryUnderstandingPipeline


pipeline = QueryUnderstandingPipeline()

result = pipeline.run(
    "近五年基于大语言模型的药物发现研究"
)

print(result)
```

需要输出标准 JSON 时使用：

```python
print(result.to_json(indent=2))
```

返回结果示例：

```json
{
  "original_query": "近五年基于大语言模型的药物发现研究",
  "research_intent": "研究LLM在药物发现领域中的应用",
  "entities": [
    {
      "name": "Large Language Model",
      "type": "method",
      "importance": "high"
    },
    {
      "name": "Drug Discovery",
      "type": "domain",
      "importance": "high"
    }
  ],
  "method_constraints": [
    "LLM"
  ],
  "domain_constraints": [
    "Drug Discovery"
  ],
  "dataset_constraints": [],
  "task_type": "drug_discovery",
  "time_constraints": {
    "start_year": 2021,
    "end_year": null,
    "description": "近五年"
  },
  "sub_queries": [],
  "expanded_queries": [
    "large language model drug discovery",
    "foundation model molecular discovery"
  ],
  "keywords": [
    "LLM drug discovery",
    "foundation model molecule"
  ]
}
```

具体实体名称、时间范围、子查询和关键词由 DeepSeek 根据实际输入生成，因此可能与示例文字略有不同，但返回字段和 JSON 结构保持一致。
