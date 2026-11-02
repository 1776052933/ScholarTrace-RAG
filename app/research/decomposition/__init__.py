"""科研查询自动拆解模块的公共 API。"""

from .analyzer import ComplexityAnalyzer, QueryAnalyzer
from .decomposer import QueryDecomposer
from .llm_client import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekError,
    DeepSeekJSONError,
    DeepSeekNetworkError,
)
from .optimizer import SubQueryOptimizer
from .pipeline import QueryDecompositionPipeline
from .schema import (
    ComplexityLevel,
    ComplexityReport,
    DecompositionResult,
    QueryAnalysisInput,
    QueryUnderstandingResult,
    SubQuery,
)

__all__ = [
    "ComplexityAnalyzer",
    "ComplexityLevel",
    "ComplexityReport",
    "DeepSeekClient",
    "DeepSeekAPIError",
    "DeepSeekError",
    "DeepSeekJSONError",
    "DeepSeekNetworkError",
    "DecompositionResult",
    "QueryAnalysisInput",
    "QueryAnalyzer",
    "QueryUnderstandingResult",
    "QueryDecomposer",
    "QueryDecompositionPipeline",
    "SubQuery",
    "SubQueryOptimizer",
]
