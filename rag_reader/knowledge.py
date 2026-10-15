"""论文结构化知识的线程安全管理。"""

from collections.abc import Mapping
from threading import RLock
from typing import Any, Dict, List, Optional, Union

from .schema import PaperKnowledge


class KnowledgeAlreadyExistsError(ValueError):
    """指定论文知识已经存在。"""


class KnowledgeNotFoundError(LookupError):
    """指定论文知识不存在。"""


class KnowledgeManager:
    """保存、读取和更新 PaperKnowledge。

    当前使用内存字典。所有存储操作都集中在该类中，后续可以在不改变公开
    ``add/get/update`` 接口的前提下替换为知识图谱或持久化后端。
    """

    def __init__(self) -> None:
        self._knowledge: Dict[str, PaperKnowledge] = {}
        self._lock = RLock()

    def add(
        self,
        paper_id: Union[str, PaperKnowledge],
        knowledge: Optional[PaperKnowledge] = None,
    ) -> PaperKnowledge:
        """新增论文知识。

        支持 ``add("paper-id", knowledge)`` 和 ``add(knowledge)``；后一种
        写法使用论文标题作为存储键。
        """
        key, value = _resolve_add_arguments(paper_id, knowledge)
        with self._lock:
            if key in self._knowledge:
                raise KnowledgeAlreadyExistsError(f"knowledge already exists: {key}")
            self._knowledge[key] = value.model_copy(deep=True)
            return self._knowledge[key].model_copy(deep=True)

    def get(self, paper_id: str) -> Optional[PaperKnowledge]:
        """读取论文知识；不存在时返回 ``None``。"""
        key = _key(paper_id)
        with self._lock:
            value = self._knowledge.get(key)
            return value.model_copy(deep=True) if value is not None else None

    def update(
        self,
        paper_id: str,
        knowledge: Optional[Union[PaperKnowledge, Mapping[str, Any]]] = None,
        **changes: Any,
    ) -> PaperKnowledge:
        """整体替换或按字段更新论文知识。"""
        key = _key(paper_id)
        with self._lock:
            current = self._knowledge.get(key)
            if current is None:
                raise KnowledgeNotFoundError(f"knowledge not found: {key}")
            if isinstance(knowledge, PaperKnowledge):
                if changes:
                    raise ValueError("changes cannot be combined with a full PaperKnowledge")
                updated = knowledge
            else:
                patch = dict(knowledge or {})
                patch.update(changes)
                if not patch:
                    return current.model_copy(deep=True)
                unknown = set(patch) - set(PaperKnowledge.model_fields)
                if unknown:
                    raise ValueError("unknown PaperKnowledge fields: " + ", ".join(sorted(unknown)))
                data = current.model_dump(mode="python")
                data.update(patch)
                updated = PaperKnowledge.model_validate(data)
            self._knowledge[key] = updated.model_copy(deep=True)
            return self._knowledge[key].model_copy(deep=True)

    def list_all(self) -> List[PaperKnowledge]:
        """返回全部知识副本，供后续知识图谱导入使用。"""
        with self._lock:
            return [value.model_copy(deep=True) for value in self._knowledge.values()]

    def clear(self) -> None:
        """清空当前内存知识库。"""
        with self._lock:
            self._knowledge.clear()


def _resolve_add_arguments(
    paper_id: Union[str, PaperKnowledge],
    knowledge: Optional[PaperKnowledge],
) -> tuple[str, PaperKnowledge]:
    if isinstance(paper_id, PaperKnowledge):
        if knowledge is not None:
            raise TypeError("knowledge must be omitted when the first argument is PaperKnowledge")
        return _key(paper_id.title), paper_id
    if not isinstance(knowledge, PaperKnowledge):
        raise TypeError("knowledge must be a PaperKnowledge")
    return _key(paper_id), knowledge


def _key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("paper_id must be a string")
    result = value.strip()
    if not result:
        raise ValueError("paper_id must not be empty")
    return result
