"""论文全文下载任务状态追踪，不包含具体下载实现。"""

from enum import Enum
from typing import Union

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import DownloadTaskModel


class DownloadStatus(str, Enum):
    """下载任务状态。"""

    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class DownloadTaskNotFoundError(LookupError):
    """指定下载任务不存在。"""


TaskIdentifier = Union[int, str]


class DownloadTracker:
    """创建和更新 DownloadTaskModel，不执行论文下载。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_task(self, paper_id: str) -> DownloadTaskModel:
        """为论文创建一个待下载任务。"""
        paper_id = paper_id.strip()
        if not paper_id:
            raise ValueError("paper_id must not be empty")
        task = DownloadTaskModel(
            paper_id=paper_id,
            status=DownloadStatus.PENDING.value,
            retry_count=0,
            error=None,
        )
        self.session.add(task)
        self.session.flush()
        return task

    def update_status(
        self,
        task: TaskIdentifier,
        status: Union[DownloadStatus, str],
    ) -> DownloadTaskModel:
        """更新任务状态；任务参数可传任务 ID 或 paper_id。"""
        model = self._require_task(task)
        target = DownloadStatus(status)
        model.status = target.value
        if target is DownloadStatus.SUCCESS:
            model.error = None
        self.session.flush()
        return model

    def fail_task(self, task: TaskIdentifier, error: str) -> DownloadTaskModel:
        """将任务标记为失败并记录失败原因。"""
        reason = error.strip()
        if not reason:
            raise ValueError("error must not be empty")
        model = self._require_task(task)
        model.status = DownloadStatus.FAILED.value
        model.error = reason
        self.session.flush()
        return model

    def retry_task(self, task: TaskIdentifier) -> DownloadTaskModel:
        """重新排队失败任务，重试次数加一并清除旧错误。"""
        model = self._require_task(task)
        if model.status != DownloadStatus.FAILED.value:
            raise ValueError("only FAILED tasks can be retried")
        model.retry_count += 1
        model.status = DownloadStatus.PENDING.value
        model.error = None
        self.session.flush()
        return model

    def _require_task(self, task: TaskIdentifier) -> DownloadTaskModel:
        model = self._find_task(task)
        if model is None:
            raise DownloadTaskNotFoundError(f"download task not found: {task}")
        return model

    def _find_task(self, task: TaskIdentifier) -> DownloadTaskModel | None:
        if isinstance(task, bool):
            raise TypeError("task must be an integer task ID or string paper_id")
        if isinstance(task, int):
            return self.session.get(DownloadTaskModel, task)
        if not isinstance(task, str):
            raise TypeError("task must be an integer task ID or string paper_id")
        paper_id = task.strip()
        if not paper_id:
            return None
        statement = (
            select(DownloadTaskModel)
            .where(DownloadTaskModel.paper_id == paper_id)
            .order_by(DownloadTaskModel.id.desc())
            .limit(1)
        )
        return self.session.scalar(statement)
