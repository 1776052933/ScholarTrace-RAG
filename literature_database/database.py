"""SQLAlchemy 数据库引擎、Session 和建表管理。"""

import os
from typing import Any, Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


DEFAULT_DATABASE_URL = "sqlite:///literature.db"


class DatabaseManager:
    """初始化数据库并统一创建 SQLAlchemy Session。"""

    def __init__(
        self,
        database_url: Optional[str] = None,
        *,
        echo: bool = False,
        engine_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.database_url = (
            database_url
            or os.getenv("DATABASE_URL")
            or DEFAULT_DATABASE_URL
        ).strip()
        if not self.database_url:
            raise ValueError("database_url must not be empty")

        options: dict[str, Any] = {
            "echo": echo,
            "pool_pre_ping": True,
        }
        if self.database_url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}
        if engine_options:
            # 显式传入项可覆盖默认值，便于 PostgreSQL 设置连接池参数。
            options.update(engine_options)

        self.engine: Engine = create_engine(self.database_url, **options)
        if self.database_url.startswith("sqlite"):
            _enable_sqlite_foreign_keys(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    def get_session(self) -> Session:
        """创建独立 Session；调用方负责 commit/rollback 和 close。

        返回的 Session 支持上下文管理器：

        ``with database.get_session() as session: ...``
        """
        return self.session_factory()

    def create_tables(self) -> None:
        """创建当前 Base 注册的全部 ORM 数据表。"""
        Base.metadata.create_all(bind=self.engine)

    def initialize_database(self) -> None:
        """初始化数据库结构，作为 ``create_tables`` 的语义化别名。"""
        self.create_tables()

    def dispose(self) -> None:
        """释放 Engine 持有的连接池资源。"""
        self.engine.dispose()


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """SQLite 默认关闭外键约束，此处为每个新连接显式启用。"""

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
