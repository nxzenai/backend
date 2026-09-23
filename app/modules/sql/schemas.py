from pydantic import BaseModel

from app.modules.sql.models import (
    SQLResult,
    TableSchema,
)


class SQLExecuteRequest(BaseModel):

    query: str


class SQLExecuteResponse(SQLResult):

    execution_time: float
    message: str | None = None
    database_changed: bool = False


class SchemaResponse(BaseModel):

    tables: list[TableSchema]
    active_database: str
    databases: list[str]
