from pydantic import BaseModel

from app.modules.sql.models import (
    SQLResult,
    TableSchema,
)


class SQLExecuteRequest(BaseModel):

    query: str


class SQLExecuteResponse(SQLResult):

    execution_time: float


class SchemaResponse(BaseModel):

    tables: list[TableSchema]