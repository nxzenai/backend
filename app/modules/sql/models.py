from pydantic import BaseModel


class SQLResult(BaseModel):

    columns: list[str]

    rows: list[list]


class TableSchema(BaseModel):

    name: str

    columns: list[str]