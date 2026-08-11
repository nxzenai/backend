from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PaginationData(BaseModel, Generic[T]):
    items: list[T]
    page: int
    page_size: int
    total: int
    total_pages: int


class PaginatedResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str
    data: PaginationData[T]