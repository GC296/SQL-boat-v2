"""Pydantic request and response models for vessel prototypes."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ShipCreate(BaseModel):
    hull_number: str = Field(..., min_length=1, max_length=50, description="舷号")
    description: str = Field(..., min_length=1, max_length=4000, description="当前图片可见的身份结构描述")


class ShipUpdate(BaseModel):
    description: str = Field(..., min_length=1, max_length=4000, description="原型结构描述")


class ShipBulkCreate(BaseModel):
    ships: dict[str, str] = Field(..., description="批量数据 {hull_number: description}")


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Any = None


class ShipItem(BaseModel):
    prototype_id: str
    hull_number: str
    description: str


class ShipListResponse(BaseModel):
    total: int
    ships: list[ShipItem]


class StatsResponse(BaseModel):
    total_ships: int
    total_prototypes: int
    backend: str


class SearchResponse(BaseModel):
    total: int
    results: list[ShipItem]


class RecognizeData(BaseModel):
    hull_number: str
    description: str
    already_exists: bool = False
    existing_prototype_count: int = 0
