"""REST API routes for vessel description prototypes and image recognition."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from web.models import (
    ApiResponse,
    SearchResponse,
    ShipBulkCreate,
    ShipCreate,
    ShipItem,
    ShipListResponse,
    ShipUpdate,
    StatsResponse,
)
from web.services import ShipService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ships", tags=["ships"])
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/gif"}
MAX_FILE_SIZE = 20 * 1024 * 1024


def get_service(request: Request) -> ShipService:
    return request.app.state.ship_service


@router.get("", response_model=ShipListResponse)
async def list_ships(svc: Annotated[ShipService, Depends(get_service)]):
    ships = svc.list_ships()
    return ShipListResponse(total=len(ships), ships=[ShipItem(**item) for item in ships])


@router.get("/search", response_model=SearchResponse)
async def search_ships(
    q: Annotated[str, Query(description="搜索关键词")],
    svc: Annotated[ShipService, Depends(get_service)],
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="搜索关键词不能为空")
    results = svc.search(q)
    return SearchResponse(total=len(results), results=[ShipItem(**item) for item in results])


@router.get("/stats", response_model=StatsResponse)
async def stats(svc: Annotated[ShipService, Depends(get_service)]):
    return StatsResponse(**svc.stats())


@router.post("", response_model=ApiResponse)
async def create_ship(body: ShipCreate, svc: Annotated[ShipService, Depends(get_service)]):
    record = svc.create_ship(body.hull_number, body.description)
    if record is None:
        raise HTTPException(status_code=400, detail="舷号和结构描述不能为空，且当前数据源必须支持多原型")
    return ApiResponse(success=True, message=f"已为舷号 {body.hull_number} 添加原型 {record['prototype_id']}", data=record)


@router.post("/bulk", response_model=ApiResponse)
async def bulk_create(body: ShipBulkCreate, svc: Annotated[ShipService, Depends(get_service)]):
    result = svc.bulk_create(body.ships)
    return ApiResponse(success=True, message=f"成功添加 {result['added']} 条原型", data=result)


def _validate_upload(file: UploadFile, contents: bytes) -> None:
    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件过大，请上传 20MB 以内的图片")


@router.post("/recognize", response_model=ApiResponse)
async def recognize_ship(file: UploadFile = File(...), svc: ShipService = Depends(get_service)):
    contents = await file.read()
    _validate_upload(file, contents)
    try:
        result = svc.recognize_ship(contents, file.filename or "upload.jpg")
    except Exception as exc:
        logger.error("VLM 识别失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"识别失败: {exc}") from exc
    return ApiResponse(success=True, message="识别成功", data=result)


@router.post("/recognize-and-add", response_model=ApiResponse)
async def recognize_and_add(file: UploadFile = File(...), svc: ShipService = Depends(get_service)):
    contents = await file.read()
    _validate_upload(file, contents)
    try:
        result = svc.recognize_and_add(contents, file.filename or "upload.jpg")
    except Exception as exc:
        logger.error("VLM 识别失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"识别失败: {exc}") from exc
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return ApiResponse(success=True, message="识别结果已追加为新原型", data=result)


@router.get("/prototypes/{prototype_id}", response_model=ShipItem)
async def get_prototype(prototype_id: str, svc: Annotated[ShipService, Depends(get_service)]):
    record = svc.get_prototype(prototype_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"未找到原型: {prototype_id}")
    return ShipItem(**record)


@router.put("/prototypes/{prototype_id}", response_model=ApiResponse)
async def update_prototype(
    prototype_id: str,
    body: ShipUpdate,
    svc: Annotated[ShipService, Depends(get_service)],
):
    if not svc.update_prototype(prototype_id, body.description):
        raise HTTPException(status_code=404, detail=f"未找到原型: {prototype_id}")
    return ApiResponse(success=True, message=f"已更新原型: {prototype_id}")


@router.delete("/prototypes/{prototype_id}", response_model=ApiResponse)
async def delete_prototype(prototype_id: str, svc: Annotated[ShipService, Depends(get_service)]):
    if not svc.delete_prototype(prototype_id):
        raise HTTPException(status_code=404, detail=f"未找到原型: {prototype_id}")
    return ApiResponse(success=True, message=f"已删除原型: {prototype_id}")


@router.get("/{hull_number}", response_model=ShipItem)
async def get_ship(hull_number: str, svc: Annotated[ShipService, Depends(get_service)]):
    ship = svc.get_ship(hull_number)
    if ship is None:
        raise HTTPException(status_code=404, detail=f"未找到舷号: {hull_number}")
    return ShipItem(**ship)


@router.delete("/{hull_number}", response_model=ApiResponse)
async def delete_ship(hull_number: str, svc: Annotated[ShipService, Depends(get_service)]):
    if not svc.delete_ship(hull_number):
        raise HTTPException(status_code=404, detail=f"未找到舷号: {hull_number}")
    return ApiResponse(success=True, message=f"已删除舷号 {hull_number} 的全部原型")
