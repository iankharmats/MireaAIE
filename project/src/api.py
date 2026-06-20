"""
api.py
FastAPI сервис генерации шаржей.
Все параметры конфигурации из get_config().
"""

from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime
from time import perf_counter
from typing import Any, Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response

from .config import get_config
from .feature_extraction import warp_image
from .caricature_generator import CaricatureGenerator
from .metrics import evaluate

cfg = get_config()

# Счётчики статистики сервера
total_req         = 0
latency_ct        = 0
total_latency_ms  = 0.0
last_caricature_metrics = {
    "timestamp":  None,
    "request_id": None,
    "metrics":    None,
}

app = FastAPI(
    title       = "AIE Caricature Generation Service API",
    version     = "0.2.0",
    description = (
        "HTTP-сервис генерации карикатур. "
        "Основывается на Stable Diffusion и оценке VLLM-критиком"
    ),
    docs_url  = "/docs",
    redoc_url = None,
)

# CORS из конфига
app.add_middleware(
    CORSMiddleware,
    allow_origins     = cfg.app.cors_origins_list,
    allow_methods     = cfg.app_yaml["cors"]["allow_methods"],
    allow_headers     = cfg.app_yaml["cors"]["allow_headers"],
    allow_credentials = True,
)

# Инициализируем генератор один раз при старте — все параметры из конфига
print("⏳ Загрузка весов SD 1.5, ControlNet и Qwen2-VL в видеопамять...")
generator = CaricatureGenerator()
print("🚀 Все модели успешно загружены и готовы к инференсу!")

LOG_DIR      = "logs"
LOG_FILENAME = "api.log"


class RequestLogger:
    """Логгер запросов в JSON-файл."""

    @staticmethod
    def log_request(
        endpoint:   str,
        status:     int,
        latency_ms: Optional[float] = None,
        request_id: Optional[str]   = None,
    ) -> None:
        log_entry = {
            "timestamp":  datetime.now().isoformat(),
            "endpoint":   endpoint,
            "status":     status,
            "latency_ms": latency_ms,
            "request_id": request_id or str(uuid.uuid4()),
        }
        log_entry     = {k: v for k, v in log_entry.items() if v is not None}
        full_log_path = os.path.join(LOG_DIR, LOG_FILENAME)
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(full_log_path):
            open(full_log_path, "w", encoding="utf-8").close()

        line = f"{json.dumps(log_entry, ensure_ascii=False)}\n"
        print(line)
        with open(full_log_path, "a", encoding="utf-8") as f:
            f.write(line)


@app.get("/health", tags=["system"], summary="Health-check сервиса")
def health() -> dict[str, str]:
    global total_req
    total_req += 1
    return {"status": "ok", "service": "caricature-generation", "version": "0.2.0"}


@app.get("/metrics", tags=["system"], summary="Статистика сервера и метрики последнего шаржа")
def metrics() -> dict[str, Any]:
    global total_latency_ms, latency_ct, total_req, last_caricature_metrics
    avg = total_latency_ms / latency_ct if latency_ct else None
    return {
        "total_req":      total_req,
        "avg_latency_ms": avg,
        "last_caricature_evaluation": {
            "timestamp":  last_caricature_metrics["timestamp"],
            "request_id": last_caricature_metrics["request_id"],
            "data":       last_caricature_metrics["metrics"],
        },
    }


@app.post(
    "/generate",
    tags=["main"],
    summary="Генерация карикатуры на основе RLAIF-пайплайна",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Лучший шарж (выбор Qwen2-VL) в формате PNG.",
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
        },
        400: {"description": "Загружен не графический файл"},
        500: {"description": "Внутренняя ошибка генерации"},
    },
)
async def caricature_generation(
    file: UploadFile = File(..., description="Изображение лица для генерации шаржа"),
    strength: float  = Form(
        default=cfg.warp_strength, ge=0.0, le=3.0,
        description="Сила утрирования черт",
    ),
):
    """
    Принимает фото, прогоняет через ControlNet+SD 1.5, оценивает Qwen2-VL,
    считает CLIP+Artifact метрики и возвращает лучший вариант шаржа.
    """
    global total_req, last_caricature_metrics, total_latency_ms, latency_ct
    total_req  += 1
    request_id  = str(uuid.uuid4())
    start       = perf_counter()

    if not file.content_type.startswith("image/"):
        RequestLogger.log_request("/generate", 400, request_id=request_id)
        raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

    temp_in  = f"temp_gen_{request_id}.jpg"
    temp_out = f"temp_out_{request_id}.png"

    try:
        contents = await file.read()
        with open(temp_in, "wb") as f:
            f.write(contents)

        result = generator.generate(temp_in, warp_strength=strength)

        print(f"\n[{request_id}] Ранжирование: {' > '.join(result['ranking'])}")
        print(f"[{request_id}] Вердикт:\n{result['judge_response']}\n")

        result["best"].save(temp_out, format="PNG")

        # FID/KID отключены — поштучный инференс, пороги артефактов из конфига
        metrics_score = evaluate(
            original          = temp_in,
            caricature        = temp_out,
            device            = cfg.app.metrics_device,
            compute_fid       = False,
            compute_kid       = False,
            compute_clip      = True,
            compute_artifacts = True,
        )

        last_caricature_metrics.update({
            "timestamp":  datetime.now().isoformat(),
            "request_id": request_id,
            "metrics":    metrics_score,
        })

        with open(temp_out, "rb") as img_f:
            io_buf = io.BytesIO(img_f.read())
        io_buf.seek(0)
        status_code = 200

    except Exception as e:
        RequestLogger.log_request("/generate", 500, request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Ошибка пайплайна генерации: {e}")

    finally:
        for p in (temp_in, temp_out):
            if os.path.exists(p):
                os.remove(p)

    latency_ms       = (perf_counter() - start) * 1000.0
    latency_ct      += 1
    total_latency_ms += latency_ms

    RequestLogger.log_request("/generate", status_code, latency_ms, request_id)
    return StreamingResponse(io_buf, media_type="image/png")


@app.post(
    "/warp",
    tags=["main"],
    summary="Warp-деформация загруженного изображения",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Деформированный PNG.",
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
        },
        400: {"description": "Не графический файл"},
        500: {"description": "Ошибка пайплайна или сбой MediaPipe"},
    },
)
async def warp(
    file: UploadFile = File(..., description="Изображение лица для варпинга"),
    strength: float  = Form(
        default=cfg.warp_strength, ge=0.0, le=3.0,
        description="Сила утрирования черт",
    ),
) -> Response:
    global total_req, total_latency_ms, latency_ct
    total_req  += 1
    request_id  = str(uuid.uuid4())
    start       = perf_counter()

    if not file.content_type.startswith("image/"):
        RequestLogger.log_request("/warp", 400, request_id=request_id)
        raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

    temp_in = f"temp_input_{request_id}.jpg"
    try:
        contents = await file.read()
        print(f"[/warp] прочитано байт: {len(contents)}")
        
        with open(temp_in, "wb") as f:
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())

        print(f"[/warp] файл записан: {temp_in}")
        print(f"[/warp] файл существует: {os.path.exists(temp_in)}")
        print(f"[/warp] размер на диске: {os.path.getsize(temp_in)} байт")

        warped_bgr = warp_image(temp_in, strength)
        _, enc      = cv2.imencode(".png", warped_bgr)
        io_buf      = io.BytesIO(enc.tobytes())
        status_code = 200

    except Exception as e:
        RequestLogger.log_request("/warp", 500, request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {e}")

    finally:
        if os.path.exists(temp_in):
            os.remove(temp_in)

    latency_ms       = (perf_counter() - start) * 1000.0
    latency_ct      += 1
    total_latency_ms += latency_ms

    RequestLogger.log_request("/warp", status_code, latency_ms, request_id)
    return StreamingResponse(io_buf, media_type="image/png")