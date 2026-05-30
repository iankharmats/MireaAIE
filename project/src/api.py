from __future__ import annotations

from time import perf_counter
import io
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime
import uuid
import json
import os

from .feature_extraction import warp_image
from caricature_generator import CaricatureGenerator

# --------------- Поля сбора статистики --------------
total_req = 0
latency_ct = 0
total_latency_ms = 0
last_ok_for_model = {"score": 0.0, "time": None}

# --------------- FastAPI ---------------

app = FastAPI(
    title="AIE Caricature Generation Service API",
    version="0.2.0",
    description=(
        "HTTP-сервис генерации карикатур. "
        "Основывается на Stable Diffusion и оценке VLLM-критиком"
    ),
    docs_url="/docs",
    redoc_url=None,
)

# --------------- Предварительная конфигурация --------------
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)
MEAN_FACE_PATH = os.path.join(PROJECT_ROOT, "configs", "mean_face.json")

# Инициализируем генератор ОДИН раз при старте сервера
print("⏳ Загрузка весов SD 1.5, ControlNet и Qwen2-VL в видеопамять...")
generator = CaricatureGenerator(
    checkpoint_dir = os.path.join(PROJECT_ROOT, "notebooks/dpo_checkpoints/cycle_01"),
    mean_face_path = MEAN_FACE_PATH,
    judge_device   = "cuda:3",  # Судья на отдельной карте
)
print("🚀 Все модели успешно загружены и готовы к инференсу!")

# --------------- Логирование ---------------

LOG_DIR = 'logs'
LOG_FILENAME = 'api.log'

class RequestLogger:
    """Реализация логгера"""
    
    @staticmethod
    def log_request(
        endpoint: str,
        status: int,
        latency_ms: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> None:
        """Статический метод логирования запроса"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "status": status,
            "latency_ms": latency_ms,
            "request_id": request_id or str(uuid.uuid4())
        }
        
        log_entry = {k: v for k, v in log_entry.items() if v is not None}

        global LOG_DIR, LOG_FILENAME 
        full_log_path = os.path.join(LOG_DIR, LOG_FILENAME)
        
        # Создаем папку logs, если её нет
        os.makedirs(LOG_DIR, exist_ok=True) 
        
        # Проверяем существование файла по пути full_log_path
        if not os.path.exists(full_log_path):
            open(full_log_path, "w", encoding='utf-8').close()

        json_str = f'{json.dumps(log_entry, ensure_ascii=False)}\n'
        print(json_str)

        with open(os.path.join(LOG_DIR, LOG_FILENAME), 'a', encoding='utf-8') as f:
            f.write(json_str)


# ---------- Системные эндпоинты ----------

@app.get("/health", tags=["system"], summary="Health-check сервиса")
def health() -> dict[str, str]:
    global total_req
    total_req += 1
    return {
        "status": "ok",
        "service": "caricature-generation",
        "version": "0.1.0",
    }


@app.get("/metrics", tags=["system"], summary="Статистика по работе сервиса.")
def metrics() -> dict[str, Any]:
    try: 
        global total_latency_ms, latency_ct
        avg_latency_ms = total_latency_ms / latency_ct
    except:
        avg_latency_ms = None
    global total_req
    return {
        "total_req": total_req,
        "avg_latency_ms": avg_latency_ms,
    }


# ---------- Основные эндпоинты ----------

@app.post(
    "/generate", 
    tags=["main"], 
    summary="Генерация карикатуры на основе RLAIF-пайплайна",
    response_class=StreamingResponse, 
    responses={
        200: {
            "description": "Успешная генерация. Возвращает лучший шарж (выбор Qwen2-VL) в формате PNG.",
            "content": {
                "image/png": {  
                    "schema": {
                        "type": "string",
                        "format": "binary"  
                    }
                }
            }
        },
        400: {
            "description": "Ошибка валидации (загружен не графический файл)"
        },
        500: {
            "description": "Внутренняя ошибка генерации, сбой диффузии или критика"
        }
    }
)
async def caricature_generation(
    file: UploadFile = File(..., description="Выбрать изображение лица для генерации шаржа"),
    strength: float = Form(
        default=1.5, 
        ge=0.0, 
        le=3.0, 
        description="Сила утрирования черт"
    )
):
    """
    Эндпоинт принимает фото, сохраняет его во временный файл, передает в пайплайн 
    ControlNet+SD 1.5, прогоняет через критика Qwen2-VL и возвращает лучшую карикатуру.
    """
    global total_req
    total_req += 1
    request_id = str(uuid.uuid4())

    start = perf_counter()

    if not file.content_type.startswith("image/"):
        RequestLogger.log_request(endpoint="/generate", status=400, request_id=request_id)
        raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

    temp_input_path = f"temp_gen_{request_id}.jpg"
    
    try:
        # Сохраняем входящий поток байт во временный файл для генератора
        contents = await file.read()
        with open(temp_input_path, "wb") as f:
            f.write(contents)
            
        result = generator.generate(temp_input_path, warp_strength=strength)
        
        print(f"\n[Request {request_id}] Ранжирование вариантов: {' > '.join(result['ranking'])}")
        print(f"[Request {request_id}] Вердикт Qwen2-VL:\n{result['judge_response']}\n")
        
        best_img = result["best"]
        
        io_buf = io.BytesIO()
        best_img.save(io_buf, format="PNG")
        io_buf.seek(0)
        
        status_code = 200

    except Exception as e:
        status_code = 500
        RequestLogger.log_request(endpoint="/generate", status=500, request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Ошибка пайплайна генерации: {str(e)}")
        
    finally:
        # Блок гарантированно удалит временный файл с диска даже при критическом сбое
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

    global total_latency_ms, latency_ct
    latency_ms = (perf_counter() - start) * 1000.0
    latency_ct += 1
    total_latency_ms += latency_ms

    RequestLogger.log_request(
        endpoint="/generate",
        status=status_code,
        latency_ms=latency_ms,
        request_id=request_id
    )

    return StreamingResponse(io_buf, media_type="image/png")


@app.post(
    "/warp", tags=["main"], summary="Warp-деформация загруженного изображения.",
    response_class=StreamingResponse, 
    # Переопределяем отображение схемы в Swagger UI
    responses={
        200: {
            "description": "Успешное выполнение варпинга. Возвращает измененный PNG-файл.",
            "content": {
                "image/png": {  # Это переключит Media type с application/json
                    "schema": {
                        "type": "string",
                        "format": "binary"  # Это уберет "string" из Example Value
                    }
                }
            }
        },
        400: {
            "description": "Ошибка валидации входных данных (например, загружен не графический файл)"
        },
        500: {
            "description": "Внутренняя ошибка пайплайна или сбой MediaPipe"
        }
    }
)
async def warp(
    file: UploadFile = File(..., description="Выбрать изображение лица для варпинга"),
    strength: float = Form(
        default=1.5, 
        ge=0.0, 
        le=3.0, 
        description="Сила утрирования черт"
    )
) -> Response:
    """
    Эндпоинт принимает файл изображения и коэффициент деформации через форму,
    выполняет варпинг лица и возвращает бинарный файл измененного изображения (PNG).
    """
    global total_req
    total_req += 1
    request_id = str(uuid.uuid4())

    start = perf_counter()

    # Проверяем, что загружен именно графический файл
    if not file.content_type.startswith("image/"):
        RequestLogger.log_request(endpoint="/warp", status=400, request_id=request_id)
        raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

    temp_input_path = f"temp_input_{request_id}.jpg"
    try:
        # Читаем бинарный контент загруженного файла
        contents = await file.read()
        
        # Создаем временный файл, чтобы скормить его в mediapipe_prompt_extraction
        with open(temp_input_path, "wb") as f:
            f.write(contents)
            
        warped_img_bgr = warp_image(temp_input_path, strength)
        
        # Удаляем временный файл с диска после обработки
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
            
        _, encoded_img = cv2.imencode(".png", warped_img_bgr)
        io_buf = io.BytesIO(encoded_img.tobytes())
        
        status_code = 200

    except Exception as e:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

        status_code = 500
        RequestLogger.log_request(endpoint="/warp", status=500, request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

    global total_latency_ms, latency_ct
    latency_ms = (perf_counter() - start) * 1000.0
    latency_ct += 1
    total_latency_ms += latency_ms

    RequestLogger.log_request(
        endpoint="/warp",
        status=status_code,
        latency_ms=latency_ms,
        request_id=request_id
    )

    return StreamingResponse(io_buf, media_type="image/png")