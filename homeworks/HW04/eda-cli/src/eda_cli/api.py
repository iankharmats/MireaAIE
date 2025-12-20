from __future__ import annotations

from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime
from typing import Optional
import uuid
import json
import os

from .core import compute_quality_flags, missing_table, summarize_dataset
from .cli import json_summary_data

# --------------- Поля сбора статистики --------------
total_req = 0
latency_ct = 0
total_latency_ms = 0
last_ok_for_model = {"score": 0.0, "time": None}

# --------------- FastAPI ---------------

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)

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
        ok_for_model: Optional[bool] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> None:
        """Статический метод логирования запроса"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "status": status,
            "latency_ms": latency_ms,
            "ok_for_model": ok_for_model,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "request_id": request_id or str(uuid.uuid4())
        }
        
        log_entry = {k: v for k, v in log_entry.items() if v is not None}

        global LOG_DIR, LOG_FILENAME 
        os.makedirs(LOG_DIR, exist_ok=True) 
        if not os.path.exists(LOG_FILENAME):
            open(LOG_FILENAME, "w").close() 
        json_str = f'{json.dumps(log_entry, ensure_ascii=False)}\n'
        print(json_str)

        with open(os.path.join(LOG_DIR, LOG_FILENAME), 'a', encoding='utf-8') as f:
            f.write(json_str)

# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


class FlagsResponse(BaseModel):
    """Ответ словаря всех флагов качества."""

    flags: dict[str, Any] | None = Field(
        default=None,
        description="Полный словарь флагов качества датасета (например, too_few_rows, too_many_missing)",
    )


# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"],
    summary="Health-check сервиса",)
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    global total_req
    total_req += 1
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


@app.get("/metrics", tags=["system"],
    summary="Статистика по работе сервиса.",)
def metrics() -> dict[str, Any]:
    """Простая статистика по работе сервиса."""
    try: 
        global total_latency_ms, latency_ct
        avg_latency_ms = total_latency_ms / latency_ct,
    except:
        avg_latency_ms = None
    global total_req, last_ok_for_model
    return {
        "total_req": total_req,
        "avg_latency_ms": avg_latency_ms,
        "last_ok_for_model": last_ok_for_model,
    }


# ---------- Заглушка /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"],
    summary="Оценка качества датасета по агрегированным признакам.",)
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """
    global total_req
    total_req += 1
    request_id = str(uuid.uuid4())

    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков (если есть числовые и категориальные)
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
        global last_ok_for_model
        last_ok_for_model = {
            "score": score,
            "time": datetime.now()
        }
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    global total_latency_ms, latency_ct
    latency_ms = (perf_counter() - start) * 1000.0
    latency_ct += 1
    total_latency_ms += latency_ms

    # Флаги, которые могут быть полезны для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Примитивный лог — на семинаре можно обсудить, как это превратить в нормальный logger
    # print(
    #     f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
    #     f"max_missing_share={req.max_missing_share:.3f} "
    #     f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    # )

    RequestLogger.log_request(
        endpoint="/quality",
        status=200,
        latency_ms=latency_ms,
        ok_for_model=ok_for_model,
        n_rows=req.n_rows,
        n_cols=req.n_cols,
        request_id=request_id
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.

    Именно это по сути связывает S03 (CLI EDA) и S04 (HTTP-сервис).
    """
    global total_req
    total_req += 1
    request_id = str(uuid.uuid4())

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        RequestLogger.log_request(
            endpoint="/quality-from-csv",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        RequestLogger.log_request(
            endpoint="/quality-from-csv",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        RequestLogger.log_request(
            endpoint="/quality-from-csv",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    # Ожидаем, что compute_quality_flags вернёт quality_score в [0,1]
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
        global last_ok_for_model
        last_ok_for_model = {
            "score": score,
            "time": datetime.now()
        }
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    global total_latency_ms, latency_ct
    latency_ms = (perf_counter() - start) * 1000.0
    latency_ct += 1
    total_latency_ms += latency_ms

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета берём из summary (если там есть поля n_rows/n_cols),
    # иначе — напрямую из DataFrame.
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    # print(
    #     f"[quality-from-csv] filename={file.filename!r} "
    #     f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
    #     f"latency_ms={latency_ms:.1f} ms"
    # )

    RequestLogger.log_request(
        endpoint="/quality-from-csv",
        status=200,
        latency_ms=latency_ms,
        ok_for_model=ok_for_model,
        n_rows=n_rows,
        n_cols=n_cols,
        request_id=request_id
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )



# ---------- /quality-flags-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-flags-from-csv",
    response_model=FlagsResponse,
    tags=["quality"],
    summary="Полный набор флагов качества",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> FlagsResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    и возвращает словарь флагов качества quality_flags.
    """
    global total_req
    total_req += 1
    request_id = str(uuid.uuid4())

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        RequestLogger.log_request(
            endpoint="/quality-flags-from-csv",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        RequestLogger.log_request(
            endpoint="/quality-flags-from-csv",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        RequestLogger.log_request(
            endpoint="/quality-flags-from-csv",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    RequestLogger.log_request(
        endpoint="/quality-flags-from-csv",
        status=200,
        request_id=request_id
    )
    return FlagsResponse(
        flags=flags_all
    )


# ---------- /summary-from-csv: краткая json-сводка по CSV ----------


@app.post(
    "/summary-from-csv",
    tags=["quality"],
    summary="Краткая json-сводка по CSV",
)
async def json_from_csv(file: UploadFile = File(...)):
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    и возвращает краткую сводку качества, как в CLI-режимe --json-summary.
    """
    global total_req
    total_req += 1
    request_id = str(uuid.uuid4())

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        RequestLogger.log_request(
            endpoint="/summary-from-csv",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        RequestLogger.log_request(
            endpoint="/summary-from-csv",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        RequestLogger.log_request(
            endpoint="/summary-from-csv",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    data = json_summary_data(df, summary=summary, quality_flags=flags_all, name=file.filename)

    RequestLogger.log_request(
        endpoint="/summary-from-csv",
        status=200,
        request_id=request_id
    )
    return data


# ---------- /sample: вывод n случайных строк CSV ----------


@app.post(
    "/sample",
    tags=["exploration"],
    summary="Вывод n случайных строк CSV",
)
async def sample_from_csv(file: UploadFile = File(...), n: int = 3):
    """
    Эндпоинт, который принимает CSV-файл, и возвращает n случайных строчек 
    из CSV, аналог sample CLI-режима.
    """
    global total_req
    total_req += 1
    request_id = str(uuid.uuid4())

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        RequestLogger.log_request(
            endpoint="/sample",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        RequestLogger.log_request(
            endpoint="/sample",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        RequestLogger.log_request(
            endpoint="/sample",
            status=400,
            request_id=request_id
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    RequestLogger.log_request(
        endpoint="/sample",
        status=200,
        request_id=request_id
    )
    return df.sample(n=n).to_dict()

