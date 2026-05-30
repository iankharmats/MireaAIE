import os
import io
from unittest.mock import patch
import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

@pytest.fixture
def dummy_image_bytes():
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


# 1. Sanity-check

def test_health_endpoint():
    """Проверяем простейший health-check сервиса."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()


def test_metrics_endpoint():
    """Проверяем сбор базовой статистики и доступность метрик."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_req" in data
    assert "avg_latency_ms" in data


# 2. Модульные тесты

@patch("src.api.warp_image")
def test_warp_validation_success(mock_warp, dummy_image_bytes):
    """Проверяем, что эндпоинт корректно принимает параметры формы и передает их в warp_image."""
    mock_warp.return_value = np.zeros((200, 200, 3), dtype=np.uint8)

    files = {"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    data = {"strength": "2.0"}

    response = client.post("/warp", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    mock_warp.assert_called_once()


def test_warp_validation_invalid_file():
    """Проверяем защиту эндпоинта от загрузки не-графических файлов (например, текстовых логов)."""
    files = {"file": ("logs.txt", b"some text log data", "text/plain")}
    data = {"strength": "1.5"}

    response = client.post("/warp", files=files, data=data)
    
    assert response.status_code == 400
    assert "not an image" in response.json()["detail"].lower()


def test_warp_validation_wrong_strength(dummy_image_bytes):
    """Проверяем pydantic/fastapi валидацию ограничений ползунка strength (максимум 3.0)."""
    files = {"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
    data = {"strength": "5.5"}  

    response = client.post("/warp", files=files, data=data)
    
    assert response.status_code == 422


# 3. E2E тесты 

@pytest.mark.integration
def test_e2e_warp_real_processing():
    """
    Сквозной тест: отправляет реальный файл на обработку MediaPipe.
    Запускается только если тестовый файл физически существует на диске.
    """
    real_image_path = "tests/test_image.jpg"
    
    if not os.path.exists(real_image_path):
        pytest.skip("Пропуск E2E теста: отсутствует тестовое изображение 'tests/test_image.jpg'")

    with open(real_image_path, "rb") as f:
        files = {"file": ("test_image.jpg", f.read(), "image/jpeg")}
    
    data = {"strength": "1.2"}
    
    response = client.post("/warp", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0

@pytest.mark.integration
def test_e2e_generate_real_processing():
    """
    Тяжелый E2E-тест: Проверяет реальную генерацию шаржа через SD 1.5 и Qwen2-VL.
    Запускается только при наличии видеокарты и тестового изображения.
    """
    real_image_path = "tests/test_image.jpg"
    
    if not os.path.exists(real_image_path):
        pytest.skip("Пропуск тяжелого E2E: отсутствует файл 'tests/test_image.jpg'")

    with open(real_image_path, "rb") as f:
        files = {"file": ("test_face.jpg", f.read(), "image/jpeg")}
    
    # Отправляем запрос с кастомной силой утрирования
    data = {"strength": "2.0"}
    
    print("\n[E2E] Запуск реального инференса моделей (это может занять время)...")
    response = client.post("/generate", files=files, data=data)
    
    # Базовые проверки на успешный бинарный ответ картинки
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0
    print("[E2E] Реальная генерация успешно завершена!")