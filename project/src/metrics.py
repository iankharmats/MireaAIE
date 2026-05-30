"""
metrics.py

Модуль оценки качества шаржей.
Вход:  оригинальное изображение + шарж (пути или PIL Image)
Выход: словарь метрик + опциональная таблица визуализации

Метрики:
  - FID  (Frechet Inception Distance)
  - KID  (Kernel Inception Distance)
  - CLIP Similarity
  - Artifact Score (шум, резкость, экспозиция, блочность, энтропия)
"""

import os
import gc
import warnings
import urllib.request
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.fft import fft2, fftshift
from scipy.stats import entropy as scipy_entropy
from skimage.filters import laplace
from skimage.metrics import structural_similarity as ssim_fn

warnings.filterwarnings("ignore")

# ── Тип входа: путь или PIL ───────────────────────────────────────────
ImageInput = Union[str, Path, Image.Image]


# ════════════════════════════════════════════════════════════════════════
#  Утилиты загрузки
# ════════════════════════════════════════════════════════════════════════

def _to_pil(img: ImageInput, size: Optional[int] = None) -> Image.Image:
    """Конвертирует любой входной тип в PIL RGB."""
    if isinstance(img, (str, Path)):
        pil = Image.open(str(img)).convert("RGB")
    elif isinstance(img, np.ndarray):
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        raise TypeError(f"Неподдерживаемый тип: {type(img)}")
    if size is not None:
        pil = pil.resize((size, size), Image.LANCZOS)
    return pil


def _to_tensor(pil: Image.Image, device: str = "cpu",
               normalize_imagenet: bool = False) -> torch.Tensor:
    """PIL → (1, 3, H, W) float32 tensor в диапазоне [0, 1]."""
    arr = np.array(pil).astype(np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    if normalize_imagenet:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        t    = (t - mean) / std
    return t


# ════════════════════════════════════════════════════════════════════════
#  Inception V3 — общий экземпляр (ленивая загрузка)
# ════════════════════════════════════════════════════════════════════════

_inception_model = None

def _get_inception(device: str) -> torch.nn.Module:
    global _inception_model
    if _inception_model is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        print("⏳ Загрузка Inception V3...")
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        # Используем пулинг вместо классификатора → вектор 2048
        model.fc = torch.nn.Identity()
        model.eval()
        _inception_model = model
        print("✅ Inception V3 готов")
    return _inception_model.to(device)


def _inception_features(
    images : List[Image.Image],
    device : str,
    batch  : int = 8,
) -> np.ndarray:
    """
    Извлекает вектора признаков Inception V3 (2048-мерные).
    images: список PIL RGB, будут приведены к 299×299.
    """
    model = _get_inception(device)
    feats = []
    for i in range(0, len(images), batch):
        batch_pils = images[i:i + batch]
        tensors = torch.cat([
            _to_tensor(p.resize((299, 299), Image.LANCZOS),
                       device=device, normalize_imagenet=True)
            for p in batch_pils
        ], dim=0)
        with torch.no_grad():
            f = model(tensors)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)   # (N, 2048)


# ════════════════════════════════════════════════════════════════════════
#  CLIP — общий экземпляр (ленивая загрузка)
# ════════════════════════════════════════════════════════════════════════

_clip_model     = None
_clip_processor = None

def _get_clip(device: str):
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            print("⏳ Загрузка CLIP (openai/clip-vit-base-patch32)...")
            _clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            _clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).eval()
            print("✅ CLIP готов")
        except ImportError:
            raise ImportError(
                "Установите transformers: pip install transformers"
            )
    return _clip_model.to(device), _clip_processor


def _clip_image_embedding(pil: Image.Image, device: str) -> np.ndarray:
    """Возвращает L2-нормированный CLIP-вектор изображения."""
    model, processor = _get_clip(device)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy().squeeze()   # (512,)


# ════════════════════════════════════════════════════════════════════════
#  FID
# ════════════════════════════════════════════════════════════════════════

def _fid_from_features(
    feats_real : np.ndarray,   # (N, D)
    feats_fake : np.ndarray,   # (M, D)
) -> float:
    """
    Frechet Inception Distance.
    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r @ Sigma_f))

    Для пары одиночных изображений N=M=1 ковариация вырождается в 0,
    поэтому FID сводится к квадрату расстояния между средними.
    Чем меньше FID — тем ближе распределения (0 = идентичны).
    """
    from scipy.linalg import sqrtm

    mu_r, mu_f  = feats_real.mean(0), feats_fake.mean(0)
    diff_sq     = np.sum((mu_r - mu_f) ** 2)

    if feats_real.shape[0] < 2 or feats_fake.shape[0] < 2:
        # При N=1 ковариация неопределена → возвращаем только L2²
        return float(diff_sq)

    cov_r = np.cov(feats_real, rowvar=False)
    cov_f = np.cov(feats_fake, rowvar=False)

    covmean, _ = sqrtm(cov_r @ cov_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_sq + np.trace(cov_r + cov_f - 2 * covmean)
    return float(np.real(fid))


# ════════════════════════════════════════════════════════════════════════
#  KID
# ════════════════════════════════════════════════════════════════════════

def _polynomial_kernel(
    X : np.ndarray,   # (N, D)
    Y : np.ndarray,   # (M, D)
    degree : int   = 3,
    gamma  : float = None,
    coef0  : float = 1.0,
) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (gamma * X @ Y.T + coef0) ** degree
    return K


def _kid_from_features(
    feats_real : np.ndarray,
    feats_fake : np.ndarray,
) -> float:
    """
    Kernel Inception Distance (несмещённая оценка MMD² с полиномиальным ядром).
    KID = MMD²(P_real, P_fake).
    Интерпретация аналогична FID: меньше = лучше.
    Преимущество KID перед FID: несмещённая оценка при малых выборках.
    """
    # Нормализуем признаки
    X = feats_real / (np.linalg.norm(feats_real, axis=1, keepdims=True) + 1e-8)
    Y = feats_fake / (np.linalg.norm(feats_fake, axis=1, keepdims=True) + 1e-8)

    m, n = X.shape[0], Y.shape[0]

    K_XX = _polynomial_kernel(X, X)
    K_YY = _polynomial_kernel(Y, Y)
    K_XY = _polynomial_kernel(X, Y)

    # Несмещённая оценка MMD²
    if m > 1:
        mmd_XX = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    else:
        mmd_XX = 0.0

    if n > 1:
        mmd_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    else:
        mmd_YY = 0.0

    mmd_XY = np.mean(K_XY)
    kid    = mmd_XX + mmd_YY - 2 * mmd_XY
    return float(kid)


# ════════════════════════════════════════════════════════════════════════
#  CLIP Similarity
# ════════════════════════════════════════════════════════════════════════

def compute_clip_similarity(
    orig   : ImageInput,
    caric  : ImageInput,
    device : str = "cpu",
) -> float:
    """
    Косинусное сходство CLIP-эмбеддингов оригинала и шаржа.
    Диапазон: 0–1. Выше = изображения семантически ближе.
    Для хорошего шаржа: 0.6–0.85 (узнаваемо, но отличается).
    """
    emb_orig  = _clip_image_embedding(_to_pil(orig),  device)
    emb_caric = _clip_image_embedding(_to_pil(caric), device)
    sim = float(np.dot(emb_orig, emb_caric))   # уже нормированы
    return float(np.clip(sim, 0.0, 1.0))


# ════════════════════════════════════════════════════════════════════════
#  Метрики артефактов
# ════════════════════════════════════════════════════════════════════════

def _sharpness(pil: Image.Image, size: int = 256) -> float:
    """
    Дисперсия лапласиана.
    < 50   → размыто
    50–400 → норма
    > 400  → очень резкое / шумное
    """
    gray = np.array(pil.resize((size, size)).convert("L"), dtype=np.float32)
    return float(laplace(gray).var())


def _noise_level(pil: Image.Image, size: int = 256) -> float:
    """
    Std высокочастотной компоненты (разность с гауссовым размытием).
    < 5   → чисто
    5–15  → приемлемо
    > 15  → шумно
    """
    gray    = np.array(pil.resize((size, size)).convert("L"), dtype=np.float32)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return float(np.std(gray - blurred))


def _exposure(pil: Image.Image, size: int = 256) -> Tuple[float, float]:
    """(overexposed_pct, underexposed_pct) — % пикселей >250 и <5."""
    gray  = np.array(pil.resize((size, size)).convert("L"))
    total = gray.size
    return (
        float(np.sum(gray > 250) / total * 100),
        float(np.sum(gray < 5)   / total * 100),
    )


def _blockiness(pil: Image.Image, size: int = 256) -> float:
    """
    Доля энергии в блочных частотах FFT (JPEG-артефакты 8×8).
    < 0.10  → чисто
    0.10–0.15 → допустимо
    > 0.15  → заметные блоки
    """
    gray   = np.array(pil.resize((size, size)).convert("L"), dtype=np.float32)
    f      = np.abs(fftshift(fft2(gray)))
    f_log  = np.log1p(f)
    h, w   = f_log.shape
    cy, cx = h // 2, w // 2
    step   = size // 8
    block_energy = sum(
        f_log[cy + ky, cx + kx]
        for ky in range(-cy, cy, step)
        for kx in range(-cx, cx, step)
        if 0 <= cy + ky < h and 0 <= cx + kx < w
    )
    return float(block_energy / (f_log.sum() + 1e-8))


def _entropy(pil: Image.Image, size: int = 256) -> float:
    """
    Энтропия Шеннона (бит).
    5–9   → норма
    < 5   → мыльное / однотонное
    > 9   → хаотичное / шумное
    """
    gray = np.array(pil.resize((size, size)).convert("L"))
    hist = np.histogram(gray, bins=256, range=(0, 256))[0]
    hist = hist / (hist.sum() + 1e-8)
    return float(scipy_entropy(hist + 1e-8, base=2))


def _color_shift(orig: Image.Image, caric: Image.Image, size: int = 256) -> float:
    """
    Hellinger distance между HSV Hue-гистограммами.
    0 → идентичные цвета, 1 → полностью разные.
    < 0.3 → норма для шаржа.
    """
    def _hue_hist(img):
        hsv  = cv2.cvtColor(np.array(img.resize((size, size))), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
        return hist / (hist.sum() + 1e-8)

    h1, h2 = _hue_hist(orig), _hue_hist(caric)
    return float(np.clip(np.sqrt(1 - np.sum(np.sqrt(h1 * h2 + 1e-8))), 0.0, 1.0))


def _saturation_mean(pil: Image.Image, size: int = 256) -> float:
    """Средняя насыщенность HSV. Очень низкая (<20) = обесцвечено."""
    hsv = cv2.cvtColor(np.array(pil.resize((size, size))), cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


def compute_artifact_metrics(
    caric  : ImageInput,
    orig   : Optional[ImageInput] = None,
    size   : int = 256,
    config : Optional[Dict] = None,
) -> Dict:
    """
    Вычисляет все метрики артефактов для одного изображения.

    Args:
        caric:  шарж
        orig:   оригинал (нужен только для color_shift)
        size:   размер для расчётов
        config: пороги (используются дефолтные если не указан)

    Returns:
        dict с ключами:
            sharpness, noise_level, overexp_pct, underexp_pct,
            blockiness, entropy, color_shift, saturation,
            artifact_score (0–7, меньше = чище),
            artifact_flags (список найденных проблем)
    """
    cfg = {
        "blur_thresh"      : 50.0,
        "noise_thresh"     : 8.0,
        "overexp_thresh"   : 5.0,
        "underexp_thresh"  : 5.0,
        "blockiness_thresh": 0.15,
        "entropy_lo"       : 5.0,
        "entropy_hi"       : 9.0,
        "color_shift_thresh": 0.4,
        "saturation_lo"    : 20.0,
    }
    if config:
        cfg.update(config)

    caric_pil = _to_pil(caric, size=None)
    orig_pil  = _to_pil(orig,  size=None) if orig is not None else None

    sharp           = _sharpness(caric_pil, size)
    noise           = _noise_level(caric_pil, size)
    overexp, underexp = _exposure(caric_pil, size)
    block           = _blockiness(caric_pil, size)
    entr            = _entropy(caric_pil, size)
    cshift          = _color_shift(orig_pil, caric_pil, size) if orig_pil else 0.0
    sat             = _saturation_mean(caric_pil, size)

    # Подсчёт проблем
    flags = []
    if sharp    <  cfg["blur_thresh"]:       flags.append("blur")
    if noise    >  cfg["noise_thresh"]:      flags.append("noise")
    if overexp  >  cfg["overexp_thresh"]:    flags.append("overexposed")
    if underexp >  cfg["underexp_thresh"]:   flags.append("underexposed")
    if block    >  cfg["blockiness_thresh"]: flags.append("blocking")
    if not (cfg["entropy_lo"] <= entr <= cfg["entropy_hi"]): flags.append("entropy")
    if cshift   >  cfg["color_shift_thresh"]: flags.append("color_shift")
    if sat      <  cfg["saturation_lo"]:     flags.append("desaturated")

    return {
        "sharpness"     : round(sharp,   2),
        "noise_level"   : round(noise,   4),
        "overexp_pct"   : round(overexp, 3),
        "underexp_pct"  : round(underexp,3),
        "blockiness"    : round(block,   5),
        "entropy"       : round(entr,    4),
        "color_shift"   : round(cshift,  4),
        "saturation"    : round(sat,     2),
        "artifact_score": len(flags),
        "artifact_flags": flags,
    }


# ════════════════════════════════════════════════════════════════════════
#  Главная функция
# ════════════════════════════════════════════════════════════════════════

def evaluate(
    original       : ImageInput,
    caricature     : ImageInput,
    device         : str = "cpu",
    compute_fid    : bool = True,
    compute_kid    : bool = True,
    compute_clip   : bool = True,
    compute_artifacts: bool = True,
    artifact_config: Optional[Dict] = None,
    extra_real     : Optional[List[ImageInput]] = None,
    extra_fake     : Optional[List[ImageInput]] = None,
) -> Dict:
    """
    Главная функция оценки пары (оригинал, шарж).

    Args:
        original:         оригинальное изображение
        caricature:       сгенерированный шарж
        device:           "cpu" / "cuda" / "cuda:N"
        compute_fid:      считать FID
        compute_kid:      считать KID
        compute_clip:     считать CLIP similarity
        compute_artifacts:считать метрики артефактов
        artifact_config:  пороги артефактов (dict, опционально)
        extra_real:       дополнительные реальные фото для FID/KID
                          (улучшает статистику при N>1)
        extra_fake:       дополнительные шаржи для FID/KID

    Returns:
        dict с разделами:
            fid, kid, clip, artifacts, summary
    """
    result: Dict = {}

    orig_pil  = _to_pil(original)
    caric_pil = _to_pil(caricature)

    # Собираем батчи для FID/KID
    real_pils = [orig_pil]  + ([_to_pil(x) for x in extra_real] if extra_real else [])
    fake_pils = [caric_pil] + ([_to_pil(x) for x in extra_fake] if extra_fake else [])

    # ── FID ──────────────────────────────────────────────────────────
    if compute_fid or compute_kid:
        print("🔢 Извлечение Inception признаков...")
        feats_real = _inception_features(real_pils, device)
        feats_fake = _inception_features(fake_pils, device)

    if compute_fid:
        fid_val = _fid_from_features(feats_real, feats_fake)
        result["fid"] = {
            "value"      : round(fid_val, 4),
            "n_real"     : len(real_pils),
            "n_fake"     : len(fake_pils),
            "interpretation": _interpret_fid(fid_val),
            "note": (
                "При N=1 FID = L2² между Inception-векторами. "
                "Для надёжного FID нужно N≥50."
                if len(real_pils) < 2 else ""
            ),
        }
        print(f"  FID  = {fid_val:.4f}  {_interpret_fid(fid_val)}")

    # ── KID ──────────────────────────────────────────────────────────
    if compute_kid:
        kid_val = _kid_from_features(feats_real, feats_fake)
        result["kid"] = {
            "value"         : round(kid_val, 6),
            "n_real"        : len(real_pils),
            "n_fake"        : len(fake_pils),
            "interpretation": _interpret_kid(kid_val),
            "note": (
                "KID несмещён при малых N. "
                "При N=1 совпадает с нормированным MMD²."
                if len(real_pils) < 10 else ""
            ),
        }
        print(f"  KID  = {kid_val:.6f}  {_interpret_kid(kid_val)}")

    # ── CLIP Similarity ───────────────────────────────────────────────
    if compute_clip:
        print("🔢 CLIP similarity...")
        clip_val = compute_clip_similarity(orig_pil, caric_pil, device)
        result["clip"] = {
            "value"         : round(clip_val, 4),
            "interpretation": _interpret_clip(clip_val),
            "description"   : (
                "Косинусное сходство CLIP-эмбеддингов. "
                "Оптимум для шаржа: 0.60–0.85 "
                "(узнаваемо, но заметно изменено)."
            ),
        }
        print(f"  CLIP = {clip_val:.4f}  {_interpret_clip(clip_val)}")

    # ── Артефакты ─────────────────────────────────────────────────────
    if compute_artifacts:
        print("🔢 Метрики артефактов...")
        art = compute_artifact_metrics(
            caric=caric_pil,
            orig=orig_pil,
            config=artifact_config,
        )
        result["artifacts"] = art
        print(f"  Artifact score = {art['artifact_score']}/8  "
              f"flags={art['artifact_flags']}")

    # ── Сводка ───────────────────────────────────────────────────────
    result["summary"] = _build_summary(result)

    return result


# ════════════════════════════════════════════════════════════════════════
#  Интерпретация
# ════════════════════════════════════════════════════════════════════════

def _interpret_fid(fid: float) -> str:
    if   fid < 10:   return "🟢 Отлично (очень близко к реальным)"
    elif fid < 50:   return "🟢 Хорошо"
    elif fid < 150:  return "🟡 Приемлемо"
    elif fid < 300:  return "🟠 Слабо"
    else:            return "🔴 Плохо (сильное отклонение)"


def _interpret_kid(kid: float) -> str:
    if   kid < 0.01:  return "🟢 Отлично"
    elif kid < 0.05:  return "🟢 Хорошо"
    elif kid < 0.15:  return "🟡 Приемлемо"
    elif kid < 0.30:  return "🟠 Слабо"
    else:             return "🔴 Плохо"


def _interpret_clip(clip: float) -> str:
    if   clip > 0.85:  return "🟡 Слишком близко (слабый шарж)"
    elif clip > 0.60:  return "🟢 Отлично (узнаваемо + изменено)"
    elif clip > 0.40:  return "🟡 Приемлемо (заметное изменение)"
    else:              return "🔴 Плохо (личность потеряна)"


def _build_summary(result: Dict) -> Dict:
    score, total = 0, 0

    if "fid" in result:
        total += 1
        if result["fid"]["value"] < 150:
            score += 1

    if "kid" in result:
        total += 1
        if result["kid"]["value"] < 0.15:
            score += 1

    if "clip" in result:
        total += 1
        v = result["clip"]["value"]
        if 0.40 <= v <= 0.85:
            score += 1

    if "artifacts" in result:
        total += 1
        if result["artifacts"]["artifact_score"] <= 2:
            score += 1

    grades = {
        4: "🏆 Отлично",
        3: "🟢 Хорошо",
        2: "🟡 Приемлемо",
        1: "🔴 Плохо",
        0: "💀 Критично",
    }
    return {
        "score"           : score,
        "total"           : total,
        "grade"           : grades.get(score, "❓"),
        "metrics_computed": list(result.keys()),
    }


# ════════════════════════════════════════════════════════════════════════
#  Батчевая оценка нескольких пар
# ════════════════════════════════════════════════════════════════════════

def evaluate_batch(
    pairs          : List[Tuple[ImageInput, ImageInput]],
    device         : str = "cpu",
    compute_fid    : bool = True,
    compute_kid    : bool = True,
    compute_clip   : bool = True,
    compute_artifacts: bool = True,
    artifact_config: Optional[Dict] = None,
) -> Dict:
    """
    Оценивает список пар (оригинал, шарж).
    FID и KID считаются по всему батчу (статистически надёжнее).

    Args:
        pairs: список (original, caricature)

    Returns:
        dict с ключами:
            per_pair   — список результатов evaluate() для каждой пары
            aggregate  — усреднённые метрики по всему батчу
            fid_batch  — FID по всему батчу (реальные vs шаржи)
            kid_batch  — KID по всему батчу
    """
    all_orig  = [_to_pil(o) for o, _ in pairs]
    all_caric = [_to_pil(c) for _, c in pairs]

    result: Dict = {"per_pair": [], "aggregate": {}}

    # Батчевые FID / KID — надёжнее при N≥10
    if (compute_fid or compute_kid) and len(pairs) >= 2:
        print(f"🔢 Inception признаки для {len(pairs)} пар...")
        feats_real = _inception_features(all_orig,  device)
        feats_fake = _inception_features(all_caric, device)

        if compute_fid:
            fid_val = _fid_from_features(feats_real, feats_fake)
            result["fid_batch"] = {
                "value"         : round(fid_val, 4),
                "n_pairs"       : len(pairs),
                "interpretation": _interpret_fid(fid_val),
            }
            print(f"  FID (batch N={len(pairs)}) = {fid_val:.4f}")

        if compute_kid:
            kid_val = _kid_from_features(feats_real, feats_fake)
            result["kid_batch"] = {
                "value"         : round(kid_val, 6),
                "n_pairs"       : len(pairs),
                "interpretation": _interpret_kid(kid_val),
            }
            print(f"  KID (batch N={len(pairs)}) = {kid_val:.6f}")

    # Поштучная оценка
    for i, (orig_pil, caric_pil) in enumerate(zip(all_orig, all_caric)):
        print(f"\n[{i+1}/{len(pairs)}]")
        r = evaluate(
            original          = orig_pil,
            caricature        = caric_pil,
            device            = device,
            compute_fid       = compute_fid and len(pairs) < 2,
            compute_kid       = compute_kid and len(pairs) < 2,
            compute_clip      = compute_clip,
            compute_artifacts = compute_artifacts,
            artifact_config   = artifact_config,
        )
        result["per_pair"].append(r)

    # Агрегация
    agg: Dict = {}
    keys_to_avg = {
        "clip"     : ["value"],
        "artifacts": ["sharpness", "noise_level", "overexp_pct",
                      "underexp_pct", "blockiness", "entropy",
                      "color_shift", "saturation", "artifact_score"],
    }
    for section, fields in keys_to_avg.items():
        if all(section in r for r in result["per_pair"]):
            agg[section] = {}
            for f in fields:
                vals = [r[section][f] for r in result["per_pair"]
                        if isinstance(r[section].get(f), (int, float))]
                if vals:
                    agg[section][f"mean_{f}"] = round(float(np.mean(vals)), 4)
                    agg[section][f"std_{f}"]  = round(float(np.std(vals)),  4)

    result["aggregate"] = agg
    return result


# ════════════════════════════════════════════════════════════════════════
#  Визуализация таблицы
# ════════════════════════════════════════════════════════════════════════

def plot_metrics_table(
    result    : Dict,
    orig      : Optional[ImageInput] = None,
    caric     : Optional[ImageInput] = None,
    save_path : Optional[str]        = None,
    title     : str                  = "Оценка качества шаржа",
) -> None:
    """
    Строит визуальную таблицу метрик с цветовой индикацией.
    Опционально показывает изображения рядом с таблицей.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("pip install matplotlib")
        return

    def _cell_color(val, lo, hi, invert=False):
        if val is None:
            return "#eeeeee"
        in_range = lo <= val <= hi
        if invert:
            in_range = not in_range
        if in_range:
            return "#d4edda"
        margin = (hi - lo) * 0.5
        near   = (lo - margin) <= val <= (hi + margin)
        return "#fff3cd" if near else "#f8d7da"

    # Строки таблицы: [Метрика, Значение, Оптимум, Оценка, Цвет]
    rows      = []
    row_clrs  = []

    def _add(name, val, lo, hi, optimum_str, invert=False, fmt=".4f"):
        v_str  = f"{val:{fmt}}" if isinstance(val, float) else str(val)
        grade  = result.get("summary", {}).get("grade", "")
        clr    = _cell_color(val, lo, hi, invert=invert) if isinstance(val, float) else "#eeeeee"
        interp = ""
        if   clr == "#d4edda": interp = "🟢"
        elif clr == "#fff3cd": interp = "🟡"
        elif clr == "#f8d7da": interp = "🔴"
        rows.append([name, v_str, optimum_str, interp])
        row_clrs.append([clr] * 4)

    # ── Секция FID ────────────────────────────────────────────────────
    if "fid" in result:
        rows.append(["── FID ──────────────────", "", "", ""]); row_clrs.append(["#e9ecef"]*4)
        _add("FID ↓",
             result["fid"]["value"], 0, 150,
             "< 150 (< 50 отлично)", fmt=".2f")
        rows.append(["  N real/fake",
                     f"{result['fid']['n_real']}/{result['fid']['n_fake']}",
                     "≥ 50 для надёжности", ""])
        row_clrs.append(["#f8f9fa"]*4)

    # ── Секция KID ────────────────────────────────────────────────────
    if "kid" in result:
        rows.append(["── KID ──────────────────", "", "", ""]); row_clrs.append(["#e9ecef"]*4)
        _add("KID ↓",
             result["kid"]["value"], 0, 0.15,
             "< 0.15 (< 0.05 отлично)", invert=True, fmt=".6f")

    # ── Секция CLIP ───────────────────────────────────────────────────
    if "clip" in result:
        rows.append(["── CLIP ─────────────────", "", "", ""]); row_clrs.append(["#e9ecef"]*4)
        _add("CLIP Similarity ↕",
             result["clip"]["value"], 0.60, 0.85,
             "0.60–0.85 (шарж узнаваем)")

    # ── Секция Артефакты ──────────────────────────────────────────────
    if "artifacts" in result:
        art = result["artifacts"]
        rows.append(["── АРТЕФАКТЫ ────────────", "", "", ""]); row_clrs.append(["#e9ecef"]*4)

        _add("Artifact Score ↓", art["artifact_score"],    0,  2,  "≤ 2/8", invert=True, fmt="d")
        _add("Sharpness ↑",      art["sharpness"],         50, 400,"50–400",  fmt=".1f")
        _add("Noise Level ↓",    art["noise_level"],       0,  8,  "< 8",  invert=True)
        _add("Overexp % ↓",      art["overexp_pct"],       0,  5,  "< 5%", invert=True)
        _add("Underexp % ↓",     art["underexp_pct"],      0,  5,  "< 5%", invert=True)
        _add("Blockiness ↓",     art["blockiness"],        0,  0.15,"< 0.15", invert=True)
        _add("Entropy ↕",        art["entropy"],           5,  9,  "5–9 бит")
        _add("Color Shift ↓",    art["color_shift"],       0,  0.4,"< 0.4", invert=True)
        _add("Saturation ↑",     art["saturation"],        20, 200,"20–200", fmt=".1f")

        if art["artifact_flags"]:
            rows.append(["  Проблемы",
                         ", ".join(art["artifact_flags"]), "", "⚠️"])
            row_clrs.append(["#fff3cd"]*4)

    # ── Итог ─────────────────────────────────────────────────────────
    if "summary" in result:
        s = result["summary"]
        rows.append(["── ИТОГ ─────────────────", "", "", ""]); row_clrs.append(["#343a40"]*4)
        rows.append(["Балл", f"{s['score']}/{s['total']}", "", s["grade"]])
        clr = {"🏆":"#d4edda","🟢":"#d4edda","🟡":"#fff3cd","🔴":"#f8d7da"}.get(
            s["grade"][0], "#ffffff"
        )
        row_clrs.append([clr]*4)

    # ── Рисуем ───────────────────────────────────────────────────────
    show_images = orig is not None and caric is not None
    fig_h = max(6, len(rows) * 0.38 + 2)
    if show_images:
        fig, (ax_imgs, ax_tbl) = plt.subplots(
            1, 2, figsize=(18, fig_h),
            gridspec_kw={"width_ratios": [1, 2]}
        )
        # Панель с изображениями
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=ax_imgs.get_subplotspec(), hspace=0.1
        )
        ax_imgs.remove()
        for ri, (label, img_input) in enumerate([("Оригинал", orig), ("Шарж", caric)]):
            ax = fig.add_subplot(gs_inner[ri])
            ax.imshow(_to_pil(img_input))
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.axis("off")
    else:
        fig, ax_tbl = plt.subplots(figsize=(14, fig_h))

    ax_tbl.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    col_labels = ["Метрика", "Значение", "Оптимум", ""]
    tbl = ax_tbl.table(
        cellText    = rows,
        colLabels   = col_labels,
        cellColours = row_clrs,
        loc         = "center",
        cellLoc     = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Заголовок таблицы
    for j in range(4):
        tbl[(0, j)].set_facecolor("#343a40")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Заголовки секций — серый фон, жирный текст
    for i, row in enumerate(rows):
        if row[0].startswith("──"):
            for j in range(4):
                cell = tbl[(i + 1, j)]
                cell.set_facecolor("#e9ecef")
                cell.set_text_props(fontweight="bold", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Таблица сохранена: {save_path}")
    plt.show()