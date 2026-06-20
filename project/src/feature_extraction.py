"""
feature_extraction.py
Модуль извлечения признаков лица для ControlNet-пайплайна.
MediaPipe Face Landmarker (478 точек) + IDW-варпинг + генерация промпта.
"""

import json
import os
import random
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

warnings.filterwarnings("ignore")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeFaceAggregator:
    """
    Агрегатор признаков лиц на базе MediaPipe Face Landmarker (478 точек).
    Поддерживает два режима:
      - накопление базы: add_face() → compute_mean_face() → save_mean_face()
      - использование готовой статистики: load_mean_face() или mean_face_path в __init__
    """

    def __init__(
        self,
        model_path:     Optional[str] = None,
        mean_face_path: Optional[str] = None,
    ):
        # Путь к модели: аргумент → конфиг → дефолт
        if model_path is None:
            try:
                from .config import get_config
                model_path = str(get_config().face_landmarker_path)
            except Exception:
                model_path = "artifacts/pretrained_models/face_landmarker.task"

        if not os.path.exists(model_path):
            self._download_model(model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Хранилища для режима накопления базы
        self.all_landmarks          = []
        self.all_original_landmarks = []
        self.all_features           = []
        self._cached_stats          = None

        # Индексы ключевых точек для анатомических зон
        self.indices = {
            "face_contour":        [10, 338, 297, 332, 284, 251, 389, 356, 454, 23],
            "forehead_top":        [10, 338, 297],
            "forehead_bottom":     [151, 108, 69],
            "left_eyebrow_upper":  [70, 63, 105, 66, 107],
            "left_eyebrow_lower":  [46, 53, 52, 65, 55],
            "right_eyebrow_upper": [336, 296, 334, 293, 300],
            "right_eyebrow_lower": [282, 283, 285, 295, 282],
            "left_eye":            [33, 133, 157, 158, 159, 160, 161, 173],
            "right_eye":           [362, 263, 387, 386, 385, 384, 398, 466],
            "nose_tip":            [1, 2, 98, 327, 326, 94],
            "nose_bridge":         [168, 6, 195, 5, 4],
            "nose_base":           [94, 97, 2, 326, 327, 294],
            "mouth_outer":         [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321],
            "mouth_corners":       [61, 291],
            "chin":                [152, 148, 149, 150, 136, 172, 138, 135, 169],
            "left_cheekbone":      [50, 101, 100, 117, 118],
            "right_cheekbone":     [280, 331, 330, 348, 347],
            "jawline":             [172, 136, 150, 149, 148, 152, 377, 378, 379, 365],
        }

        self.feature_names = [
            "left_eye_width", "left_eye_height", "right_eye_width", "right_eye_height",
            "nose_width", "nose_height", "mouth_width", "mouth_height",
            "left_eyebrow_thickness", "right_eyebrow_thickness", "jaw_width",
            "smile_angle", "face_ratio",
        ]

        if mean_face_path is not None:
            self.load_mean_face(mean_face_path)

    def _download_model(self, model_path: str):
        """Скачивает face_landmarker.task если файл не найден."""
        try:
            from .config import get_config
            url = get_config().face_landmarker_url
        except Exception:
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        print(f"Скачивание модели → {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        print("Модель скачана")

    def load_mean_face(self, filepath: str):
        """Загружает предвычисленную статистику выборки из JSON."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл статистики не найден: {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)
        self._cached_stats = {
            "mean_landmarks":   np.array(data["mean_landmarks"],   dtype=np.float32),
            "median_landmarks": np.array(data.get("median_landmarks", data["mean_landmarks"]), dtype=np.float32),
            "std_landmarks":    np.array(data["std_landmarks"],    dtype=np.float32),
            "mean_features":    np.array(data["mean_features"],    dtype=np.float32),
            "median_features":  np.array(data.get("median_features", data["mean_features"]), dtype=np.float32),
            "std_features":     np.array(data["std_features"],     dtype=np.float32),
            "feature_names":    data["feature_names"],
            "num_samples":      data["num_samples"],
        }
        self.feature_names = data["feature_names"]
        print(
            f"Статистика загружена из {filepath} "
            f"(выборка: {data['num_samples']} лиц)"
        )

    def extract_landmarks_from_image(self, image_path: str) -> np.ndarray:
        """Извлекает 478 landmark-точек из изображения через MediaPipe."""
        # np.fromfile + imdecode вместо cv2.imread — работает с любыми путями на Windows
        buf   = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        h, w = image.shape[:2]
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result   = self.detector.detect(mp_image)
        if not result.face_landmarks:
            raise ValueError(f"Лицо не найдено: {image_path}")

        return np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in result.face_landmarks[0]]
        )

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Нормализация: центрирование и масштаб относительно межзрачкового расстояния."""
        left_eye     = landmarks[33]
        right_eye    = landmarks[263]
        eye_center   = (left_eye + right_eye) / 2
        eye_distance = np.linalg.norm(left_eye - right_eye)
        if eye_distance > 0:
            return (landmarks - eye_center) / eye_distance
        return landmarks - eye_center

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Угол в вершине p2 между лучами p2→p1 и p2→p3, в градусах."""
        v1    = p1 - p2
        v2    = p3 - p2
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    def extract_face_features(
        self, landmarks: np.ndarray, print_output: bool = False
    ) -> Dict[str, float]:
        """
        Извлекает 13 инвариантных геометрических признаков лица.
        Все величины нормированы на ширину/высоту лица → устойчивы к масштабу.
        """
        pts = self.normalize_landmarks(landmarks) if np.max(landmarks) > 10 else landmarks

        face_width  = np.linalg.norm(pts[454] - pts[234])
        face_height = np.linalg.norm(pts[152] - pts[10])

        features = {
            "left_eye_width":          np.linalg.norm(pts[33]  - pts[133]) / face_width,
            "left_eye_height":         np.linalg.norm(pts[159] - pts[145]) / face_height,
            "right_eye_width":         np.linalg.norm(pts[263] - pts[362]) / face_width,
            "right_eye_height":        np.linalg.norm(pts[386] - pts[374]) / face_height,
            "nose_width":              np.linalg.norm(pts[331] - pts[102]) / face_width,
            "nose_height":             np.linalg.norm(pts[168] - pts[2])   / face_height,
            # толщина губ по внешнему контуру (точки 0 и 17)
            "mouth_width":             np.linalg.norm(pts[291] - pts[61])  / face_width,
            "mouth_height":            np.linalg.norm(pts[0]   - pts[17])  / face_height,
            "left_eyebrow_thickness":  np.linalg.norm(pts[52]  - pts[105]) / face_height,
            "right_eyebrow_thickness": np.linalg.norm(pts[282] - pts[334]) / face_height,
            "jaw_width":               np.linalg.norm(pts[172] - pts[397]) / face_width,
            "smile_angle":             self._calculate_angle(pts[61], pts[13], pts[291]),
            "face_ratio":              face_height / face_width,
        }

        if print_output:
            for name, val in features.items():
                print(f"{name:30}: {val:.4f}")

        return features

    def add_face(self, landmarks: np.ndarray):
        """Добавляет лицо в накопительную базу (сбрасывает кэш статистики)."""
        normalized = self.normalize_landmarks(landmarks)
        features   = self.extract_face_features(landmarks)
        self.all_landmarks.append(normalized)
        self.all_original_landmarks.append(landmarks)
        self.all_features.append(list(features.values()))
        self._cached_stats = None

    def compute_mean_face(self) -> Dict:
        """Возвращает кэшированную или вычисляет статистику по накопленной базе."""
        if self._cached_stats is not None:
            return self._cached_stats
        if not self.all_landmarks:
            raise ValueError(
                "Агрегатор пуст. Загрузите JSON или добавьте лица через add_face()."
            )
        lm = np.array(self.all_landmarks)
        ft = np.array(self.all_features)
        return {
            "mean_landmarks":   np.mean(lm,   axis=0),
            "median_landmarks": np.median(lm, axis=0),
            "std_landmarks":    np.std(lm,    axis=0),
            "mean_features":    np.mean(ft,   axis=0),
            "median_features":  np.median(ft, axis=0),
            "std_features":     np.std(ft,    axis=0),
            "feature_names":    self.feature_names,
            "num_samples":      len(self.all_landmarks),
        }

    def compare_with_mean(self, image_path: str) -> Dict[str, Dict]:
        """Сравнивает признаки лица на фото со средней нормой выборки."""
        landmarks = self.extract_landmarks_from_image(image_path)
        features  = self.extract_face_features(landmarks)
        stats     = self.compute_mean_face()

        deviations = {}
        for i, name in enumerate(stats["feature_names"]):
            val    = features[name]
            mean   = stats["mean_features"][i]
            median = stats["median_features"][i]
            std    = stats["std_features"][i]
            deviations[name] = {
                "value":             val,
                "mean":              mean,
                "median":            median,
                "std":               std,
                "deviation_percent": ((val - mean) / mean * 100) if abs(mean) > 1e-6 else 0,
                "z_score":           (val - mean) / (std + 1e-6),
            }
        return deviations

    def get_exaggeration_vector(self, image_path: str, strength: float = 1.0) -> np.ndarray:
        """
        Вычисляет вектор преувеличения (478, 2) с компенсацией наклона головы
        и синусоидальной нелинейностью.
        """
        landmarks  = self.extract_landmarks_from_image(image_path)
        normalized = self.normalize_landmarks(landmarks)
        stats      = self.compute_mean_face()

        left_eye  = normalized[33]
        right_eye = normalized[263]
        angle     = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

        cos_a, sin_a    = np.cos(-angle), np.sin(-angle)
        R_align         = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        current_center  = (left_eye + right_eye) / 2
        aligned_current = np.dot(normalized - current_center, R_align.T)

        mean_center  = (stats["mean_landmarks"][33] + stats["mean_landmarks"][263]) / 2
        aligned_mean = stats["mean_landmarks"] - mean_center

        deviation        = aligned_current - aligned_mean
        max_expected_dev = np.max(stats.get("std_landmarks", 0.5)) * 2.0
        normalized_dev   = np.clip(deviation / max_expected_dev, -1.0, 1.0)
        sinusoidal_dev   = np.sin(normalized_dev * (np.pi / 2.0))
        exag_aligned     = sinusoidal_dev * max_expected_dev * strength

        cos_b, sin_b = np.cos(angle), np.sin(angle)
        R_back       = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
        exaggeration = np.dot(exag_aligned, R_back.T)

        # Веса важности анатомических зон
        weights = np.ones(478)
        for idx in self.indices["left_eye"] + self.indices["right_eye"]:
            weights[idx] = 2.5
        for idx in self.indices["nose_tip"] + self.indices["nose_base"] + self.indices["nose_bridge"]:
            weights[idx] = 2.2
        for idx in self.indices["mouth_outer"]:
            weights[idx] = 1.8
        for idx in self.indices["face_contour"]:
            weights[idx] = 0.4

        return exaggeration * weights[:, np.newaxis]

    def save_mean_face(self, filepath: str):
        """Сохраняет текущую статистику базы в JSON."""
        stats = self.compute_mean_face()
        data  = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in stats.items()}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Среднее лицо сохранено: {filepath}")

    def warp_face(
        self,
        image_path:        str,
        caricature_params: Dict,
        target_size:       Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        IDW-варпинг методом Шепарда.
        target_size: (h, w) — высота × ширина.
        Возвращает RGB numpy array.
        """
        ex_vector    = caricature_params["exaggeration_vector"]
        sigma_factor = caricature_params.get("sigma_factor", 0.35)
        grid_step    = caricature_params.get("grid_step", 1)

        # np.fromfile + imdecode вместо cv2.imread — работает с любыми путями на Windows
        buf       = np.fromfile(image_path, dtype=np.uint8)
        image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"warp_face: не удалось прочитать {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w      = image_rgb.shape[:2]

        orig_landmarks = self.extract_landmarks_from_image(image_path)
        ex_np          = np.array(ex_vector, dtype=np.float32)

        left_eye      = orig_landmarks[33]
        right_eye     = orig_landmarks[263]
        eye_distance  = max(np.linalg.norm(left_eye - right_eye), 1e-6)
        pixel_offsets = ex_np * eye_distance

        edge_points = np.array([
            [0, 0],       [w // 2, 0],     [w - 1, 0],
            [0, h // 2],                   [w - 1, h // 2],
            [0, h - 1],   [w // 2, h - 1], [w - 1, h - 1],
        ], dtype=np.float32)

        map_x, map_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        total_dx     = np.zeros_like(map_x)
        total_dy     = np.zeros_like(map_y)
        total_weight = np.zeros_like(map_x)

        sigma_sq = 2 * (eye_distance * sigma_factor) ** 2
        for idx in range(0, 478, grid_step):
            sx, sy = orig_landmarks[idx]
            ox, oy = pixel_offsets[idx]
            r_sq   = (map_x - sx) ** 2 + (map_y - sy) ** 2
            w_     = np.exp(-r_sq / sigma_sq)
            total_dx     += ox * w_
            total_dy     += oy * w_
            total_weight += w_

        if caricature_params.get("background_stabilization", True):
            for ep in edge_points:
                ex, ey = ep
                r_sq   = (map_x - ex) ** 2 + (map_y - ey) ** 2
                total_weight += 1.0 / (r_sq + 1e-6)

        total_weight = np.maximum(total_weight, 1e-6)
        dx = total_dx / total_weight
        dy = total_dy / total_weight

        warped_map_x = np.clip(map_x - dx, 0, w - 1)
        warped_map_y = np.clip(map_y - dy, 0, h - 1)
        warped       = cv2.remap(
            image_rgb, warped_map_x, warped_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        target_h, target_w = target_size
        if warped.shape[0] != target_h or warped.shape[1] != target_w:
            # cv2.resize принимает (width, height)
            warped = cv2.resize(warped, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return warped


def get_canonical_groups() -> Dict[str, List[int]]:
    """
    Возвращает словарь анатомических групп точек Face Mesh.
    Индексы на основе официальных констант MediaPipe.
    """

    def _from_connections(connections) -> List[int]:
        indices = set()
        for a, b in connections:
            indices.add(a)
            indices.add(b)
        return list(indices)

    _LIPS = frozenset([
        (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,320),
        (320,307),(307,375),(375,321),(61,185),(185,40),(40,39),(39,37),(37,0),
        (0,267),(267,269),(269,270),(270,409),(409,291),(78,95),(95,88),(88,178),
        (178,87),(87,14),(14,317),(317,402),(402,318),(318,324),(324,308),(78,191),
        (191,80),(80,81),(81,82),(82,13),(13,312),(312,311),(311,310),(310,415),(415,308),
    ])
    _LEFT_EYE = frozenset([
        (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),(381,382),(382,362),
        (263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),(398,362),
    ])
    _LEFT_EYEBROW = frozenset([
        (276,283),(283,282),(282,295),(295,285),(300,293),(293,334),(334,296),(296,336),
    ])
    _RIGHT_EYE = frozenset([
        (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
        (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,133),
    ])
    _RIGHT_EYEBROW = frozenset([
        (46,53),(53,52),(52,65),(65,55),(70,63),(63,105),(105,66),(66,107),
    ])
    _FACE_OVAL = frozenset([
        (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),(356,454),
        (454,323),(323,361),(361,288),(288,397),(397,365),(365,379),(379,378),(378,400),
        (400,377),(377,152),(152,148),(148,176),(176,149),(149,150),(150,136),(136,172),
        (172,58),(58,132),(132,93),(93,234),(234,127),(127,162),(162,21),(21,54),
        (54,103),(103,67),(67,109),(109,10),
    ])

    return {
        "Губы":         _from_connections(_LIPS),
        "Левый глаз":   _from_connections(_LEFT_EYE),
        "Левая бровь":  _from_connections(_LEFT_EYEBROW),
        "Правый глаз":  _from_connections(_RIGHT_EYE),
        "Правая бровь": _from_connections(_RIGHT_EYEBROW),
        "Овал лица":    _from_connections(_FACE_OVAL),
        "Нос":     [1,2,4,5,6,19,94,97,98,102,129,168,195,197,203,209,326,327,331,358,423,429],
        "Челюсть": [58,132,136,148,149,150,152,172,176,288,365,377,378,379,397,400],
        "Зрачки":  list(range(468, 478)),
    }


def get_caricature_parameters(
    aggregator,
    groups:   Dict[str, List[int]],
    img_path: str,
    strength: Optional[float] = None,
) -> Dict:
    """
    Собирает полный набор параметров варпинга.
    sigma_factor, grid_step, background_stabilization берутся из конфига.
    """
    try:
        from .config import get_config
        cfg                      = get_config()
        strength                 = strength if strength is not None else cfg.warp_strength
        sigma_factor             = cfg.sigma_factor
        grid_step                = cfg.grid_step
        background_stabilization = cfg.background_stabilization  # ← была синтаксическая ошибка здесь
    except Exception:
        strength                 = strength if strength is not None else 1.3
        sigma_factor             = 0.35
        grid_step                = 1
        background_stabilization = True

    ex_vector = aggregator.get_exaggeration_vector(img_path, strength=strength)
    return {
        "exaggeration_vector":      ex_vector,
        "groups":                   groups,
        "sigma_factor":             sigma_factor,
        "grid_step":                grid_step,
        "background_stabilization": background_stabilization,
    }


def generate_dynamic_caricature_prompt(
    deviations: Dict[str, Dict],
    top_k:      int = 5,
    gender:     str = "human",
    art_style:  str = "hyperrealism",
) -> str:
    """
    Генерирует текстовый промпт для SD на основе топ-k отклонений признаков лица.
    """
    style_templates = {
        "3d_pixar": {
            "core":    "A professional 3D caricature portrait of a {expression} {gender}, cute Pixar style, flawless 3D render, claymation aesthetic",
            "details": "highly detailed clothing texture, glossy eyes, volumetric studio lighting, soft shadows, octane render, masterpiece, 8k resolution",
        },
        "hyperrealism": {
            "core":    "A hyper-realistic studio photograph of a caricatured {expression} {gender}",
            "details": "highly detailed skin texture, visible pores, individual hair strands, expressive eyes with realistic reflections, professional cinematic lighting, shot on 85mm lens, f/1.8, dramatic rim light, dark studio background, photorealistic masterpiece",
        },
        "digital_art": {
            "core":    "A stylized digital caricature illustration of a {expression} {gender}, modern comic book art style",
            "details": "clean bold outlines, smooth digital painting, rich vibrant color palette, dynamic cell shading, wacom drawing style, masterpiece, trending on artstation, highly artistic",
        },
    }

    feature_prompters = {
        ("mouth_width",      True):  ["wide cheerful smile", "big ear-to-ear grin", "laughing expression"],
        ("mouth_width",      False): ["tiny compressed lips", "pursed small mouth", "subtle ironic smirk"],
        ("mouth_height",     True):  ["open mouth in astonishment", "gasping expression", "wide laughing mouth"],
        ("mouth_height",     False): ["tightly locked flat lips", "stern tight mouth", "determined facial expression"],
        ("left_eye_height",  True):  ["wide-eyed surprised look", "huge expressive eyes", "staring intense eyes"],
        ("right_eye_height", True):  ["wide-eyed surprised look", "huge expressive eyes", "staring intense eyes"],
        ("left_eye_height",  False): ["squinting eyes", "sleepy relaxed gaze", "clever narrow eyes"],
        ("right_eye_height", False): ["squinting eyes", "sleepy relaxed gaze", "clever narrow eyes"],
        ("nose_width",       True):  ["prominent broad nose", "large stylized nose"],
        ("nose_width",       False): ["cute tiny button nose", "slender sharp nose"],
        ("nose_height",      True):  ["long majestic nose", "elongated sharp nose"],
        ("face_ratio",       True):  ["elongated narrow face shape", "stretched thin face oval"],
        ("face_ratio",       False): ["round chubby face shape", "wide square jawline structure"],
        ("jaw_width",        True):  ["gigantic heroic jawline", "broad massive chin", "strong jaw structure"],
        ("jaw_width",        False): ["very weak narrow chin", "pointed sharp chin structure"],
    }

    sorted_features = sorted(
        deviations.items(),
        key=lambda x: abs(x[1]["deviation_percent"]),
        reverse=True,
    )

    expression_tags = []
    for name, stats in sorted_features[:top_k]:
        dev = stats["deviation_percent"]
        if abs(dev) > 10:
            key = (name, dev > 0)
            if key in feature_prompters:
                tag = random.choice(feature_prompters[key])
                if tag not in expression_tags:
                    expression_tags.append(tag)

    if not expression_tags:
        expression_tags = ["cheerful", "expressive look"]

    expression_str = ", ".join(expression_tags)
    template       = style_templates.get(art_style, style_templates["hyperrealism"])
    final_prompt   = (
        template["core"].format(expression=expression_str, gender=gender)
        + ", "
        + template["details"]
    )

    print(f"expression_str: {expression_str}")
    return final_prompt


def mediapipe_prompt_extraction(
    image_path:           str,
    exageration_strength: Optional[float] = None,
) -> Dict:
    """Полный пайплайн: варп → промпт → сохранение feature_map и prompt_params."""
    from .config import get_config
    cfg      = get_config()
    strength = exageration_strength if exageration_strength is not None else cfg.warp_strength

    p = str(cfg.mean_face_path)
    if not os.path.exists(p):
        raise FileNotFoundError(
            "Запустите notebooks/feature_extraction.ipynb для создания mean_face.json"
        )

    buf   = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    aggregator = MediaPipeFaceAggregator(mean_face_path=p)
    deviations = aggregator.compare_with_mean(image_path)
    prompt     = generate_dynamic_caricature_prompt(deviations)
    neg_prompt = cfg.negative_prompt

    groups     = get_canonical_groups()
    params     = get_caricature_parameters(aggregator, groups, image_path, strength=strength)
    warped_img = aggregator.warp_face(
        image_path, caricature_params=params, target_size=image.shape[:2]
    )

    warped_bgr        = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
    name              = Path(image_path).stem
    feature_maps_dir  = cfg.feature_maps_dir
    prompt_params_dir = cfg.prompt_params_dir

    save_path = str(feature_maps_dir / f"{name}.png")
    # imencode + tofile вместо cv2.imwrite — работает с любыми путями
    cv2.imencode(".png", warped_bgr)[1].tofile(save_path)

    cn = cfg.generation["controlnet"]
    controlnet_input = {
        "control_weight":        float(cn["weight"]),
        "starting_control_step": float(cn["start"]),
        "ending_control_step":   float(cn["end"]),
        "feature_map_dir":       save_path,
        "text_promt":            prompt,
        "negative_promt":        neg_prompt,
    }
    with open(prompt_params_dir / f"{name}.json", "w") as f:
        json.dump(controlnet_input, f, indent=4)

    return controlnet_input


def warp_image(image_path: str, exageration_strength: Optional[float] = None) -> np.ndarray:
    """Варп-деформация для эндпоинта /warp. Возвращает BGR numpy array."""
    from .config import get_config
    cfg      = get_config()
    strength = exageration_strength if exageration_strength is not None else cfg.warp_strength

    # Безопасное чтение через numpy — работает с любыми путями на Windows
    buf   = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось декодировать изображение: '{image_path}'")

    h, w = image.shape[:2]

    p = str(cfg.mean_face_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Не найден файл статистики: {p}")

    aggregator = MediaPipeFaceAggregator(mean_face_path=p)
    groups     = get_canonical_groups()
    params     = get_caricature_parameters(aggregator, groups, image_path, strength=strength)
    warped_img = aggregator.warp_face(
        image_path,
        caricature_params=params,
        target_size=(h, w),
    )

    return cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)