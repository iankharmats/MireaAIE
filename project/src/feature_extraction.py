"""Модуль извлечения признаков из входного изображения для подачи на вход ControlNet"""

import numpy as np
import cv2
import json
import os
import random
from typing import List, Dict, Tuple, Optional
import mediapipe as mp
import warnings
import urllib.request
warnings.filterwarnings('ignore')

# Импорты для нового API MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# для визуализации лэндмарков по группам
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Основной класс
class MediaPipeFaceAggregator:
    """
    Агрегатор лиц с использованием MediaPipe Face Landmarker (478 точек)
    Новая версия с mp.tasks API
    """
    
    def __init__(self, model_path: str = "artifacts/pretrained_models/face_landmarker.task",  mean_face_path: Optional[str] = None):
        # Скачиваем модель если её нет
        if not os.path.exists(model_path):
            self._download_model(model_path)
        
        # Инициализация детектора MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Внутренние хранилища для режима накопления базы данных
        self.all_landmarks = []          # нормализованные (N, 478, 2)
        self.all_original_landmarks = [] # в пикселях (N, 478, 2)
        self.all_features = []           # признаки лиц (N, M)
        
        # Поле для хранения кэшированной статистики
        self._cached_stats = None
        
        # Индексы ключевых точек MediaPipe (внутренний маппинг)
        self.indices = {
            'face_contour': [10, 338, 297, 332, 284, 251, 389, 356, 454, 23],
            'forehead_top': [10, 338, 297],
            'forehead_bottom': [151, 108, 69],
            'left_eyebrow_upper': [70, 63, 105, 66, 107],
            'left_eyebrow_lower': [46, 53, 52, 65, 55],
            'right_eyebrow_upper': [336, 296, 334, 293, 300],
            'right_eyebrow_lower': [282, 283, 285, 295, 282],
            'left_eye': [33, 133, 157, 158, 159, 160, 161, 173],
            'right_eye': [362, 263, 387, 386, 385, 384, 398, 466],
            'nose_tip': [1, 2, 98, 327, 326, 94],
            'nose_bridge': [168, 6, 195, 5, 4],
            'nose_base': [94, 97, 2, 326, 327, 294],
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321],
            'mouth_corners': [61, 291],
            'chin': [152, 148, 149, 150, 136, 172, 138, 135, 169],
            'left_cheekbone': [50, 101, 100, 117, 118],
            'right_cheekbone': [280, 331, 330, 348, 347],
            'jawline': [172, 136, 150, 149, 148, 152, 377, 378, 379, 365],
        }
        
        self.feature_names = [
            'left_eye_width', 'left_eye_height', 'right_eye_width', 'right_eye_height',
            'nose_width', 'nose_height', 'mouth_width', 'mouth_height',
            'left_eyebrow_thickness', 'right_eyebrow_thickness', 'jaw_width',
            'smile_angle', 'face_ratio'
        ]
        
        # Если передан путь к файлу статистики, загружаем его моментально
        if mean_face_path is not None:
            self.load_mean_face(mean_face_path)

    def _download_model(self, model_path: str):
        """Скачивает модель FaceLandmarker"""
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print(f"Скачивание модели {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        print("✅ Модель скачана")

    def load_mean_face(self, filepath: str):
        """Загружает готовую предобработанную статистику выборки из JSON"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл статистики не найден по пути: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Восстанавливаем матрицы в формат numpy массивов
        self._cached_stats = {
            'mean_landmarks': np.array(data['mean_landmarks'], dtype=np.float32),
            'median_landmarks': np.array(data.get('median_landmarks', data['mean_landmarks']), dtype=np.float32),
            'std_landmarks': np.array(data['std_landmarks'], dtype=np.float32),
            'mean_features': np.array(data['mean_features'], dtype=np.float32),
            'median_features': np.array(data.get('median_features', data['mean_features']), dtype=np.float32),
            'std_features': np.array(data['std_features'], dtype=np.float32),
            'feature_names': data['feature_names'],
            'num_samples': data['num_samples']
        }
        self.feature_names = data['feature_names']
        print(f"Статистика базы успешно импортирована из {filepath} (Выборка: {data['num_samples']} лиц). Расчет датасета пропущен!")

    def extract_landmarks_from_image(self, image_path: str) -> np.ndarray:
        """Извлекает 478 точек из изображения через MediaPipe"""
        image = cv2.imread(image_path)
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
        result = self.detector.detect(mp_image)
        
        if not result.face_landmarks:
            raise ValueError(f"Лицо не найдено на кадре: {image_path}")
        
        landmarks_pixels = [[int(lm.x * w), int(lm.y * h)] for lm in result.face_landmarks[0]]
        return np.array(landmarks_pixels)
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Нормализация landmarks: центрирование и масштабирование относительно зрачков"""
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        eye_center = (left_eye + right_eye) / 2
        eye_distance = np.linalg.norm(left_eye - right_eye)
        
        if eye_distance > 0:
            return (landmarks - eye_center) / eye_distance
        return landmarks - eye_center
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Вычисляет угол между тремя точками в градусах"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    
    def extract_face_features(self, landmarks: np.ndarray, print_output: bool = False) -> Dict[str, float]:
        """Извлекает инвариантные геометрические признаки лица (устойчивые к наклонам)"""
        features = {}
        pts = self.normalize_landmarks(landmarks) if np.max(landmarks) > 10 else landmarks

        face_width = np.linalg.norm(pts[454] - pts[234]) 
        face_height = np.linalg.norm(pts[152] - pts[10]) 

        # Глаза
        features['left_eye_width'] = np.linalg.norm(pts[33] - pts[133]) / face_width
        features['left_eye_height'] = np.linalg.norm(pts[159] - pts[145]) / face_height
        features['right_eye_width'] = np.linalg.norm(pts[263] - pts[362]) / face_width
        features['right_eye_height'] = np.linalg.norm(pts[386] - pts[374]) / face_height

        # Нос
        features['nose_width'] = np.linalg.norm(pts[331] - pts[102]) / face_width
        features['nose_height'] = np.linalg.norm(pts[168] - pts[2]) / face_height

        # Рот (Исправленный FIXED расчет полной толщины губ по внешнему контуру 0 и 17)
        features['mouth_width'] = np.linalg.norm(pts[291] - pts[61]) / face_width
        features['mouth_height'] = np.linalg.norm(pts[0] - pts[17]) / face_height

        # Брови
        features['left_eyebrow_thickness'] = np.linalg.norm(pts[52] - pts[105]) / face_height
        features['right_eyebrow_thickness'] = np.linalg.norm(pts[282] - pts[334]) / face_height

        # Челюсть
        features['jaw_width'] = np.linalg.norm(pts[172] - pts[397]) / face_width

        # Углы и соотношения сторон
        features['smile_angle'] = self._calculate_angle(pts[61], pts[13], pts[291])
        features['face_ratio'] = face_height / face_width

        if print_output:
            for name, val in features.items():
                print(f"{name:25}: {val:.4f}")

        return features
    
    def add_face(self, landmarks: np.ndarray):
        """Добавляет лицо в оперативную базу накопления статистики (сбрасывает кэш JSON)"""
        normalized = self.normalize_landmarks(landmarks)
        features = self.extract_face_features(landmarks)
        
        self.all_landmarks.append(normalized)
        self.all_original_landmarks.append(landmarks)
        self.all_features.append(list(features.values()))
        self._cached_stats = None # Сбрасываем кэш, так как база изменилась

    def compute_mean_face(self) -> Dict:
        """Возвращает кэшированную или динамически рассчитывает статистику выборки"""
        if self._cached_stats is not None:
            return self._cached_stats
            
        if len(self.all_landmarks) == 0:
            raise ValueError("Ошибка: Агрегатор пуст. Загрузите файл через инициализатор или добавьте лица методом .add_face()")
        
        all_landmarks = np.array(self.all_landmarks)
        all_features = np.array(self.all_features)
        
        return {
            'mean_landmarks': np.mean(all_landmarks, axis=0),
            'median_landmarks': np.median(all_landmarks, axis=0),
            'std_landmarks': np.std(all_landmarks, axis=0),
            'mean_features': np.mean(all_features, axis=0),
            'median_features': np.median(all_features, axis=0),
            'std_features': np.std(all_features, axis=0),
            'feature_names': self.feature_names,
            'num_samples': len(self.all_landmarks)
        }
    
    def compare_with_mean(self, image_path: str) -> Dict[str, Dict]:
        """Сравнивает лицо на изображении со средней нормой выборки"""
        landmarks = self.extract_landmarks_from_image(image_path)
        features = self.extract_face_features(landmarks)
        stats = self.compute_mean_face()
        
        deviations = {}
        for i, name in enumerate(stats['feature_names']):
            feature_val = features[name]
            mean_val = stats['mean_features'][i]
            median_val = stats['median_features'][i]
            std_val = stats['std_features'][i]
            
            dev_pct = ((feature_val - mean_val) / mean_val * 100) if abs(mean_val) > 1e-6 else 0
            z_score = (feature_val - mean_val) / (std_val + 1e-6)
            
            deviations[name] = {
                'value': feature_val,
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'deviation_percent': dev_pct,
                'z_score': z_score
            }
        return deviations
    
    def get_exaggeration_vector(self, image_path: str, strength: float = 1.0) -> np.ndarray:
        """Вычисляет инвариантный вектор преувеличения с синусоидальной нелинейностью"""
        landmarks = self.extract_landmarks_from_image(image_path)
        normalized = self.normalize_landmarks(landmarks)
        stats = self.compute_mean_face()
        
        # Компенсация наклона головы
        left_eye = normalized[33]
        right_eye = normalized[263]
        angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R_align = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        current_center = (left_eye + right_eye) / 2
        aligned_current = np.dot(normalized - current_center, R_align.T)
        
        mean_left = stats['mean_landmarks'][33]
        mean_right = stats['mean_landmarks'][263]
        mean_center = (mean_left + mean_right) / 2
        aligned_mean = stats['mean_landmarks'] - mean_center
        
        deviation = aligned_current - aligned_mean
        max_expected_dev = np.max(stats.get('std_landmarks', 0.5)) * 2.0
        
        normalized_dev = np.clip(deviation / max_expected_dev, -1.0, 1.0)
        sinusoidal_dev = np.sin(normalized_dev * (np.pi / 2.0))
        
        exaggeration_aligned = sinusoidal_dev * max_expected_dev * strength
        
        cos_b, sin_b = np.cos(angle), np.sin(angle)
        R_back = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
        exaggeration = np.dot(exaggeration_aligned, R_back.T)
        
        # Взвешивание анатомических зон важности
        importance_weights = np.ones(478)
        for idx in self.indices['left_eye'] + self.indices['right_eye']:
            importance_weights[idx] = 2.5  
        for idx in self.indices['nose_tip'] + self.indices['nose_base'] + self.indices['nose_bridge']:
            importance_weights[idx] = 2.2  
        for idx in self.indices['mouth_outer']:
            importance_weights[idx] = 1.8  
        for idx in self.indices['face_contour']:
            importance_weights[idx] = 0.4  
                
        return exaggeration * importance_weights[:, np.newaxis]
    
    def save_mean_face(self, filepath: str):
        """Сохраняет рассчитанную текущую статистику базы данных в файл JSON"""
        stats = self.compute_mean_face()
        data = {
            'mean_landmarks': stats['mean_landmarks'].tolist(),
            'median_landmarks': stats['median_landmarks'].tolist(),
            'std_landmarks': stats['std_landmarks'].tolist(),
            'mean_features': stats['mean_features'].tolist(),
            'median_features': stats['median_features'].tolist(),
            'std_features': stats['std_features'].tolist(),
            'feature_names': stats['feature_names'],
            'num_samples': stats['num_samples']
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Среднее лицо успешно сохранено в файл: {filepath}")
    
    def warp_face(self, image_path: str, caricature_params: Dict, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Выполняет анатомически стабильный IDW-варпинг изображения методом Шепарда"""
        ex_vector = caricature_params['exaggeration_vector']
        sigma_factor = caricature_params.get('sigma_factor', 0.35)
        grid_step = caricature_params.get('grid_step', 1)
        
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        orig_landmarks = self.extract_landmarks_from_image(image_path)
        ex_vector_np = np.array(ex_vector, dtype=np.float32)
        
        left_eye = orig_landmarks[33]
        right_eye = orig_landmarks[263]
        eye_distance = max(np.linalg.norm(left_eye - right_eye), 1e-6)
        
        pixel_offsets = ex_vector_np * eye_distance
        
        # Стабилизация фона по периметру
        edge_points = np.array([
            [0, 0], [w // 2, 0], [w - 1, 0],
            [0, h // 2],         [w - 1, h // 2],
            [0, h - 1], [w // 2, h - 1], [w - 1, h - 1]
        ], dtype=np.float32)
        src_pts = np.vstack([orig_landmarks, edge_points]).astype(np.float32)

        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        total_dx = np.zeros_like(map_x)
        total_dy = np.zeros_like(map_y)
        total_weight = np.zeros_like(map_x)
        
        sparse_indices = list(range(0, 478, grid_step))
        sigma_sq = 2 * (eye_distance * sigma_factor) ** 2
        
        for idx in sparse_indices:
            sx, sy = src_pts[idx]
            ox, oy = pixel_offsets[idx] if idx < 478 else (0, 0)
            
            r_sq = (map_x - sx) ** 2 + (map_y - sy) ** 2
            weight = np.exp(-r_sq / sigma_sq)
            
            total_dx += ox * weight
            total_dy += oy * weight
            total_weight += weight

        if caricature_params.get('background_stabilization', True):
            for edge_pt in edge_points:
                ex, ey = edge_pt
                r_sq = (map_x - ex) ** 2 + (map_y - ey) ** 2
                edge_weight = 1.0 / (r_sq + 1e-6) 
                total_weight += edge_weight

        total_weight = np.maximum(total_weight, 1e-6)
        dx = total_dx / total_weight
        dy = total_dy / total_weight

        warped_map_x = np.clip(map_x - dx, 0, w - 1)
        warped_map_y = np.clip(map_y - dy, 0, h - 1)
        
        warped_img = cv2.remap(image_rgb, warped_map_x, warped_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        if warped_img.shape[:2] != target_size:
            warped_img = cv2.resize(warped_img, target_size, interpolation=cv2.INTER_AREA)
            
        return warped_img

# Вспомогательные функции

def get_canonical_groups():
    """
    Генерирует группы на основе официальных констант MediaPipe Face Mesh.
    Это гарантирует 100% точность индексов.
    """
    
    # Вспомогательная функция для превращения набора связей в список уникальных точек
    def get_indices(connections):
        indices = set()
        for connection in connections:
            indices.add(connection[0])
            indices.add(connection[1])
        return list(indices)

    groups = {
        # Основные контуры (из FACEMESH_CONTOURS)
        'Губы': get_indices(mp_face_mesh.FACEMESH_LIPS),
        'Левый глаз': get_indices(mp_face_mesh.FACEMESH_LEFT_EYE),
        'Левая бровь': get_indices(mp_face_mesh.FACEMESH_LEFT_EYEBROW),
        'Правый глаз': get_indices(mp_face_mesh.FACEMESH_RIGHT_EYE),
        'Правая бровь': get_indices(mp_face_mesh.FACEMESH_RIGHT_EYEBROW),
        'Овал лица': get_indices(mp_face_mesh.FACEMESH_FACE_OVAL),
        
        # Детализация (из FACEMESH_TESSELATION - это вся сетка)
        # Здесь мы добавляем нос и челюсть через фиксированные индексы, 
        # так как в connections они не выделены отдельными именованными константами
        'Нос': [
            1, 2, 98, 327, 326, 97, 4, 5, 195, 197, # центр
            102, 129, 203, 209, 429, 423, 358, 331    # крылья
        ],
        'Челюсть': [58, 172, 136, 150, 149, 148, 152, 377, 378, 379, 365, 397, 288],
        # 'Переносица': [168, 6, 197, 195, 5]
    }

    # Ирисы 
    try:
        groups['Зрачки'] = get_indices(mp_face_mesh.FACEMESH_IRISES)
    except AttributeError:
        pass

    return groups

def get_caricature_parameters(aggregator, groups: Dict[str, List[int]], img_path: str, strength: float = 1.3) -> Dict:
    """
    Генерирует полный набор параметров для создания карикатуры.
    Автоматически рассчитывает внутренний вектор преувеличения MediaPipe.
    """
    # Вычисляем вектор преувеличения через встроенный метод агрегатора
    ex_vector = aggregator.get_exaggeration_vector(img_path, strength=strength)
    
    return {
        'exaggeration_vector': ex_vector,
        'groups': groups,
        'sigma_factor': 0.35, # Плавность затухания деформации (0.6 - 0.8)
        'grid_step': 1, # Шаг прореживания сетки (1 - идеально плавно, 4 - быстро)
        'background_stabilization': True
    }

# Сбор промта ControlNet
def generate_dynamic_caricature_prompt(deviations: Dict[str, Dict], 
                                        top_k: int = 5,
                                      gender: str = "human", 
                                      art_style: str = "hyperrealism") -> str:
    """
    Автоматически собирает эталонный промт для ControlNet / Stable Diffusion
    на основе топ-3 экстраординарных признаков лица.
    
    Args:
        deviations: Словарь отклонений от aggregator.compare_with_mean()
        gender: Пол персонажа ("man", "woman", "boy", "girl", "human" по умолчанию)
        art_style: Желаемый стиль ("3d_pixar", "hyperrealism", "digital_art")
    """
    
    # 1. Базовые художественные стили (шаблоны окружения, текстуры и света)
    style_templates = {
        "3d_pixar": {
            "core": "A professional 3D caricature portrait of a {expression} {gender}, cute Pixar style, flawless 3D render, claymation aesthetic",
            "details": "highly detailed clothing texture, glossy eyes, volumetric studio lighting, soft shadows, octane render, masterpiece, 8k resolution"
        },
        "hyperrealism": {
            "core": "A hyper-realistic studio photograph of a caricatured {expression} {gender}",
            "details": "highly detailed skin texture, visible pores, individual hair strands, expressive eyes with realistic reflections, professional cinematic lighting, shot on 85mm lens, f/1.8, dramatic rim light, dark studio background, photorealistic masterpiece"
        },
        "digital_art": {
            "core": "A stylized digital caricature illustration of a {expression} {gender}, modern comic book art style",
            "details": "clean bold outlines, smooth digital painting, rich vibrant color palette, dynamic cell shading, wacom drawing style, masterpiece, trending on artstation, highly artistic"
        }
    }
    
    # 2. Маппинг анатомических отклонений в эмоциональные текстовые маркеры
    feature_prompters = {
        ("mouth_width", True): ["wide cheerful smile", "big ear-to-ear grin", "laughing expression"],
        ("mouth_width", False): ["tiny compressed lips", "pursed small mouth", "subtle ironic smirk"],
        
        ("mouth_height", True): ["open mouth in astonishment", "gasping expression", "wide laughing mouth"],
        ("mouth_height", False): ["tightly locked flat lips", "stern tight mouth", "determined facial expression"],
        
        ("left_eye_height", True): ["wide-eyed surprised look", "huge expressive eyes", "staring intense eyes"],
        ("right_eye_height", True): ["wide-eyed surprised look", "huge expressive eyes", "staring intense eyes"],
        ("left_eye_height", False): ["squinting eyes", "sleepy relaxed gaze", "clever narrow eyes"],
        ("right_eye_height", False): ["squinting eyes", "sleepy relaxed gaze", "clever narrow eyes"],
        
        ("nose_width", True): ["prominent broad nose", "large stylized nose"],
        ("nose_width", False): ["cute tiny button nose", "slender sharp nose"],
        
        ("nose_height", True): ["long majestic nose", "elongated sharp nose"],
        
        ("face_ratio", True): ["elongated narrow face shape", "stretched thin face oval"],
        ("face_ratio", False): ["round chubby face shape", "wide square jawline structure"],
        
        ("jaw_width", True): ["gigantic heroic jawline", "broad massive chin", "strong jaw structure"],
        ("jaw_width", False): ["very weak narrow chin", "pointed sharp chin structure"],
    }
    
    # 3. Находим Топ-k экстраординарных признака (сортировка по модулю отклонения)
    sorted_features = sorted(
        deviations.items(),
        key=lambda x: abs(x[1]['deviation_percent']),
        reverse=True
    )
    
    # 4. Формируем строку выражения лица (expression) на основе топ-отклонений
    expression_tags = []
    
    # Проверяем топ-k признака на наличие текстовых ассоциаций
    for name, stats in sorted_features[:top_k]:
        dev_percent = stats['deviation_percent']
        
        # Учитываем только значимые отклонения
        if abs(dev_percent) > 10:
            is_positive = dev_percent > 0
            key = (name, is_positive)
            
            if key in feature_prompters:
                # Случайным образом берем один из синонимов, чтобы разнообразить генерацию
                chosen_tag = random.choice(feature_prompters[key])
                if chosen_tag not in expression_tags:
                    expression_tags.append(chosen_tag)
                    
    # Если лицо слишком «среднее» и явных черт нет, задаем нейтрально-позитивный тон
    if not expression_tags:
        expression_tags = ["cheerful", "expressive look"]
        
    # Соединяем теги через запятую
    expression_str = ", ".join(expression_tags)
    
    # 5. Собираем финальный текстовый промт
    selected_template = style_templates.get(art_style, style_templates["hyperrealism"])
    
    core_prompt = selected_template["core"].format(expression=expression_str, gender=gender)
    final_prompt = f"{core_prompt}, {selected_template['details']}"

    print(f'expression_str: {expression_str}')
    
    return final_prompt

def mediapipe_prompt_extraction(image_path: str, exageration_strength: int = 1.5):
    """Ключевая функция модуля - составление карты признаков и промта для ControlNet"""
    image = cv2.imread(image_path)

    # Передаем ссылку на JSON-файл при создании объекта
    p = "configs/mean_face.json"
    if os.path.exists(p):
        aggregator = MediaPipeFaceAggregator(mean_face_path=p)
    else:
        raise FileNotFoundError('Перед использованием модуля запустите ноутбук notebooks/feature_extraction.ipynb для подсчета статистик!')

    deviations = aggregator.compare_with_mean(image_path) 
    prompt = generate_dynamic_caricature_prompt(deviations, top_k=5)
    neg_promt = "normal proportions, regular anatomy, symmetrical face, 3D render, anime, cartoon, illustration, drawing, painting, bad anatomy, smooth skin, plastic skin, blurry, low resolution, watermark"

    groups = get_canonical_groups()
    params = get_caricature_parameters(aggregator, groups, image_path, strength=exageration_strength)
    warped_img = aggregator.warp_face(image_path, caricature_params=params, target_size=image.shape[:-1])

    warped_bgr = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
    name = image_path.split('.')[0].split('/')[-1]

    save_path = f'configs/controlnet/feature_maps/{name}.png'
    cv2.imwrite(save_path, warped_bgr)

    controlnet_input = {
        'control_weight': 0.8,
        'starting_control_step': 0.0,
        'ending_control_step': 0.8,
        'feature_map_dir': save_path,
        'text_promt': prompt,
        'negative_promt': neg_promt
    }

    # Сохраняем
    with open(f'configs/controlnet/prompt_params/{name}.json', 'w') as f:
        json.dump(controlnet_input, f, indent=4)
    
    return controlnet_input

def warp_image(image_path: str, exageration_strength: int = 1.5):
    """Функция для визуализации варпа на веб-сервисе"""
    image = cv2.imread(image_path)

    # Передаем ссылку на JSON-файл при создании объекта
    p = "configs/mean_face.json"
    if os.path.exists(p):
        aggregator = MediaPipeFaceAggregator(mean_face_path=p)
    else:
        import inspect
        frame = inspect.currentframe()
        caller_dir = os.path.dirname(os.path.abspath(inspect.getfile(frame)))
        raise FileNotFoundError(f'Не найден файл configs/mean_face.json. Тек. директория: {os.getcwd()}. Директория файла, откуда вызвана функция: {caller_dir}')
    groups = get_canonical_groups()
    params = get_caricature_parameters(aggregator, groups, image_path, strength=exageration_strength)
    warped_img = aggregator.warp_face(image_path, caricature_params=params, target_size=image.shape[:-1])

    warped_bgr = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
    return warped_bgr