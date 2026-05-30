"""
caricature_generator.py

Модуль генерации шаржей.
Использует feature_extraction.py для варпа и построения промпта,
дообученную SD-модель для генерации и Qwen2-VL для оценки результатов.
"""

import os
import gc
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# Импорт из модуля caricature_generator
from feature_extraction import (
    MediaPipeFaceAggregator,
    get_canonical_groups,
    get_caricature_parameters,
    generate_dynamic_caricature_prompt,
)


 
#  Константы генерации

CROP_SIZE          = 512
DENOISE_STRENGTH   = 0.85
GUIDANCE_SCALE     = 9.0
IP_ADAPTER_SCALE   = 0.35
LORA_SCALE         = 0.4
NUM_STEPS          = 35
SEED_A             = 67
SEED_B             = 999
SEED_C             = 42
CANNY_LO           = 60
CANNY_HI           = 150
CONTROLNET_WEIGHT  = 0.6
CONTROLNET_START   = 0.0
CONTROLNET_END     = 0.65



NEGATIVE_PROMPT = (
    "normal proportions, regular anatomy, symmetrical face, "
    "3D render, anime, cartoon, illustration, drawing, painting, "
    "bad anatomy, smooth skin, plastic skin, blurry, low resolution, watermark"
)

# Промпт для судьи
JUDGE_PROMPT = """You are an expert caricature critic.
You will see four images:
- IMAGE 1: Original photograph (reference face)
- IMAGE 2: Caricature variant A
- IMAGE 3: Caricature variant B
- IMAGE 4: Caricature variant C

For each caricature (A, B, C) briefly describe:
  1. Which features are exaggerated
  2. How funny and recognizable it is
  3. Overall artistic quality

Then rank them from best to worst.

You MUST end your response with exactly this format:
RANKING: X > Y > Z
where X, Y, Z are A, B, C in your preferred order."""


 
#  Вспомогательные утилиты

def _get_canny(image_pil: Image.Image) -> Image.Image:
    """Строит карту контуров Canny из PIL-изображения."""
    gray  = np.array(image_pil.convert("L"))
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def _square_crop(
    image_pil: Image.Image,
    bbox: np.ndarray,
    size: int = CROP_SIZE,
    padding: float = 0.30,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Квадратный кроп вокруг bbox с отступом, ресайз до size×size."""
    iw, ih       = image_pil.size
    x1, y1, x2, y2 = bbox.astype(int)
    pad          = int(max(x2 - x1, y2 - y1) * padding)
    cx, cy       = (x1 + x2) // 2, (y1 + y2) // 2
    half         = max(x2 - x1, y2 - y1) // 2 + pad
    box          = (
        max(0, cx - half), max(0, cy - half),
        min(iw - 1, cx + half), min(ih - 1, cy + half),
    )
    return image_pil.crop(box).resize((size, size), Image.LANCZOS), box


def _build_face_embeds(face, device: str) -> torch.Tensor:
    """IP-Adapter FaceID embeds: (2, 1, 512) — uncond + cond."""
    cond   = torch.from_numpy(face.normed_embedding).unsqueeze(0).unsqueeze(0)
    uncond = torch.zeros_like(cond)
    return torch.cat([uncond, cond], dim=0).to(dtype=torch.float16, device=device)


def _parse_ranking(response: str) -> List[str]:
    """Парсит строку RANKING: X > Y > Z из ответа судьи."""
    match = re.search(
        r"RANKING\s*:\s*([ABC])\s*>\s*([ABC])\s*>\s*([ABC])",
        response, re.IGNORECASE,
    )
    if match:
        return [match.group(i).upper() for i in range(1, 4)]

    # Fallback: собираем упоминания A/B/C по порядку появления
    letters = re.findall(r"\b([ABC])\b", response.upper())
    seen, ranking = set(), []
    for letter in letters:
        if letter not in seen:
            seen.add(letter)
            ranking.append(letter)
    # Дополняем отсутствующие
    for missing in ["A", "B", "C"]:
        if missing not in ranking:
            ranking.append(missing)

    print(f"  [judge] тег RANKING не найден, fallback: {ranking}")
    return ranking


 
#  Загрузка моделей
 

def load_pipeline(checkpoint_dir: Optional[str] = None, device: str = "cuda"):
    """
    Загружает SD 1.5 + ControlNet Canny + IP-Adapter FaceID.
    Если checkpoint_dir указан — применяет дообученные LoRA-веса.
    """
    from diffusers import (
        StableDiffusionControlNetImg2ImgPipeline,
        ControlNetModel,
        EulerAncestralDiscreteScheduler,
    )

    gc.collect()
    torch.cuda.empty_cache()

    print("[1/3] Загрузка ControlNet Canny...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    print("[2/3] Загрузка Stable Diffusion 1.5...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    print("[3/3] Загрузка IP-Adapter FaceID...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sd15.bin",
        image_encoder_folder=None,
    )
    pipe.load_lora_weights(
        "h94/IP-Adapter-FaceID",
        weight_name="ip-adapter-faceid_sd15_lora.safetensors",
    )
    pipe.fuse_lora(lora_scale=LORA_SCALE)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

    # Применяем дообученный чекпоинт если передан
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            print(f"Чекпоинт не найден: {ckpt_path}. Используем базовую модель.")
        else:
            from peft import PeftModel
            print(f"Применяем чекпоинт: {ckpt_path}")
            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(ckpt_path))
            pipe.unet = pipe.unet.merge_and_unload()
            pipe.unet = pipe.unet.to(device)

    print("Pipeline готов!")
    return pipe


def load_insightface(device: str = "cuda"):
    """Загружает InsightFace для получения face embedding."""
    from insightface.app import FaceAnalysis

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith("cuda") else
        ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace загружен")
    return app


def load_judge(device: str = "cuda"):
    """Загружает Qwen2-VL-7B-Instruct для оценки шаржей."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    gc.collect()
    torch.cuda.empty_cache()

    print("Загрузка судьи Qwen2-VL-7B-Instruct...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()
    print(f"Судья загружен на {device}")
    return model, processor


 
#  Генерация трёх вариантов
 

def _generate_variants(
    pipe,
    warped_pil:      Image.Image,
    face,
    original_pil:    Image.Image,
    positive_prompt: str,
    negative_prompt: str,
    device:          str,
) -> Dict[str, Image.Image]:
    """
    Генерирует три варианта шаржа с seed A/B/C.
    Возвращает словарь PIL-изображений.
    """
    warped_crop, box = _square_crop(warped_pil, face.bbox)
    orig_crop = original_pil.crop(box).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
    canny     = _get_canny(warped_crop)
    embeds    = _build_face_embeds(face, device)

    common_kwargs = dict(
        prompt                        = positive_prompt,
        negative_prompt               = negative_prompt,
        image                         = warped_crop,
        control_image                 = canny,
        ip_adapter_image_embeds       = [embeds],
        num_inference_steps           = NUM_STEPS,
        strength                      = DENOISE_STRENGTH,
        guidance_scale                = GUIDANCE_SCALE,
        controlnet_conditioning_scale = CONTROLNET_WEIGHT,
        control_guidance_start        = CONTROLNET_START,
        control_guidance_end          = CONTROLNET_END,
    )

    variants: Dict[str, Image.Image] = {}
    for label, seed in [("A", SEED_A), ("B", SEED_B), ("C", SEED_C)]:
        print(f"Генерация варианта {label} (seed={seed})...")
        generator = torch.Generator(device=device).manual_seed(seed)
        variants[f"caricature_{label}"] = pipe(**common_kwargs, generator=generator).images[0]

    del embeds
    torch.cuda.empty_cache()

    variants["original_crop"] = orig_crop
    variants["warped_crop"]   = warped_crop
    variants["canny"]         = canny
    return variants


 
#  Оценка через VLLM
 

def _run_judge(
    model,
    processor,
    original_crop: Image.Image,
    car_a:         Image.Image,
    car_b:         Image.Image,
    car_c:         Image.Image,
) -> Dict:
    """Запрашиваем у Qwen2-VL ранжирование трёх вариантов шаржа."""
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "text",  "text": "IMAGE 1 (Original):"},
            {"type": "image", "image": original_crop},
            {"type": "text",  "text": "IMAGE 2 (Caricature A):"},
            {"type": "image", "image": car_a},
            {"type": "text",  "text": "IMAGE 3 (Caricature B):"},
            {"type": "image", "image": car_b},
            {"type": "text",  "text": "IMAGE 4 (Caricature C):"},
            {"type": "image", "image": car_c},
            {"type": "text",  "text": JUDGE_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated     = output_ids[0][inputs["input_ids"].shape[1]:]
    full_response = processor.decode(generated, skip_special_tokens=True).strip()
    ranking       = _parse_ranking(full_response)
    scores        = {label: rank + 1 for rank, label in enumerate(ranking)}

    return {
        "ranking":       ranking,   # ['A', 'C', 'B'] — от лучшего к худшему
        "scores":        scores,    # {'A': 1, 'C': 2, 'B': 3}
        "full_response": full_response,
    }


 
#  Главный класс
 

class CaricatureGenerator:
    """
    Генератор шаржей на основе дообученной SD-модели.

    Использует feature_extraction.py для:
      - варп-трансформации входного изображения (MediaPipeFaceAggregator)
      - построения текстового промпта (generate_dynamic_caricature_prompt)

    Пример использования:
        gen = CaricatureGenerator(
            checkpoint_dir="artifacts/models/dpo_checkpoints/cycle_015",
        )
        result = gen.generate("photo.jpg")
        result["best"].save("best_caricature.png")
        print("Победитель:", result["best_label"])
        print("Ранжирование:", result["ranking"])
    """

    def __init__(
        self,
        checkpoint_dir:  Optional[str] = None,
        mean_face_path:  Optional[str] = None,
        gen_device:      str           = "cuda",
        judge_device:    str           = "cuda",
        warp_strength:   float         = 2.0,
        art_style:       str           = "hyperrealism",
    ):
        """
        Args:
            checkpoint_dir: путь к папке с дообученным LoRA-чекпоинтом
                            (например "artifacts/models/dpo_checkpoints/cycle_015")
            mean_face_path: путь к JSON со статистикой выборки
                            (по умолчанию "configs/mean_face.json")
            gen_device:     GPU для генерации ("cuda", "cuda:0", ...)
            judge_device:   GPU для судьи ("cuda", "cuda:1", ...)
            warp_strength:  сила деформации лица (рекомендуется 1.5–2.5)
            art_style:      стиль промпта — "hyperrealism", "3d_pixar", "digital_art"
        """
        self.gen_device    = gen_device
        self.judge_device  = judge_device
        self.warp_strength = warp_strength
        self.art_style     = art_style

        # Путь к статистике выборки
        _mean_face = (mean_face_path)

        # Агрегатор из feature_extraction.py
        print("Инициализация MediaPipe агрегатора...")
        if not os.path.exists(_mean_face):
            raise FileNotFoundError(
                f"Файл статистики не найден: {_mean_face}\n"
                "Запустите notebooks/feature_extraction.ipynb для его создания."
            )
        self.aggregator = MediaPipeFaceAggregator(mean_face_path=_mean_face)
        self.groups     = get_canonical_groups()

        # InsightFace
        self.face_app = load_insightface(gen_device)

        # SD Pipeline
        _ckpt = (checkpoint_dir) if checkpoint_dir else None
        self.pipe = load_pipeline(_ckpt, device=gen_device)

        # Судья
        self._judge_model: Optional[object] = None
        self._judge_proc:  Optional[object] = None

        print("Генератор инициализирован!")

    # Управление судьёй

    def _ensure_judge(self) -> None:
        if self._judge_model is None:
            self._judge_model, self._judge_proc = load_judge(self.judge_device)

    def _release_judge(self) -> None:
        if self._judge_model is not None:
            try:
                self._judge_model.to("cpu")
            except Exception:
                pass
            del self._judge_model, self._judge_proc
            self._judge_model = None
            self._judge_proc  = None
            gc.collect()
            torch.cuda.empty_cache()
            print("Судья выгружен из памяти")

    # ── Основной метод генерации ──────────────────────────────────────

    def generate(
        self,
        image_path:          str,
        warp_strength:       Optional[float] = None,
        top_k:               int             = 5,
        gender:              str             = "human",
        release_judge_after: bool            = True,
    ) -> Dict:
        """
        Принимает путь к фото, возвращает три варианта шаржа с оценкой VLLM.

        Args:
            image_path:           путь к входному изображению
            warp_strength:        сила деформации (переопределяет значение из __init__)
            top_k:                сколько топ-отклонений использовать для промпта
            gender:               пол для промпта ("human", "man", "woman", ...)
            release_judge_after:  выгружать ли судью из VRAM после оценки

        Returns:
            dict со следующими ключами:
                caricature_A/B/C  — три варианта шаржа (PIL Image)
                warped            — варп-трансформация входного фото (PIL Image)
                original_crop     — кроп оригинала по bbox лица (PIL Image)
                canny             — карта контуров для ControlNet (PIL Image)
                best              — лучший вариант по оценке судьи (PIL Image)
                best_label        — метка лучшего варианта: 'A', 'B' или 'C'
                ranking           — список ['X','Y','Z'] от лучшего к худшему
                scores            — {'A':1, 'B':3, 'C':2} (1 = лучший)
                judge_response    — полный текст оценки от Qwen2-VL
                prompt            — использованный positive промпт
                negative_prompt   — использованный negative промпт
                deviations        — словарь отклонений признаков лица
        """
        image_path = str(image_path)
        strength   = warp_strength if warp_strength is not None else self.warp_strength

        # ── 1. Варп через feature_extraction.py ──────────────────────
        print(f"Варп-трансформация (strength={strength})...")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")
        h, w = img_bgr.shape[:2]

        caricature_params = get_caricature_parameters(
            self.aggregator,
            self.groups,
            image_path,
            strength=strength,
        )
        warped_np  = self.aggregator.warp_face(
            image_path,
            caricature_params=caricature_params,
            target_size=(h, w),
        )
        warped_pil  = Image.fromarray(warped_np)           # RGB
        original_pil = Image.fromarray(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        )

        # ── 2. Промпт через feature_extraction.py ────────────────────
        print("📝 Построение промпта...")
        deviations     = self.aggregator.compare_with_mean(image_path)
        positive_prompt = generate_dynamic_caricature_prompt(
            deviations,
            top_k=top_k,
            gender=gender,
            art_style=self.art_style,
        )
        print(f"   → {positive_prompt[:100]}...")

        # ── 3. InsightFace: детекция лица ────────────────────────────
        faces = self.face_app.get(img_bgr)
        if not faces:
            raise ValueError(
                f"InsightFace не обнаружил лицо на изображении: {image_path}"
            )
        face = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )[0]

        # ── 4. Генерация трёх вариантов ──────────────────────────────
        print("Генерация вариантов A / B / C...")
        variants = _generate_variants(
            pipe            = self.pipe,
            warped_pil      = warped_pil,
            face            = face,
            original_pil    = original_pil,
            positive_prompt = positive_prompt,
            negative_prompt = NEGATIVE_PROMPT,
            device          = self.gen_device,
        )

        # ── 5. Оценка судьёй ─────────────────────────────────────────
        print("Оценка вариантов (Qwen2-VL)...")
        self._ensure_judge()
        judgment = _run_judge(
            model         = self._judge_model,
            processor     = self._judge_proc,
            original_crop = variants["original_crop"],
            car_a         = variants["caricature_A"],
            car_b         = variants["caricature_B"],
            car_c         = variants["caricature_C"],
        )
        if release_judge_after:
            self._release_judge()

        best_label = judgment["ranking"][0]
        print(
            f"Готово! Победитель: {best_label} | "
            f"Ранжирование: {' > '.join(judgment['ranking'])}"
        )

        return {
            "caricature_A":   variants["caricature_A"],
            "caricature_B":   variants["caricature_B"],
            "caricature_C":   variants["caricature_C"],
            "warped":         warped_pil,
            "original_crop":  variants["original_crop"],
            "canny":          variants["canny"],
            "best":           variants[f"caricature_{best_label}"],
            "best_label":     best_label,
            "ranking":        judgment["ranking"],
            "scores":         judgment["scores"],
            "judge_response": judgment["full_response"],
            "prompt":         positive_prompt,
            "negative_prompt":NEGATIVE_PROMPT,
            "deviations":     deviations,
        }


 
#  Функциональный API (быстрый вызов без явного создания класса)
 

_GENERATOR_INSTANCE: Optional[CaricatureGenerator] = None


def generate_caricature(
    image_path:     str,
    checkpoint_dir: Optional[str] = None,
    mean_face_path: Optional[str] = None,
    warp_strength:  float          = 2.0,
    gen_device:     str            = "cuda",
    judge_device:   str            = "cuda",
    art_style:      str            = "hyperrealism",
    gender:         str            = "human",
) -> Dict:
    """
    Функциональный API — создаёт глобальный экземпляр при первом вызове
    и переиспользует его при последующих.

    Пример:
        from caricature_generator import generate_caricature

        result = generate_caricature(
            "photo.jpg",
            checkpoint_dir="artifacts/models/dpo_checkpoints/cycle_015",
        )
        result["best"].save("output.png")
        print("Лучший вариант:", result["best_label"])
        print("Ранжирование:",   result["ranking"])
        print("Судья:\n",        result["judge_response"])
    """
    global _GENERATOR_INSTANCE

    if _GENERATOR_INSTANCE is None:
        _GENERATOR_INSTANCE = CaricatureGenerator(
            checkpoint_dir = checkpoint_dir,
            mean_face_path = mean_face_path,
            gen_device     = gen_device,
            judge_device   = judge_device,
            warp_strength  = warp_strength,
            art_style      = art_style,
        )

    return _GENERATOR_INSTANCE.generate(
        image_path    = image_path,
        warp_strength = warp_strength,
        gender        = gender,
    )