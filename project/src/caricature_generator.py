"""
caricature_generator.py
Генерация шаржей: варп - промпт - SD+ControlNet+IP-Adapter - Qwen2-VL судья.
"""

import gc
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

from .config import get_config
from .feature_extraction import (
    MediaPipeFaceAggregator,
    generate_dynamic_caricature_prompt,
    get_canonical_groups,
    get_caricature_parameters,
)

# Конфиг на уровне модуля — читается один раз при импорте
_cfg  = get_config()
_gen  = _cfg.generation
_mdls = _cfg.models

# Гиперпараметры из конфига в модульные константы для удобства
CROP_SIZE         = _cfg.crop_size
DENOISE_STRENGTH  = _cfg.denoise_strength
GUIDANCE_SCALE    = _cfg.guidance_scale
NUM_STEPS         = _cfg.num_steps
SEED_A, SEED_B, SEED_C = (int(s) for s in _cfg.seeds[:3])
IP_ADAPTER_SCALE  = _cfg.ip_adapter_scale
LORA_SCALE        = _cfg.lora_scale
CANNY_LO          = _cfg.canny_lo
CANNY_HI          = _cfg.canny_hi
CONTROLNET_WEIGHT = _cfg.controlnet_weight
CONTROLNET_START  = _cfg.controlnet_start
CONTROLNET_END    = _cfg.controlnet_end
JUDGE_MAX_TOKENS  = _cfg.judge_max_new_tokens
NEGATIVE_PROMPT   = _cfg.negative_prompt
_DEFAULT_TOP_K    = _cfg.prompt_top_k
_DEFAULT_GENDER   = _cfg.default_gender

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


def _get_canny(image_pil: Image.Image) -> Image.Image:
    """Карта контуров Canny из PIL-изображения."""
    gray  = np.array(image_pil.convert("L"))
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def _square_crop(
    image_pil: Image.Image,
    bbox:      np.ndarray,
    padding:   float = 0.30,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Квадратный кроп вокруг bbox с отступом, ресайз до CROP_SIZE."""
    iw, ih          = image_pil.size
    x1, y1, x2, y2 = bbox.astype(int)
    pad  = int(max(x2 - x1, y2 - y1) * padding)
    cx   = (x1 + x2) // 2
    cy   = (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2 + pad
    box  = (
        max(0, cx - half), max(0, cy - half),
        min(iw - 1, cx + half), min(ih - 1, cy + half),
    )
    return image_pil.crop(box).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS), box


def _build_face_embeds(face, device: str) -> torch.Tensor:
    """IP-Adapter FaceID embeds: (2, 1, 512) — uncond + cond."""
    cond   = torch.from_numpy(face.normed_embedding).unsqueeze(0).unsqueeze(0)
    uncond = torch.zeros_like(cond)
    return torch.cat([uncond, cond], dim=0).to(dtype=torch.float16, device=device)


def _parse_ranking(response: str) -> List[str]:
    """Парсит RANKING: X > Y > Z из ответа судьи, fallback — по порядку упоминания."""
    match = re.search(
        r"RANKING\s*:\s*([ABC])\s*>\s*([ABC])\s*>\s*([ABC])",
        response, re.IGNORECASE,
    )
    if match:
        return [match.group(i).upper() for i in range(1, 4)]

    letters = re.findall(r"\b([ABC])\b", response.upper())
    seen, ranking = set(), []
    for letter in letters:
        if letter not in seen:
            seen.add(letter)
            ranking.append(letter)
    for missing in ["A", "B", "C"]:
        if missing not in ranking:
            ranking.append(missing)
    print(f"  [judge] тег RANKING не найден, fallback: {ranking}")
    return ranking


def load_pipeline(checkpoint_dir: Optional[str] = None, device: str = "cuda"):
    """
    Загружает SD 1.5 + ControlNet Canny + IP-Adapter FaceID.
    Model ID берутся из models.yaml. Если передан checkpoint_dir — применяет LoRA.
    """
    from diffusers import (
        ControlNetModel,
        EulerAncestralDiscreteScheduler,
        StableDiffusionControlNetImg2ImgPipeline,
    )

    hf = _mdls["huggingface"]
    gc.collect()
    torch.cuda.empty_cache()

    print("[1/3] Загрузка ControlNet Canny...")
    controlnet = ControlNetModel.from_pretrained(
        hf["controlnet"], torch_dtype=torch.float16, use_safetensors=True,
    ).to(device)

    print("[2/3] Загрузка Stable Diffusion 1.5...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        hf["sd_base"],
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print("[3/3] Загрузка IP-Adapter FaceID...")
    pipe.load_ip_adapter(
        hf["ip_adapter_repo"],
        subfolder=None,
        weight_name=hf["ip_adapter_weight"],
        image_encoder_folder=None,
    )
    pipe.load_lora_weights(hf["ip_adapter_repo"], weight_name=hf["ip_adapter_lora"])
    pipe.fuse_lora(lora_scale=LORA_SCALE)
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            print(f"Чекпоинт не найден: {ckpt_path}. Используем базовую модель.")
        else:
            from peft import PeftModel
            print(f"Применяем чекпоинт: {ckpt_path}")
            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(ckpt_path))
            pipe.unet = pipe.unet.merge_and_unload().to(device)

    print("Pipeline готов!")
    return pipe


def load_insightface(device: str = "cuda"):
    """Загружает InsightFace для face embedding. Параметры из models.yaml."""
    from insightface.app import FaceAnalysis

    hf        = _mdls["huggingface"]
    det_size  = int(_mdls["insightface"]["det_size"])
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith("cuda") else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name=hf["insightface_model"], providers=providers)
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    print("InsightFace загружен")
    return app


def load_judge(device: str = "cuda"):
    """Загружает Qwen2-VL для оценки шаржей. Параметры из generation.yaml."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    gc.collect()
    torch.cuda.empty_cache()

    judge_id = _mdls["huggingface"]["judge"]
    print(f"Загрузка судьи {judge_id}...")
    processor = AutoProcessor.from_pretrained(
        judge_id,
        min_pixels=_cfg.judge_min_pixels,
        max_pixels=_cfg.judge_max_pixels,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        judge_id, torch_dtype=torch.float16, device_map={"": device},
    )
    model.eval()
    print(f"Судья загружен на {device}")
    return model, processor


def _generate_variants(
    pipe:            object,
    warped_pil:      Image.Image,
    face:            object,
    original_pil:    Image.Image,
    positive_prompt: str,
    negative_prompt: str,
    device:          str,
) -> Dict[str, Image.Image]:
    """Генерирует три варианта шаржа (A/B/C) с разными seed'ами из конфига."""
    warped_crop, box = _square_crop(warped_pil, face.bbox)
    orig_crop        = original_pil.crop(box).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
    canny            = _get_canny(warped_crop)
    embeds           = _build_face_embeds(face, device)

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
        gen = torch.Generator(device=device).manual_seed(seed)
        variants[f"caricature_{label}"] = pipe(**common_kwargs, generator=gen).images[0]

    del embeds
    torch.cuda.empty_cache()

    variants["original_crop"] = orig_crop
    variants["warped_crop"]   = warped_crop
    variants["canny"]         = canny
    return variants


def _run_judge(
    model:         object,
    processor:     object,
    original_crop: Image.Image,
    car_a:         Image.Image,
    car_b:         Image.Image,
    car_c:         Image.Image,
) -> Dict:
    """Отправляет 4 изображения в Qwen2-VL и получает ранжирование вариантов."""
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=JUDGE_MAX_TOKENS,
            do_sample=False, temperature=None, top_p=None,
        )

    generated     = output_ids[0][inputs["input_ids"].shape[1]:]
    full_response = processor.decode(generated, skip_special_tokens=True).strip()
    ranking       = _parse_ranking(full_response)
    scores        = {label: rank + 1 for rank, label in enumerate(ranking)}

    return {"ranking": ranking, "scores": scores, "full_response": full_response}


class CaricatureGenerator:
    """
    Генератор шаржей на основе SD 1.5 + ControlNet + IP-Adapter + Qwen2-VL.
    Все дефолтные значения берутся из конфига — параметры __init__ позволяют
    переопределить их для конкретного экземпляра.

    Пример:
        gen = CaricatureGenerator()
        result = gen.generate("photo.jpg")
        result["best"].save("caricature.png")
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str]   = None,
        mean_face_path: Optional[str]   = None,
        gen_device:     Optional[str]   = None,
        judge_device:   Optional[str]   = None,
        warp_strength:  Optional[float] = None,
        art_style:      Optional[str]   = None,
    ):
        self.gen_device    = gen_device    or _cfg.app.generation_device
        self.judge_device  = judge_device  or _cfg.app.judge_device
        self.warp_strength = warp_strength if warp_strength is not None else _cfg.warp_strength
        self.art_style     = art_style     or _cfg.art_style

        _mean_face = mean_face_path or str(_cfg.mean_face_path)
        _ckpt      = checkpoint_dir or (str(_cfg.checkpoint_dir) if _cfg.checkpoint_dir else None)

        print("Инициализация MediaPipe агрегатора...")
        if not os.path.exists(_mean_face):
            raise FileNotFoundError(
                f"Файл статистики не найден: {_mean_face}\n"
                "Запустите notebooks/feature_extraction.ipynb для его создания."
            )
        self.aggregator = MediaPipeFaceAggregator(mean_face_path=_mean_face)
        self.groups     = get_canonical_groups()
        self.face_app   = load_insightface(self.gen_device)
        self.pipe       = load_pipeline(_ckpt, device=self.gen_device)

        self._judge_model = None
        self._judge_proc  = None
        print("Генератор инициализирован!")

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

    def generate(
        self,
        image_path:          str,
        warp_strength:       Optional[float] = None,
        top_k:               int             = _DEFAULT_TOP_K,
        gender:              str             = _DEFAULT_GENDER,
        release_judge_after: bool            = True,
    ) -> Dict:
        """
        Полный пайплайн генерации шаржа.

        Returns:
            dict: caricature_A/B/C, warped, original_crop, canny,
                  best, best_label, ranking, scores, judge_response,
                  prompt, negative_prompt, deviations
        """
        image_path = str(image_path)
        strength   = warp_strength if warp_strength is not None else self.warp_strength

        # 1. Варп
        print(f"Варп-трансформация (strength={strength})...")
        buf     = np.fromfile(image_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")   
        h, w = img_bgr.shape[:2]

        params     = get_caricature_parameters(self.aggregator, self.groups, image_path, strength)
        warped_np  = self.aggregator.warp_face(image_path, caricature_params=params, target_size=(h, w))
        warped_pil = Image.fromarray(warped_np)
        orig_pil   = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # 2. Промпт
        print("Построение промпта...")
        deviations      = self.aggregator.compare_with_mean(image_path)
        positive_prompt = generate_dynamic_caricature_prompt(
            deviations, top_k=top_k, gender=gender, art_style=self.art_style,
        )
        print(f"   → {positive_prompt[:100]}...")

        # 3. InsightFace: выбираем самое крупное лицо
        faces = self.face_app.get(img_bgr)
        if not faces:
            raise ValueError(f"InsightFace не обнаружил лицо: {image_path}")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # 4. Генерация A / B / C
        print("Генерация вариантов A / B / C...")
        variants = _generate_variants(
            pipe=self.pipe, warped_pil=warped_pil, face=face,
            original_pil=orig_pil, positive_prompt=positive_prompt,
            negative_prompt=NEGATIVE_PROMPT, device=self.gen_device,
        )

        # 5. Судья
        print("Оценка вариантов (Qwen2-VL)...")
        self._ensure_judge()
        judgment = _run_judge(
            model=self._judge_model, processor=self._judge_proc,
            original_crop=variants["original_crop"],
            car_a=variants["caricature_A"],
            car_b=variants["caricature_B"],
            car_c=variants["caricature_C"],
        )
        if release_judge_after:
            self._release_judge()

        best_label = judgment["ranking"][0]
        print(f"Победитель: {best_label} | Ранжирование: {' > '.join(judgment['ranking'])}")

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
            "negative_prompt": NEGATIVE_PROMPT,
            "deviations":     deviations,
        }


# Функциональный API

_GENERATOR_INSTANCE: Optional[CaricatureGenerator] = None


def generate_caricature(
    image_path:     str,
    checkpoint_dir: Optional[str]   = None,
    mean_face_path: Optional[str]   = None,
    warp_strength:  Optional[float] = None,
    gen_device:     Optional[str]   = None,
    judge_device:   Optional[str]   = None,
    art_style:      Optional[str]   = None,
    gender:         Optional[str]   = None,
) -> Dict:
    """
    Функциональный API — создаёт глобальный экземпляр при первом вызове.
    None-параметры подхватываются из конфига через CaricatureGenerator.__init__.
    """
    global _GENERATOR_INSTANCE
    if _GENERATOR_INSTANCE is None:
        _GENERATOR_INSTANCE = CaricatureGenerator(
            checkpoint_dir=checkpoint_dir,
            mean_face_path=mean_face_path,
            gen_device=gen_device,
            judge_device=judge_device,
            warp_strength=warp_strength,
            art_style=art_style,
        )
    return _GENERATOR_INSTANCE.generate(
        image_path=image_path,
        warp_strength=warp_strength,
        gender=gender or _DEFAULT_GENDER,
    )