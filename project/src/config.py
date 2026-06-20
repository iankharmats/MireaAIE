"""
config.py
Единая точка загрузки всех конфигов проекта.
Использует python-dotenv + PyYAML + pydantic-settings.
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

# src/ → корень проекта
ROOT_DIR    = Path(__file__).parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"


def _resolve_env(value: str) -> str:
    """Заменяет ${VAR:default} на значение переменной окружения."""
    pattern = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")
    def _replace(m):
        var, default = m.group(1), m.group(2) or ""
        return os.environ.get(var, default)
    return pattern.sub(_replace, value)


def _resolve_dict(d: Any) -> Any:
    """Рекурсивно применяет _resolve_env ко всем строкам словаря."""
    if isinstance(d, dict):
        return {k: _resolve_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_resolve_dict(i) for i in d]
    if isinstance(d, str):
        return _resolve_env(d)
    return d


def _load_yaml(name: str) -> Dict:
    path = CONFIGS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _resolve_dict(raw)


class AppSettings(BaseSettings):
    """Настройки из .env — загружаются автоматически через pydantic-settings."""

    app_host:       str  = "0.0.0.0"
    app_port:       int  = 8000
    app_workers:    int  = 1
    app_reload:     bool = False
    app_log_level:  str  = "info"

    secret_key:     str  = "dev-secret-key"
    api_key:        str  = ""
    cors_origins:   str  = "http://localhost:3000"

    generation_gpu: int  = 0
    judge_gpu:      int  = 1
    metrics_gpu:    int  = 2

    dataset_dir:    str  = "dataset/Humans"
    output_dir:     str  = "outputs"
    checkpoint_dir: str  = ""
    mean_face_path: str  = "configs/mean_face.json"

    hf_token:       str  = ""
    hf_home:        str  = "artifacts/hf_cache"
    openai_api_key: str  = ""

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"
        extra             = "ignore"

    @property
    def generation_device(self) -> str:
        return f"cuda:{self.generation_gpu}" if self.generation_gpu >= 0 else "cpu"

    @property
    def judge_device(self) -> str:
        return f"cuda:{self.judge_gpu}" if self.judge_gpu >= 0 else "cpu"

    @property
    def metrics_device(self) -> str:
        return f"cuda:{self.metrics_gpu}" if self.metrics_gpu >= 0 else "cpu"

    @property
    def cors_origins_list(self):
        return [o.strip() for o in self.cors_origins.split(",")]


class Config:
    """
    Единая точка доступа ко всем конфигам.

    Использование:
        from src.config import get_config
        cfg = get_config()
        cfg.warp_strength        # из generation.yaml
        cfg.app.generation_device  # из .env
    """

    def __init__(self):
        self.app        = AppSettings()
        self.generation = _load_yaml("generation.yaml")
        self.models     = _load_yaml("models.yaml")
        self.app_yaml   = _load_yaml("app.yaml")
        self.metrics    = _load_yaml("metrics.yaml")

    # Warp
    @property
    def warp_strength(self) -> float:
        return float(self.generation["warp"]["strength"])

    @property
    def sigma_factor(self) -> float:
        return float(self.generation["warp"]["sigma_factor"])

    @property
    def grid_step(self) -> int:
        return int(self.generation["warp"]["grid_step"])

    @property
    def background_stabilization(self) -> bool:
        return bool(self.generation["warp"]["background_stabilization"])

    # Diffusion
    @property
    def crop_size(self) -> int:
        return int(self.generation["diffusion"]["crop_size"])

    @property
    def denoise_strength(self) -> float:
        return float(self.generation["diffusion"]["denoise_strength"])

    @property
    def guidance_scale(self) -> float:
        return float(self.generation["diffusion"]["guidance_scale"])

    @property
    def num_steps(self) -> int:
        return int(self.generation["diffusion"]["num_steps"])

    @property
    def seeds(self):
        return self.generation["diffusion"]["seeds"]

    # ControlNet
    @property
    def controlnet_weight(self) -> float:
        return float(self.generation["controlnet"]["weight"])

    @property
    def controlnet_start(self) -> float:
        return float(self.generation["controlnet"]["start"])

    @property
    def controlnet_end(self) -> float:
        return float(self.generation["controlnet"]["end"])

    @property
    def canny_lo(self) -> int:
        return int(self.generation["controlnet"]["canny_lo"])

    @property
    def canny_hi(self) -> int:
        return int(self.generation["controlnet"]["canny_hi"])

    # IP-Adapter
    @property
    def ip_adapter_scale(self) -> float:
        return float(self.generation["ip_adapter"]["scale"])

    @property
    def lora_scale(self) -> float:
        return float(self.generation["ip_adapter"]["lora_scale"])

    # Prompt
    @property
    def prompt_top_k(self) -> int:
        return int(self.generation["prompt"]["top_k"])

    @property
    def deviation_threshold(self) -> float:
        return float(self.generation["prompt"]["deviation_threshold"])

    @property
    def art_style(self) -> str:
        return self.generation["prompt"]["art_style"]

    @property
    def default_gender(self) -> str:
        return self.generation["prompt"]["gender"]

    @property
    def negative_prompt(self) -> str:
        return self.generation["prompt"]["negative_prompt"].strip()

    # Judge
    @property
    def judge_model_id(self) -> str:
        return self.generation["judge"]["model_id"]

    @property
    def judge_max_new_tokens(self) -> int:
        return int(self.generation["judge"]["max_new_tokens"])

    @property
    def judge_min_pixels(self) -> int:
        return int(self.generation["judge"]["min_pixels"])

    @property
    def judge_max_pixels(self) -> int:
        return int(self.generation["judge"]["max_pixels"])

    # Пути
    @property
    def mean_face_path(self) -> Path:
        return ROOT_DIR / self.app.mean_face_path

    @property
    def checkpoint_dir(self):
        d = self.app.checkpoint_dir
        return (ROOT_DIR / d) if d else None

    @property
    def output_dir(self) -> Path:
        p = ROOT_DIR / self.app.output_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def face_landmarker_path(self) -> Path:
        return ROOT_DIR / self.models["paths"]["face_landmarker"]

    @property
    def face_landmarker_url(self) -> str:
        return self.models["paths"]["face_landmarker_url"].strip()

    @property
    def feature_maps_dir(self) -> Path:
        p = ROOT_DIR / self.models["paths"]["feature_maps_dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def prompt_params_dir(self) -> Path:
        p = ROOT_DIR / self.models["paths"]["prompt_params_dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    # HuggingFace
    @property
    def sd_base(self) -> str:
        return self.models["huggingface"]["sd_base"]

    @property
    def controlnet_model(self) -> str:
        return self.models["huggingface"]["controlnet"]

    @property
    def ip_adapter_repo(self) -> str:
        return self.models["huggingface"]["ip_adapter_repo"]

    @property
    def ip_adapter_weight(self) -> str:
        return self.models["huggingface"]["ip_adapter_weight"]

    @property
    def ip_adapter_lora(self) -> str:
        return self.models["huggingface"]["ip_adapter_lora"]

    @property
    def insightface_model(self) -> str:
        return self.models["huggingface"]["insightface_model"]

    @property
    def insightface_det_size(self) -> int:
        return int(self.models["insightface"]["det_size"])

    # Метрики
    @property
    def artifact_thresholds(self) -> Dict:
        """Словарь порогов для compute_artifact_metrics()."""
        return dict(self.metrics["thresholds"]["artifacts"])

    @property
    def min_faceid_similarity(self) -> float:
        return float(self.metrics["thresholds"]["identity"]["min_faceid_similarity"])

    @property
    def clip_sim_range(self):
        clip = self.metrics["thresholds"]["clip"]
        return float(clip["sim_lo"]), float(clip["sim_hi"])


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Singleton — возвращает единственный экземпляр конфига."""
    return Config()