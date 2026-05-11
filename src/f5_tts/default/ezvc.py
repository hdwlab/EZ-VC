import os

from cached_path import cached_path
from f5_tts.infer.utils_xeus import ApplyKmeans, load_xeus_model
from omegaconf import OmegaConf
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
)
from hydra.utils import get_class


def load_ezvc(device="cpu"):
    vocoder_name = "bigvgan"

    # load XEUS model
    xeus_model = load_xeus_model(device).eval()
    apply_kmeans = ApplyKmeans(device)
    vocoder = load_vocoder(vocoder_name=vocoder_name, device=device)

    # load TTS model
    model_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../configs/F5TTS_Base_EZ-VC.yaml"))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    ema_model = load_model(
        model_cls,
        model_arc,
        str(cached_path("hf://SPRINGLab/EZ-VC/model_2700000.safetensors")),
        mel_spec_type=vocoder_name,
        vocab_file=str(cached_path("hf://SPRINGLab/EZ-VC/vocab.txt")),
        device=device
    )

    return xeus_model, apply_kmeans, ema_model, vocoder