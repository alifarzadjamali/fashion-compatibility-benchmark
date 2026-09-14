from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderSpec:
    model_key: str
    checkpoint: str
    revision: str | None
    family: str
    input_resolution: int
    embedding_dim: int
    trust_remote_code: bool = False
    use_safetensors: bool | None = True
    backend: str = "huggingface"
    cuda_dtype: str = "float16"


PRIMARY_ENCODERS = {
    "resnet50": EncoderSpec(
        "resnet50",
        "torchvision ResNet50_Weights.IMAGENET1K_V2",
        "11ad3fa6",
        "supervised_cnn",
        224,
        2048,
        use_safetensors=None,
        backend="torchvision",
    ),
    "dinov3_vitl16": EncoderSpec(
        "dinov3_vitl16",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "ea8dc2863c51be0a264bab82070e3e8836b02d51",
        "self_supervised_vision",
        224,
        1024,
        cuda_dtype="bfloat16",
    ),
    "clip_vitl14_336": EncoderSpec(
        "clip_vitl14_336",
        "openai/clip-vit-large-patch14-336",
        "ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "generic_vision_language",
        336,
        768,
        use_safetensors=False,
    ),
    "siglip2_b16_384": EncoderSpec(
        "siglip2_b16_384",
        "google/siglip2-base-patch16-384",
        "f775b65a79762255128c981547af89addcfe0f88",
        "generic_vision_language",
        384,
        768,
    ),
    "fashionclip2": EncoderSpec(
        "fashionclip2",
        "patrickjohncyh/fashion-clip",
        "7e3ba62ce16b379a1ab479346b66f192e76f51b7",
        "fashion_vision_language",
        224,
        512,
    ),
    "marqo_fashionsiglip": EncoderSpec(
        "marqo_fashionsiglip",
        "Marqo/marqo-fashionSigLIP",
        "c56244cc94f92419e8369fa71efdaf403b124ce8",
        "fashion_vision_language",
        224,
        768,
        backend="open_clip_pinned",
    ),
    "gr_lite": EncoderSpec(
        "gr_lite",
        "srpone/gr-lite",
        "a8057f8aadaaab91670ad37949f6e1b5aa187e29",
        "fashion_retrieval",
        336,
        1024,
        trust_remote_code=True,
        cuda_dtype="bfloat16",
    ),
}

PRIMARY_MODEL_KEYS = tuple(PRIMARY_ENCODERS)


def get_encoder_spec(model_key: str) -> EncoderSpec:
    try:
        return PRIMARY_ENCODERS[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown primary model key: {model_key}") from exc
