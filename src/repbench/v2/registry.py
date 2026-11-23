"""Candidate v2 roster. Immutable revisions, separate from the frozen v1 roster."""

from dataclasses import replace

from repbench.encoders.registry import PRIMARY_ENCODERS, EncoderSpec

V2_ENCODERS = {
    key: replace(value, cuda_dtype="bfloat16") for key, value in PRIMARY_ENCODERS.items()
}
V2_ENCODERS.update(
    {
        "dinov3_vitl16_336": replace(
            V2_ENCODERS["dinov3_vitl16"], model_key="dinov3_vitl16_336", input_resolution=336
        ),
        "pe_core_l14_336": EncoderSpec(
            "pe_core_l14_336",
            "timm/PE-Core-L-14-336",
            "8eff41b3f687e50a323662c2dda5eb3588c6dd35",
            "generic_vision_language",
            336,
            1024,
            backend="open_clip_hub",
            cuda_dtype="bfloat16",
        ),
        "siglip2_so400m_384": EncoderSpec(
            "siglip2_so400m_384",
            "google/siglip2-so400m-patch16-384",
            "dd658faac399427308559e2c3ac1e99cbe43845d",
            "generic_vision_language",
            384,
            1152,
            cuda_dtype="bfloat16",
        ),
        "siglip_b16_224": EncoderSpec(
            "siglip_b16_224",
            "timm/ViT-B-16-SigLIP",
            "41f575766f40e752fdd1383e9565b7f02388c1c4",
            "generic_vision_language",
            224,
            768,
            backend="open_clip_hub",
            cuda_dtype="bfloat16",
        ),
        "openclip_b32_laion2b": EncoderSpec(
            "openclip_b32_laion2b",
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "1a25a446712ba5ee05982a381eed697ef9b435cf",
            "generic_vision_language",
            224,
            512,
            backend="open_clip_hub",
            cuda_dtype="bfloat16",
        ),
        "marqo_fashionclip": EncoderSpec(
            "marqo_fashionclip",
            "Marqo/marqo-fashionCLIP",
            "44f4c655124ed71e90cbf528d82f508d69d8a81b",
            "fashion_vision_language",
            224,
            512,
            backend="open_clip_hub",
            cuda_dtype="bfloat16",
        ),
        "qwen3_vl_emb_2b": EncoderSpec(
            "qwen3_vl_emb_2b",
            "Qwen/Qwen3-VL-Embedding-2B",
            "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
            "mllm_embedding",
            0,
            2048,
            backend="qwen_embedding",
            cuda_dtype="bfloat16",
        ),
        "qwen3_vl_emb_8b": EncoderSpec(
            "qwen3_vl_emb_8b",
            "Qwen/Qwen3-VL-Embedding-8B",
            "2c4565515e0f265c6511776e7193b22c0968ddc7",
            "mllm_embedding",
            0,
            4096,
            backend="qwen_embedding",
            cuda_dtype="bfloat16",
        ),
    }
)
V2_ENCODERS["marqo_fashionsiglip"] = replace(
    V2_ENCODERS["marqo_fashionsiglip"], backend="open_clip_hub"
)
ZERO_SHOT = EncoderSpec(
    "qwen3_vl_8b_zeroshot",
    "Qwen/Qwen3-VL-8B-Instruct",
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "zero_shot_mllm",
    0,
    0,
    backend="qwen_generation",
    cuda_dtype="bfloat16",
)
