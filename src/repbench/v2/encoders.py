"""Uniform bf16/SDPA interface, including CPU preprocessing for isolated timing."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

from repbench.utils.paths import configure_cache_environment

configure_cache_environment()

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image


def gr_sdpa(attention, x):
    batch, length, _ = x.shape

    def project(layer):
        return (
            layer(x).reshape(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        )

    values = F.scaled_dot_product_attention(
        project(attention.q_proj),
        project(attention.k_proj),
        project(attention.v_proj),
        dropout_p=0.0,
    )
    return attention.o_proj(values.transpose(1, 2).reshape(batch, length, -1))


class V2Encoder:
    def __init__(self, spec, *, device="cuda", dtype=torch.bfloat16):
        self.spec, self.device, self.dtype = spec, torch.device(device), dtype
        self.processor = self.transform = None
        self.attention_method = "sdpa"
        if spec.backend == "torchvision":
            from torchvision.models import ResNet50_Weights, resnet50

            weights = ResNet50_Weights.IMAGENET1K_V2
            full = resnet50(weights=weights)
            self.model = torch.nn.Sequential(*list(full.children())[:-1])
            self.transform = weights.transforms()
            self.attention_method = "not applicable (CNN)"
        elif spec.backend == "open_clip_hub":
            import open_clip

            config_path = hf_hub_download(
                spec.checkpoint, "open_clip_config.json", revision=spec.revision
            )
            weights = hf_hub_download(
                spec.checkpoint, "open_clip_model.safetensors", revision=spec.revision
            )
            config = json.loads(Path(config_path).read_text())
            # Register the pinned architecture locally: hf-hub default loading would
            # resolve floating main in some open_clip versions.
            name = "repbench_" + spec.model_key + "_" + spec.revision
            local = Path(os.environ["HF_HOME"]) / "repbench_configs" / (name + ".json")
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                local.write_text(json.dumps(config["model_cfg"]))
            open_clip.add_model_config(local)
            pre = config["preprocess_cfg"]
            self.model, _, self.transform = open_clip.create_model_and_transforms(
                name,
                pretrained=weights,
                precision="fp32",
                device="cpu",
                image_mean=pre.get("mean"),
                image_std=pre.get("std"),
                image_interpolation=pre.get("interpolation", "bicubic"),
                image_resize_mode=pre.get("resize_mode", "shortest"),
            )
            # open_clip's attention uses torch SDPA when available.
            for module in self.model.modules():
                if hasattr(module, "fused_attn"):
                    module.fused_attn = True
        elif spec.backend.startswith("qwen_"):
            from repbench.v2.qwen import load_qwen

            self.model, self.processor = load_qwen(spec, dtype)
        else:
            from transformers import AutoImageProcessor, AutoModel

            if spec.model_key == "gr_lite":
                from repbench.encoders.hf_encoder import _GRLiteProcessor

                self.processor = _GRLiteProcessor()
            else:
                self.processor = AutoImageProcessor.from_pretrained(
                    spec.checkpoint,
                    revision=spec.revision,
                    use_fast=spec.model_key.startswith("dinov3_"),
                )
                if spec.model_key == "dinov3_vitl16_336":
                    self.processor.size = {"height": 336, "width": 336}
            kwargs = {
                "revision": spec.revision,
                "dtype": dtype,
                "trust_remote_code": spec.trust_remote_code,
                "use_safetensors": spec.use_safetensors,
            }
            if spec.model_key != "gr_lite":
                kwargs["attn_implementation"] = "sdpa"
            self.model = AutoModel.from_pretrained(spec.checkpoint, **kwargs)
            if spec.model_key == "gr_lite":
                for layer in self.model.layer:
                    layer.attention.forward = types.MethodType(gr_sdpa, layer.attention)
                self.attention_method = (
                    "sdpa; algebraic replacement of pinned GR-Lite eager attention"
                )
        self.model = self.model.eval().requires_grad_(False).to(self.device, dtype=dtype)

    def prepare(self, images):
        """CPU-only decode/resize/normalize, no device transfers."""
        decoded = []
        for image in images:
            if isinstance(image, Image.Image):
                decoded.append(image.convert("RGB"))
            else:
                with Image.open(image) as opened:
                    decoded.append(opened.convert("RGB"))
        if self.spec.backend.startswith("qwen_"):
            from repbench.v2.qwen import prepare_qwen

            return prepare_qwen(
                self.processor, decoded, embedding=self.spec.backend == "qwen_embedding"
            )
        if self.transform is not None:
            return {"pixel_values": torch.stack([self.transform(i) for i in decoded])}
        return dict(self.processor(images=decoded, return_tensors="pt"))

    def to_device(self, inputs):
        return {
            k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }

    def forward(self, inputs):
        if self.spec.backend == "torchvision":
            result = self.model(inputs["pixel_values"]).flatten(1)
        elif self.spec.backend == "open_clip_hub":
            result = self.model.encode_image(inputs["pixel_values"])
        elif self.spec.backend == "qwen_embedding":
            outputs = self.model(**inputs, use_cache=False)
            mask = inputs["attention_mask"]
            positions = mask.shape[1] - 1 - mask.flip(1).argmax(1)
            result = outputs.last_hidden_state[
                torch.arange(len(mask), device=mask.device), positions
            ]
        elif self.spec.backend == "qwen_generation":
            outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
            return torch.log_softmax(outputs.logits[:, -1].float(), dim=-1)
        elif hasattr(self.model, "get_image_features"):
            result = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            if not isinstance(result, torch.Tensor):
                result = result.pooler_output
        else:
            outputs = self.model(**inputs)
            result = outputs.pooler_output
            if result is None:
                result = outputs.last_hidden_state[:, 0]
        return F.normalize(result.float(), dim=-1)

    def image_parameter_count(self):
        if self.spec.backend.startswith("qwen_"):
            return sum(p.numel() for p in self.model.visual.parameters())
        if hasattr(self.model, "visual"):
            return sum(p.numel() for p in self.model.visual.parameters())
        if hasattr(self.model, "vision_model"):
            count = sum(p.numel() for p in self.model.vision_model.parameters())
            if hasattr(self.model, "visual_projection"):
                count += sum(p.numel() for p in self.model.visual_projection.parameters())
            return count
        return sum(p.numel() for p in self.model.parameters())
