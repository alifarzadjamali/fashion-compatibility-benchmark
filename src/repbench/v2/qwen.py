"""Image-only adaptation of the pinned Qwen embedding reference implementation.

Reference: Qwen/Qwen3-VL-Embedding-2B, revision
9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda, scripts/qwen3_vl_embedding.py
(Apache-2.0). Same system instruction, image sizing, chat template and last-token
pooling. Reject preprocessing failures instead of silently embedding NULL text.
"""

from __future__ import annotations

from typing import ClassVar

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)


class QwenEmbeddingModel(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping: ClassVar[dict] = {}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    @property
    def visual(self):
        return self.model.visual

    def forward(self, **kwargs):
        return self.model(**kwargs)


def load_qwen(spec, dtype):
    cls = (
        QwenEmbeddingModel if spec.backend == "qwen_embedding" else Qwen3VLForConditionalGeneration
    )
    model = cls.from_pretrained(
        spec.checkpoint, revision=spec.revision, dtype=dtype, attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(
        spec.checkpoint,
        revision=spec.revision,
        padding_side="right" if spec.backend == "qwen_embedding" else "left",
    )
    return model, processor


def prepare_qwen(processor, images, *, embedding):
    from qwen_vl_utils import process_vision_info

    from repbench.v2.images import QWEN_MAX_PIXELS, QWEN_MIN_PIXELS

    conversations = []
    for image in images:
        content = [
            {
                "type": "image",
                "image": image,
                "min_pixels": QWEN_MIN_PIXELS,
                "max_pixels": QWEN_MAX_PIXELS,  # uniform 384x384 budget (decision F)
            }
        ]
        if embedding:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Represent the user's input."}],
                }
            ]
        else:
            conversation = []
            content.append({"type": "text", "text": "Choose a letter: A or B. Answer:"})
        conversation.append({"role": "user", "content": content})
        conversations.append(conversation)
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    pixels, videos = process_vision_info(conversations, image_patch_size=16)
    if pixels is None or videos is not None:
        raise ValueError("Image-only processing failed")
    inputs = processor(text=text, images=pixels, padding=True, do_resize=False, return_tensors="pt")
    if inputs["input_ids"].shape[1] > 8192:
        raise ValueError("Input exceeds pinned max_length; refusing silent image-token truncation")
    return dict(inputs)
