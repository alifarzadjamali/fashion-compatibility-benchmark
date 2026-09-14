from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url

from repbench.encoders.registry import PRIMARY_ENCODERS

ROWS = [
    {
        "model_key": "resnet50",
        "model_checkpoint": "torchvision ResNet50_Weights.IMAGENET1K_V2",
        "family": "supervised CNN",
        "parameter_count": 25557032,
        "native_embedding_dimension": 2048,
        "official_input_resolution": 224,
        "preprocessing": "resize 232 bilinear; center crop 224; ImageNet mean/std",
        "training_data_provenance": "ImageNet-1K supervised training; V2 torchvision recipe",
        "known_polyvore_training_exposure": "no",
        "possible_polyvore_exposure": "unlikely",
        "known_polyvore_evaluation_exposure": "no",
        "license_usage_terms": "torchvision code BSD-3-Clause; weight-specific terms not stated; ImageNet data has separate terms",
        "source": "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html",
        "uncertainty": "No distinct license statement for the pretrained weights in the weight metadata.",
    },
    {
        "model_key": "dinov3_vitl16",
        "model_checkpoint": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "family": "self-supervised vision transformer",
        "parameter_count": 303129600,
        "native_embedding_dimension": 1024,
        "official_input_resolution": 224,
        "preprocessing": "squash resize 224 bilinear; ImageNet mean/std; normalized CLS pooler",
        "training_data_provenance": "ViT-L distilled from DINOv3 ViT-7B; LVD-1689M curated from 17B public Instagram images",
        "known_polyvore_training_exposure": "no",
        "possible_polyvore_exposure": "possible incidental web/repost exposure",
        "known_polyvore_evaluation_exposure": "no",
        "license_usage_terms": "DINOv3 License; research publication acknowledgement and derivative redistribution conditions apply",
        "source": "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "uncertainty": "LVD-1689M image membership is not public, so incidental Polyvore-image exposure cannot be excluded.",
    },
    {
        "model_key": "clip_vitl14_336",
        "model_checkpoint": "openai/clip-vit-large-patch14-336",
        "family": "generic contrastive vision-language transformer",
        "parameter_count": 427616513,
        "native_embedding_dimension": 768,
        "official_input_resolution": 336,
        "preprocessing": "resize and center crop 336 bicubic; OpenAI CLIP mean/std; projected image embedding",
        "training_data_provenance": "OpenAI CLIP family trained on 400M public internet image-text pairs",
        "known_polyvore_training_exposure": "no",
        "possible_polyvore_exposure": "possible incidental web exposure",
        "known_polyvore_evaluation_exposure": "no",
        "license_usage_terms": "OpenAI CLIP repository MIT; Hugging Face checkpoint card does not state separate weight terms",
        "source": "https://huggingface.co/openai/clip-vit-large-patch14-336",
        "uncertainty": "Exact training set is undisclosed and the checkpoint card itself says training data are unknown.",
    },
    {
        "model_key": "siglip2_b16_384",
        "model_checkpoint": "google/siglip2-base-patch16-384",
        "family": "generic sigmoid-loss vision-language transformer",
        "parameter_count": 375479810,
        "native_embedding_dimension": 768,
        "official_input_resolution": 384,
        "preprocessing": "squash resize 384 bilinear; scale to [-1,1]; projected image embedding",
        "training_data_provenance": "SigLIP2 pretrained on WebLI",
        "known_polyvore_training_exposure": "no",
        "possible_polyvore_exposure": "possible incidental web exposure",
        "known_polyvore_evaluation_exposure": "no",
        "license_usage_terms": "Apache-2.0 per official checkpoint metadata",
        "source": "https://huggingface.co/google/siglip2-base-patch16-384",
        "uncertainty": "WebLI membership is not fully enumerated; incidental Polyvore exposure cannot be excluded.",
    },
    {
        "model_key": "fashionclip2",
        "model_checkpoint": "patrickjohncyh/fashion-clip",
        "family": "fashion-specialized contrastive vision-language transformer",
        "parameter_count": 151277439,
        "native_embedding_dimension": 512,
        "official_input_resolution": 224,
        "preprocessing": "resize and center crop 224 bicubic; CLIP mean/std; projected image embedding",
        "training_data_provenance": "LAION CLIP ViT-B/32 base fine-tuned on >800K Farfetch product image-text pairs",
        "known_polyvore_training_exposure": "no",
        "possible_polyvore_exposure": "possible only through undisclosed LAION web membership",
        "known_polyvore_evaluation_exposure": "no documented evaluation exposure for training; model card reports other fashion benchmarks",
        "license_usage_terms": "MIT per checkpoint metadata; model card says deployment was not assessed",
        "source": "https://huggingface.co/patrickjohncyh/fashion-clip",
        "uncertainty": "Farfetch data are not public; LAION pretraining can contain incidental web duplicates.",
    },
    {
        "model_key": "marqo_fashionsiglip",
        "model_checkpoint": "Marqo/marqo-fashionSigLIP",
        "family": "fashion-specialized sigmoid-loss vision-language transformer",
        "parameter_count": 203155968,
        "native_embedding_dimension": 768,
        "official_input_resolution": 224,
        "preprocessing": "squash resize 224 bicubic; mean/std 0.5; normalized OpenCLIP image embedding",
        "training_data_provenance": "ViT-B/16 SigLIP WebLI base; fashion GCL fine-tuning corpus provenance not fully disclosed",
        "known_polyvore_training_exposure": "no documented training use",
        "possible_polyvore_exposure": "possibly_exposed",
        "known_polyvore_evaluation_exposure": "yes; official repository evaluates on a Polyvore retrieval dataset",
        "license_usage_terms": "Apache-2.0 per checkpoint metadata",
        "source": "https://huggingface.co/Marqo/marqo-fashionSigLIP",
        "uncertainty": "Fine-tuning image sources are insufficiently disclosed to rule out Polyvore training exposure.",
    },
    {
        "model_key": "gr_lite",
        "model_checkpoint": "srpone/gr-lite",
        "family": "fashion-retrieval vision transformer",
        "parameter_count": 303129600,
        "native_embedding_dimension": 1024,
        "official_input_resolution": 336,
        "preprocessing": "squash resize 336 bilinear; ImageNet mean/std; L2-normalized CLS embedding",
        "training_data_provenance": "DINOv3 ViT-L/16 derivative; paper reports 1.3M open-source plus 0.5M in-house fashion images",
        "known_polyvore_training_exposure": "no documented Polyvore source",
        "possible_polyvore_exposure": "possible via crawled/in-house data and DINOv3 upstream web data",
        "known_polyvore_evaluation_exposure": "no documented Polyvore evaluation",
        "license_usage_terms": "DINOv3 License governs weights as a derivative, despite Apache-2.0 Hugging Face front matter",
        "source": "https://arxiv.org/html/2601.14706#S4.SS1",
        "uncertainty": "Release code/output (1024-d CLS, no projection module) conflicts with paper text describing a 512-d projection; paper table/model card use 1024-d. Training-data totals also conflict within the paper.",
    },
]

WEIGHT_FILES = {
    "dinov3_vitl16": "model.safetensors",
    "clip_vitl14_336": "pytorch_model.bin",
    "siglip2_b16_384": "model.safetensors",
    "fashionclip2": "model.safetensors",
    "marqo_fashionsiglip": "open_clip_model.safetensors",
    "gr_lite": "model.safetensors",
}

REMOTE_CODE = {
    "marqo_fashionsiglip": ["marqo_fashionSigLIP.py"],
    "gr_lite": ["configuration_gr_lite.py", "modeling_gr_lite.py"],
}


def main() -> None:
    token = get_token()
    if not token:
        raise RuntimeError("No local Hugging Face token found")
    api = HfApi(token=token)
    account = api.whoami()["name"]
    access_rows = []
    for key, spec in PRIMARY_ENCODERS.items():
        if spec.backend == "torchvision":
            access_rows.append(
                {
                    "model_key": key,
                    "checkpoint": spec.checkpoint,
                    "revision": spec.revision,
                    "accessible": True,
                    "weight_file": "resnet50-11ad3fa6.pth",
                    "weight_size": 102530333,
                    "safetensors": False,
                    "remote_code_inspected": False,
                    "remote_code_executed": False,
                }
            )
            continue
        info = api.model_info(spec.checkpoint, revision=spec.revision, files_metadata=True)
        if info.sha != spec.revision:
            raise RuntimeError(f"Revision mismatch for {key}: {info.sha}")
        weight_file = WEIGHT_FILES[key]
        metadata = get_hf_file_metadata(
            hf_hub_url(spec.checkpoint, weight_file, revision=spec.revision), token=token
        )
        code_hashes = {}
        for filename in REMOTE_CODE.get(key, []):
            path = hf_hub_download(
                spec.checkpoint, filename, revision=spec.revision, token=token
            )
            code_hashes[filename] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        access_rows.append(
            {
                "model_key": key,
                "checkpoint": spec.checkpoint,
                "revision": info.sha,
                "accessible": True,
                "weight_file": weight_file,
                "weight_size": metadata.size,
                "safetensors": weight_file.endswith(".safetensors"),
                "remote_code_inspected": bool(code_hashes),
                "remote_code_executed": spec.trust_remote_code,
                "remote_code_sha256": code_hashes,
            }
        )
    output = {
        "audit_time_utc": datetime.now(UTC).isoformat(),
        "authenticated_account": account,
        "all_accessible": all(row["accessible"] for row in access_rows),
        "models": access_rows,
    }
    Path("artifacts/checkpoint_access_audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    revisions = {key: spec.revision for key, spec in PRIMARY_ENCODERS.items()}
    frame = pd.DataFrame(ROWS)
    frame.insert(2, "exact_revision", frame.model_key.map(revisions))
    frame.to_csv("artifacts/model_audit.csv", index=False)


if __name__ == "__main__":
    main()
