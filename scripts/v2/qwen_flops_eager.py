"""Post-run sweep: GFLOPs/image for the two Qwen3-VL-Embedding encoders, whose profile
entry is null because torch's FlopCounterMode SDPA rule rejects grouped-query attention
shapes. Same image (profile ``images_sampled[0]``), same count_flops(), but attention is
dispatched to the mathematically equivalent eager implementation so it can be counted."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from profile_efficiency_v2 import count_flops  # noqa: E402
from repbench.v2.encoders import V2Encoder  # noqa: E402
from repbench.v2.registry import V2_ENCODERS  # noqa: E402

profile_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for key in ("qwen3_vl_emb_2b", "qwen3_vl_emb_8b"):
    image = json.loads((profile_dir / f"{key}.json").read_text())["images_sampled"][0]
    enc = V2Encoder(V2_ENCODERS[key])
    for module in enc.model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
    result = count_flops(enc, image)
    result.update(model_key=key, image=image, attention_for_counting="eager (counting only)")
    (out / f"{key}.json").write_text(json.dumps(result, indent=2))
    print(key, result.get("gflops_per_image"), result.get("error"))
    del enc
