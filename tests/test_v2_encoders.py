from types import SimpleNamespace

import pytest
import torch
from PIL import UnidentifiedImageError

from repbench.encoders.registry import PRIMARY_ENCODERS
from repbench.v2.encoders import V2Encoder, gr_sdpa
from repbench.v2.registry import V2_ENCODERS, ZERO_SHOT


def test_roster_pins_and_true_parent_resolution():
    assert len(V2_ENCODERS) == 15
    for spec in [*V2_ENCODERS.values(), ZERO_SHOT]:
        assert spec.revision
        assert spec.cuda_dtype == "bfloat16"
        if spec.backend != "torchvision":
            assert len(spec.revision) == 40
    parent = V2_ENCODERS["dinov3_vitl16"]
    higher = V2_ENCODERS["dinov3_vitl16_336"]
    assert parent.checkpoint == higher.checkpoint
    assert parent.revision == higher.revision
    assert higher.input_resolution == 336
    assert PRIMARY_ENCODERS["clip_vitl14_336"].cuda_dtype == "float16"


def test_gr_sdpa_matches_pinned_eager_math():
    torch.manual_seed(42)
    attention = SimpleNamespace(num_heads=2, head_dim=4, scale=4**-0.5)
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        setattr(attention, name, torch.nn.Linear(8, 8))
    x = torch.randn(3, 7, 8)

    def project(layer):
        return layer(x).reshape(3, 7, 2, 4).permute(0, 2, 1, 3)

    q, k, v = [project(getattr(attention, n)) for n in ["q_proj", "k_proj", "v_proj"]]
    eager = ((q @ k.transpose(-2, -1)) * attention.scale).softmax(-1) @ v
    eager = attention.o_proj(eager.transpose(1, 2).reshape(3, 7, 8))
    torch.testing.assert_close(gr_sdpa(attention, x), eager, rtol=1e-5, atol=1e-6)


def test_hf_revision_resolution_and_sdpa_are_used(monkeypatch):
    import transformers

    calls = []
    processor = SimpleNamespace(size={"height": 224, "width": 224})
    monkeypatch.setattr(
        transformers.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: processor
    )

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.nn.Linear(3, 3)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load)
    spec = V2_ENCODERS["dinov3_vitl16_336"]
    V2Encoder(spec, device="cpu", dtype=torch.float32)
    assert processor.size == {"height": 336, "width": 336}
    assert calls[0][1]["revision"] == spec.revision
    assert calls[0][1]["attn_implementation"] == "sdpa"


def test_qwen_corrupt_input_fails_instead_of_embedding_null(tmp_path):
    encoder = object.__new__(V2Encoder)
    encoder.spec = V2_ENCODERS["qwen3_vl_emb_2b"]
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        encoder.prepare([bad])
