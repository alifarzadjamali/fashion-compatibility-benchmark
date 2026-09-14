from repbench.encoders.registry import PRIMARY_MODEL_KEYS


def test_primary_roster_is_locked_and_excludes_dinov2():
    assert PRIMARY_MODEL_KEYS == (
        "resnet50",
        "dinov3_vitl16",
        "clip_vitl14_336",
        "siglip2_b16_384",
        "fashionclip2",
        "marqo_fashionsiglip",
        "gr_lite",
    )
    assert all("dinov2" not in model for model in PRIMARY_MODEL_KEYS)
