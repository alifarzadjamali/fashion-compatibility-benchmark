import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from repbench.v2.mllm_zeroshot import (
    LETTERS,
    MCQuestion,
    ZeroShotScorer,
    append_result,
    build_conversation,
    letter_token_ids,
    option_permutation,
    read_results,
    sample_question_ids,
    score_letters,
    summarize,
)

VOCAB = 64
TOKENS = {"A": 10, "B": 11, "C": 12, "D": 13, "E": 14, " A": 20, " B": 21, " C": 22, " D": 23}
TOKENS[" E"] = 24


class FakeTokenizer:
    def __init__(self, tokens=TOKENS):
        self.tokens = tokens

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.tokens[text]] if text in self.tokens else [1, 2]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        parts = []
        for item in conversation[0]["content"]:
            parts.append("<img>" if item["type"] == "image" else item["text"])
        return " ".join(parts) + " <assistant>"

    def __call__(self, text, images, **kwargs):
        self.calls.append((text, images, kwargs))
        return {"input_ids": torch.ones((1, 7 + len(images)), dtype=torch.long)}


class FakeModel:
    """Returns crafted logits: favours a chosen letter token."""

    def __init__(self, favoured_token):
        self.favoured_token = favoured_token
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        logits = torch.zeros((1, 1, VOCAB))
        logits[0, -1, self.favoured_token] = 5.0
        logits[0, -1, 21] = 3.0  # " B" (space variant) second
        return SimpleNamespace(logits=logits)


def images(n, value):
    return tuple(Image.new("RGB", (16, 16), (value + i, 0, 0)) for i in range(n))


def question(qid="7", correct=2):
    return MCQuestion(qid, images(3, 0), images(4, 100), correct)


def test_sample_ids_deterministic_and_bounded():
    a = sample_question_ids(10_000, 2000, 5)
    assert a == sample_question_ids(10_000, 2000, 5)
    assert a != sample_question_ids(10_000, 2000, 6)
    assert len(set(a)) == 2000 and a == sorted(a) and max(a) < 10_000
    assert sample_question_ids(50, 2000, 5) == list(range(50))


def test_permutation_determinism():
    p = option_permutation(1, "q1", 4)
    assert sorted(p) == [0, 1, 2, 3]
    assert p == option_permutation(1, "q1", 4)
    perms = {tuple(option_permutation(1, f"q{i}", 4)) for i in range(200)}
    assert len(perms) > 12  # genuinely randomized across questions
    seeds = [option_permutation(s, "q1", 4) for s in range(20)]
    assert len({tuple(p) for p in seeds}) > 1
    # Letter positions are balanced over many questions.
    counts = np.zeros(4)
    for i in range(4000):
        counts[option_permutation(3, str(i), 4).index(0)] += 1
    assert counts.min() > 850


def test_prompt_construction():
    q = question()
    perm = [2, 0, 3, 1]
    conv = build_conversation(q, perm, min_pixels=100, max_pixels=400)
    assert len(conv) == 1 and conv[0]["role"] == "user"
    content = conv[0]["content"]
    imgs = [c for c in content if c["type"] == "image"]
    assert len(imgs) == 7
    assert all(c["min_pixels"] == 100 and c["max_pixels"] == 400 for c in imgs)
    assert [c["image"] for c in imgs[:3]] == list(q.partial_images)
    # Each candidate label is directly followed by the permuted candidate image.
    for letter, original in zip("ABCD", perm, strict=True):
        position = next(i for i, c in enumerate(content) if c.get("text") == f"Candidate {letter}:")
        assert content[position + 1]["image"] is q.candidate_images[original]
    assert "(A, B, C or D)" in content[-1]["text"]
    with pytest.raises(ValueError):
        build_conversation(q, [0, 0, 1, 2], min_pixels=1, max_pixels=2)


def test_letter_tokens_resolution():
    ids = letter_token_ids(FakeTokenizer(), 4)
    assert ids == {"bare": [10, 11, 12, 13], "space": [20, 21, 22, 23]}
    broken = dict(TOKENS)
    del broken["C"]
    with pytest.raises(ValueError, match="single token"):
        letter_token_ids(FakeTokenizer(broken), 4)
    collide = dict(TOKENS, B=10)
    with pytest.raises(ValueError, match="collide"):
        letter_token_ids(FakeTokenizer(collide), 4)


def test_letter_scoring_maps_back_through_permutation():
    ids = letter_token_ids(FakeTokenizer(), 4)
    logprobs = np.full(VOCAB, -20.0)
    logprobs[[10, 11, 12, 13]] = [-3.0, -0.1, -2.0, -4.0]  # letter B wins
    perm = [3, 2, 0, 1]
    result = score_letters(logprobs, ids, perm, correct_index=2)
    assert result["predicted_letter"] == "B"
    assert result["predicted_index"] == 2 and result["correct"]
    assert result["correct_letter"] == "B"
    assert result["letter_logprobs"]["B"] == -0.1
    wrong = score_letters(logprobs, ids, perm, correct_index=0)
    assert not wrong["correct"] and wrong["correct_letter"] == "C"


def test_scorer_end_to_end_with_fake_model():
    processor = FakeProcessor()
    model = FakeModel(favoured_token=12)  # bare "C"
    scorer = ZeroShotScorer(
        model, processor, min_pixels=64, max_pixels=256, vision_info=lambda conv: ["x"] * 7
    )
    q = question(correct=2)
    record = scorer.score(q, option_seed=11)
    perm = option_permutation(11, "7", 4)
    assert record["permutation"] == perm
    assert record["predicted_letter"] == "C"
    assert record["predicted_index"] == perm[2]
    assert record["correct"] == (perm[2] == 2)
    assert model.kwargs["logits_to_keep"] == 1 and model.kwargs["use_cache"] is False
    text = processor.calls[0][0][0]
    assert text.count("<img>") == 7 and text.endswith("<assistant>")
    assert processor.calls[0][2]["do_resize"] is False
    assert record["input_tokens"] == 14 and record["seconds"] >= 0
    # Log-probs are normalized: probabilities over the vocabulary sum to 1.
    assert 0 < record["letter_probability_mass"] <= 1
    tight = ZeroShotScorer(model, processor, max_input_tokens=5, vision_info=lambda conv: ["x"] * 7)
    with pytest.raises(ValueError, match="truncation"):
        tight.score(q, option_seed=11)


def test_jsonl_resume_drops_torn_line_and_summary(tmp_path):
    path = tmp_path / "r.jsonl"
    records = []
    for i in range(3):
        r = {
            "question_id": str(i),
            "correct": i != 1,
            "seconds": 1.0,
            "preprocess_seconds": 0.5,
            "forward_seconds": 0.5,
            "allocator_peak_bytes": 100 + i,
            "letter_probability_mass": 0.9,
            "predicted_index_combined": 0,
            "correct_index": 0,
            "predicted_letter": "A",
            "input_tokens": 10,
        }
        append_result(path, r)
        records.append(r)
    with path.open("a") as stream:
        stream.write('{"question_id": "3", "corr')
    assert [r["question_id"] for r in read_results(path)] == ["0", "1", "2"]
    assert path.read_text().endswith("\n")
    summary = summarize(records, {"x": 1})
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["allocator_peak_bytes"] == 102
    assert summary["predicted_letter_counts"]["A"] == 3
    assert set(summary["predicted_letter_counts"]) == set(LETTERS)


def _script():
    spec = importlib.util.spec_from_file_location(
        "mllm_script", Path("scripts/v2/mllm_zeroshot.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_ids_and_test_guard(tmp_path):
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    outfits = [
        {"set_id": str(s), "items": [{"item_id": f"i{s}_{k}", "index": k} for k in range(1, 4)]}
        for s in range(30)
    ]
    (protocol / "test.json").write_text(json.dumps(outfits))
    fitb = [
        {
            "question": [f"{s}_1", f"{s}_2"],
            "answers": [f"{s}_3", f"{(s + 1) % 30}_3", f"{(s + 2) % 30}_3", f"{(s + 3) % 30}_3"],
        }
        for s in range(30)
    ]
    (protocol / "fill_in_blank_test.json").write_text(json.dumps(fitb))
    script = _script()
    ids = tmp_path / "ids.json"
    argv = ["freeze-ids", "--protocol-root", str(protocol), "--split", "test", "--n", "10"]
    script.main([*argv, "--seed", "3", "--output", str(ids)])
    frozen = json.loads(ids.read_text())
    assert len(frozen["question_ids"]) == 10 and frozen["total_questions"] == 30
    with pytest.raises(FileExistsError):
        script.main([*argv, "--seed", "3", "--output", str(ids)])
    with pytest.raises(SystemExit, match="protocol freeze"):
        script.main(
            [
                "score",
                "--ids",
                str(ids),
                "--protocol-root",
                str(protocol),
                "--images-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
