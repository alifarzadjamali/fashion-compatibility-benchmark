"""Zero-shot FITB scoring with an instruction-tuned MLLM (HANDOFF_V2 §6.2).

The model sees the partial outfit and the candidates labelled with letters in a
seed-randomized but fixed order, and the answer is read from the next-token
log-probabilities of the option letters at the final prompt position. No free
text is generated or parsed.

Pure helpers (question sampling, option permutation, conversation construction,
letter-token resolution, letter scoring) are separated from model execution so
they can be unit tested with fakes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LETTERS = "ABCDE"
# Qwen3-VL: 16 px patches merged 2x2, so one visual token covers 32x32 pixels.
PIXELS_PER_VISUAL_TOKEN = 32 * 32
DEFAULT_MIN_PIXELS = 4 * PIXELS_PER_VISUAL_TOKEN
DEFAULT_MAX_PIXELS = 144 * PIXELS_PER_VISUAL_TOKEN  # 384x384, uniform budget (decision F)
DEFAULT_MAX_INPUT_TOKENS = 16384
PROMPT_VERSION = "fitb-letter-v1"
INTRO_TEXT = "Here are the items of a partial outfit:"
CANDIDATE_TEXT = (
    "Exactly one of the following candidate items completes the outfit so that it is "
    "stylistically compatible."
)


def question_text(letters):
    listed = ", ".join(letters[:-1]) + " or " + letters[-1]
    return (
        f"Which candidate ({listed}) best completes the outfit? "
        "Answer with the single letter of the candidate only."
    )


@dataclass(frozen=True)
class MCQuestion:
    """A multiple-choice completion question; images are paths or PIL images."""

    question_id: str
    partial_images: tuple
    candidate_images: tuple
    correct_index: int
    extra: dict | None = None


def sample_question_ids(total, n, seed):
    """Deterministic seeded subset of question positions, returned sorted."""
    if n >= total:
        return list(range(total))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(total, size=n, replace=False))


def option_permutation(option_seed, question_id, n_options):
    """Fixed per (seed, question) order. Entry j = original candidate shown at letter j."""
    digest = hashlib.sha256(f"{option_seed}:{question_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return [int(i) for i in rng.permutation(n_options)]


def build_conversation(question, permutation, *, min_pixels, max_pixels):
    """Single-turn chat: partial outfit images, then lettered candidates, then the ask."""
    n = len(question.candidate_images)
    if sorted(permutation) != list(range(n)):
        raise ValueError("Permutation must reorder all candidates")
    if n > len(LETTERS):
        raise ValueError(f"At most {len(LETTERS)} options are supported")
    letters = list(LETTERS[:n])

    def image(value):
        return {"type": "image", "image": value, "min_pixels": min_pixels, "max_pixels": max_pixels}

    content = [{"type": "text", "text": INTRO_TEXT}]
    content += [image(value) for value in question.partial_images]
    content.append({"type": "text", "text": CANDIDATE_TEXT})
    for letter, original in zip(letters, permutation, strict=True):
        content.append({"type": "text", "text": f"Candidate {letter}:"})
        content.append(image(question.candidate_images[original]))
    content.append({"type": "text", "text": question_text(letters)})
    return [{"role": "user", "content": content}]


def letter_token_ids(tokenizer, n_options):
    """Resolve each letter, bare and with a leading space, to exactly one token id."""
    variants = {}
    for name, prefix in (("bare", ""), ("space", " ")):
        ids = []
        for letter in LETTERS[:n_options]:
            encoded = tokenizer.encode(prefix + letter, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Letter {prefix + letter!r} is not a single token: {encoded}")
            ids.append(int(encoded[0]))
        if len(set(ids)) != n_options:
            raise ValueError(f"Letter tokens collide for variant {name}: {ids}")
        variants[name] = ids
    return variants


def score_letters(logprobs, token_ids, permutation, correct_index):
    """Primary score = bare-letter log-prob (answer follows 'assistant\\n').

    The space variant and the per-letter logsumexp of both are recorded as
    secondary evidence only; the decision rule does not use them.
    """
    logprobs = np.asarray(logprobs, dtype=np.float64)
    bare = [float(logprobs[i]) for i in token_ids["bare"]]
    space = [float(logprobs[i]) for i in token_ids["space"]]
    combined = [float(np.logaddexp(a, b)) for a, b in zip(bare, space, strict=True)]
    position = int(np.argmax(bare))
    predicted = int(permutation[position])
    return {
        "letter_logprobs": dict(zip(LETTERS, bare, strict=False)),
        "letter_logprobs_space": dict(zip(LETTERS, space, strict=False)),
        "letter_logprobs_combined": dict(zip(LETTERS, combined, strict=False)),
        "letter_probability_mass": float(np.exp(combined).sum()),
        "predicted_letter": LETTERS[position],
        "predicted_index": predicted,
        "predicted_index_combined": int(permutation[int(np.argmax(combined))]),
        "correct_letter": LETTERS[permutation.index(correct_index)],
        "correct_index": int(correct_index),
        "correct": predicted == int(correct_index),
    }


def _load_image(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        image = value
    else:
        with Image.open(value) as opened:
            opened.load()
            image = opened.copy()
    from repbench.v2.images import to_rgb_on_white

    return to_rgb_on_white(image)


def qwen_vision_info(conversations):
    from qwen_vl_utils import process_vision_info

    pixels, videos = process_vision_info(conversations, image_patch_size=16)
    if pixels is None or videos is not None:
        raise ValueError("Image-only processing failed")
    return pixels


class ZeroShotScorer:
    """Runs one question at a time (batch 1: no padding effects on the last position)."""

    def __init__(
        self,
        model,
        processor,
        *,
        device="cpu",
        min_pixels=DEFAULT_MIN_PIXELS,
        max_pixels=DEFAULT_MAX_PIXELS,
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
        vision_info=qwen_vision_info,
    ):
        if not 0 < min_pixels <= max_pixels:
            raise ValueError("Require 0 < min_pixels <= max_pixels")
        self.model, self.processor = model, processor
        self.device = device
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.max_input_tokens = max_input_tokens
        self.vision_info = vision_info
        self._token_ids = {}

    def token_ids(self, n_options):
        if n_options not in self._token_ids:
            self._token_ids[n_options] = letter_token_ids(self.processor.tokenizer, n_options)
        return self._token_ids[n_options]

    def prepare(self, question, permutation):
        loaded = MCQuestion(
            question.question_id,
            tuple(_load_image(v) for v in question.partial_images),
            tuple(_load_image(v) for v in question.candidate_images),
            question.correct_index,
        )
        conversation = build_conversation(
            loaded, permutation, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        images = self.vision_info([conversation])
        inputs = self.processor(
            text=[text], images=images, padding=False, do_resize=False, return_tensors="pt"
        )
        length = int(inputs["input_ids"].shape[1])
        if length > self.max_input_tokens:
            raise ValueError(
                f"Prompt has {length} tokens > {self.max_input_tokens}; refusing truncation"
            )
        return dict(inputs), length

    def next_token_logprobs(self, inputs):
        import torch

        moved = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**moved, use_cache=False, logits_to_keep=1)
            logits = outputs.logits[0, -1].float()
            return torch.log_softmax(logits, dim=-1).cpu().numpy()

    def score(self, question, option_seed):
        import torch

        n = len(question.candidate_images)
        permutation = option_permutation(option_seed, question.question_id, n)
        token_ids = self.token_ids(n)
        started = time.perf_counter()
        inputs, length = self.prepare(question, permutation)
        prepared = time.perf_counter()
        logprobs = self.next_token_logprobs(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        finished = time.perf_counter()
        if not np.isfinite(logprobs).all():
            raise ValueError(f"Non-finite log-probabilities for question {question.question_id}")
        record = {
            "question_id": question.question_id,
            "permutation": permutation,
            "n_options": n,
            "n_partial_images": len(question.partial_images),
            "input_tokens": length,
            **score_letters(logprobs, token_ids, permutation, question.correct_index),
            "seconds": finished - started,
            "preprocess_seconds": prepared - started,
            "forward_seconds": finished - prepared,
            "allocator_peak_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
        }
        if question.extra:
            record["extra"] = question.extra
        return record


# ----------------------------------------------------------------------------- runs


def read_results(path):
    """Read append-only JSONL; drop (and truncate) only a torn final line."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    records = []
    lines = raw.split(b"\n")
    complete, tail = lines[:-1], lines[-1]
    for number, line in enumerate(complete, 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt record at {path}:{number}; inspect manually") from exc
    if tail.strip():
        # A crash during append leaves an unterminated line; it is incomplete by definition.
        with path.open("r+b") as stream:
            stream.truncate(len(raw) - len(tail))
    return records


def append_result(path, record):
    import os

    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def bootstrap_accuracy_ci(correct, reps=2000, seed=20261118, level=0.95):
    correct = np.asarray(correct, dtype=float)
    if len(correct) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), size=(reps, len(correct)))].mean(axis=1)
    alpha = (1 - level) / 2
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))]


def summarize(records, config):
    correct = [bool(r["correct"]) for r in records]
    seconds = np.asarray([r["seconds"] for r in records], dtype=float)
    peaks = [r["allocator_peak_bytes"] for r in records if r.get("allocator_peak_bytes")]
    mass = np.asarray([r["letter_probability_mass"] for r in records], dtype=float)
    return {
        "config": config,
        "questions": len(records),
        "accuracy": float(np.mean(correct)),
        "accuracy_ci95_question_bootstrap": bootstrap_accuracy_ci(correct),
        "ci_note": "Descriptive iid question bootstrap (2000 reps); the §6.3 script is primary.",
        "accuracy_combined_letter_variants": float(
            np.mean([r["predicted_index_combined"] == r["correct_index"] for r in records])
        ),
        "seconds_per_question_mean": float(seconds.mean()),
        "seconds_per_question_median": float(np.median(seconds)),
        "seconds_total": float(seconds.sum()),
        "preprocess_seconds_per_question_mean": float(
            np.mean([r["preprocess_seconds"] for r in records])
        ),
        "forward_seconds_per_question_mean": float(
            np.mean([r["forward_seconds"] for r in records])
        ),
        "allocator_peak_bytes": int(max(peaks)) if peaks else None,
        "allocator_peak_label": "torch.cuda.max_memory_allocated (allocator peak, not device total)",
        "letter_probability_mass_median": float(np.median(mass)),
        "letter_probability_mass_min": float(mass.min()),
        "predicted_letter_counts": {
            letter: sum(r["predicted_letter"] == letter for r in records) for letter in LETTERS
        },
        "input_tokens_max": int(max(r["input_tokens"] for r in records)),
    }


def config_record(**values):
    return {"prompt_version": PROMPT_VERSION, **values}
