from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from repbench.data.polyvore import FITBQuestion


def evaluate_fitb(questions: Sequence[FITBQuestion], scorer: Callable[[tuple[str, ...]], float]):
    predictions = []
    for question in questions:
        scores = [
            scorer(question.question_item_ids + (candidate,))
            for candidate in question.candidate_item_ids
        ]
        predictions.append(int(np.argmax(scores)))
    correct = np.fromiter(
        (p == q.correct_index for p, q in zip(predictions, questions)), dtype=bool
    )
    return {"accuracy": float(correct.mean()), "predictions": predictions, "correct": correct}


def random_fitb_accuracy(candidate_counts: Sequence[int], seed: int) -> float:
    rng = np.random.default_rng(seed)
    draws = [int(rng.integers(count)) for count in candidate_counts]
    # Intended for official questions where the correct index is checked separately by caller.
    return float(np.mean([draw == 0 for draw in draws]))
