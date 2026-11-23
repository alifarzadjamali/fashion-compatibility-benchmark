"""One validation-only model-selection rule for all v2 analyses."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def select_models(rows: Sequence[Mapping], *, dataset: str, families: Mapping[str, str]) -> dict:
    """Require complete clean-validation native CP results; lexical ties are explicit."""
    if not dataset.endswith("_clean"):
        raise ValueError("Selection requires a clean training corpus")
    eligible = []
    seen = set()
    for row in rows:
        if row["dataset"] != dataset:
            continue
        if row["split"] != "valid" or row["metric"] != "cp_auc" or row["setting"] != "native":
            raise ValueError("Selection input must contain only native validation CP AUC")
        key, value = row["model_key"], float(row["value"])
        if key in seen or key not in families or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid, duplicated or unregistered selection row: {key}")
        seen.add(key)
        eligible.append((key, value))
    if seen != set(families):
        raise ValueError(f"Incomplete roster: missing {sorted(set(families) - seen)}")

    def best(group):
        candidates = [(k, v) for k, v in eligible if group(k)]
        if not candidates:
            raise ValueError("No eligible model in selection group")
        return min(candidates, key=lambda pair: (-pair[1], pair[0]))[0]

    return {
        "dataset": dataset,
        "overall": best(lambda k: True),
        "generic": best(lambda k: families[k] == "generic_vision_language"),
        "fashion": best(lambda k: families[k] in {"fashion_vision_language", "fashion_retrieval"}),
        "tie_break": "lexicographic model_key after exact validation AUC tie",
    }
