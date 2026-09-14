"""Strict ingestion for the official Polyvore Outfits disjoint protocol."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Outfit:
    set_id: str
    item_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompatibilityExample:
    label: int
    item_ids: tuple[str, ...]


@dataclass(frozen=True)
class FITBQuestion:
    question_item_ids: tuple[str, ...]
    candidate_item_ids: tuple[str, ...]
    correct_index: int


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_item_map(outfits: Sequence[Mapping]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for outfit in outfits:
        set_id = str(outfit["set_id"])
        for item in outfit["items"]:
            key = f"{set_id}_{int(item['index'])}"
            item_id = str(item["item_id"])
            if key in mapping and mapping[key] != item_id:
                raise ValueError(f"Ambiguous official item key: {key}")
            mapping[key] = item_id
    return mapping


def _resolve(keys: Iterable[str], key_to_item: Mapping[str, str]) -> tuple[str, ...]:
    resolved = []
    for key in keys:
        if key not in key_to_item:
            raise KeyError(f"Official question references unknown item key: {key}")
        resolved.append(key_to_item[key])
    return tuple(resolved)


class PolyvoreDisjoint:
    """Read a Polyvore protocol while keeping images in the canonical raw-data root.

    With no ``split_root`` this reads the packaged historical ``disjoint`` files.  A
    separate split root is used for derived protocols so the historical benchmark can
    never be overwritten accidentally.
    """

    SPLITS = ("train", "valid", "test")

    def __init__(
        self,
        root: str | Path,
        split_root: str | Path | None = None,
        dataset_name: str = "historical_polyvore_d",
    ):
        self.root = Path(root)
        self.split_root = Path(split_root) if split_root is not None else self.root / "disjoint"
        self.images_root = self.root / "images"
        self.dataset_name = dataset_name

    def outfits(self, split: str) -> list[Outfit]:
        raw = self._raw_outfits(split)
        return [
            Outfit(
                set_id=str(row["set_id"]),
                item_ids=tuple(str(item["item_id"]) for item in row["items"]),
            )
            for row in raw
        ]

    def compatibility(self, split: str) -> list[CompatibilityExample]:
        raw = self._raw_outfits(split)
        item_map = _canonical_item_map(raw)
        path = self._first_existing(
            self.split_root / f"compatibility_{split}.txt",
            self.split_root / f"compatibility_{split}_new.txt",
        )
        examples: list[CompatibilityExample] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] not in {"0", "1"}:
                raise ValueError(f"Invalid label at {path}:{line_number}")
            examples.append(CompatibilityExample(int(fields[0]), _resolve(fields[1:], item_map)))
        return examples

    def fitb(self, split: str) -> list[FITBQuestion]:
        raw = self._raw_outfits(split)
        item_map = _canonical_item_map(raw)
        path = self._first_existing(
            self.split_root / f"fill_in_blank_{split}.json",
            self.split_root / f"fill_in_the_blank_{split}.json",
        )
        questions = []
        for row in _load_json(path):
            question_keys = tuple(str(x) for x in row["question"])
            answer_keys = tuple(str(x) for x in row["answers"])
            ground_truth_set = question_keys[0].split("_", 1)[0]
            correct = [
                i for i, key in enumerate(answer_keys) if key.split("_", 1)[0] == ground_truth_set
            ]
            if len(correct) != 1:
                raise ValueError(
                    f"FITB question must have exactly one answer, found {len(correct)}"
                )
            questions.append(
                FITBQuestion(
                    _resolve(question_keys, item_map), _resolve(answer_keys, item_map), correct[0]
                )
            )
        return questions

    def image_path(self, item_id: str) -> Path:
        return self.images_root / f"{item_id}.jpg"

    def item_ids(self, split: str) -> set[str]:
        return {item_id for outfit in self.outfits(split) for item_id in outfit.item_ids}

    def _raw_outfits(self, split: str) -> list[dict]:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}")
        path = self._first_existing(
            self.split_root / f"{split}.json",
            self.split_root / ("validation.json" if split == "valid" else f"{split}.json"),
        )
        data = _load_json(path)
        if not isinstance(data, list):
            raise TypeError(f"Expected a list in {path}")
        return data

    @staticmethod
    def _first_existing(*paths: Path) -> Path:
        for path in paths:
            if path.is_file():
                return path
        raise FileNotFoundError(
            "None of the required official files exists: " + ", ".join(map(str, paths))
        )
