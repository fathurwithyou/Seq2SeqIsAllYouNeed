from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN)

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return clean_caption(text).split()


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def start_id(self) -> int:
        return self.token_to_id[START_TOKEN]

    @property
    def end_id(self) -> int:
        return self.token_to_id[END_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, text: str, *, add_special: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize(text)]
        if add_special:
            ids = [self.start_id, *ids, self.end_id]
        return ids

    def decode(self, ids: Iterable[int], *, strip_special: bool = True) -> str:
        tokens: list[str] = []
        for index in ids:
            index = int(index)
            if index < 0 or index >= self.size:
                continue
            token = self.id_to_token[index]
            if strip_special and token in {PAD_TOKEN, START_TOKEN}:
                continue
            if strip_special and token == END_TOKEN:
                break
            tokens.append(token)
        return " ".join(tokens)

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump({"token_to_id": self.token_to_id}, handle, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        token_to_id = {token: int(index) for token, index in data["token_to_id"].items()}
        id_to_token = [""] * len(token_to_id)
        for token, index in token_to_id.items():
            id_to_token[index] = token
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @classmethod
    def build(
        cls,
        captions: Iterable[str],
        *,
        min_count: int = 1,
        max_size: int | None = None,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(tokenize(caption))
        kept = [
            (token, count)
            for token, count in counter.most_common()
            if count >= min_count and token not in SPECIAL_TOKENS
        ]
        if max_size is not None:
            kept = kept[: max(max_size - len(SPECIAL_TOKENS), 0)]
        id_to_token = list(SPECIAL_TOKENS) + [token for token, _ in kept]
        token_to_id = {token: index for index, token in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)


__all__ = [
    "Vocabulary",
    "clean_caption",
    "tokenize",
    "PAD_TOKEN",
    "START_TOKEN",
    "END_TOKEN",
    "UNK_TOKEN",
    "SPECIAL_TOKENS",
]
