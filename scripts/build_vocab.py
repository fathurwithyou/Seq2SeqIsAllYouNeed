from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.captioning import download_flickr8k_dataset
from seq2seq.utils import Vocabulary, tokenize


def _load_captions(captions_file: Path, train_images: set[str] | None) -> dict[str, list[str]]:
    captions: dict[str, list[str]] = {}
    with captions_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = row["image"].strip()
            caption = row["caption"].strip()
            if train_images is not None and image not in train_images:
                continue
            captions.setdefault(image, []).append(caption)
    return captions


def _load_split(split_file: Path | None) -> set[str] | None:
    if split_file is None:
        return None
    return {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def _load_image_list(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", default=None, help="Path to Flickr8k captions.txt")
    parser.add_argument("--flickr-root", default=None, help="Path to Flickr8k root. Defaults to KaggleHub download.")
    parser.add_argument("--output", required=True, help="Path for vocabulary JSON")
    parser.add_argument("--captions-output", required=True, help="Path for per-image captions JSON")
    parser.add_argument("--train-split", default=None, help="Optional file listing training image filenames")
    parser.add_argument("--image-list", default=None, help="Optional file listing image filenames to include.")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=None)
    args = parser.parse_args()

    flickr_root = Path(args.flickr_root) if args.flickr_root else None
    if args.captions:
        captions_file = Path(args.captions)
    else:
        flickr_root = flickr_root or download_flickr8k_dataset()
        captions_file = flickr_root / "captions.txt"

    train_set = _load_split(Path(args.train_split)) if args.train_split else None
    image_set = _load_image_list(Path(args.image_list)) if args.image_list else None
    if train_set is not None and image_set is not None:
        train_set = train_set & image_set
    elif image_set is not None:
        train_set = image_set
    captions = _load_captions(captions_file, train_set)

    flat_captions = [caption for texts in captions.values() for caption in texts]
    vocab = Vocabulary.build(flat_captions, min_count=args.min_count, max_size=args.max_size)

    vocab_path = Path(args.output)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)

    captions_payload = {
        image: [tokenize(caption) for caption in texts]
        for image, texts in captions.items()
    }
    with Path(args.captions_output).open("w", encoding="utf-8") as handle:
        json.dump(captions_payload, handle, ensure_ascii=False)
    print(f"Vocabulary size: {vocab.size}")
    print(f"Images: {len(captions_payload)}  Captions: {len(flat_captions)}")


if __name__ == "__main__":
    main()
