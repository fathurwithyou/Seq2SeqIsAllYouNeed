from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seq2seq-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "seq2seq-cache"))


SUMMARY_FIELDNAMES = [
    "file",
    "kind",
    "layer_type",
    "rnn_type",
    "num_conv_layers",
    "num_layers",
    "filters",
    "kernel_size",
    "pooling",
    "hidden_size",
    "max_length",
    "mode",
    "decoding",
    "test_macro_f1",
    "keras_macro_f1",
    "scratch_macro_f1",
    "prediction_agreement",
    "bleu_4",
    "meteor",
    "seconds_per_image",
    "num_evaluated",
    "final_loss",
    "final_val_loss",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_training(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    config = payload.get("config", {})
    history = payload.get("history", {})
    row: dict[str, Any] = {
        "file": str(path),
        "kind": "training",
        "layer_type": config.get("layer_type"),
        "rnn_type": config.get("rnn_type"),
        "num_conv_layers": config.get("num_conv_layers"),
        "num_layers": config.get("num_layers"),
        "filters": "-".join(str(v) for v in config.get("filters", [])),
        "kernel_size": config.get("kernel_size"),
        "pooling": config.get("pooling"),
        "hidden_size": config.get("hidden_size"),
        "max_length": config.get("max_length"),
        "test_macro_f1": payload.get("test_macro_f1"),
        "final_loss": history.get("loss", [None])[-1] if history.get("loss") else None,
        "final_val_loss": history.get("val_loss", [None])[-1] if history.get("val_loss") else None,
    }
    return row


def _summarize_eval(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = []
    for mode, metrics in payload.get("results", {}).items():
        rows.append(
            {
                "file": str(path),
                "kind": "caption_eval",
                "mode": mode,
                "decoding": payload.get("decoding"),
                "max_length": payload.get("max_length"),
                "bleu_4": metrics.get("bleu_4"),
                "meteor": metrics.get("meteor"),
                "seconds_per_image": metrics.get("seconds_per_image"),
                "num_evaluated": payload.get("num_evaluated"),
            }
        )
    return rows


def _summarize_cnn_eval(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    config = payload.get("config", {})
    return {
        "file": str(path),
        "kind": "cnn_scratch_eval",
        "layer_type": config.get("layer_type"),
        "num_conv_layers": config.get("num_conv_layers"),
        "filters": "-".join(str(v) for v in config.get("filters", [])),
        "kernel_size": config.get("kernel_size"),
        "pooling": config.get("pooling"),
        "keras_macro_f1": payload.get("keras_macro_f1"),
        "scratch_macro_f1": payload.get("scratch_macro_f1"),
        "prediction_agreement": payload.get("prediction_agreement"),
        "num_evaluated": payload.get("num_evaluated"),
    }


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = SUMMARY_FIELDNAMES + sorted(
        {key for row in rows for key in row} - set(SUMMARY_FIELDNAMES)
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_losses(paths: list[Path], output_dir: Path) -> None:
    if not paths:
        return

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        payload = _read_json(path)
        history = payload.get("history", {})
        if "loss" not in history:
            continue
        plt.figure()
        plt.plot(history["loss"], label="train loss")
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(path.stem)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{path.stem}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--output", default="artifacts/summary.csv")
    parser.add_argument("--plot-losses", action="store_true")
    parser.add_argument("--plots-dir", default="artifacts/plots")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    json_paths = sorted(artifacts_dir.rglob("*.json"))
    rows: list[dict[str, Any]] = []
    training_paths: list[Path] = []
    for path in json_paths:
        payload = _read_json(path)
        if "history" in payload:
            rows.append(_summarize_training(path))
            training_paths.append(path)
        elif "results" in payload:
            rows.extend(_summarize_eval(path))
        elif payload.get("kind") == "cnn_scratch_eval":
            rows.append(_summarize_cnn_eval(path))

    _write_csv(rows, Path(args.output))
    if args.plot_losses:
        _plot_losses(training_paths, Path(args.plots_dir))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
