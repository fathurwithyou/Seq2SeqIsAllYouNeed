from __future__ import annotations

import argparse
import shlex
import subprocess


def _cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _cnn_commands(args) -> list[list[str]]:
    base = [
        "uv",
        "run",
        "python",
        "scripts/train_cnn.py",
        "--output-dir",
        args.cnn_output,
        "--epochs",
        str(args.cnn_epochs),
        "--batch-size",
        str(args.batch_size),
        "--target-size",
        str(args.cnn_target_size[0]),
        str(args.cnn_target_size[1]),
        "--dense-units",
        str(args.cnn_dense_units),
    ]
    if args.intel_root:
        base.extend(["--data-root", args.intel_root])
    variants: list[dict] = [
        {"name": "baseline_shared", "layer_type": "conv2d", "num_layers": 1, "filters": [8], "kernel": 3, "pooling": "max"},
        {"name": "non_shared", "layer_type": "locally_connected", "num_layers": 1, "filters": [8], "kernel": 3, "pooling": "max"},
        {"name": "depth_1", "layer_type": "conv2d", "num_layers": 1, "filters": [8], "kernel": 3, "pooling": "max"},
        {"name": "depth_2", "layer_type": "conv2d", "num_layers": 2, "filters": [8, 12], "kernel": 3, "pooling": "max"},
        {"name": "depth_3", "layer_type": "conv2d", "num_layers": 3, "filters": [8, 12, 16], "kernel": 3, "pooling": "max"},
        {"name": "filters_small", "layer_type": "conv2d", "num_layers": 2, "filters": [4, 8], "kernel": 3, "pooling": "max"},
        {"name": "filters_medium", "layer_type": "conv2d", "num_layers": 2, "filters": [8, 12], "kernel": 3, "pooling": "max"},
        {"name": "filters_large", "layer_type": "conv2d", "num_layers": 2, "filters": [12, 16], "kernel": 3, "pooling": "max"},
        {"name": "kernel_3", "layer_type": "conv2d", "num_layers": 2, "filters": [8, 12], "kernel": 3, "pooling": "max"},
        {"name": "kernel_5", "layer_type": "conv2d", "num_layers": 2, "filters": [8, 12], "kernel": 5, "pooling": "max"},
        {"name": "kernel_7", "layer_type": "conv2d", "num_layers": 2, "filters": [8, 12], "kernel": 7, "pooling": "max"},
        {"name": "pool_max", "layer_type": "conv2d", "num_layers": 2, "filters": [8, 12], "kernel": 3, "pooling": "max"},
        {"name": "pool_average", "layer_type": "conv2d", "num_layers": 2, "filters": [8, 12], "kernel": 3, "pooling": "average"},
    ]
    commands = []
    seen: set[tuple] = set()
    for variant in variants:
        key = (
            variant["layer_type"],
            variant["num_layers"],
            tuple(variant["filters"]),
            variant["kernel"],
            variant["pooling"],
        )
        if key in seen:
            continue
        seen.add(key)
        commands.append(
            base
            + [
                "--layer-type",
                variant["layer_type"],
                "--num-conv-layers",
                str(variant["num_layers"]),
                "--filters",
                *[str(value) for value in variant["filters"]],
                "--kernel-size",
                str(variant["kernel"]),
                "--pooling",
                variant["pooling"],
            ]
        )
    return commands


def _caption_train_commands(args) -> list[list[str]]:
    base = [
        "uv",
        "run",
        "python",
        "scripts/train_caption.py",
        "--features",
        args.features,
        "--paths-file",
        args.paths_file,
        "--vocab",
        args.vocab,
        "--captions-json",
        args.captions_json,
        "--output-dir",
        args.caption_output,
        "--epochs",
        str(args.caption_epochs),
        "--batch-size",
        str(args.batch_size),
        "--embed-dim",
        str(args.caption_embed_dim),
        "--max-length",
        str(args.caption_max_length),
    ]
    commands = []
    for rnn_type in ("rnn", "lstm"):
        for num_layers in (1, 2, 3):
            for hidden_size in args.caption_hidden_sizes:
                commands.append(
                    base
                    + [
                        "--rnn-type",
                        rnn_type,
                        "--num-layers",
                        str(num_layers),
                        "--hidden-size",
                        str(hidden_size),
                    ]
                )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("cnn", "caption", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--intel-root", default=None)
    parser.add_argument("--features", default="artifacts/flickr8k_features.npy")
    parser.add_argument("--paths-file", default="artifacts/flickr8k_paths.txt")
    parser.add_argument("--vocab", default="artifacts/vocab.json")
    parser.add_argument("--captions-json", default="artifacts/captions_tokens.json")
    parser.add_argument("--cnn-output", default="artifacts/cnn")
    parser.add_argument("--caption-output", default="artifacts/captioning")
    parser.add_argument("--cnn-epochs", type=int, default=5)
    parser.add_argument("--caption-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cnn-target-size", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--cnn-dense-units", type=int, default=32)
    parser.add_argument("--caption-hidden-sizes", type=int, nargs=2, default=(32, 64))
    parser.add_argument("--caption-embed-dim", type=int, default=64)
    parser.add_argument("--caption-max-length", type=int, default=12)
    args = parser.parse_args()

    commands: list[list[str]] = []
    if args.suite in {"cnn", "all"}:
        commands.extend(_cnn_commands(args))
    if args.suite in {"caption", "all"}:
        commands.extend(_caption_train_commands(args))

    for command in commands:
        print(_cmd(command))
        if not args.dry_run:
            subprocess.run(command, check=True)

    if args.dry_run:
        print(f"\n{len(commands)} commands generated.")
    else:
        print(f"\n{len(commands)} commands completed.")


if __name__ == "__main__":
    main()
