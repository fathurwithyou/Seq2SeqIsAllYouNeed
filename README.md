# Seq2SeqIsAllYouNeed

This repository contains the implementation for IF3270 Machine Learning Major
Assignment 2: Convolutional Neural Network (CNN), Simple RNN, and LSTM. The main
goal is to validate from-scratch forward propagation with NumPy and compare it
against Keras models used for training and weight transfer.

Supported experiments:

- Intel Image Classification with CNN.
- Comparison between shared-parameter `Conv2D` and non-shared-parameter
  `LocallyConnected2D`.
- Flickr8k image captioning with a pre-inject encoder-decoder architecture.
- Simple RNN and LSTM decoders.
- Keras vs from-scratch inference comparison.
- Metric summaries, loss plots, and LaTeX report generation.

## Repository Structure

```text
.
├── src/
│   ├── seq2seq/              # from-scratch NumPy-based framework
│   │   ├── layers/           # Conv2D, pooling, Dense, Embedding, RNN, LSTM
│   │   ├── ops/              # NumPy operation kernels
│   │   ├── metrics/          # macro F1, BLEU-4, METEOR
│   │   ├── saving/           # Keras weight loading into NumPy layers
│   │   └── utils/            # image, text, sequence, and dataset utilities
│   └── experiments/
│       ├── cnn/              # Keras CNN model builders
│       ├── captioning/       # captioning decoder and weight loader
│       └── experiments.ipynb # main experiment notebook
├── scripts/                  # training, evaluation, and summary pipelines
├── tests/                    # unit tests and Keras parity tests
├── artifacts/                # weights, histories, evaluations, plots, summary
└── doc/                      # LaTeX report, PDF, and references.bib
```

## Requirements

- Python 3.11 or newer.
- `uv` for environment management.
- Internet access for the first KaggleHub dataset download or pretrained Keras
  weight download.
- MacTeX or another TeX distribution if you want to compile the LaTeX report.

Check that `uv` is installed:

```bash
uv --version
```

## Setup

```bash
uv venv
uv pip install -e ".[full,dev]"
```

If you run the notebook from another environment, make sure the notebook kernel
points to this project environment.

## Running Tests

Run the test suite with:

```bash
uv run pytest
```

The tests cover CNN layers, pooling, Flatten, Dense, Embedding, Simple RNN,
LSTM, metrics, utilities, Keras weight loading, and Keras-vs-NumPy output
comparisons.

## Running Experiments from the Notebook

The main notebook is located at:

```text
src/experiments/experiments.ipynb
```

General workflow:

1. Open the notebook.
2. Set dataset paths if the datasets already exist locally.
3. Leave dataset paths empty if you want the scripts to use KaggleHub.
4. Start with small parameters.
5. Run the cells sequentially.

Recommended lightweight configuration:

```python
FAST_MODE = True
RUN_HEAVY = True
CNN_EPOCHS = 5
CAPTION_EPOCHS = 5
FEATURE_ENCODER = "mobilenet_v2"
FEATURE_LIMIT = 1000
```

Note: if `FEATURE_LIMIT` is used, it must be a number. Do not pass `--limit`
without a value.

## Datasets

If dataset paths are not provided, the scripts download the datasets through
KaggleHub.

CNN dataset:

```python
import kagglehub

path = kagglehub.dataset_download("puneet6060/intel-image-classification")
print("Path to dataset files:", path)
```

Captioning dataset:

```python
import kagglehub

path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)
```

To speed up captioning experiments, use `mobilenet_v2` and limit the number of
extracted image features with `--limit`.

## CNN Experiments

Small CNN training example:

```bash
uv run python scripts/train_cnn.py \
  --output-dir artifacts/cnn \
  --layer-type conv2d \
  --num-conv-layers 1 \
  --filters 8 \
  --kernel-size 3 \
  --pooling max \
  --target-size 64 64 \
  --dense-units 32 \
  --epochs 5 \
  --batch-size 8
```

Three-layer variation:

```bash
uv run python scripts/train_cnn.py \
  --output-dir artifacts/cnn \
  --layer-type conv2d \
  --num-conv-layers 3 \
  --filters 8 12 16 \
  --kernel-size 3 \
  --pooling max \
  --target-size 64 64 \
  --dense-units 32 \
  --epochs 5 \
  --batch-size 8
```

`LocallyConnected2D` example:

```bash
uv run python scripts/train_cnn.py \
  --output-dir artifacts/cnn \
  --layer-type locally_connected \
  --num-conv-layers 1 \
  --filters 8 \
  --kernel-size 3 \
  --pooling max \
  --target-size 64 64 \
  --dense-units 32 \
  --epochs 5 \
  --batch-size 8
```

From-scratch CNN forward evaluation:

```bash
uv run python scripts/eval_cnn_scratch.py \
  --weights artifacts/cnn/conv2d_1x8_k3_max.weights.h5 \
  --config artifacts/cnn/conv2d_1x8_k3_max.history.json \
  --output artifacts/cnn/eval_conv2d_1x8_k3_max.json
```

## Image Captioning Experiments

Extract image features with a lightweight encoder:

```bash
uv run python scripts/extract_features.py \
  --output artifacts/flickr8k_features.npy \
  --paths-output artifacts/flickr8k_paths.txt \
  --encoder mobilenet_v2 \
  --batch-size 32 \
  --limit 1000
```

Build vocabulary and caption tokens for the extracted image list:

```bash
uv run python scripts/build_vocab.py \
  --output artifacts/vocab.json \
  --captions-output artifacts/captions_tokens.json \
  --image-list artifacts/flickr8k_paths.txt \
  --min-count 2
```

Train a small RNN decoder:

```bash
uv run python scripts/train_caption.py \
  --features artifacts/flickr8k_features.npy \
  --paths-file artifacts/flickr8k_paths.txt \
  --vocab artifacts/vocab.json \
  --captions-json artifacts/captions_tokens.json \
  --output-dir artifacts/captioning \
  --rnn-type rnn \
  --num-layers 1 \
  --hidden-size 32 \
  --embed-dim 64 \
  --max-length 12 \
  --epochs 5 \
  --batch-size 8
```

Train a small LSTM decoder:

```bash
uv run python scripts/train_caption.py \
  --features artifacts/flickr8k_features.npy \
  --paths-file artifacts/flickr8k_paths.txt \
  --vocab artifacts/vocab.json \
  --captions-json artifacts/captions_tokens.json \
  --output-dir artifacts/captioning \
  --rnn-type lstm \
  --num-layers 1 \
  --hidden-size 32 \
  --embed-dim 64 \
  --max-length 12 \
  --epochs 5 \
  --batch-size 8
```

Evaluate the decoder with the from-scratch implementation:

```bash
uv run python scripts/eval_caption.py \
  --weights artifacts/captioning/lstm_L1_H32.weights.h5 \
  --config artifacts/captioning/lstm_L1_H32.history.json \
  --vocab artifacts/vocab.json \
  --captions-json artifacts/captions_tokens.json \
  --features artifacts/flickr8k_features.npy \
  --paths-file artifacts/flickr8k_paths.txt \
  --mode both \
  --decoding greedy \
  --max-length 12 \
  --num-examples 10 \
  --output artifacts/captioning/eval_lstm_L1_H32_both_greedy_max12.json
```

Beam search example:

```bash
uv run python scripts/eval_caption.py \
  --weights artifacts/captioning/lstm_L1_H32.weights.h5 \
  --config artifacts/captioning/lstm_L1_H32.history.json \
  --vocab artifacts/vocab.json \
  --captions-json artifacts/captions_tokens.json \
  --features artifacts/flickr8k_features.npy \
  --paths-file artifacts/flickr8k_paths.txt \
  --mode scratch \
  --decoding beam \
  --beam-width 3 \
  --max-length 12 \
  --output artifacts/captioning/eval_lstm_L1_H32_scratch_beam_max12.json
```

## Result Summary

After experiments generate `.history.json` and evaluation `.json` files, create
`summary.csv` and loss plots with:

```bash
uv run python scripts/summarize_results.py \
  --artifacts-dir artifacts \
  --output artifacts/summary.csv \
  --plot-losses \
  --plots-dir artifacts/plots
```

When this command is run from a notebook environment, the script sets
`MPLBACKEND=Agg` so plots can be generated from a terminal process.

## Report

The report source is located at:

```text
doc/Laporan Tugas Besar 2.tex
```

Compile the report with:

```bash
cd doc
pdflatex "Laporan Tugas Besar 2.tex"
bibtex "Laporan Tugas Besar 2"
pdflatex "Laporan Tugas Besar 2.tex"
pdflatex "Laporan Tugas Besar 2.tex"
```

Generated PDF:

```text
doc/Laporan Tugas Besar 2.pdf
```

## Task Distribution

| Student ID | Name | Contribution |
| --- | ---- | ------------ |
| 13523097 | Shanice Feodora Tjahjono | CNN layer implementation, pooling tests, and CNN analysis. |
| 13523105 | Muhammad Fathur Rizky | RNN/LSTM implementation, captioning pipeline, and Keras vs from-scratch evaluation. |
| 13523121 | Ahmad Wicaksono | Experiment setup, metrics, notebook workflow, result summaries, and report writing. |

## Implementation Notes

- The from-scratch implementation uses NumPy for core forward propagation.
- Keras is used for model training and weight saving.
- Keras weights are loaded into NumPy layers to validate forward propagation.
- README commands use small parameters so they can run faster. For stronger
  results, increase the amount of data, hidden size, number of filters, and
  number of epochs.
