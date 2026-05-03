from __future__ import annotations

import numpy as np

from seq2seq.layers import Dense, Embedding, LSTM, SimpleRNN
from seq2seq.models import Model
from seq2seq.utils.sequence_utils import pad_sequences


class CaptioningDecoder(Model):
    def __init__(
        self,
        feature_dim: int,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        *,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if rnn_type not in ("rnn", "lstm"):
            raise ValueError("rnn_type must be 'rnn' or 'lstm'")
        self.rnn_type = rnn_type
        self.feature_dim = int(feature_dim)
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.image_projection = Dense(
            embed_dim,
            input_dim=feature_dim,
            use_bias=True,
            name="image_projection",
            seed=seed,
        )
        self.embedding = Embedding(
            vocab_size,
            embed_dim,
            mask_zero=True,
            seed=seed,
            name="embedding",
        )

        recurrent_seed = None if seed is None else seed + 7
        if rnn_type == "lstm":
            self.recurrent = LSTM(
                hidden_size,
                num_layers=num_layers,
                return_sequences=True,
                input_dim=embed_dim,
                seed=recurrent_seed,
                name="recurrent",
            )
        else:
            self.recurrent = SimpleRNN(
                hidden_size,
                num_layers=num_layers,
                return_sequences=True,
                input_dim=embed_dim,
                seed=recurrent_seed,
                name="recurrent",
            )
        self.output = Dense(
            vocab_size,
            input_dim=hidden_size,
            use_bias=True,
            activation="softmax",
            name="output",
            seed=seed,
        )

    @staticmethod
    def _as_feature_batch(features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None, :]
        if feats.ndim != 2:
            raise ValueError(f"features must be (N, F); got {feats.shape}")
        return feats

    def build_inputs(self, features: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        feats = self._as_feature_batch(features)
        tokens = np.asarray(token_ids, dtype=np.int64)
        if tokens.ndim != 2:
            raise ValueError(f"token_ids must be (N, T); got {tokens.shape}")
        if feats.shape[0] != tokens.shape[0]:
            raise ValueError("batch size mismatch between features and tokens")
        image_step = self.image_projection(feats)[:, None, :]
        embedded = self.embedding(tokens)
        return np.concatenate([image_step, embedded], axis=1)

    def call(self, features: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        hidden = self.recurrent(self.build_inputs(features, token_ids))
        return self.output(hidden)

    def next_token_distribution(self, features: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        return self(features, token_ids)[:, -1, :]

    def greedy_decode(
        self,
        features: np.ndarray,
        *,
        start_id: int,
        end_id: int,
        max_length: int = 20,
    ) -> list[list[int]]:
        feats = self._as_feature_batch(features)
        batch_size = feats.shape[0]
        sequences = [[start_id] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_length):
            current = pad_sequences(sequences, value=0, padding="post")
            next_ids = self.next_token_distribution(feats, current).argmax(axis=-1).tolist()
            for index in range(batch_size):
                if finished[index]:
                    continue
                token = int(next_ids[index])
                sequences[index].append(token)
                if token == end_id:
                    finished[index] = True
            if all(finished):
                break
        return sequences

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, rnn_type={self.rnn_type!r}"
        )


def beam_search_decode(
    decoder: CaptioningDecoder,
    feature: np.ndarray,
    *,
    start_id: int,
    end_id: int,
    beam_width: int = 3,
    max_length: int = 20,
    length_normalisation: float = 0.7,
) -> list[int]:
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    feature = CaptioningDecoder._as_feature_batch(feature)
    sequences: list[tuple[float, list[int], bool]] = [(0.0, [start_id], False)]

    for _ in range(max_length):
        candidates: list[tuple[float, list[int], bool]] = []
        for log_prob, sequence, finished in sequences:
            if finished:
                candidates.append((log_prob, sequence, True))
                continue
            probs = decoder.next_token_distribution(feature, np.array([sequence], dtype=np.int64))[0]
            k = min(beam_width, probs.shape[0])
            top = np.argpartition(-probs, k - 1)[:k]
            for token in top:
                token = int(token)
                lp = float(np.log(max(probs[token], 1e-20)))
                candidates.append((log_prob + lp, sequence + [token], token == end_id))

        def score(item):
            log_prob, sequence, _ = item
            length = max(len(sequence) - 1, 1)
            return log_prob / (length ** length_normalisation)

        candidates.sort(key=score, reverse=True)
        sequences = candidates[:beam_width]
        if all(item[2] for item in sequences):
            break

    best = max(
        sequences,
        key=lambda item: item[0] / max(len(item[1]) - 1, 1) ** length_normalisation,
    )
    return best[1][1:]
