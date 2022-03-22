import os
from random import shuffle
from typing import List, Tuple

import numpy as np
from cellx.augmentation.image import (
    augment_random_boundary,
    augment_random_flip,
    augment_random_rot90,
)
from cellx.augmentation.utils import append_conditional_augmentation
from cellx.tools.dataset import build_dataset, per_channel_normalize

from .config import ConfigBase
from .synthetic import create_synthetic_trajectory

VALIDATE_FRACTION = 0.1


def encoder_training_dataset(config: ConfigBase):
    """Encoder training dataset."""
    dataset = build_dataset(
        config.src_dir,
        output_shape=config.input_shape,
        output_dtype=config.input_dtype
    )
    dataset = dataset.shuffle(
        buffer_size=config.batch_size * 1000, reshuffle_each_iteration=True
    )
    dataset = append_conditional_augmentation(
        dataset,
        [augment_random_boundary],
        accept_probability=0.1,
    )
    dataset = dataset.map(augment_random_flip, num_parallel_calls=4)
    dataset = dataset.map(augment_random_rot90, num_parallel_calls=4)
    dataset = dataset.map(per_channel_normalize, num_parallel_calls=4)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(1)
    dataset = dataset.repeat()
    return dataset


def encoder_validation_dataset(config: ConfigBase, batch_size: int = 1):
    """Encoder validation dataset."""
    dataset = build_dataset(
        config.src_dir,
        output_shape=config.input_shape,
        output_dtype=config.input_dtype
    )
    dataset = dataset.shuffle(
        buffer_size=config.batch_size * 1000, reshuffle_each_iteration=True
    )
    dataset = dataset.map(per_channel_normalize, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    return dataset.as_numpy_iterator()


def trim_encoding(x: np.ndarray, max_len: int, cutoff: int) -> np.ndarray:
    """Trim an encoding to the correct dimensions.

    Parameters
    ----------
    x : array
        The encoding to be trimmed.
    max_len : int
        The number of timesteps.
    cutoff : int
        The position of the cutoff.

    Returns
    -------
    trimmed : array
        The per-channel normalized glimpse.
    """
    trimmed = x[: cutoff + 1, ...]
    trimmed = trimmed[-max_len:, ...]

    if trimmed.shape[0] < max_len:
        pad = max_len - trimmed.shape[0]
        trimmed = np.pad(trimmed, ((pad, 0), (0, 0), (0, 0)))
        trimmed[:pad, ..., 1] = 1  # var=1, mean=0 for pad

    assert trimmed.shape == (max_len, x.shape[1], x.shape[-1])

    return trimmed


class TauVAEDataset:
    """Dataset function for training the TauVAE.

    Parameters
    ----------
    config : config.ConfigBase
        A configuration.

    """

    def __init__(self, config: ConfigBase):
        self._config = config
        self._encoding_dir = config.src_dir / "encodings"
        self._encoded = {}
        self._validation = {}

        # determine the ground truth labels
        self._labels = [
            label
            for label in os.listdir(self._encoding_dir)
            if os.path.isdir(self._encoding_dir / label)
        ]

        # load all of the encodings
        for label in self._labels:
            self._encoded[label] = []
            files = [
                file
                for file in os.listdir(self._encoding_dir / label)
                if file.endswith(".npz")
            ]
            for file in files:
                data = np.load(self._encoding_dir / label / file)
                self._encoded[label].append(dict(data))

                # shuffle the dataset
                shuffle(self._encoded[label])

        # use a fraction of the data for validation
        n_validate = int(
            VALIDATE_FRACTION * min([len(d) for d in self._encoded.values()])
        )
        n_validate = max([1, n_validate])

        # split the data for a validation set
        for label in self._labels:
            self._validation[label] = [
                self._encoded[label].pop(0) for _ in range(n_validate)
            ]

        self._n_validate = n_validate

    @property
    def labels(self) -> List[str]:
        """Return a list of ground truth labels as stings."""
        return self._labels + ["synthetic"]

    def create_synthetic(self):
        """Create a synthetic data sample."""
        rnd_label = np.random.choice(self._labels)
        data = np.random.choice(self._encoded[rnd_label])["encoding"]
        idx = np.random.randint(data.shape[0])
        init_encoding = data[idx, ...]
        assert init_encoding.shape == (32, 2)
        encoding = create_synthetic_trajectory(init_encoding, noise=0.2)
        return encoding

    def batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return a batch of augmented data."""

        # choose the samples with uniform probability
        labels = np.random.choice(self.labels, size=self._config.batch_size)
        batch = []

        for label in labels:
            if label == "synthetic":
                encoding = self.create_synthetic()
            else:
                idx = np.random.randint(len(self._encoded[label]))
                data = self._encoded[label][idx]
                cutoff = max(0, data["cutoff"] - np.random.randint(0, 5))
                encoding = trim_encoding(data["encoding"], self._config.max_len, cutoff)

            # do random cropping augmentation to simulate short trajectories
            if np.random.random() < 0.5:
                rnd_crop = np.random.randint(1, encoding.shape[0] - 10)
                encoding[:rnd_crop, ..., 0] = 0.0
                encoding[:rnd_crop, ..., 1] = 1.0

            batch.append(encoding)

        batch = np.stack(batch, axis=0)
        numeric_labels = np.array([self.labels.index(label) for label in labels])

        return batch, numeric_labels

    def validation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the validation set."""

        # make a synthetic dataset
        synthetic = [self.create_synthetic() for _ in range(self._n_validate)]
        self._validation["synthetic"] = synthetic

        validation = []
        labels = []
        for numeric_label, label in enumerate(self.labels):
            for idx in range(self._n_validate):
                data = self._validation[label][idx]
                if label == "synthetic":
                    encoding = data[..., 0]
                else:
                    encoding = trim_encoding(
                        data["encoding"], self._config.max_len, data["cutoff"]
                    )
                    encoding = encoding[..., 0]

                validation.append(encoding)
                labels.append(numeric_label)

        validation = np.stack(validation, axis=0)

        return validation, np.array(labels)


def temporal_training_dataset(config: ConfigBase):
    """Temporal model training dataset."""

    dataset = TauVAEDataset(config)
    validation = dataset.validation()
    noise = config.noise if config.use_probabilistic_encoder else 0.0

    def generator(dataset, noise):
        def sampler(encoding, noise: float = 1.0):
            mean = encoding[..., 0]
            log_var = encoding[..., 1]
            epsilon = np.random.normal(size=mean.shape) * noise
            return mean + np.exp(0.5 * log_var) * epsilon

        while True:
            # get a batch and either sample from the encoding or provide the
            # latent encoding
            batch, labels = dataset.batch()
            encodings = sampler(batch, noise)

            yield encodings, labels

    return generator(dataset, noise=noise), validation
