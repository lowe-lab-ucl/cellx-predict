import dataclasses
import os
from functools import partial
from pathlib import Path
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
from cellx.utils import CallableEnum

from .config import ConfigBase
from .prepare import normalize_glimpse
from .synthetic import create_synthetic_trajectory

VALIDATE_FRACTION = 0.1


class OutputType(CallableEnum):
    """Enumerated class to handle different outputs of the `TauVAE` model.

    Options are:
        OutputType.LABEL
            A numeric label representing the state to be predicted.
        OutputType.ENCODING
            An encoding representing the state to be predicted.
        OutputType.IMAGE
            An image representing the state to be predicted.
    """

    LABEL = partial(lambda x: x.class_label)
    ENCODING = partial(lambda x: x.encoding[-1, ..., 0])
    IMAGE = partial(lambda x: x.last_frame)


def encoder_training_dataset(config: ConfigBase):
    """Encoder training dataset."""
    dataset = build_dataset(
        config.src_dir, output_shape=config.input_shape, output_dtype=config.input_dtype
    )
    dataset = dataset.shuffle(
        buffer_size=config.batch_size * 1000, reshuffle_each_iteration=True
    )

    if config.augment_boundary:
        dataset = append_conditional_augmentation(
            dataset,
            [augment_random_boundary],
            accept_probability=0.1,
        )

    if config.augment_flip:
        dataset = dataset.map(augment_random_flip, num_parallel_calls=4)
        dataset = dataset.map(augment_random_rot90, num_parallel_calls=4)

    if config.augment_normalize:
        dataset = dataset.map(per_channel_normalize, num_parallel_calls=4)

    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(1)
    dataset = dataset.repeat()
    return dataset


def encoder_validation_dataset(config: ConfigBase, batch_size: int = 1):
    """Encoder validation dataset."""
    dataset = build_dataset(
        config.src_dir, output_shape=config.input_shape, output_dtype=config.input_dtype
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


@dataclasses.dataclass
class Encoding:
    """A dataclass to encapsulate a glimpse encoding from the probabilistic
    encoder.

    Parameters
    ----------
    label : str
        A string label representing the class of the encoding.
    class_label : int
        A numeric label representing the class of the encoding.
    filename : Path
        The filename of the original data.
    encoding : np.array, (N, D, 2)
        The encoding of the glimpse, with N timesteps and D latent dimensions.
        The last two dimensions represent the mean and variance of the encoding.
    cutoff : int
        The index into the first dimension to specify the cutoff.
    rotation : int
        Integer number of rotations applied, e.g. k=2.
    last_frame : np.array
        A numpy array representing the last frame of the glimpse. Ignores the
        cutoff.
    """

    label: str
    class_label: int
    filename: Path
    encoding: np.array
    cutoff: int
    rotation: int
    last_frame: np.array

    @staticmethod
    def load(filename: os.PathLike) -> "Encoding":
        """Load and return an `Encoding` dataclass from the numpy file."""
        fields = dataclasses.fields(Encoding)
        with np.load(filename) as np_data:
            data = {k.name: None for k in fields if k in np_data.keys()}
            for field in fields:
                try:
                    value = field.type(np_data[field.name])
                except TypeError:
                    value = str(np_data[field.name])
                    value = field.type(value)
                data[field.name] = value

        # normalize the final frame of the movie
        data["last_frame"] = normalize_glimpse(
            data["last_frame"][np.newaxis, ...].astype(np.float32)
        )[0, ...]

        return Encoding(**data)

    def trim(self, max_len: int) -> "Encoding":
        """Trim the encoding according to the cutoff."""
        return trim_encoding(self.encoding, max_len, self.cutoff)


def shuffle_batch(batch: list, labels: list):
    """Shuffle a batch."""
    batch = np.stack(batch, axis=0)
    labels = np.stack(labels, axis=0)

    # shuffle all of the examples
    idx = list(range(batch.shape[0]))
    shuffle(idx)
    idx = tuple(idx)

    batch = batch[idx, ...]
    labels = labels[idx, ...]

    assert batch.shape[0] == labels.shape[0]
    return batch, labels


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
        self._output_type = OutputType[config.output_type.upper()]

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
                # data = np.load(self._encoding_dir / label / file)
                data = Encoding.load(self._encoding_dir / label / file)
                assert data.class_label == self._labels.index(label)
                self._encoded[label].append(data)

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

        # if the output type is `IMAGE`, don't try to generate synthetic data
        if self._output_type == OutputType.IMAGE:
            return self._labels

        # for all other types, include the synthetic data
        # NOTE(arl): we may want to make this use the `--use_synthetic` flag
        return self._labels + ["synthetic"]

    def create_synthetic(self) -> Encoding:
        """Create a synthetic data sample."""

        rnd_label = np.random.choice(self._labels)
        data = np.random.choice(self._encoded[rnd_label]).encoding
        idx = np.random.randint(data.shape[0])
        init_encoding = data[idx, ...]
        assert init_encoding.shape == (32, 2)
        encoding = create_synthetic_trajectory(
            init_encoding, length=self._config.max_len + 10, noise=0.2
        )

        synthetic = Encoding(
            label="synthetic",
            class_label=self.labels.index("synthetic"),
            filename="",
            encoding=encoding,
            cutoff=self._config.max_len,
            rotation=0,
            last_frame=np.zeros(self._config.input_shape, dtype=np.uint8),
        )

        return synthetic

    def batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return a batch of augmented data."""

        # choose the samples with uniform probability
        labels = np.random.choice(self.labels, size=self._config.batch_size)
        numeric_labels = []
        batch = []

        for label in labels:
            if label == "synthetic":
                data = self.create_synthetic()
                cutoff = data.cutoff
            else:
                idx = np.random.randint(len(self._encoded[label]))
                data = self._encoded[label][idx]
                cutoff = max(0, data.cutoff - np.random.randint(0, 5))

            encoding = trim_encoding(data.encoding, self._config.max_len, cutoff)

            # do random cropping augmentation to simulate short trajectories
            if np.random.random() < 0.5:
                rnd_crop = np.random.randint(1, encoding.shape[0] - 10)
                encoding[:rnd_crop, ..., 0] = 0.0
                encoding[:rnd_crop, ..., 1] = 1.0

            batch.append(encoding)
            numeric_labels.append(self._output_type(data))

        batch, numeric_labels = shuffle_batch(batch, numeric_labels)

        return batch, numeric_labels

    def validation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the validation set."""

        # make a synthetic dataset
        if "synthetic" in self.labels:
            synthetic = [self.create_synthetic() for _ in range(self._n_validate)]
            self._validation["synthetic"] = synthetic

        validation = []
        numeric_labels = []

        for label in self.labels:
            for idx in range(self._n_validate):
                data = self._validation[label][idx]
                encoding = trim_encoding(
                    data.encoding, self._config.max_len, data.cutoff
                )
                encoding = encoding[..., 0]

                validation.append(encoding)
                numeric_label = self._output_type(data)
                numeric_labels.append(numeric_label)

        validation, numeric_labels = shuffle_batch(validation, numeric_labels)

        return validation, numeric_labels


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
