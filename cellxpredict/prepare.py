import os

import numpy as np
from skimage import io
from tqdm import tqdm

from .config import ConfigBase
from .models import _build_encoder


def normalize_glimpse(x: np.ndarray) -> np.ndarray:
    """Normalize a glimpse to zero mean, unit variance, for encoding.

    Parameters
    ----------
    x : array
        The glimpse to be normalized. Should be organized as (T, W, H, C)

    Returns
    -------
    normalize : array
        The per-channel normalized glimpse.
    """
    if not x.ndim == 4:
        raise ValueError("Input must be 4-dimensional, TWHC.")

    normalized = np.zeros_like(x).astype(np.float32)
    px = np.prod(x.shape[1:3])

    def a_std(d):
        return np.max([np.std(d), 1.0 / np.sqrt(px)])

    def nrm(d):
        return np.clip((d - np.mean(d)) / a_std(d), -4.0, 4.0)

    for dim in range(x.shape[-1]):
        for idx in range(x.shape[0]):  # TODO(arl): this should be vectorized
            normalized[idx, ..., dim] = nrm(x[idx, ..., dim])
    return normalized


def prepare_temporal(config: ConfigBase):
    """Encode glimpses for the temporal model using the trained encoder.

    Parameters
    ----------
    config : ConfigBase
        The configuration.
    """

    # set up the folder structure
    img_dir = config.src_dir / "data"
    meta_dir = config.src_dir / "metadata"
    encoding_dir = config.src_dir / "encodings"

    # set up the models
    model = _build_encoder(config)
    model.summary()

    labels = [label for label in os.listdir(img_dir) if os.path.isdir(img_dir / label)]

    rotations = [0, 1, 2, 3] if config.use_rotations else [0]

    if not labels:
        raise IOError(f"Could not find training data in {img_dir}")

    for label in labels:
        # find the raw image data
        files = [file for file in os.listdir(img_dir / label) if file.endswith(".tif")]

        # make the directory if it doesn't exist
        if not os.path.exists(encoding_dir / label):
            os.makedirs(encoding_dir / label)

        for file in tqdm(files, desc=f"Preparing {label}"):
            # load the image data and make the channel axis the last
            img_file = img_dir / label / file
            img = io.imread(str(img_file))  # need to be str, not sure why!
            img = np.swapaxes(img, 1, -1)
            img = normalize_glimpse(img)

            if img.ndim != 4:
                raise ValueError("Input must be 4-dimensional, TWHC.")

            # also get the cutoff value
            fn, _ = os.path.splitext(file)
            meta_file = f"{fn}.npz"
            data = np.load(meta_dir / label / meta_file)
            cutoff = data["cutoff"]

            for rotation in rotations:
                # rotate k times in the xy plane
                img = np.rot90(img, k=rotation, axes=(1, 2))

                # get the last frame of the sequence
                last_frame = img[-1, ...]
                assert last_frame.shape == config.input_shape

                z_mean, z_log_var, _ = model.predict(img)
                z = np.stack([z_mean, z_log_var], axis=-1)

                encoding = {
                    "label": label,
                    "class_label": labels.index(label),
                    "filename": file,
                    "encoding": z,
                    "cutoff": cutoff,
                    "rotation": rotation,
                    "last_frame": last_frame,
                }

                # now save out the encoding
                encoding_file = f"{fn}_k{rotation}.npz"
                np.savez(encoding_dir / label / encoding_file, **encoding)
