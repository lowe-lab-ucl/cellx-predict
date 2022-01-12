import sys

import numpy as np
from cellx.callbacks import (
    tensorboard_confusion_matrix_callback,
    tensorboard_montage_callback,
)
from cellx.train import create_tensorboard_log_dir
from tensorflow import keras as K
from tqdm import tqdm

from .config import ConfigBase
from .models import _build_autoencoder, _build_encoder, _build_temporal

# TODO(arl): remove these hard-coded values for release
NUM_IMAGES = 1_030_766
MONTAGE_SAMPLES = 32
TEMPORAL_STEPS_PER_EPOCH = 100


def train_encoder(config: ConfigBase):
    """Train the encoder model."""
    from .dataset import encoder_training_dataset, encoder_validation_dataset

    # set up the model
    model = _build_autoencoder(config)
    # model.summary()

    # set up the datasets and augmentation
    train_dataset = encoder_training_dataset(config)
    validation_dataset = encoder_validation_dataset(config)

    # sample some images for the montage callback in tensorboard
    montage_dataset = np.concatenate(
        [next(validation_dataset) for i in range(MONTAGE_SAMPLES)], axis=0
    )

    # set up callbacks
    tensorboard_callback = K.callbacks.TensorBoard(
        log_dir=config.log_dir, write_graph=True
    )

    montage_callback = tensorboard_montage_callback(
        model, montage_dataset, config.log_dir
    )

    # train the model
    model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=NUM_IMAGES // config.batch_size,
        callbacks=[tensorboard_callback, montage_callback],
    )

    # save the model weights
    model_filename = config.model_dir / config.filename("weights")
    model.encoder.save_weights(model_filename.with_suffix(".h5"))

    decoder_filename = config.filename("weights").replace("encoder", "decoder")
    model_filename = config.model_dir / decoder_filename
    model.decoder.save_weights(model_filename.with_suffix(".h5"))


def train_projector(config: ConfigBase):
    """Train the projector model."""
    from sklearn.decomposition import PCA

    from .dataset import encoder_validation_dataset

    # set up the model
    model = _build_encoder(config)
    model.summary()

    # set up the datasets and augmentation
    projection_dataset = encoder_validation_dataset(config, batch_size=512)

    # encode the images
    encodings = []
    for x in tqdm(projection_dataset):
        z, _, _ = model.predict(x)
        encodings.append(z)

    encodings = np.concatenate(encodings, axis=0)

    # build the PCA solver, save the components
    pca = PCA(n_components=config.latent_dims)
    pca.fit(encodings)
    components_filename = config.model_dir / config.filename("components")
    np.savez(
        components_filename.with_suffix(".npz"),
        components=pca.components_.T, # NOTE(arl): the transform must be transposed
        mean=pca.mean_,
    )


def train_temporal(config: ConfigBase):
    """Train the temporal model."""

    from .dataset import temporal_training_dataset

    # set up the models
    model = _build_temporal(config)
    model.summary()

    # set up the datasets and augmentation
    train_dataset, validation_dataset = temporal_training_dataset(config)

    # set up callbacks
    tensorboard_callback = K.callbacks.TensorBoard(
        log_dir=config.log_dir, write_graph=True
    )

    confusion_callback = tensorboard_confusion_matrix_callback(
        model,
        validation_dataset[0],
        validation_dataset[1],
        config.log_dir,
        class_names=["apoptosis", "mitosis", "synthetic"],
        is_binary=False,
    )

    # train the model
    model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=TEMPORAL_STEPS_PER_EPOCH,
        callbacks=[tensorboard_callback, confusion_callback],
        validation_data=validation_dataset,
    )

    # save the model weights
    model_filename = config.model_dir / config.filename("weights")
    model.save_weights(model_filename.with_suffix(".h5"))


def train(config: ConfigBase):
    """Train the correct model."""
    # start by clearing the session, just in case
    K.backend.clear_session()

    # set up a log directory
    config.log_dir = create_tensorboard_log_dir(config.log_dir)

    # get the training function
    train_fn = getattr(sys.modules[__name__], f"train_{config.model.lower()}")
    train_fn(config)
